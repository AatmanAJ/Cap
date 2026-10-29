use cidre::{arc, blocks, ns, sc};
use objc::runtime::Object;

const APP_ICON_SIZE: f64 = 64.0;
const NS_BITMAP_IMAGE_FILE_TYPE_PNG: u64 = 4;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudioApp {
    pub bundle_id: String,
    pub name: String,
    pub icon_png: Option<Vec<u8>>,
}

pub fn app_audio_supported() -> bool {
    scap_screencapturekit::is_system_audio_supported()
}

pub fn list_audio_apps() -> Vec<AudioApp> {
    use cocoa::{
        appkit::{NSApplicationActivationPolicy, NSApplicationActivationPolicyRegular},
        base::id,
        foundation::NSArray,
    };
    use objc::{class, msg_send, rc::autoreleasepool, sel, sel_impl};

    let own_pid = std::process::id() as i32;

    let apps = autoreleasepool(|| unsafe {
        let workspace: id = msg_send![class!(NSWorkspace), sharedWorkspace];
        let running: id = msg_send![workspace, runningApplications];
        if running.is_null() {
            return Vec::new();
        }

        (0..NSArray::count(running))
            .filter_map(|index| {
                let app: id = running.objectAtIndex(index);
                let policy: NSApplicationActivationPolicy = msg_send![app, activationPolicy];
                let pid: i32 = msg_send![app, processIdentifier];
                if policy != NSApplicationActivationPolicyRegular || pid == own_pid {
                    return None;
                }

                let bundle_id = ns_string_to_string(msg_send![app, bundleIdentifier])?;
                let name = ns_string_to_string(msg_send![app, localizedName])
                    .unwrap_or_else(|| bundle_id.clone());
                let icon_png = image_png(msg_send![app, icon]);

                Some(AudioApp {
                    bundle_id,
                    name,
                    icon_png,
                })
            })
            .collect::<Vec<_>>()
    });

    sorted_unique_apps(apps)
}

fn sorted_unique_apps(mut apps: Vec<AudioApp>) -> Vec<AudioApp> {
    apps.retain(|app| !app.bundle_id.is_empty());
    apps.sort_by(|a, b| {
        a.name
            .to_lowercase()
            .cmp(&b.name.to_lowercase())
            .then_with(|| a.bundle_id.cmp(&b.bundle_id))
    });
    let mut seen = std::collections::HashSet::new();
    apps.retain(|app| seen.insert(app.bundle_id.to_lowercase()));
    apps
}

unsafe fn ns_string_to_string(value: cocoa::base::id) -> Option<String> {
    use cocoa::foundation::NSString;

    if value.is_null() {
        return None;
    }
    let utf8 = unsafe { NSString::UTF8String(value) };
    if utf8.is_null() {
        return None;
    }
    let value = unsafe { std::ffi::CStr::from_ptr(utf8) }
        .to_string_lossy()
        .into_owned();
    (!value.is_empty()).then_some(value)
}

unsafe fn image_png(image: cocoa::base::id) -> Option<Vec<u8>> {
    use cocoa::{
        base::{id, nil},
        foundation::{NSPoint, NSRect, NSSize},
    };
    use objc::{class, msg_send, sel, sel_impl};

    if image.is_null() {
        return None;
    }

    unsafe {
        let mut bounds = NSRect::new(
            NSPoint::new(0.0, 0.0),
            NSSize::new(APP_ICON_SIZE, APP_ICON_SIZE),
        );
        let cg_image: *mut std::ffi::c_void = msg_send![
            image,
            CGImageForProposedRect: &mut bounds
            context: nil
            hints: nil
        ];
        if cg_image.is_null() {
            return None;
        }

        let bitmap: id = msg_send![class!(NSBitmapImageRep), alloc];
        let bitmap: id = msg_send![bitmap, initWithCGImage: cg_image];
        if bitmap.is_null() {
            return None;
        }
        let bitmap: id = msg_send![bitmap, autorelease];

        let png: id = msg_send![
            bitmap,
            representationUsingType: NS_BITMAP_IMAGE_FILE_TYPE_PNG
            properties: nil
        ];
        if png.is_null() {
            return None;
        }

        let length: usize = msg_send![png, length];
        let bytes: *const u8 = msg_send![png, bytes];
        if bytes.is_null() || length == 0 {
            return None;
        }
        Some(std::slice::from_raw_parts(bytes, length).to_vec())
    }
}

/// Helpers are not matched: ScreenCaptureKit only lists apps that own a
/// window, so windowless helpers such as `com.apple.WebKit.GPU` (which plays
/// Safari's web audio) can't be added to the filter at all.
pub fn bundle_id_selected(bundle_id: &str, selected: &[String]) -> bool {
    !bundle_id.is_empty() && selected.iter().any(|id| id.eq_ignore_ascii_case(bundle_id))
}

pub fn normalized_pids(pids: impl IntoIterator<Item = i32>) -> Vec<i32> {
    let mut pids = pids.into_iter().collect::<Vec<_>>();
    pids.sort_unstable();
    pids.dedup();
    pids
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AppAudioStep {
    Keep,
    Stop,
    Start,
    Update,
}

/// No stream exists while none of the selected apps runs, so a filter is never
/// built from an empty app list. A stream that has not started yet, or was
/// stopped by the system, is rebuilt rather than updated in place.
pub fn next_step(
    active_pids: Option<&[i32]>,
    pids: &[i32],
    started: bool,
    rebuild: bool,
) -> AppAudioStep {
    match active_pids {
        Some(active) if !rebuild && active == pids => AppAudioStep::Keep,
        Some(_) if pids.is_empty() => AppAudioStep::Stop,
        None if pids.is_empty() => AppAudioStep::Keep,
        Some(_) if started && !rebuild => AppAudioStep::Update,
        _ => AppAudioStep::Start,
    }
}

pub fn selected_running_apps(
    content: &sc::ShareableContent,
    selected: &[String],
) -> Vec<arc::R<sc::RunningApp>> {
    content
        .apps()
        .iter()
        .filter(|app| bundle_id_selected(&app.bundle_id().to_string(), selected))
        .map(|app| app.retained())
        .collect()
}

/// Shareable content that counts off-screen windows too, so an app playing
/// audio while it is minimized, hidden or on another Space is still listed.
pub fn shareable_content_with_offscreen_apps()
-> impl Future<Output = Result<arc::R<sc::ShareableContent>, arc::R<ns::Error>>> + Send + 'static {
    use objc::{
        class, msg_send,
        runtime::{NO, YES},
        sel, sel_impl,
    };

    let (future, mut block) = blocks::result::<sc::ShareableContent>();
    let block: *mut blocks::ResultCh<sc::ShareableContent> = &mut *block;
    unsafe {
        let _: () = msg_send![
            class!(SCShareableContent),
            getShareableContentExcludingDesktopWindows: YES
            onScreenWindowsOnly: NO
            completionHandler: block.cast::<std::ffi::c_void>()
        ];
    }
    future
}

fn object_ptr<T>(value: &T) -> *mut Object {
    std::ptr::from_ref(value).cast_mut().cast()
}

/// `initWithDisplay:includingApplications:exceptingWindows:`, which cidre does
/// not bind. A stream with this filter delivers only these apps' audio.
pub fn display_including_apps_filter(
    display: &sc::Display,
    apps: &[arc::R<sc::RunningApp>],
) -> Option<arc::R<sc::ContentFilter>> {
    use objc::{class, msg_send, sel, sel_impl};

    let apps = ns::Array::from_slice_retained(apps);
    let windows = ns::Array::<sc::Window>::new();
    unsafe {
        let filter: *mut Object = msg_send![class!(SCContentFilter), alloc];
        if filter.is_null() {
            return None;
        }
        let filter: *mut Object = msg_send![
            filter,
            initWithDisplay: object_ptr(display)
            includingApplications: object_ptr(apps.as_ref())
            exceptingWindows: object_ptr(windows.as_ref())
        ];
        if filter.is_null() {
            return None;
        }
        Some(std::mem::transmute::<*mut Object, arc::R<sc::ContentFilter>>(filter))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn selection(ids: &[&str]) -> Vec<String> {
        ids.iter().map(|id| id.to_string()).collect()
    }

    #[test]
    fn selected_app_matches_exactly_and_ignoring_case() {
        let selected = selection(&["com.apple.Music"]);
        assert!(bundle_id_selected("com.apple.Music", &selected));
        assert!(bundle_id_selected("com.apple.music", &selected));
        assert!(!bundle_id_selected("com.apple.Safari", &selected));
    }

    #[test]
    fn only_the_selected_bundle_matches() {
        let selected = selection(&["com.google.Chrome"]);
        assert!(!bundle_id_selected("com.google.Chrome.helper", &selected));
        assert!(!bundle_id_selected("com.google.ChromeBeta", &selected));
        assert!(!bundle_id_selected("com.google", &selected));
    }

    #[test]
    fn no_stream_exists_until_a_selected_app_runs() {
        assert_eq!(next_step(None, &[], false, false), AppAudioStep::Keep);
        assert_eq!(next_step(None, &[], true, false), AppAudioStep::Keep);
        assert_eq!(next_step(None, &[42], false, false), AppAudioStep::Start);
        assert_eq!(next_step(None, &[42], true, false), AppAudioStep::Start);
    }

    #[test]
    fn a_running_stream_follows_launches_and_quits() {
        assert_eq!(
            next_step(Some(&[42]), &[42], true, false),
            AppAudioStep::Keep
        );
        assert_eq!(
            next_step(Some(&[42]), &[42, 77], true, false),
            AppAudioStep::Update
        );
        assert_eq!(
            next_step(Some(&[42]), &[77], true, false),
            AppAudioStep::Update
        );
        assert_eq!(next_step(Some(&[42]), &[], true, false), AppAudioStep::Stop);
    }

    #[test]
    fn an_unstarted_or_stopped_stream_is_rebuilt() {
        assert_eq!(
            next_step(Some(&[42]), &[42, 77], false, false),
            AppAudioStep::Start
        );
        assert_eq!(
            next_step(Some(&[42]), &[42], true, true),
            AppAudioStep::Start
        );
        assert_eq!(next_step(Some(&[42]), &[], true, true), AppAudioStep::Stop);
        assert_eq!(next_step(None, &[], true, true), AppAudioStep::Keep);
    }

    #[test]
    fn empty_ids_never_match() {
        assert!(!bundle_id_selected("", &selection(&["com.apple.Music"])));
        assert!(!bundle_id_selected("com.apple.Music", &selection(&[""])));
        assert!(!bundle_id_selected("com.apple.Music", &[]));
    }

    fn app(bundle_id: &str, name: &str) -> AudioApp {
        AudioApp {
            bundle_id: bundle_id.to_string(),
            name: name.to_string(),
            icon_png: None,
        }
    }

    #[test]
    fn app_list_is_sorted_by_name_with_one_entry_per_bundle() {
        let apps = sorted_unique_apps(vec![
            app("com.apple.Safari", "Safari"),
            app("com.apple.Music", "Music"),
            app("", "Unbundled"),
            app("com.spotify.client", "spotify"),
            app("com.apple.Safari", "Safari"),
            app("com.apple.music", "Music"),
        ]);

        assert_eq!(
            apps.iter()
                .map(|app| app.bundle_id.as_str())
                .collect::<Vec<_>>(),
            vec!["com.apple.Music", "com.apple.Safari", "com.spotify.client"]
        );
    }

    #[test]
    fn pid_sets_compare_regardless_of_order_and_duplicates() {
        assert_eq!(normalized_pids([42, 7, 42, 9]), vec![7, 9, 42]);
        assert_eq!(normalized_pids([9, 7]), normalized_pids([7, 9]));
        assert_ne!(normalized_pids([7, 9]), normalized_pids([7, 10]));
        assert!(normalized_pids([]).is_empty());
    }
}
