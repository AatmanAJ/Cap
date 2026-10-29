use super::{
    KeyPressDisplay, KeyPressEvent, KeyboardEvents, KeyboardTrackSegment, is_modifier_key,
    special_key_symbol,
};

const CONTROL_KEYS: &[&str] = &["LControl", "RControl"];
const ALT_KEYS: &[&str] = &["LAlt", "RAlt", "LOption", "ROption"];
const SHIFT_KEYS: &[&str] = &["LShift", "RShift"];
const META_KEYS: &[&str] = &["LMeta", "RMeta", "Meta", "Command", "RCommand"];

const STANDALONE_KEYS: &[&str] = &[
    "Escape", "Enter", "Return", "Tab", "Up", "Down", "Left", "Right", "Home", "End", "PageUp",
    "PageDown",
];

const MAC_MODIFIER_SYMBOLS: &[&str] = &["⌃", "⌥", "⇧", "⌘"];
const MODIFIER_WORDS: &[&str] = &["Win+", "Super+", "Ctrl+", "Alt+", "Shift+"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyLabelPlatform {
    Mac,
    Windows,
    Linux,
}

impl KeyLabelPlatform {
    pub fn from_recording(platform: Option<&crate::Platform>) -> Self {
        match platform.cloned().unwrap_or_default() {
            crate::Platform::MacOS => Self::Mac,
            crate::Platform::Windows => Self::Windows,
            crate::Platform::Linux => Self::Linux,
        }
    }
}

fn is_function_key(key: &str) -> bool {
    key.strip_prefix('F')
        .and_then(|number| number.parse::<u8>().ok())
        .is_some_and(|number| (1..=24).contains(&number))
}

fn is_standalone_shortcut_key(key: &str) -> bool {
    STANDALONE_KEYS.contains(&key) || is_function_key(key)
}

fn key_label(key: &str, platform: KeyLabelPlatform) -> String {
    if is_function_key(key) || key == "Space" {
        return key.to_string();
    }
    if platform != KeyLabelPlatform::Mac {
        let word = match key {
            "Escape" => Some("Esc"),
            "Enter" | "Return" => Some("Enter"),
            "Tab" => Some("Tab"),
            "Backspace" => Some("Backspace"),
            "Delete" => Some("Del"),
            "Home" => Some("Home"),
            "End" => Some("End"),
            "PageUp" => Some("PgUp"),
            "PageDown" => Some("PgDn"),
            _ => None,
        };
        if let Some(word) = word {
            return word.to_string();
        }
    }
    if let Some(symbol) = special_key_symbol(key) {
        return symbol.to_string();
    }
    key.to_uppercase()
}

fn shortcut_label(
    held: &[&str],
    key: &str,
    show_special_keys: bool,
    platform: KeyLabelPlatform,
) -> Option<String> {
    let has = |names: &[&str]| held.iter().any(|held| names.contains(held));
    // Windows and most Linux layouts type characters such as @ or € with
    // AltGr, which reports as RAlt (plus a synthetic LControl on Windows), so
    // treating it as a shortcut would leak typed text.
    let alt_graph = platform != KeyLabelPlatform::Mac && held.contains(&"RAlt");
    let control = !alt_graph && has(CONTROL_KEYS);
    let alt = !alt_graph && has(ALT_KEYS);
    let shift = has(SHIFT_KEYS);
    let meta = has(META_KEYS);

    let shown_alone = show_special_keys && is_standalone_shortcut_key(key);
    if !(control || alt || meta || shown_alone) {
        return None;
    }

    let mut label = String::new();
    match platform {
        KeyLabelPlatform::Mac => {
            for (on, symbol) in [(control, "⌃"), (alt, "⌥"), (shift, "⇧"), (meta, "⌘")] {
                if on {
                    label.push_str(symbol);
                }
            }
        }
        KeyLabelPlatform::Windows | KeyLabelPlatform::Linux => {
            let meta_name = if platform == KeyLabelPlatform::Windows {
                "Win"
            } else {
                "Super"
            };
            for (on, name) in [
                (meta, meta_name),
                (control, "Ctrl"),
                (alt, "Alt"),
                (shift, "Shift"),
            ] {
                if on {
                    label.push_str(name);
                    label.push('+');
                }
            }
        }
    }
    label.push_str(&key_label(key, platform));
    Some(label)
}

pub fn group_shortcut_events(
    events: &KeyboardEvents,
    linger_duration_ms: f64,
    show_special_keys: bool,
    platform: KeyLabelPlatform,
) -> Vec<KeyboardTrackSegment> {
    let mut ordered: Vec<&KeyPressEvent> = events
        .presses
        .iter()
        .filter(|event| event.time_ms.is_finite())
        .collect();
    ordered.sort_by(|a, b| a.time_ms.total_cmp(&b.time_ms));

    let mut held: Vec<&str> = Vec::new();
    let mut pressed: Vec<(&str, usize)> = Vec::new();
    let mut segments: Vec<KeyboardTrackSegment> = Vec::new();
    let mut first = 0;
    while first < ordered.len() {
        let time_ms = ordered[first].time_ms;
        let last = first + ordered[first..].partition_point(|event| event.time_ms <= time_ms);
        let batch = &ordered[first..last];
        first = last;

        // One capture poll can report a modifier and the key it modifies at
        // the same timestamp in either order, so every modifier change in the
        // batch applies before its keys are labelled.
        for event in batch.iter().filter(|event| is_modifier_key(&event.key)) {
            if event.down {
                if !held.contains(&event.key.as_str()) {
                    held.push(&event.key);
                }
            } else {
                held.retain(|key| *key != event.key);
            }
        }

        for event in batch.iter().filter(|event| !is_modifier_key(&event.key)) {
            if !event.down {
                if let Some(position) = pressed.iter().position(|(key, _)| *key == event.key) {
                    let (_, index) = pressed.remove(position);
                    let segment = &mut segments[index];
                    segment.end = segment.end.max((time_ms + linger_duration_ms) / 1000.0);
                }
                continue;
            }
            pressed.retain(|(key, _)| *key != event.key);
            let Some(display_text) = shortcut_label(&held, &event.key, show_special_keys, platform)
            else {
                continue;
            };
            segments.push(KeyboardTrackSegment {
                id: format!("kb-{}", segments.len() + 1),
                start: time_ms / 1000.0,
                end: (time_ms + linger_duration_ms) / 1000.0,
                display_text,
                keys: vec![KeyPressDisplay {
                    key: event.key.clone(),
                    time_offset: 0.0,
                }],
                fade_duration_override: None,
                position_override: None,
                color_override: None,
                background_color_override: None,
                font_size_override: None,
                uppercase_override: None,
            });
            pressed.push((&event.key, segments.len() - 1));
        }
    }
    segments
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeycapLabel<'a> {
    pub text: &'a str,
    pub modifier: bool,
}

pub fn keycap_labels(display_text: &str) -> Vec<KeycapLabel<'_>> {
    let mut labels = Vec::new();
    let mut rest = display_text;
    loop {
        if let Some(symbol) = MAC_MODIFIER_SYMBOLS
            .iter()
            .find(|symbol| rest.starts_with(**symbol))
        {
            labels.push(KeycapLabel {
                text: &rest[..symbol.len()],
                modifier: true,
            });
            rest = &rest[symbol.len()..];
        } else if let Some(word) = MODIFIER_WORDS
            .iter()
            .find(|word| rest.len() > word.len() && rest.starts_with(**word))
        {
            labels.push(KeycapLabel {
                text: &rest[..word.len() - 1],
                modifier: true,
            });
            rest = &rest[word.len()..];
        } else {
            break;
        }
    }
    if !rest.is_empty() {
        labels.push(KeycapLabel {
            text: rest,
            modifier: false,
        });
    }
    labels
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(key: &str, time_ms: f64, down: bool) -> KeyPressEvent {
        KeyPressEvent {
            key: key.to_string(),
            key_code: key.to_string(),
            time_ms,
            down,
        }
    }

    fn tap(presses: &mut Vec<KeyPressEvent>, key: &str, time_ms: f64) {
        presses.push(event(key, time_ms, true));
        presses.push(event(key, time_ms + 40.0, false));
    }

    fn labels(presses: Vec<KeyPressEvent>, platform: KeyLabelPlatform) -> Vec<String> {
        group_shortcut_events(&KeyboardEvents { presses }, 800.0, true, platform)
            .into_iter()
            .map(|segment| segment.display_text)
            .collect()
    }

    #[test]
    fn ordinary_typing_and_shift_capitals_are_hidden() {
        let mut presses = Vec::new();
        for (index, key) in [
            "h",
            "u",
            "n",
            "t",
            "e",
            "r",
            "2",
            "Space",
            "Backspace",
            "Delete",
        ]
        .iter()
        .enumerate()
        {
            tap(&mut presses, key, index as f64 * 100.0);
        }
        presses.push(event("LShift", 1_000.0, true));
        tap(&mut presses, "p", 1_010.0);
        tap(&mut presses, "1", 1_100.0);
        tap(&mut presses, "Backspace", 1_200.0);
        presses.push(event("LShift", 1_300.0, false));
        presses.push(event("CapsLock", 1_400.0, true));

        assert!(labels(presses, KeyLabelPlatform::Mac).is_empty());
    }

    #[test]
    fn modifier_combos_use_apple_order_on_mac() {
        let mut presses = Vec::new();
        for modifier in ["Meta", "LShift", "LAlt", "LControl"] {
            presses.push(event(modifier, 0.0, true));
        }
        tap(&mut presses, "t", 10.0);
        for modifier in ["Meta", "LShift", "LAlt", "LControl"] {
            presses.push(event(modifier, 100.0, false));
        }
        presses.push(event("RCommand", 200.0, true));
        presses.push(event("LShift", 200.0, true));
        tap(&mut presses, "[", 210.0);

        assert_eq!(labels(presses, KeyLabelPlatform::Mac), ["⌃⌥⇧⌘T", "⇧⌘["]);
    }

    #[test]
    fn modifier_combos_use_words_on_windows_and_linux() {
        let mut presses = Vec::new();
        for modifier in ["LShift", "Meta", "LControl"] {
            presses.push(event(modifier, 0.0, true));
        }
        tap(&mut presses, "s", 10.0);
        presses.push(event("LShift", 50.0, false));
        presses.push(event("Meta", 50.0, false));
        tap(&mut presses, "+", 60.0);
        presses.push(event("LControl", 100.0, false));
        presses.push(event("LAlt", 200.0, true));
        tap(&mut presses, "F4", 210.0);

        assert_eq!(
            labels(presses.clone(), KeyLabelPlatform::Windows),
            ["Win+Ctrl+Shift+S", "Ctrl++", "Alt+F4"]
        );
        assert_eq!(
            labels(presses, KeyLabelPlatform::Linux),
            ["Super+Ctrl+Shift+S", "Ctrl++", "Alt+F4"]
        );
    }

    #[test]
    fn standalone_special_keys_and_shift_navigation_are_kept() {
        let mut presses = Vec::new();
        for (index, key) in ["Escape", "Enter", "Tab", "Up", "PageDown", "F5", "F13"]
            .iter()
            .enumerate()
        {
            tap(&mut presses, key, index as f64 * 100.0);
        }
        presses.push(event("RShift", 1_000.0, true));
        tap(&mut presses, "Tab", 1_010.0);
        tap(&mut presses, "Right", 1_100.0);
        presses.push(event("RShift", 1_200.0, false));

        assert_eq!(
            labels(presses.clone(), KeyLabelPlatform::Mac),
            ["⎋", "⏎", "⇥", "↑", "⇟", "F5", "F13", "⇧⇥", "⇧→"]
        );
        assert_eq!(
            labels(presses.clone(), KeyLabelPlatform::Windows),
            [
                "Esc",
                "Enter",
                "Tab",
                "↑",
                "PgDn",
                "F5",
                "F13",
                "Shift+Tab",
                "Shift+→"
            ]
        );
        assert!(
            group_shortcut_events(
                &KeyboardEvents { presses },
                800.0,
                false,
                KeyLabelPlatform::Mac
            )
            .is_empty()
        );
    }

    #[test]
    fn hiding_special_keys_keeps_modifier_combos_with_them() {
        let presses = vec![
            event("Meta", 0.0, true),
            event("Enter", 10.0, true),
            event("Enter", 50.0, false),
            event("Meta", 60.0, false),
            event("Enter", 100.0, true),
        ];
        let segments = group_shortcut_events(
            &KeyboardEvents { presses },
            800.0,
            false,
            KeyLabelPlatform::Mac,
        );
        assert_eq!(segments.len(), 1);
        assert_eq!(segments[0].display_text, "⌘⏎");
    }

    #[test]
    fn alt_graph_typing_is_hidden_off_mac() {
        let presses = vec![
            event("LControl", 0.0, true),
            event("RAlt", 0.0, true),
            event("q", 10.0, true),
            event("q", 50.0, false),
            event("LControl", 60.0, false),
            event("RAlt", 60.0, false),
            event("RAlt", 100.0, true),
            event("Meta", 100.0, true),
            event("e", 110.0, true),
        ];
        assert_eq!(
            labels(presses.clone(), KeyLabelPlatform::Windows),
            ["Win+E"]
        );
        assert_eq!(
            labels(presses.clone(), KeyLabelPlatform::Linux),
            ["Super+E"]
        );
        assert_eq!(labels(presses, KeyLabelPlatform::Mac), ["⌃⌥Q", "⌥⌘E"]);
    }

    #[test]
    fn modifier_state_at_one_timestamp_applies_before_its_keys() {
        let presses = vec![
            event("t", 0.0, true),
            event("Meta", 0.0, true),
            event("t", 30.0, false),
            event("w", 100.0, true),
            event("Meta", 100.0, false),
        ];
        assert_eq!(labels(presses, KeyLabelPlatform::Mac), ["⌘T"]);
    }

    #[test]
    fn modifiers_alone_do_not_create_segments() {
        let presses = vec![
            event("Meta", 0.0, true),
            event("Meta", 50.0, false),
            event("LControl", 100.0, true),
            event("LShift", 110.0, true),
            event("LShift", 150.0, false),
            event("LControl", 160.0, false),
        ];
        assert!(labels(presses, KeyLabelPlatform::Mac).is_empty());
    }

    #[test]
    fn held_shortcut_lingers_after_its_key_is_released() {
        let presses = vec![
            event("Meta", 0.0, true),
            event("Tab", 100.0, true),
            event("Tab", 1_600.0, false),
            event("Meta", 1_700.0, false),
            event("Meta", 3_000.0, true),
            event("z", 3_000.0, true),
            event("Meta", 3_020.0, false),
            event("z", 3_030.0, false),
            event("Meta", 4_000.0, true),
            event("c", 4_100.0, true),
        ];
        let segments = group_shortcut_events(
            &KeyboardEvents { presses },
            800.0,
            true,
            KeyLabelPlatform::Mac,
        );
        let timing: Vec<_> = segments
            .iter()
            .map(|segment| (segment.display_text.as_str(), segment.start, segment.end))
            .collect();
        assert_eq!(
            timing,
            [("⌘⇥", 0.1, 2.4), ("⌘Z", 3.0, 3.83), ("⌘C", 4.1, 4.9)]
        );
    }

    #[test]
    fn repeated_presses_stay_separate_segments() {
        let mut presses = vec![event("Meta", 0.0, true)];
        for index in 0..3 {
            tap(&mut presses, "z", 100.0 + index as f64 * 150.0);
        }
        let segments = group_shortcut_events(
            &KeyboardEvents { presses },
            800.0,
            true,
            KeyLabelPlatform::Mac,
        );
        let ids: Vec<_> = segments.iter().map(|segment| segment.id.as_str()).collect();
        assert_eq!(ids, ["kb-1", "kb-2", "kb-3"]);
        assert!(segments.iter().all(|segment| segment.display_text == "⌘Z"));
        assert!(segments.windows(2).all(|pair| pair[0].end > pair[1].start));
    }

    #[test]
    fn unordered_and_non_finite_events_are_sorted_or_dropped() {
        let presses = vec![
            event("c", 120.0, true),
            event("x", f64::NAN, true),
            event("Meta", 100.0, true),
            event("c", 160.0, false),
        ];
        assert_eq!(labels(presses, KeyLabelPlatform::Mac), ["⌘C"]);
    }

    #[test]
    fn keycap_labels_split_modifiers_from_the_key() {
        let texts = |display: &str| -> Vec<String> {
            keycap_labels(display)
                .iter()
                .map(|label| label.text.to_string())
                .collect()
        };
        assert_eq!(texts("⌃⌥⇧⌘T"), ["⌃", "⌥", "⇧", "⌘", "T"]);
        assert_eq!(texts("⇧hi"), ["⇧", "hi"]);
        assert_eq!(texts("Win+Ctrl+Shift+S"), ["Win", "Ctrl", "Shift", "S"]);
        assert_eq!(texts("Ctrl++"), ["Ctrl", "+"]);
        assert_eq!(texts("Super+Alt+F4"), ["Super", "Alt", "F4"]);
        assert_eq!(texts("hello"), ["hello"]);
        assert_eq!(texts("⇧"), ["⇧"]);
        assert_eq!(texts("Shift+"), ["Shift+"]);
        assert!(keycap_labels("").is_empty());

        let modifiers: Vec<bool> = keycap_labels("Ctrl+Shift+Tab")
            .iter()
            .map(|label| label.modifier)
            .collect();
        assert_eq!(modifiers, [true, true, false]);
        assert!(keycap_labels("⌘").iter().all(|label| label.modifier));
        assert!(!keycap_labels("Shift+")[0].modifier);
    }

    #[test]
    fn platform_follows_the_recording() {
        assert_eq!(
            KeyLabelPlatform::from_recording(Some(&crate::Platform::Windows)),
            KeyLabelPlatform::Windows
        );
        assert_eq!(
            KeyLabelPlatform::from_recording(Some(&crate::Platform::Linux)),
            KeyLabelPlatform::Linux
        );
        assert_eq!(
            KeyLabelPlatform::from_recording(Some(&crate::Platform::MacOS)),
            KeyLabelPlatform::Mac
        );
    }
}
