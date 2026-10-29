use std::{
    collections::{HashMap, HashSet},
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

use cap_project::{MaskSegment, ProjectConfiguration, RecordingMeta, XY};
use cap_redaction::{
    Detection, DetectorOptions, LumaFrame, Rect, SampleWindow, Thumbnail,
    masks::{MaskPlan, track_to_masks},
    track::{Tracker, TrackerConfig},
};
use cap_rendering::{
    DecodedFrame, DecodedSegmentFrames, DisplayPlacement, PixelFormat, ProjectUniforms,
    RenderVideoConstants, ZoomTransformTimeline,
};

use crate::{SegmentMedia, create_segments_without_audio};

/// How often a frame is decoded to look for changes.
const PROBE_INTERVAL: f64 = 0.1;
/// How often a changing screen is read again.
const SAMPLE_INTERVAL: f64 = 0.5;
/// How often it is read while something on it is moving (scrolling).
const MOTION_SAMPLE_INTERVAL: f64 = 0.2;
/// Share of the frame that has to change between two probes for it to be a
/// cut, which is read straight away.
const SCENE_CHANGE: f64 = 0.3;
const MOTION: f64 = 0.004;
/// Frames wider than this are halved before text recognition.
const MAX_OCR_WIDTH: u32 = 2600;
/// The timeline rate placements are evaluated at, matching playback.
const PLACEMENT_FPS: u32 = 60;
/// How often a mask's output placement is probed, so it follows zooms.
const PLACEMENT_STEP: f64 = 1.0 / 30.0;

#[derive(Clone, Debug, Default)]
pub struct SensitiveScanOptions {
    pub detectors: DetectorOptions,
}

#[derive(Clone, Debug)]
pub struct SensitiveScanResult {
    /// One mask per tracked item, lanes not yet assigned.
    pub masks: Vec<MaskSegment>,
    pub probed_frames: usize,
    pub recognition_passes: usize,
    pub elapsed: Duration,
}

/// Whether this build can scan recordings for sensitive information.
pub fn sensitive_scan_available() -> bool {
    cap_redaction::ocr_available()
}

/// Reads the edited timeline for email addresses, card numbers, keys and
/// faces, and returns a blur mask for each that follows it on screen.
///
/// Runs its own decoders so the editor's playback is untouched. `progress`
/// receives 0..=1; setting `cancel` stops the scan with an error.
pub async fn scan_for_sensitive_info(
    recording_meta: &RecordingMeta,
    constants: Arc<RenderVideoConstants>,
    project: Arc<ProjectConfiguration>,
    template: MaskSegment,
    options: SensitiveScanOptions,
    progress: impl Fn(f64) + Send,
    cancel: Arc<AtomicBool>,
) -> Result<SensitiveScanResult, String> {
    let started = Instant::now();
    let worker = OcrWorker::spawn(options.detectors)?;
    let duration = project
        .timeline
        .as_ref()
        .map(|timeline| timeline.duration())
        .unwrap_or(0.0);
    if duration <= 0.0 {
        return Err("The recording has no timeline to scan".into());
    }
    let segments = create_segments_without_audio(recording_meta, &constants.meta, false).await?;
    for segment in segments.iter() {
        segment.decoders.prefer_sequential_access();
    }

    let mut placements = Placements::new(&constants, &project, &segments, duration);
    let mut tracker = Tracker::new(TrackerConfig::default());
    let mut window: Vec<(f64, Arc<Thumbnail>)> = Vec::new();
    let mut last_pass: Option<(f64, Arc<Thumbnail>, Vec<Detection>)> = None;
    let mut source_size = (0u32, 0u32);
    let mut probed_frames = 0;
    let mut recognition_passes = 0;
    let mut fetched_clips = HashSet::new();

    let probes = (duration / PROBE_INTERVAL).ceil() as usize;
    for index in 0..=probes {
        if cancel.load(Ordering::Relaxed) {
            return Err("Scan cancelled".into());
        }
        let is_last = index == probes;
        let time = if is_last {
            (duration - 1e-3).max(0.0)
        } else {
            index as f64 * PROBE_INTERVAL
        };
        if !is_last && time >= duration {
            continue;
        }
        let Some((source_time, segment)) = project.get_segment_time(time) else {
            continue;
        };
        let clip = segment.recording_clip;
        let Some(media) = segments.get(clip as usize) else {
            continue;
        };
        let offsets = project
            .clips
            .iter()
            .find(|candidate| candidate.index == clip)
            .map(|candidate| candidate.offsets)
            .unwrap_or_default();
        let initial = fetched_clips.insert(clip);
        let Some(frames) = media
            .decoders
            .get_frames_offline(source_time as f32, offsets, initial)
            .await
        else {
            continue;
        };
        let Some(luma) = frames.screen_frame.as_ref().and_then(luma_from_frame) else {
            continue;
        };
        source_size = (luma.width, luma.height);
        let luma = if luma.width > MAX_OCR_WIDTH {
            luma.half()
        } else {
            luma
        };
        probed_frames += 1;
        let thumbnail = Arc::new(Thumbnail::from_luma(&luma));
        let scene_cut = window
            .last()
            .is_some_and(|(_, previous)| previous.difference(&thumbnail) > SCENE_CHANGE);
        window.push((time, thumbnail.clone()));

        let interval = if tracker.last_motion() > MOTION {
            MOTION_SAMPLE_INTERVAL
        } else {
            SAMPLE_INTERVAL
        };
        let decision = match &last_pass {
            None => Some(true),
            Some((pass_time, pass_thumbnail, _)) => {
                let unchanged = pass_thumbnail.difference(&thumbnail) == 0.0;
                let due = time - pass_time >= interval - 1e-6 || is_last;
                if scene_cut && !unchanged {
                    Some(true)
                } else if due {
                    Some(!unchanged)
                } else {
                    None
                }
            }
        };
        let Some(recognize) = decision else {
            progress((time / duration).clamp(0.0, 1.0));
            continue;
        };

        let detections = if recognize {
            recognition_passes += 1;
            worker.analyze(luma).await?
        } else {
            last_pass
                .as_ref()
                .map(|(_, _, detections)| detections.clone())
                .unwrap_or_default()
        };
        tracker.push(
            &SampleWindow {
                frames: std::mem::take(&mut window),
            },
            &detections,
        );
        window.push((time, thumbnail.clone()));
        last_pass = Some((time, thumbnail, detections));
        progress((time / duration).clamp(0.0, 1.0));
    }

    let tracks = tracker.finish(duration);
    let plan = MaskPlan {
        source_size,
        total_duration: duration,
        template: &template,
        step: PLACEMENT_STEP,
    };
    let masks = tracks
        .iter()
        .flat_map(|track| {
            track_to_masks(track, &plan, &mut |time, rect| placements.map(time, rect))
        })
        .collect();
    progress(1.0);

    Ok(SensitiveScanResult {
        masks,
        probed_frames,
        recognition_passes,
        elapsed: started.elapsed(),
    })
}

fn luma_from_frame(frame: &DecodedFrame) -> Option<LumaFrame> {
    let (width, height) = (frame.width(), frame.height());
    if width == 0 || height == 0 {
        return None;
    }
    match frame.format() {
        PixelFormat::Nv12 | PixelFormat::Yuv420p => {
            LumaFrame::from_plane(frame.y_plane()?, width, height, frame.y_stride() as usize)
        }
        PixelFormat::Rgba => {
            let data = frame.data();
            LumaFrame::from_rgba(data, width, height, data.len() / height as usize)
        }
    }
}

/// Where the screen sits in the output at a timeline time, cached per
/// rendered frame.
struct Placements<'a> {
    constants: &'a RenderVideoConstants,
    project: &'a Arc<ProjectConfiguration>,
    segments: &'a [SegmentMedia],
    duration: f64,
    resolution_base: XY<u32>,
    zoom: HashMap<u32, ZoomTransformTimeline>,
    cache: HashMap<u32, DisplayPlacement>,
}

impl<'a> Placements<'a> {
    fn new(
        constants: &'a RenderVideoConstants,
        project: &'a Arc<ProjectConfiguration>,
        segments: &'a [SegmentMedia],
        duration: f64,
    ) -> Self {
        let (width, height) = ProjectUniforms::get_base_size(&constants.options, project);
        Self {
            constants,
            project,
            segments,
            duration,
            resolution_base: XY::new(width, height),
            zoom: HashMap::new(),
            cache: HashMap::new(),
        }
    }

    fn placement(&mut self, time: f64) -> Option<DisplayPlacement> {
        let frame_number = (time * f64::from(PLACEMENT_FPS)).round().max(0.0) as u32;
        if let Some(placement) = self.cache.get(&frame_number) {
            return Some(*placement);
        }
        let frame_time = f64::from(frame_number) / f64::from(PLACEMENT_FPS);
        let (source_time, segment) = self.project.get_segment_time(frame_time)?;
        let clip = segment.recording_clip;
        let media = self.segments.get(clip as usize)?;
        let screen_size = self.constants.options.screen_size;
        let zoom = self.zoom.entry(clip).or_insert_with(|| {
            ZoomTransformTimeline::from_project_for_clip(
                self.project,
                &media.cursor,
                self.duration,
                screen_size,
                clip,
            )
        });
        zoom.ensure_precomputed_until((frame_number as f32 + 1.0) / PLACEMENT_FPS as f32);
        let frames = DecodedSegmentFrames {
            screen_size,
            screen_frame: None,
            camera_frame: None,
            segment_time: source_time as f32,
            recording_time: source_time as f32,
            segment_has_camera: false,
        };
        let uniforms = ProjectUniforms::new(
            self.constants,
            self.project,
            frame_number,
            PLACEMENT_FPS,
            self.resolution_base,
            &media.cursor,
            &frames,
            self.duration,
            zoom,
        );
        let placement = uniforms.display_placement();
        self.cache.insert(frame_number, placement);
        Some(placement)
    }

    fn map(&mut self, time: f64, rect: Rect) -> Option<Rect> {
        let placement = self.placement(time)?;
        if !placement.visible {
            return None;
        }
        let (x0, y0) = placement.source_to_output(rect.x, rect.y);
        let (x1, y1) = placement.source_to_output(rect.right(), rect.bottom());
        Some(Rect::new(x0, y0, x1 - x0, y1 - y0))
    }
}

#[cfg(target_os = "macos")]
type OcrReply = tokio::sync::oneshot::Sender<Result<Vec<Detection>, String>>;

/// Text recognition on a thread of its own: Vision blocks for tens of
/// milliseconds a frame and keeps per-thread state.
struct OcrWorker {
    #[cfg(target_os = "macos")]
    requests: std::sync::mpsc::Sender<(LumaFrame, OcrReply)>,
}

impl OcrWorker {
    #[cfg(target_os = "macos")]
    fn spawn(detectors: DetectorOptions) -> Result<Self, String> {
        let (requests, receiver) = std::sync::mpsc::channel::<(LumaFrame, OcrReply)>();
        std::thread::Builder::new()
            .name("sensitive-scan-ocr".into())
            .spawn(move || {
                let mut analyzer = cap_redaction::vision::VisionAnalyzer::new(detectors);
                while let Ok((frame, reply)) = receiver.recv() {
                    let _ = reply.send(analyzer.analyze(&frame));
                }
            })
            .map_err(|error| format!("Could not start text recognition: {error}"))?;
        Ok(Self { requests })
    }

    #[cfg(not(target_os = "macos"))]
    fn spawn(_detectors: DetectorOptions) -> Result<Self, String> {
        Err("Scanning for sensitive information needs macOS".into())
    }

    #[cfg(target_os = "macos")]
    async fn analyze(&self, frame: LumaFrame) -> Result<Vec<Detection>, String> {
        let (reply, response) = tokio::sync::oneshot::channel();
        self.requests
            .send((frame, reply))
            .map_err(|_| "Text recognition stopped".to_string())?;
        response
            .await
            .map_err(|_| "Text recognition stopped".to_string())?
    }

    #[cfg(not(target_os = "macos"))]
    async fn analyze(&self, _frame: LumaFrame) -> Result<Vec<Detection>, String> {
        Err("Scanning for sensitive information needs macOS".into())
    }
}
