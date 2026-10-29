use cap_project::{ProjectConfiguration, SceneMode, SceneSegment, StudioRecordingMeta};
use serde::{Deserialize, Serialize};
use std::{
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, RwLock},
    time::UNIX_EPOCH,
};

use crate::scene::MIN_GAP_FOR_TRANSITION;

pub const CAMERA_ANALYSIS_FILE: &str = "camera-analysis.json";
const CAMERA_ANALYSIS_VERSION: u32 = 1;

pub const SILENCE_SHOW_LEAD_SECS: f64 = 0.3;
pub const SILENCE_MIN_SPEECH_SECS: f64 = 0.2;
pub const SILENCE_TRANSITION_SECS: f64 = 0.3;
const SILENCE_MIN_HIDDEN_SECS: f64 = 1.0;
const SILENCE_GRID_SECS: f64 = 0.02;

pub const FACE_SAMPLE_INTERVAL_SECS: f64 = 0.1;
const FACE_SMOOTHING_SECS: f64 = 0.45;
const FACE_HEADROOM: f32 = 0.42;

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceFingerprint {
    pub len: u64,
    pub modified_ms: u64,
}

impl SourceFingerprint {
    pub fn of(path: &Path) -> Option<Self> {
        let metadata = std::fs::metadata(path).ok()?;
        let modified_ms = metadata
            .modified()
            .ok()
            .and_then(|time| time.duration_since(UNIX_EPOCH).ok())
            .map_or(0, |duration| duration.as_millis() as u64);
        Some(Self {
            len: metadata.len(),
            modified_ms,
        })
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct SpeechTrack {
    pub source: SourceFingerprint,
    pub spans: Vec<(f64, f64)>,
}

#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq)]
pub struct FaceBox {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

impl FaceBox {
    fn center(self) -> [f32; 2] {
        [self.x + self.width / 2.0, self.y + self.height / 2.0]
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct FaceTrack {
    pub source: SourceFingerprint,
    pub interval: f64,
    pub samples: Vec<Option<FaceBox>>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct SegmentCameraAnalysis {
    #[serde(default)]
    pub speech: Option<SpeechTrack>,
    #[serde(default)]
    pub faces: Option<FaceTrack>,
}

#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
pub struct CameraAnalysis {
    pub version: u32,
    pub segments: Vec<SegmentCameraAnalysis>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SegmentSources {
    pub mic: Option<PathBuf>,
    pub camera: Option<PathBuf>,
}

pub fn segment_sources(project_path: &Path, meta: &StudioRecordingMeta) -> Vec<SegmentSources> {
    match meta {
        StudioRecordingMeta::SingleSegment { segment } => vec![SegmentSources {
            mic: segment
                .audio
                .as_ref()
                .map(|audio| audio.path.to_path(project_path)),
            camera: segment
                .camera
                .as_ref()
                .map(|camera| camera.path.to_path(project_path)),
        }],
        StudioRecordingMeta::MultipleSegments { inner } => inner
            .segments
            .iter()
            .map(|segment| SegmentSources {
                mic: segment
                    .mic
                    .as_ref()
                    .map(|audio| audio.path.to_path(project_path)),
                camera: segment
                    .camera
                    .as_ref()
                    .map(|camera| camera.path.to_path(project_path)),
            })
            .collect(),
    }
}

impl CameraAnalysis {
    pub fn path(project_path: &Path) -> PathBuf {
        project_path.join(CAMERA_ANALYSIS_FILE)
    }

    pub fn load(project_path: &Path, sources: &[SegmentSources]) -> Self {
        let Some(mut analysis) = std::fs::read(Self::path(project_path))
            .ok()
            .and_then(|bytes| serde_json::from_slice::<Self>(&bytes).ok())
            .filter(|analysis| analysis.version == CAMERA_ANALYSIS_VERSION)
        else {
            return Self::empty(sources.len());
        };
        analysis
            .segments
            .resize_with(sources.len(), Default::default);
        for (segment, source) in analysis.segments.iter_mut().zip(sources) {
            let mic = source.mic.as_deref().and_then(SourceFingerprint::of);
            if segment.speech.as_ref().map(|speech| speech.source) != mic {
                segment.speech = None;
            }
            let camera = source.camera.as_deref().and_then(SourceFingerprint::of);
            if segment.faces.as_ref().map(|faces| faces.source) != camera {
                segment.faces = None;
            }
        }
        analysis
    }

    pub fn empty(segments: usize) -> Self {
        Self {
            version: CAMERA_ANALYSIS_VERSION,
            segments: vec![SegmentCameraAnalysis::default(); segments],
        }
    }

    pub fn save(&self, project_path: &Path) -> std::io::Result<()> {
        let path = Self::path(project_path);
        let temporary = path.with_extension("json.tmp");
        std::fs::write(&temporary, serde_json::to_vec(self)?)?;
        std::fs::rename(temporary, path)
    }

    pub fn has_speech(&self, sources: &[SegmentSources]) -> bool {
        sources
            .iter()
            .zip(&self.segments)
            .all(|(source, segment)| source.mic.is_none() || segment.speech.is_some())
    }

    pub fn has_faces(&self, sources: &[SegmentSources]) -> bool {
        sources
            .iter()
            .zip(&self.segments)
            .all(|(source, segment)| source.camera.is_none() || segment.faces.is_some())
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SilenceSettings {
    pub hide_after: f64,
    pub show_lead: f64,
    pub min_speech: f64,
    pub transition: f64,
}

impl SilenceSettings {
    pub fn new(hide_after: f64) -> Self {
        Self {
            hide_after,
            show_lead: SILENCE_SHOW_LEAD_SECS,
            min_speech: SILENCE_MIN_SPEECH_SECS,
            transition: SILENCE_TRANSITION_SECS,
        }
    }
}

/// Timeline windows where the camera should be hidden, from speech runs
/// already on the timeline clock. Speech shorter than `min_speech` is
/// ignored, a silence must outlast `hide_after` before the camera leaves, and
/// the camera is back `show_lead` before the next speech.
pub fn silence_windows(
    speech: &[(f64, f64)],
    duration: f64,
    settings: SilenceSettings,
) -> Vec<(f64, f64)> {
    let mut windows = Vec::new();
    if !(duration.is_finite() && duration > 0.0) {
        return windows;
    }
    let mut silence_start = 0.0;
    let speech = speech
        .iter()
        .filter(|(start, end)| end - start >= settings.min_speech);
    let mut push = |silence_start: f64, silence_end: f64, lead: f64| {
        let start = silence_start + settings.hide_after;
        let end = silence_end - lead;
        if end - start >= SILENCE_MIN_HIDDEN_SECS.max(settings.transition * 2.0) {
            windows.push((start, end));
        }
    };
    for &(start, end) in speech {
        if start > silence_start {
            push(silence_start, start.min(duration), settings.show_lead);
        }
        silence_start = silence_start.max(end);
    }
    if silence_start < duration {
        push(silence_start, duration, 0.0);
    }
    windows
}

/// Speech runs on the timeline clock, sampled through the timeline's own
/// frame mapping so clip speed, transitions and holds map exactly as the
/// renderer does. A clip whose recording has no speech analysis counts as
/// speaking throughout, so it never hides the camera.
pub fn timeline_speech(
    project: &ProjectConfiguration,
    analysis: &CameraAnalysis,
    duration: f64,
) -> Vec<(f64, f64)> {
    let steps = (duration / SILENCE_GRID_SECS).ceil().max(0.0) as usize;
    let mut runs: Vec<(f64, f64)> = Vec::new();
    for step in 0..steps {
        let time = step as f64 * SILENCE_GRID_SECS;
        let speaking = match source_at(project, time) {
            Some((source_time, clip)) => {
                let offset = project
                    .clips
                    .iter()
                    .find(|config| config.index == clip)
                    .map_or(0.0, |config| f64::from(config.offsets.mic));
                match analysis
                    .segments
                    .get(clip as usize)
                    .and_then(|segment| segment.speech.as_ref())
                {
                    Some(speech) => contains(&speech.spans, source_time + offset),
                    None => true,
                }
            }
            None => true,
        };
        if !speaking {
            continue;
        }
        let end = (time + SILENCE_GRID_SECS).min(duration);
        match runs.last_mut() {
            Some(last) if (last.1 - time).abs() < SILENCE_GRID_SECS / 2.0 => last.1 = end,
            _ => runs.push((time, end)),
        }
    }
    runs
}

fn source_at(project: &ProjectConfiguration, time: f64) -> Option<(f64, u32)> {
    if project.timeline.is_none() {
        return Some((time, 0));
    }
    project
        .get_segment_time(time)
        .map(|(source_time, segment)| (source_time, segment.recording_clip))
}

fn contains(spans: &[(f64, f64)], time: f64) -> bool {
    let index = spans.partition_point(|(start, _)| *start <= time);
    index > 0 && time < spans[index - 1].1
}

/// Hidden-camera scene segments for the silent windows, kept clear of the
/// user's own scene segments (and their transitions) so a manual layout
/// always wins and the two never blend into each other.
pub fn silence_scene_segments(
    windows: &[(f64, f64)],
    explicit: &[SceneSegment],
    transition: f64,
) -> Vec<SceneSegment> {
    let mut blocked: Vec<(f64, f64)> = explicit
        .iter()
        .map(|segment| {
            let margin = transition + MIN_GAP_FOR_TRANSITION;
            (
                segment.start - segment.transition_in.max(0.0) - margin,
                segment.end + segment.transition_out.max(0.0) + margin,
            )
        })
        .collect();
    blocked.sort_by(|a, b| a.0.total_cmp(&b.0));

    let mut segments = Vec::new();
    for &(start, end) in windows {
        let mut pieces = vec![(start, end)];
        for &(block_start, block_end) in &blocked {
            pieces = pieces
                .into_iter()
                .flat_map(|(start, end)| {
                    if block_end <= start || block_start >= end {
                        vec![(start, end)]
                    } else {
                        [(start, block_start), (block_end, end)]
                            .into_iter()
                            .filter(|(start, end)| end > start)
                            .collect()
                    }
                })
                .collect();
        }
        for (start, end) in pieces {
            if end - start < SILENCE_MIN_HIDDEN_SECS.max(transition * 2.0) {
                continue;
            }
            segments.push(SceneSegment {
                start: start + transition,
                end,
                mode: SceneMode::HideCamera,
                split_layout: None,
                transition_in: transition,
                transition_out: transition,
            });
        }
    }
    segments
}

fn merge_scene_segments(
    explicit: &[SceneSegment],
    silence: Vec<SceneSegment>,
) -> Vec<SceneSegment> {
    let mut segments: Vec<SceneSegment> = explicit.iter().cloned().chain(silence).collect();
    segments.sort_by(|a, b| a.start.total_cmp(&b.start));
    segments
}

/// A face track smoothed offline: gaps hold the last face, a 3-tap median
/// drops single-sample detector outliers, and a critically damped spring run
/// forward then backward settles jitter without lagging behind the motion.
#[derive(Clone, Debug, PartialEq)]
pub struct SmoothedFaceTrack {
    interval: f64,
    centers: Vec<[f32; 2]>,
}

impl SmoothedFaceTrack {
    pub fn new(track: &FaceTrack) -> Option<Self> {
        let interval = track.interval;
        if !(interval.is_finite() && interval > 0.0) {
            return None;
        }
        let first = track.samples.iter().flatten().next()?.center();
        let mut last = first;
        let held: Vec<[f32; 2]> = track
            .samples
            .iter()
            .map(|sample| {
                if let Some(face) = sample {
                    last = face.center();
                }
                last
            })
            .collect();
        let median: Vec<[f32; 2]> = (0..held.len())
            .map(|index| {
                let window = &held[index.saturating_sub(1)..(index + 2).min(held.len())];
                [0, 1].map(|axis| median_of(window.iter().map(|center| center[axis])))
            })
            .collect();
        let forward = spring_pass(median.iter().copied(), interval);
        let mut backward = spring_pass(forward.iter().rev().copied(), interval);
        backward.reverse();
        Some(Self {
            interval,
            centers: backward,
        })
    }

    pub fn sample(&self, time: f64) -> [f32; 2] {
        let position = (time / self.interval).max(0.0);
        let index = position.floor() as usize;
        let Some(&current) = self.centers.get(index) else {
            return *self.centers.last().unwrap_or(&[0.5, 0.5]);
        };
        let next = self.centers.get(index + 1).copied().unwrap_or(current);
        let t = (position - index as f64) as f32;
        [0, 1].map(|axis| current[axis] + (next[axis] - current[axis]) * t)
    }
}

fn median_of(values: impl Iterator<Item = f32>) -> f32 {
    let mut values: Vec<f32> = values.collect();
    values.sort_by(f32::total_cmp);
    values[values.len() / 2]
}

fn spring_pass(targets: impl Iterator<Item = [f32; 2]>, interval: f64) -> Vec<[f32; 2]> {
    let omega = 2.0 / FACE_SMOOTHING_SECS;
    let mut position: Option<[f64; 2]> = None;
    let mut velocity = [0.0f64; 2];
    targets
        .map(|target| {
            let current = position.get_or_insert([f64::from(target[0]), f64::from(target[1])]);
            for axis in 0..2 {
                let (next, speed) = critically_damped_step(
                    current[axis],
                    velocity[axis],
                    f64::from(target[axis]),
                    omega,
                    interval,
                );
                current[axis] = next;
                velocity[axis] = speed;
            }
            [current[0] as f32, current[1] as f32]
        })
        .collect()
}

fn critically_damped_step(
    position: f64,
    velocity: f64,
    target: f64,
    omega: f64,
    dt: f64,
) -> (f64, f64) {
    let offset = position - target;
    let decay = (-omega * dt).exp();
    let shared = velocity + omega * offset;
    let next = target + (offset + shared * dt) * decay;
    let speed = (velocity - omega * shared * dt) * decay;
    (next, speed)
}

/// The camera crop zoomed by `zoom` and panned so the face sits centred
/// horizontally and a little above centre, clamped to stay inside the frame.
pub fn face_crop(base: [f32; 4], frame: [f32; 2], center: [f32; 2], zoom: f32) -> [f32; 4] {
    let zoom = if zoom.is_finite() { zoom.max(1.0) } else { 1.0 };
    let width = (base[2] - base[0]) / zoom;
    let height = (base[3] - base[1]) / zoom;
    let face = [center[0] * frame[0], center[1] * frame[1]];
    let x = (face[0] - width / 2.0).clamp(0.0, (frame[0] - width).max(0.0));
    let y = (face[1] - height * FACE_HEADROOM).clamp(0.0, (frame[1] - height).max(0.0));
    [x, y, x + width, y + height]
}

struct CachedScenes {
    key: u64,
    segments: Arc<Vec<SceneSegment>>,
}

#[derive(Default)]
pub struct CameraAnalysisStore {
    analysis: RwLock<Arc<LoadedAnalysis>>,
    scenes: Mutex<Option<CachedScenes>>,
}

#[derive(Default)]
struct LoadedAnalysis {
    generation: u64,
    analysis: CameraAnalysis,
    faces: Vec<Option<SmoothedFaceTrack>>,
}

impl CameraAnalysisStore {
    pub fn load(project_path: &Path, meta: &StudioRecordingMeta) -> Self {
        let store = Self::default();
        let sources = segment_sources(project_path, meta);
        store.set(CameraAnalysis::load(project_path, &sources));
        store
    }

    pub fn set(&self, analysis: CameraAnalysis) {
        let faces = analysis
            .segments
            .iter()
            .map(|segment| segment.faces.as_ref().and_then(SmoothedFaceTrack::new))
            .collect();
        let mut current = self.analysis.write().unwrap_or_else(|e| e.into_inner());
        *current = Arc::new(LoadedAnalysis {
            generation: current.generation + 1,
            analysis,
            faces,
        });
    }

    pub fn analysis(&self) -> CameraAnalysis {
        self.loaded().analysis.clone()
    }

    pub fn with_analysis<R>(&self, read: impl FnOnce(&CameraAnalysis) -> R) -> R {
        read(&self.loaded().analysis)
    }

    fn loaded(&self) -> Arc<LoadedAnalysis> {
        Arc::clone(&self.analysis.read().unwrap_or_else(|e| e.into_inner()))
    }

    /// The scene segments the renderer should use: the timeline's own, plus
    /// hidden-camera segments over silences when the camera is set to hide
    /// while the speaker is silent. Cached until the timeline, the setting or
    /// the analysis changes.
    pub fn scene_segments(
        &self,
        project: &ProjectConfiguration,
        duration: f64,
    ) -> Option<Arc<Vec<SceneSegment>>> {
        let camera = &project.camera;
        if camera.hide || !camera.hide_when_silent {
            return None;
        }
        let loaded = self.loaded();
        if loaded
            .analysis
            .segments
            .iter()
            .all(|segment| segment.speech.is_none())
        {
            return None;
        }
        let key = silence_key(project, duration, loaded.generation);
        let mut cached = self.scenes.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(cached) = cached.as_ref().filter(|cached| cached.key == key) {
            return Some(Arc::clone(&cached.segments));
        }
        let explicit = project
            .timeline
            .as_ref()
            .map_or(&[][..], |timeline| timeline.scene_segments.as_slice());
        let settings = SilenceSettings::new(f64::from(camera.hide_when_silent_delay_clamped()));
        let speech = timeline_speech(project, &loaded.analysis, duration);
        let windows = silence_windows(&speech, duration, settings);
        let silence = silence_scene_segments(&windows, explicit, settings.transition);
        let segments = Arc::new(merge_scene_segments(explicit, silence));
        *cached = Some(CachedScenes {
            key,
            segments: Arc::clone(&segments),
        });
        Some(segments)
    }

    /// Normalized face centre in the camera frame shown at `timeline_time`.
    pub fn face_center(
        &self,
        project: &ProjectConfiguration,
        meta: &StudioRecordingMeta,
        timeline_time: f64,
    ) -> Option<[f32; 2]> {
        let (source_time, clip) = source_at(project, timeline_time)?;
        let loaded = self.loaded();
        let track = loaded.faces.get(clip as usize)?.as_ref()?;
        let clip_offset = project
            .clips
            .iter()
            .find(|config| config.index == clip)
            .map_or(0.0, |config| f64::from(config.offsets.camera));
        let decoder_offset = match meta {
            StudioRecordingMeta::MultipleSegments { inner }
                if (clip as usize) < inner.segments.len() =>
            {
                crate::segment_timing::segment_video_timing(meta, clip as usize).camera_offset
            }
            _ => 0.0,
        };
        Some(track.sample(source_time + clip_offset + decoder_offset))
    }
}

fn silence_key(project: &ProjectConfiguration, duration: f64, generation: u64) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    generation.hash(&mut hasher);
    duration.to_bits().hash(&mut hasher);
    project
        .camera
        .hide_when_silent_delay
        .to_bits()
        .hash(&mut hasher);
    for clip in &project.clips {
        clip.index.hash(&mut hasher);
        clip.offsets.mic.to_bits().hash(&mut hasher);
    }
    if let Some(timeline) = &project.timeline {
        for segment in &timeline.segments {
            segment.recording_clip.hash(&mut hasher);
            segment.start.to_bits().hash(&mut hasher);
            segment.end.to_bits().hash(&mut hasher);
            segment.timescale.to_bits().hash(&mut hasher);
        }
        for transition in &timeline.transitions {
            transition.segment_index.hash(&mut hasher);
            transition.duration.to_bits().hash(&mut hasher);
        }
        for (start, end) in timeline.hold_windows() {
            start.to_bits().hash(&mut hasher);
            end.to_bits().hash(&mut hasher);
        }
        for segment in &timeline.scene_segments {
            segment.start.to_bits().hash(&mut hasher);
            segment.end.to_bits().hash(&mut hasher);
            segment.transition_in.to_bits().hash(&mut hasher);
            segment.transition_out.to_bits().hash(&mut hasher);
            std::mem::discriminant(&segment.mode).hash(&mut hasher);
        }
    }
    hasher.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::{InterpolatedScene, SceneSegmentsCursor};
    use cap_project::{TimelineConfiguration, TimelineSegment};

    fn settings() -> SilenceSettings {
        SilenceSettings::new(2.0)
    }

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    #[test]
    fn long_silences_hide_after_the_delay_and_return_before_speech() {
        let speech = [(1.0, 5.0), (12.0, 15.0)];
        let windows = silence_windows(&speech, 20.0, settings());
        assert_eq!(windows.len(), 2);
        assert!(approx(windows[0].0, 7.0) && approx(windows[0].1, 11.7));
        assert!(approx(windows[1].0, 17.0) && approx(windows[1].1, 20.0));
    }

    #[test]
    fn short_pauses_never_hide() {
        let speech = [(0.0, 3.0), (4.5, 8.0), (10.5, 12.0)];
        assert!(silence_windows(&speech, 12.0, settings()).is_empty());
    }

    #[test]
    fn a_silence_just_over_the_delay_needs_room_for_both_fades() {
        let speech = [(0.0, 1.0), (4.0, 6.0)];
        assert!(silence_windows(&speech, 6.0, settings()).is_empty());
        let speech = [(0.0, 1.0), (4.4, 6.0)];
        let windows = silence_windows(&speech, 6.0, settings());
        assert_eq!(windows.len(), 1);
        assert!(approx(windows[0].0, 3.0) && approx(windows[0].1, 4.1));
    }

    #[test]
    fn blips_shorter_than_the_minimum_do_not_bring_the_camera_back() {
        let speech = [(0.0, 2.0), (6.0, 6.15), (12.0, 14.0)];
        let windows = silence_windows(&speech, 14.0, settings());
        assert_eq!(windows.len(), 1);
        assert!(approx(windows[0].0, 4.0) && approx(windows[0].1, 11.7));

        let speech = [(0.0, 2.0), (6.0, 6.3), (12.0, 14.0)];
        let windows = silence_windows(&speech, 14.0, settings());
        assert_eq!(windows.len(), 2);
        assert!(approx(windows[0].1, 5.7) && approx(windows[1].0, 8.3));
    }

    #[test]
    fn leading_silence_and_empty_or_invalid_input() {
        let windows = silence_windows(&[(6.0, 8.0)], 8.0, settings());
        assert_eq!(windows.len(), 1);
        assert!(approx(windows[0].0, 2.0) && approx(windows[0].1, 5.7));
        assert_eq!(silence_windows(&[], 10.0, settings()), vec![(2.0, 10.0)]);
        assert!(silence_windows(&[], f64::NAN, settings()).is_empty());
        assert!(silence_windows(&[], 0.0, settings()).is_empty());
    }

    fn project_with(segments: Vec<TimelineSegment>) -> ProjectConfiguration {
        let mut timeline: TimelineConfiguration =
            serde_json::from_value(serde_json::json!({ "segments": [], "zoomSegments": [] }))
                .unwrap();
        timeline.segments = segments;
        ProjectConfiguration {
            timeline: Some(timeline),
            ..Default::default()
        }
    }

    fn clip(recording_clip: u32, start: f64, end: f64, timescale: f64) -> TimelineSegment {
        TimelineSegment {
            recording_clip,
            timescale,
            start,
            end,
            name: None,
            speed_audio_mode: None,
            volume: None,
            hide_cursor: None,
        }
    }

    fn analysis_with(spans: Vec<Vec<(f64, f64)>>) -> CameraAnalysis {
        CameraAnalysis {
            version: CAMERA_ANALYSIS_VERSION,
            segments: spans
                .into_iter()
                .map(|spans| SegmentCameraAnalysis {
                    speech: Some(SpeechTrack {
                        source: SourceFingerprint {
                            len: 0,
                            modified_ms: 0,
                        },
                        spans,
                    }),
                    faces: None,
                })
                .collect(),
        }
    }

    #[test]
    fn speech_maps_through_cuts_speed_and_mic_offsets() {
        let mut project = project_with(vec![clip(0, 10.0, 20.0, 1.0), clip(0, 30.0, 40.0, 2.0)]);
        let analysis = analysis_with(vec![vec![(12.0, 14.0), (32.0, 36.0)]]);
        let runs = timeline_speech(&project, &analysis, 15.0);
        assert_eq!(runs.len(), 2);
        assert!((runs[0].0 - 2.0).abs() < 0.03 && (runs[0].1 - 4.0).abs() < 0.03);
        assert!((runs[1].0 - 11.0).abs() < 0.03 && (runs[1].1 - 13.0).abs() < 0.03);

        project.clips.push(cap_project::ClipConfiguration {
            index: 0,
            offsets: cap_project::ClipOffsets {
                mic: 1.0,
                ..Default::default()
            },
            ..Default::default()
        });
        let runs = timeline_speech(&project, &analysis, 15.0);
        assert!((runs[0].0 - 1.0).abs() < 0.03 && (runs[0].1 - 3.0).abs() < 0.03);
    }

    #[test]
    fn clips_without_speech_analysis_keep_the_camera() {
        let project = project_with(vec![clip(0, 0.0, 10.0, 1.0), clip(1, 0.0, 10.0, 1.0)]);
        let mut analysis = analysis_with(vec![vec![]]);
        analysis.segments.push(SegmentCameraAnalysis::default());
        let runs = timeline_speech(&project, &analysis, 20.0);
        assert_eq!(runs.len(), 1);
        assert!((runs[0].0 - 10.0).abs() < 0.03 && approx(runs[0].1, 20.0));
    }

    fn scene(start: f64, end: f64, mode: SceneMode) -> SceneSegment {
        SceneSegment {
            start,
            end,
            mode,
            split_layout: None,
            transition_in: 0.3,
            transition_out: 0.3,
        }
    }

    #[test]
    fn silence_segments_fade_out_at_the_window_start_and_are_back_by_its_end() {
        let segments = silence_scene_segments(&[(7.0, 11.7)], &[], 0.3);
        assert_eq!(segments.len(), 1);
        let opacity = |time: f64| {
            InterpolatedScene::new(SceneSegmentsCursor::new(time, &segments))
                .regular_camera_transition_opacity()
        };
        assert!(approx(opacity(6.99), 1.0));
        assert!(opacity(7.15) > 0.1 && opacity(7.15) < 0.9);
        assert!(opacity(7.31) < 0.01);
        assert!(opacity(11.0) < 0.01);
        assert!(opacity(11.55) > 0.1 && opacity(11.55) < 0.9);
        assert!(approx(opacity(11.71), 1.0));
    }

    #[test]
    fn manual_scene_segments_win_over_silence() {
        let explicit = [scene(10.0, 14.0, SceneMode::Default)];
        let segments = silence_scene_segments(&[(4.0, 20.0)], &explicit, 0.3);
        assert_eq!(segments.len(), 2);
        assert!(segments[0].end <= 10.0 - 0.3 - 0.3 - MIN_GAP_FOR_TRANSITION + 1e-9);
        assert!(segments[1].start - 0.3 >= 14.0 + 0.3 + 0.3 + MIN_GAP_FOR_TRANSITION - 1e-9);
        let merged = merge_scene_segments(&explicit, segments);
        let at = |time: f64| InterpolatedScene::new(SceneSegmentsCursor::new(time, &merged));
        assert!(approx(at(12.0).regular_camera_transition_opacity(), 1.0));
        assert!(at(6.0).regular_camera_transition_opacity() < 0.01);
        assert!(at(18.0).regular_camera_transition_opacity() < 0.01);

        let covered = silence_scene_segments(&[(9.0, 15.0)], &explicit, 0.3);
        assert!(covered.is_empty());
    }

    #[test]
    fn store_caches_until_the_timeline_or_setting_changes() {
        let store = CameraAnalysisStore::default();
        let mut project = project_with(vec![clip(0, 0.0, 20.0, 1.0)]);
        store.set(analysis_with(vec![vec![(0.0, 2.0), (15.0, 18.0)]]));
        assert!(store.scene_segments(&project, 20.0).is_none());
        project.camera.hide_when_silent = true;
        let first = store.scene_segments(&project, 20.0).unwrap();
        let again = store.scene_segments(&project, 20.0).unwrap();
        assert!(Arc::ptr_eq(&first, &again));
        assert_eq!(first.len(), 1);
        assert!(matches!(first[0].mode, SceneMode::HideCamera));
        project.camera.hide_when_silent_delay = 5.0;
        let changed = store.scene_segments(&project, 20.0).unwrap();
        assert!(!Arc::ptr_eq(&first, &changed));
        assert!(approx(changed[0].start, 2.0 + 5.0 + 0.3));
        project.camera.hide = true;
        assert!(store.scene_segments(&project, 20.0).is_none());
    }

    fn face(x: f32, y: f32) -> Option<FaceBox> {
        Some(FaceBox {
            x: x - 0.1,
            y: y - 0.1,
            width: 0.2,
            height: 0.2,
        })
    }

    fn track(samples: Vec<Option<FaceBox>>) -> FaceTrack {
        FaceTrack {
            source: SourceFingerprint {
                len: 0,
                modified_ms: 0,
            },
            interval: FACE_SAMPLE_INTERVAL_SECS,
            samples,
        }
    }

    #[test]
    fn smoothing_settles_on_a_still_face_and_ignores_outliers() {
        let mut samples = vec![face(0.3, 0.4); 60];
        samples[30] = face(0.9, 0.9);
        let smoothed = SmoothedFaceTrack::new(&track(samples)).unwrap();
        for step in 0..60 {
            let center = smoothed.sample(step as f64 * FACE_SAMPLE_INTERVAL_SECS);
            assert!((center[0] - 0.3).abs() < 1e-4 && (center[1] - 0.4).abs() < 1e-4);
        }
    }

    #[test]
    fn smoothing_damps_jitter_and_follows_a_move_without_overshoot() {
        let samples: Vec<_> = (0..200)
            .map(|index| {
                let jitter = if index % 2 == 0 { 0.02 } else { -0.02 };
                let x = if index < 100 { 0.3 } else { 0.7 };
                face(x + jitter, 0.5)
            })
            .collect();
        let smoothed = SmoothedFaceTrack::new(&track(samples)).unwrap();
        let at = |index: usize| smoothed.sample(index as f64 * FACE_SAMPLE_INTERVAL_SECS)[0];
        for index in 20..80 {
            assert!((at(index) - 0.3).abs() < 0.006, "{index}: {}", at(index));
        }
        for index in 130..180 {
            assert!((at(index) - 0.7).abs() < 0.006, "{index}: {}", at(index));
        }
        let mut previous = at(85);
        for index in 86..115 {
            let value = at(index);
            assert!(value >= previous - 0.006, "not monotonic at {index}");
            assert!((0.29..=0.71).contains(&value));
            previous = value;
        }
        let midway = at(100);
        assert!(midway > 0.4 && midway < 0.6, "{midway}");
        let largest_step = (1..200)
            .map(|index| (at(index) - at(index - 1)).abs())
            .fold(0.0f32, f32::max);
        assert!(largest_step < 0.05, "{largest_step}");
    }

    #[test]
    fn missing_detections_hold_the_last_face() {
        let mut samples = vec![None; 10];
        samples.extend(vec![face(0.6, 0.5); 20]);
        samples.extend(vec![None; 30]);
        let smoothed = SmoothedFaceTrack::new(&track(samples)).unwrap();
        assert!((smoothed.sample(0.0)[0] - 0.6).abs() < 1e-4);
        assert!((smoothed.sample(5.5)[0] - 0.6).abs() < 1e-4);
        assert!((smoothed.sample(100.0)[0] - 0.6).abs() < 1e-4);
        assert!(SmoothedFaceTrack::new(&track(vec![None; 10])).is_none());
    }

    #[test]
    fn face_crop_zooms_pans_and_stays_inside_the_frame() {
        let frame = [1920.0, 1080.0];
        let square = [420.0, 0.0, 1500.0, 1080.0];
        let crop = face_crop(square, frame, [0.5, 0.5], 1.0);
        assert_eq!(crop, [420.0, 0.0, 1500.0, 1080.0]);
        let crop = face_crop(square, frame, [0.3, 0.5], 1.0);
        assert!((crop[0] - (576.0 - 540.0)).abs() < 1e-3 && crop[1] == 0.0);
        let crop = face_crop(square, frame, [0.5, 0.4], 1.5);
        assert!((crop[2] - crop[0] - 720.0).abs() < 1e-3);
        assert!((crop[3] - crop[1] - 720.0).abs() < 1e-3);
        assert!((crop[1] - (432.0 - 720.0 * FACE_HEADROOM)).abs() < 1e-3);
        let crop = face_crop(square, frame, [0.99, 0.99], 1.6);
        assert!(crop[2] <= 1920.0 + 1e-3 && crop[3] <= 1080.0 + 1e-3);
        assert!(crop[0] >= 0.0 && crop[1] >= 0.0);
        let crop = face_crop(square, frame, [0.0, 0.0], f32::NAN);
        assert_eq!(crop, [0.0, 0.0, 1080.0, 1080.0]);
    }

    #[test]
    fn stale_analysis_is_dropped_when_the_source_changes() {
        let dir = tempfile::tempdir().unwrap();
        let mic = dir.path().join("mic.m4a");
        std::fs::write(&mic, b"abc").unwrap();
        let sources = vec![SegmentSources {
            mic: Some(mic.clone()),
            camera: None,
        }];
        let mut analysis = CameraAnalysis::empty(1);
        analysis.segments[0].speech = Some(SpeechTrack {
            source: SourceFingerprint::of(&mic).unwrap(),
            spans: vec![(1.0, 2.0)],
        });
        analysis.save(dir.path()).unwrap();
        let loaded = CameraAnalysis::load(dir.path(), &sources);
        assert_eq!(loaded, analysis);
        assert!(loaded.has_speech(&sources));
        std::fs::write(&mic, b"abcdef").unwrap();
        let loaded = CameraAnalysis::load(dir.path(), &sources);
        assert!(loaded.segments[0].speech.is_none());
        assert!(!loaded.has_speech(&sources));
        assert_eq!(
            CameraAnalysis::load(&dir.path().join("missing"), &sources),
            CameraAnalysis::empty(1)
        );
    }
}
