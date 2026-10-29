use cap_project::{
    ClipSourceWindow, ProjectConfiguration, SceneMode, SceneSegment, StudioRecordingMeta,
};
use serde::{Deserialize, Serialize};
use std::{
    hash::{Hash, Hasher},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, RwLock},
    time::UNIX_EPOCH,
};

use crate::scene::MIN_GAP_FOR_TRANSITION;

pub const CAMERA_ANALYSIS_FILE: &str = "camera-analysis.json";
const CAMERA_ANALYSIS_VERSION: u32 = 2;

pub const SILENCE_SHOW_LEAD_SECS: f64 = 0.3;
pub const SILENCE_MIN_SPEECH_SECS: f64 = 0.2;
pub const SILENCE_TRANSITION_SECS: f64 = 0.3;
const SILENCE_MIN_HIDDEN_SECS: f64 = 1.0;

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
    #[serde(default)]
    pub timing_offset: f64,
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
        static WRITES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let path = Self::path(project_path);
        let temporary = project_path.join(format!(
            ".{CAMERA_ANALYSIS_FILE}.{}.{}.tmp",
            std::process::id(),
            WRITES.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let result = std::fs::write(&temporary, serde_json::to_vec(self)?)
            .and_then(|()| std::fs::rename(&temporary, path));
        if result.is_err() {
            std::fs::remove_file(&temporary).ok();
        }
        result
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

pub fn timeline_speech(
    project: &ProjectConfiguration,
    analysis: &CameraAnalysis,
    duration: f64,
) -> Vec<(f64, f64)> {
    let windows = match &project.timeline {
        Some(timeline) => timeline.clip_source_windows(),
        None => vec![ClipSourceWindow {
            recording_clip: 0,
            output_start: 0.0,
            output_end: duration,
            source_start: 0.0,
            timescale: 1.0,
        }],
    };
    let mut runs: Vec<(f64, f64)> = Vec::new();
    for window in windows {
        if !(window.timescale.is_finite() && window.timescale > 0.0) {
            continue;
        }
        let speech = analysis
            .segments
            .get(window.recording_clip as usize)
            .and_then(|segment| segment.speech.as_ref());
        let Some(speech) = speech else {
            runs.push((window.output_start, window.output_end));
            continue;
        };
        let offset = project
            .clips
            .iter()
            .find(|config| config.index == window.recording_clip)
            .map_or(0.0, |config| f64::from(config.offsets.mic))
            + speech.timing_offset;
        let source_end =
            window.source_start + (window.output_end - window.output_start) * window.timescale;
        let first = speech
            .spans
            .partition_point(|(_, end)| end - offset <= window.source_start);
        for &(start, end) in &speech.spans[first..] {
            let (start, end) = (start - offset, end - offset);
            if start >= source_end {
                break;
            }
            let start = start.max(window.source_start);
            let end = end.min(source_end);
            if end > start {
                runs.push((
                    window.output_start + (start - window.source_start) / window.timescale,
                    window.output_start + (end - window.source_start) / window.timescale,
                ));
            }
        }
    }
    merge_spans(runs)
}

fn merge_spans(mut spans: Vec<(f64, f64)>) -> Vec<(f64, f64)> {
    spans.sort_by(|a, b| a.0.total_cmp(&b.0));
    let mut merged: Vec<(f64, f64)> = Vec::with_capacity(spans.len());
    for (start, end) in spans {
        match merged.last_mut() {
            Some(last) if start <= last.1 + 1e-6 => last.1 = last.1.max(end),
            _ => merged.push((start, end)),
        }
    }
    merged
}

fn intersect_spans(a: &[(f64, f64)], b: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let mut result = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        let start = a[i].0.max(b[j].0);
        let end = a[i].1.min(b[j].1);
        if end > start {
            result.push((start, end));
        }
        if a[i].1 < b[j].1 {
            i += 1;
        } else {
            j += 1;
        }
    }
    result
}

fn source_at(project: &ProjectConfiguration, time: f64) -> Option<(f64, u32)> {
    if project.timeline.is_none() {
        return Some((time, 0));
    }
    project
        .get_segment_time(time)
        .map(|(source_time, segment)| (source_time, segment.recording_clip))
}

// Style segments can replace the whole camera config, so the silence setting
// is resolved per range the same way the renderer resolves it per frame.
fn silence_regimes(project: &ProjectConfiguration, duration: f64) -> Vec<(f64, f64, Option<f64>)> {
    let mut cuts = vec![0.0, duration];
    if let Some(timeline) = &project.timeline {
        for segment in &timeline.style_segments {
            if segment.overrides.camera.is_some() {
                cuts.extend([segment.start, segment.end]);
            }
        }
    }
    cuts.retain(|cut| cut.is_finite() && (0.0..=duration).contains(cut));
    cuts.sort_by(f64::total_cmp);
    cuts.dedup();
    cuts.windows(2)
        .filter(|pair| pair[1] > pair[0])
        .map(|pair| {
            let camera = project.camera_at((pair[0] + pair[1]) / 2.0);
            let delay = (!camera.hide && camera.hide_when_silent)
                .then(|| f64::from(camera.hide_when_silent_delay_clamped()));
            (pair[0], pair[1], delay)
        })
        .collect()
}

pub fn silence_windows_for_project(
    project: &ProjectConfiguration,
    analysis: &CameraAnalysis,
    duration: f64,
) -> Vec<(f64, f64)> {
    let regimes = silence_regimes(project, duration);
    if regimes.iter().all(|(_, _, delay)| delay.is_none()) {
        return Vec::new();
    }
    let speech = timeline_speech(project, analysis, duration);
    let mut delays: Vec<f64> = regimes.iter().filter_map(|(_, _, delay)| *delay).collect();
    delays.sort_by(f64::total_cmp);
    delays.dedup();
    let mut windows = Vec::new();
    for delay in delays {
        let ranges: Vec<(f64, f64)> = regimes
            .iter()
            .filter(|(_, _, regime)| *regime == Some(delay))
            .map(|(start, end, _)| (*start, *end))
            .collect();
        let all = silence_windows(&speech, duration, SilenceSettings::new(delay));
        windows.extend(intersect_spans(&all, &merge_spans(ranges)));
    }
    merge_spans(windows)
}

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

pub fn merge_scene_segments(
    explicit: &[SceneSegment],
    silence: &[SceneSegment],
) -> Vec<SceneSegment> {
    let mut segments: Vec<SceneSegment> = explicit.iter().chain(silence).cloned().collect();
    segments.sort_by(|a, b| a.start.total_cmp(&b.start));
    segments
}

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

pub fn face_crop(base: [f32; 4], frame: [f32; 2], center: [f32; 2], zoom: f32) -> [f32; 4] {
    let zoom = if zoom.is_finite() { zoom.max(1.0) } else { 1.0 };
    let width = (base[2] - base[0]) / zoom;
    let height = (base[3] - base[1]) / zoom;
    let face = [center[0] * frame[0], center[1] * frame[1]];
    let x = (face[0] - width / 2.0).clamp(0.0, (frame[0] - width).max(0.0));
    let y = (face[1] - height * FACE_HEADROOM).clamp(0.0, (frame[1] - height).max(0.0));
    [x, y, x + width, y + height]
}

// Starting a split morph from the face crop (not the centred shape crop) keeps
// the first transition frame from jumping; scaling the pane crop instead of
// lerping corners keeps its aspect, so the morph never stretches.
pub fn blend_from_face_crop(
    pane: [f32; 4],
    face: [f32; 4],
    shape: [f32; 4],
    frame: [f32; 2],
    t: f32,
) -> [f32; 4] {
    let t = t.clamp(0.0, 1.0);
    let lerp = |a: f32, b: f32| a + (b - a) * t;
    let zoomed = (face[2] - face[0]) / (shape[2] - shape[0]).max(f32::EPSILON);
    let scale = lerp(zoomed, 1.0);
    let width = (pane[2] - pane[0]) * scale;
    let height = (pane[3] - pane[1]) * scale;
    let center = [
        lerp((face[0] + face[2]) / 2.0, (pane[0] + pane[2]) / 2.0),
        lerp((face[1] + face[3]) / 2.0, (pane[1] + pane[3]) / 2.0),
    ];
    let x = (center[0] - width / 2.0).clamp(0.0, (frame[0] - width).max(0.0));
    let y = (center[1] - height / 2.0).clamp(0.0, (frame[1] - height).max(0.0));
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

    pub fn silence_scene_segments(
        &self,
        project: &ProjectConfiguration,
        fallback_duration: f64,
    ) -> Option<Arc<Vec<SceneSegment>>> {
        if !project.wants_speech_analysis() {
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
        let duration = project
            .timeline
            .as_ref()
            .map_or(fallback_duration, |timeline| timeline.duration());
        let key = silence_key(project, duration, loaded.generation);
        let mut cached = self.scenes.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(cached) = cached.as_ref().filter(|cached| cached.key == key) {
            return Some(Arc::clone(&cached.segments));
        }
        let explicit = project
            .timeline
            .as_ref()
            .map_or(&[][..], |timeline| timeline.scene_segments.as_slice());
        let windows = silence_windows_for_project(project, &loaded.analysis, duration);
        let segments = Arc::new(silence_scene_segments(
            &windows,
            explicit,
            SILENCE_TRANSITION_SECS,
        ));
        *cached = Some(CachedScenes {
            key,
            segments: Arc::clone(&segments),
        });
        Some(segments)
    }

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

fn hash_camera(camera: &cap_project::Camera, hasher: &mut impl Hasher) {
    camera.hide.hash(hasher);
    camera.hide_when_silent.hash(hasher);
    camera.hide_when_silent_delay.to_bits().hash(hasher);
}

fn silence_key(project: &ProjectConfiguration, duration: f64, generation: u64) -> u64 {
    let mut hasher = std::hash::DefaultHasher::new();
    generation.hash(&mut hasher);
    duration.to_bits().hash(&mut hasher);
    hash_camera(&project.camera, &mut hasher);
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
        for segment in &timeline.text_segments {
            segment.start.to_bits().hash(&mut hasher);
            segment.end.to_bits().hash(&mut hasher);
            segment.enabled.hash(&mut hasher);
            std::mem::discriminant(&segment.layout).hash(&mut hasher);
        }
        for segment in &timeline.style_segments {
            segment.start.to_bits().hash(&mut hasher);
            segment.end.to_bits().hash(&mut hasher);
            segment.enabled.hash(&mut hasher);
            segment.track.hash(&mut hasher);
            if let Some(camera) = &segment.overrides.camera {
                hash_camera(camera, &mut hasher);
            }
        }
        for segment in &timeline.scene_segments {
            segment.start.to_bits().hash(&mut hasher);
            segment.end.to_bits().hash(&mut hasher);
            segment.transition_in.to_bits().hash(&mut hasher);
            segment.transition_out.to_bits().hash(&mut hasher);
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
                        timing_offset: 0.0,
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

        let mut repaired = analysis.clone();
        repaired.segments[0].speech.as_mut().unwrap().timing_offset = -0.5;
        let runs = timeline_speech(&project, &repaired, 15.0);
        assert!((runs[0].0 - 1.5).abs() < 1e-9 && (runs[0].1 - 3.5).abs() < 1e-9);
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
        let merged = merge_scene_segments(&explicit, &segments);
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
        assert!(store.silence_scene_segments(&project, 20.0).is_none());
        project.camera.hide_when_silent = true;
        let first = store.silence_scene_segments(&project, 20.0).unwrap();
        let again = store.silence_scene_segments(&project, 20.0).unwrap();
        assert!(Arc::ptr_eq(&first, &again));
        assert_eq!(first.len(), 1);
        assert!(matches!(first[0].mode, SceneMode::HideCamera));
        project.camera.hide_when_silent_delay = 5.0;
        let changed = store.silence_scene_segments(&project, 20.0).unwrap();
        assert!(!Arc::ptr_eq(&first, &changed));
        assert!(approx(changed[0].start, 2.0 + 5.0 + 0.3));
        project.camera.hide = true;
        assert!(store.silence_scene_segments(&project, 20.0).is_none());
    }

    #[test]
    fn editing_a_manual_split_layout_reaches_the_merged_segments() {
        let store = CameraAnalysisStore::default();
        let mut project = project_with(vec![clip(0, 0.0, 40.0, 1.0)]);
        project.camera.hide_when_silent = true;
        store.set(analysis_with(vec![vec![(0.0, 2.0), (35.0, 38.0)]]));
        let mut split = scene(20.0, 24.0, SceneMode::SplitScreen);
        split.split_layout = Some(cap_project::SplitLayout::default());
        project.timeline.as_mut().unwrap().scene_segments = vec![split];

        let silence = store.silence_scene_segments(&project, 40.0).unwrap();
        assert!(
            silence
                .iter()
                .all(|segment| matches!(segment.mode, SceneMode::HideCamera))
        );

        project.timeline.as_mut().unwrap().scene_segments[0]
            .split_layout
            .as_mut()
            .unwrap()
            .camera_zoom = 2.5;
        let silence = store.silence_scene_segments(&project, 40.0).unwrap();
        let merged =
            merge_scene_segments(&project.timeline.as_ref().unwrap().scene_segments, &silence);
        let manual = merged
            .iter()
            .find(|segment| matches!(segment.mode, SceneMode::SplitScreen))
            .unwrap();
        assert_eq!(manual.split_layout.unwrap().camera_zoom, 2.5);
    }

    #[test]
    fn style_segments_switch_hide_when_silent_on_and_off() {
        let store = CameraAnalysisStore::default();
        let mut project = project_with(vec![clip(0, 0.0, 60.0, 1.0)]);
        store.set(analysis_with(vec![vec![
            (0.0, 1.0),
            (10.0, 11.0),
            (20.0, 21.0),
            (30.0, 31.0),
            (40.0, 41.0),
            (59.0, 60.0),
        ]]));
        let enabled = cap_project::Camera {
            hide_when_silent: true,
            hide_when_silent_delay: 3.0,
            ..Default::default()
        };
        project.timeline.as_mut().unwrap().style_segments = vec![cap_project::StyleSegment {
            start: 20.0,
            end: 41.0,
            overrides: cap_project::StyleOverrides {
                camera: Some(enabled),
                ..Default::default()
            },
            ..Default::default()
        }];
        assert!(project.wants_speech_analysis());
        let windows = silence_windows_for_project(&project, &store.analysis(), 60.0);
        assert_eq!(windows.len(), 2, "{windows:?}");
        assert!(approx(windows[0].0, 24.0) && approx(windows[0].1, 29.7));
        assert!(approx(windows[1].0, 34.0) && approx(windows[1].1, 39.7));

        project.camera.hide_when_silent = true;
        let windows = silence_windows_for_project(&project, &store.analysis(), 60.0);
        assert_eq!(windows.len(), 5, "{windows:?}");
        assert!(approx(windows[0].0, 3.0) && approx(windows[0].1, 9.7));
        assert!(approx(windows[2].0, 24.0));

        project.timeline.as_mut().unwrap().style_segments[0]
            .overrides
            .camera
            .as_mut()
            .unwrap()
            .hide = true;
        let windows = silence_windows_for_project(&project, &store.analysis(), 60.0);
        assert!(
            windows
                .iter()
                .all(|(start, end)| *end <= 20.0 || *start >= 41.0)
        );
    }

    #[test]
    fn duration_comes_from_the_timeline_so_preview_and_export_agree() {
        let store = CameraAnalysisStore::default();
        let mut project = project_with(vec![clip(0, 0.0, 20.0, 1.0)]);
        project.camera.hide_when_silent = true;
        store.set(analysis_with(vec![vec![(0.0, 2.0)]]));
        let preview = store.silence_scene_segments(&project, 0.0).unwrap();
        let export = store.silence_scene_segments(&project, 20.0).unwrap();
        let bounds = |segments: &[SceneSegment]| -> Vec<(f64, f64)> {
            segments
                .iter()
                .map(|segment| (segment.start, segment.end))
                .collect()
        };
        assert_eq!(bounds(&preview), bounds(&export));
        assert_eq!(preview.len(), 1);
        assert!(approx(preview[0].end, 20.0));
    }

    #[test]
    fn interval_mapping_matches_the_frame_mapping() {
        let mut project = project_with(vec![
            clip(0, 1.0, 9.0, 1.0),
            clip(1, 3.0, 13.0, 2.0),
            clip(0, 20.0, 26.0, 0.5),
        ]);
        let timeline = project.timeline.as_mut().unwrap();
        timeline.transitions = serde_json::from_value(serde_json::json!([
            { "segmentIndex": 1, "type": "cross-fade", "duration": 0.8 }
        ]))
        .unwrap();
        timeline.text_segments = serde_json::from_value(serde_json::json!([
            { "start": 6.0, "end": 8.5, "layout": "fullscreen", "content": "Title" }
        ]))
        .unwrap();
        let timeline = project.timeline.as_ref().unwrap();
        assert!(!timeline.hold_windows().is_empty());
        let windows = timeline.clip_source_windows();
        let duration = timeline.duration();
        let mut step = 0.0;
        while step < duration {
            let expected = project
                .get_segment_time(step)
                .map(|(time, segment)| (time, segment.recording_clip));
            let in_hold = timeline
                .hold_windows()
                .iter()
                .any(|(start, end)| step >= *start && step < *end);
            let mapped = windows
                .iter()
                .find(|w| step >= w.output_start && step < w.output_end)
                .map(|w| {
                    (
                        w.source_start + (step - w.output_start) * w.timescale,
                        w.recording_clip,
                    )
                });
            if in_hold {
                assert!(mapped.is_none(), "{step}");
            } else {
                let (expected, mapped) = (expected.unwrap(), mapped.unwrap());
                assert_eq!(expected.1, mapped.1, "{step}");
                assert!(
                    (expected.0 - mapped.0).abs() < 1e-6,
                    "{step}: {expected:?} {mapped:?}"
                );
            }
            step += 0.013;
        }
    }

    #[test]
    fn mapping_a_long_cut_up_timeline_is_fast() {
        let segments: Vec<_> = (0..200)
            .map(|index| clip(0, index as f64 * 40.0, index as f64 * 40.0 + 36.0, 1.0))
            .collect();
        let mut project = project_with(segments);
        project.camera.hide_when_silent = true;
        let spans: Vec<(f64, f64)> = (0..4000)
            .map(|index| (index as f64 * 2.0, index as f64 * 2.0 + 1.2))
            .collect();
        let store = CameraAnalysisStore::default();
        store.set(analysis_with(vec![spans]));
        let started = std::time::Instant::now();
        let segments = store.silence_scene_segments(&project, 0.0).unwrap();
        let elapsed = started.elapsed();
        assert!(segments.is_empty());
        assert!(
            elapsed < std::time::Duration::from_millis(50),
            "{elapsed:?}"
        );
    }

    #[test]
    fn concurrent_saves_never_leave_a_corrupt_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().to_path_buf();
        let threads: Vec<_> = (0..8)
            .map(|index| {
                let path = path.clone();
                std::thread::spawn(move || {
                    let mut analysis = analysis_with(vec![vec![(0.0, index as f64 + 1.0)]]);
                    analysis.segments[0].speech.as_mut().unwrap().spans = (0..2000)
                        .map(|span| (span as f64, span as f64 + 0.5))
                        .collect();
                    for _ in 0..5 {
                        analysis.save(&path).unwrap();
                    }
                })
            })
            .collect();
        for thread in threads {
            thread.join().unwrap();
        }
        let bytes = std::fs::read(CameraAnalysis::path(&path)).unwrap();
        serde_json::from_slice::<CameraAnalysis>(&bytes).unwrap();
        let leftovers = std::fs::read_dir(&path)
            .unwrap()
            .filter(|entry| entry.as_ref().unwrap().file_name() != CAMERA_ANALYSIS_FILE)
            .count();
        assert_eq!(leftovers, 0);
    }

    #[test]
    fn split_morph_starts_from_the_face_crop_without_a_jump() {
        let frame = [1920.0, 1080.0];
        let shape = [420.0, 0.0, 1500.0, 1080.0];
        let face = face_crop(shape, frame, [0.3, 0.4], 1.3);
        let pane_at_start = shape;
        assert_eq!(
            blend_from_face_crop(pane_at_start, face, shape, frame, 0.0),
            face
        );
        let pane = [0.0, 100.0, 960.0, 1000.0];
        assert_eq!(blend_from_face_crop(pane, face, shape, frame, 1.0), pane);
        let mid = blend_from_face_crop(pane, face, shape, frame, 0.5);
        let aspect = (mid[2] - mid[0]) / (mid[3] - mid[1]);
        assert!((aspect - 960.0 / 900.0).abs() < 1e-3);
        assert!(mid[0] >= 0.0 && mid[2] <= 1920.0 && mid[1] >= 0.0 && mid[3] <= 1080.0);
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
            timing_offset: 0.0,
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
