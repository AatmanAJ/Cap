use crate::{CaptionsData, KeyPressEvent, KeyboardEvents, TimelineConfiguration};
use std::ops::Range;

#[cfg(test)]
mod tests;

pub const MIN_TYPING_SPEED: f64 = 1.5;
pub const MAX_TYPING_SPEED: f64 = 8.0;
pub const DEFAULT_TYPING_SPEED: f64 = 3.0;

/// Clip edges closer than this to a sped-up range are absorbed into it rather
/// than left behind as a sliver of normal-speed video.
const CLIP_EDGE_SNAP: f64 = 0.1;

// A modifier whose release was never recorded would otherwise turn every
// later keystroke into a "shortcut" and hide all typing after it.
const HELD_MODIFIER_TIMEOUT: f64 = 5.0;

pub const SPEECH_FRAME_SECS: f64 = 0.01;
const MIC_FLOOR_DB: f32 = -60.0;
const MIC_NOISE_PERCENTILE: f64 = 0.2;
const MIC_SPEECH_ABOVE_NOISE_DB: f32 = 12.0;
const MIC_SPEECH_MIN_DB: f32 = -50.0;
const MIC_SPEECH_MIN_FRAMES: usize = 8;
const MIC_SPEECH_MERGE_GAP: f64 = 0.3;

const SHORTCUT_MODIFIERS: &[&str] = &[
    "LControl", "RControl", "LAlt", "RAlt", "LOption", "ROption", "LMeta", "RMeta", "Meta",
    "Command", "RCommand",
];

const NEUTRAL_KEYS: &[&str] = &[
    "LShift", "RShift", "CapsLock", "Function", "Fn", "Up", "Down", "Left", "Right", "Home", "End",
    "PageUp", "PageDown",
];

const TYPING_KEYS: &[&str] = &["Space", "Backspace", "Delete", "Enter", "Return", "Tab"];

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TypingRunSettings {
    /// The longest pause between two keystrokes of the same run.
    pub max_key_gap: f64,
    /// The shortest run (first to last keystroke) worth speeding up.
    pub min_duration: f64,
    pub min_keys: usize,
    /// Runs this close together become one, unless a shortcut sits between.
    pub merge_gap: f64,
    pub padding: f64,
    /// Kept at normal speed on each side of detected speech.
    pub speech_padding: f64,
}

impl Default for TypingRunSettings {
    fn default() -> Self {
        Self {
            max_key_gap: 1.2,
            min_duration: 2.0,
            min_keys: 6,
            merge_gap: 2.5,
            padding: 0.25,
            speech_padding: 0.3,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpeechSource {
    Transcript,
    Microphone,
    TranscriptAndMicrophone,
    None,
}

/// A stretch of typing in one recording clip, in that clip's own time (the
/// time base of [`crate::TimelineSegment::start`] / `end`).
#[derive(Clone, Debug, PartialEq)]
pub struct TypingSection {
    pub recording_clip: u32,
    pub start: f64,
    pub end: f64,
}

/// One range of a timeline clip to speed up, in the clip's source time.
#[derive(Clone, Debug, PartialEq)]
pub struct PlannedTypingSpeedup {
    pub clip_index: usize,
    pub start: f64,
    pub end: f64,
}

impl PlannedTypingSpeedup {
    pub fn duration(&self) -> f64 {
        self.end - self.start
    }
}

pub fn typing_speedup_time_saved(plan: &[PlannedTypingSpeedup], speed: f64) -> f64 {
    if !speed.is_finite() || speed <= 1.0 {
        return 0.0;
    }
    plan.iter()
        .map(|range| range.duration() * (1.0 - 1.0 / speed))
        .sum()
}

pub fn is_typing_key(key: &str) -> bool {
    key.chars().count() == 1 || TYPING_KEYS.contains(&key)
}

fn ordered_events(events: &KeyboardEvents) -> Vec<&KeyPressEvent> {
    let mut ordered: Vec<_> = events
        .presses
        .iter()
        .filter(|event| event.time_ms.is_finite())
        .collect();
    ordered.sort_by(|a, b| a.time_ms.total_cmp(&b.time_ms));
    ordered
}

struct Burst {
    start: f64,
    end: f64,
    keys: usize,
}

/// Typing runs in one recording clip's time: ordinary keystrokes no more than
/// `max_key_gap` apart, at least `min_duration` long, merged across short
/// pauses and padded. A shortcut (a key pressed with Control, Option or
/// Command held) or a non-typing key such as Escape ends a run and keeps the
/// runs on either side apart.
pub fn detect_typing_runs(
    events: &KeyboardEvents,
    capture_offset: f64,
    settings: &TypingRunSettings,
) -> Vec<Range<f64>> {
    let mut held: Vec<(&str, f64)> = Vec::new();
    let mut breaks = Vec::new();
    let mut bursts: Vec<Burst> = Vec::new();
    let mut current: Option<Burst> = None;

    for event in ordered_events(events) {
        let time = event.time_ms / 1000.0 - capture_offset;
        let key = event.key.as_str();
        if SHORTCUT_MODIFIERS.contains(&key) {
            held.retain(|(held_key, _)| *held_key != key);
            if event.down {
                held.push((key, time));
            }
            continue;
        }
        if !event.down || NEUTRAL_KEYS.contains(&key) {
            continue;
        }
        held.retain(|(_, since)| time - since <= HELD_MODIFIER_TIMEOUT);
        if !held.is_empty() || !is_typing_key(key) {
            bursts.extend(current.take());
            breaks.push(time);
            continue;
        }
        match &mut current {
            Some(burst) if time - burst.end <= settings.max_key_gap => {
                burst.end = time;
                burst.keys += 1;
            }
            _ => {
                bursts.extend(current.take());
                current = Some(Burst {
                    start: time,
                    end: time,
                    keys: 1,
                });
            }
        }
    }
    bursts.extend(current);

    let mut runs: Vec<Range<f64>> = Vec::new();
    for burst in bursts {
        if burst.keys < settings.min_keys || burst.end - burst.start < settings.min_duration {
            continue;
        }
        match runs.last_mut() {
            Some(previous)
                if burst.start - previous.end <= settings.merge_gap
                    && !breaks
                        .iter()
                        .any(|time| *time > previous.end && *time < burst.start) =>
            {
                previous.end = burst.end;
            }
            _ => runs.push(burst.start..burst.end),
        }
    }

    merge_ranges(
        runs.into_iter()
            .map(|run| run.start - settings.padding..run.end + settings.padding)
            .collect(),
    )
}

fn merge_ranges(mut ranges: Vec<Range<f64>>) -> Vec<Range<f64>> {
    ranges.retain(|range| {
        range.start.is_finite() && range.end.is_finite() && range.end > range.start
    });
    ranges.sort_by(|a, b| a.start.total_cmp(&b.start));
    let mut merged: Vec<Range<f64>> = Vec::with_capacity(ranges.len());
    for range in ranges {
        match merged.last_mut() {
            Some(last) if range.start <= last.end => last.end = last.end.max(range.end),
            _ => merged.push(range),
        }
    }
    merged
}

/// `ranges` with every `remove` range (widened by `padding` on each side)
/// cut out. Both inputs may be unsorted and overlapping.
pub fn subtract_ranges(
    ranges: &[Range<f64>],
    remove: &[Range<f64>],
    padding: f64,
) -> Vec<Range<f64>> {
    let remove = merge_ranges(
        remove
            .iter()
            .map(|range| range.start - padding..range.end + padding)
            .collect(),
    );
    let mut result = Vec::new();
    for range in merge_ranges(ranges.to_vec()) {
        let mut start = range.start;
        for cut in &remove {
            if cut.end <= start {
                continue;
            }
            if cut.start >= range.end {
                break;
            }
            if cut.start > start {
                result.push(start..cut.start);
            }
            start = start.max(cut.end);
            if start >= range.end {
                break;
            }
        }
        if start < range.end {
            result.push(start..range.end);
        }
    }
    result
}

/// Spoken words from a source-timed transcript, split per recording clip and
/// moved into that clip's time. `None` for a legacy transcript stored in
/// edited output time, which cannot be placed against the recordings.
pub fn transcript_speech_ranges(
    captions: &CaptionsData,
    recording_durations: &[f64],
) -> Option<Vec<Vec<Range<f64>>>> {
    if !captions.source_timed {
        return None;
    }
    let mut spoken: Vec<Range<f64>> = Vec::new();
    for segment in &captions.segments {
        if segment.words.is_empty() {
            spoken.push(f64::from(segment.start)..f64::from(segment.end));
        } else {
            spoken.extend(
                segment
                    .words
                    .iter()
                    .map(|word| f64::from(word.start)..f64::from(word.end)),
            );
        }
    }
    let spoken = merge_ranges(spoken);
    let mut offset = 0.0;
    Some(
        recording_durations
            .iter()
            .map(|duration| {
                let clip_start = offset;
                let clip_end = offset + duration.max(0.0);
                offset = clip_end;
                spoken
                    .iter()
                    .filter_map(|range| {
                        let start = range.start.max(clip_start);
                        let end = range.end.min(clip_end);
                        (end > start).then(|| start - clip_start..end - clip_start)
                    })
                    .collect()
            })
            .collect(),
    )
}

/// Mean absolute amplitude in dBFS for every [`SPEECH_FRAME_SECS`] of
/// interleaved audio, the input [`microphone_speech_ranges`] expects.
pub fn audio_frame_levels<'a>(
    slices: impl IntoIterator<Item = &'a [f32]>,
    channels: u16,
    sample_rate: u32,
) -> Vec<f32> {
    let frame_samples = ((f64::from(sample_rate) * SPEECH_FRAME_SECS).round() as usize).max(1)
        * usize::from(channels.max(1));
    fn push(sum: f32, count: usize, levels: &mut Vec<f32>) {
        let mean = sum / count as f32;
        levels.push(if mean > 0.0 {
            (20.0 * mean.log10()).max(MIC_FLOOR_DB)
        } else {
            MIC_FLOOR_DB
        });
    }

    let mut levels = Vec::new();
    let mut sum = 0.0_f32;
    let mut count = 0;
    for slice in slices {
        for sample in slice {
            sum += sample.abs();
            count += 1;
            if count == frame_samples {
                push(sum, count, &mut levels);
                sum = 0.0;
                count = 0;
            }
        }
    }
    if count > 0 {
        push(sum, count, &mut levels);
    }
    levels
}

/// Voice activity from a microphone's [`audio_frame_levels`]: stretches that
/// stay well above the track's noise floor for at least 80ms, allowing
/// single-frame dips, merged across pauses of up to 0.3s. A keystroke's click
/// on the same mic lasts 20 to 40ms, so typing alone does not read as speech.
pub fn microphone_speech_ranges(frame_levels_db: &[f32]) -> Vec<Range<f64>> {
    if frame_levels_db.is_empty() {
        return Vec::new();
    }
    let levels: Vec<f32> = frame_levels_db
        .iter()
        .map(|level| {
            if level.is_finite() {
                level.max(MIC_FLOOR_DB)
            } else {
                MIC_FLOOR_DB
            }
        })
        .collect();
    let mut sorted = levels.clone();
    sorted.sort_by(f32::total_cmp);
    let noise_index = ((sorted.len() - 1) as f64 * MIC_NOISE_PERCENTILE).round() as usize;
    let threshold = (sorted[noise_index] + MIC_SPEECH_ABOVE_NOISE_DB).max(MIC_SPEECH_MIN_DB);
    let loud: Vec<bool> = levels.iter().map(|level| *level >= threshold).collect();

    let mut ranges: Vec<Range<f64>> = Vec::new();
    let mut index = 0;
    while index < loud.len() {
        if !loud[index] {
            index += 1;
            continue;
        }
        let first = index;
        let mut last = index;
        index += 1;
        while index < loud.len() && (loud[index] || loud.get(index + 1) == Some(&true)) {
            if loud[index] {
                last = index;
            }
            index += 1;
        }
        if last - first + 1 < MIC_SPEECH_MIN_FRAMES {
            continue;
        }
        let start = first as f64 * SPEECH_FRAME_SECS;
        let end = (last + 1) as f64 * SPEECH_FRAME_SECS;
        match ranges.last_mut() {
            Some(previous) if start - previous.end <= MIC_SPEECH_MERGE_GAP => previous.end = end,
            _ => ranges.push(start..end),
        }
    }
    ranges
}

/// Speech per recording clip: the transcript's words when the project has a
/// source-timed transcript, joined with the microphone's voice activity
/// wherever a clip has a mic track (`mic_speech[i]` is `None` for a clip
/// without one). Both are used when both exist, because a transcript can drop
/// quiet or mumbled speech and the mic cannot tell talking from other loud
/// sound; either way speech is kept at normal speed.
pub fn project_speech_ranges(
    captions: Option<&CaptionsData>,
    recording_durations: &[f64],
    mic_speech: &[Option<Vec<Range<f64>>>],
) -> (SpeechSource, Vec<Vec<Range<f64>>>) {
    let transcript = captions
        .filter(|captions| !captions.segments.is_empty())
        .and_then(|captions| transcript_speech_ranges(captions, recording_durations));
    let has_mic = mic_speech.iter().any(Option::is_some);
    let source = match (transcript.is_some(), has_mic) {
        (true, true) => SpeechSource::TranscriptAndMicrophone,
        (true, false) => SpeechSource::Transcript,
        (false, true) => SpeechSource::Microphone,
        (false, false) => SpeechSource::None,
    };
    let clips = recording_durations.len().max(mic_speech.len());
    let ranges = (0..clips)
        .map(|index| {
            let mut ranges: Vec<Range<f64>> = transcript
                .as_ref()
                .and_then(|transcript| transcript.get(index))
                .cloned()
                .unwrap_or_default();
            if let Some(Some(mic)) = mic_speech.get(index) {
                ranges.extend(mic.iter().cloned());
            }
            merge_ranges(ranges)
        })
        .collect();
    (source, ranges)
}

/// Typing runs in every recording clip with speech cut out. `takes[i]` holds
/// recording clip `i`'s keyboard events and their capture offset, as
/// [`crate::load_project_keyboard_events`] returns them.
pub fn find_typing_sections(
    takes: &[(KeyboardEvents, f64)],
    speech: &[Vec<Range<f64>>],
    settings: &TypingRunSettings,
) -> Vec<TypingSection> {
    let mut sections = Vec::new();
    for (index, (events, offset)) in takes.iter().enumerate() {
        if !offset.is_finite() {
            continue;
        }
        let runs = detect_typing_runs(events, *offset, settings);
        let spoken = speech.get(index).map(Vec::as_slice).unwrap_or_default();
        sections.extend(
            subtract_ranges(&runs, spoken, settings.speech_padding)
                .into_iter()
                .filter(|range| range.end - range.start >= settings.min_duration)
                .map(|range| TypingSection {
                    recording_clip: index as u32,
                    start: range.start,
                    end: range.end,
                }),
        );
    }
    sections
}

fn transition_duration(timeline: &TimelineConfiguration, index: usize) -> f64 {
    timeline
        .effective_transition(index)
        .map_or(0.0, |transition| transition.duration)
}

/// Where `sections` land on the current timeline: each part of a section
/// that a normal-speed clip still shows, as a source range of that clip.
/// Cut material is skipped, clips that already have a speed of their own are
/// left alone, a range keeps clear of the clip's transitions the way a manual
/// split must, and a part shorter than `min_duration` is dropped. Sorted by
/// clip, then time.
pub fn plan_typing_speedup(
    timeline: &TimelineConfiguration,
    sections: &[TypingSection],
    settings: &TypingRunSettings,
) -> Vec<PlannedTypingSpeedup> {
    let mut plan = Vec::new();
    for (clip_index, clip) in timeline.segments.iter().enumerate() {
        if (clip.timescale - 1.0).abs() > 1e-9
            || !clip.start.is_finite()
            || !clip.end.is_finite()
            || clip.end <= clip.start
        {
            continue;
        }
        let incoming = transition_duration(timeline, clip_index) * 2.0;
        let outgoing = transition_duration(timeline, clip_index + 1) * 2.0;
        let first = clip.start + incoming;
        let last = clip.end - outgoing;
        let mut ranges: Vec<Range<f64>> = Vec::new();
        for section in sections
            .iter()
            .filter(|section| section.recording_clip == clip.recording_clip)
        {
            let mut start = section.start.max(first);
            let mut end = section.end.min(last);
            if incoming == 0.0 && start - clip.start < CLIP_EDGE_SNAP {
                start = clip.start;
            }
            if outgoing == 0.0 && clip.end - end < CLIP_EDGE_SNAP {
                end = clip.end;
            }
            if end - start < settings.min_duration {
                continue;
            }
            match ranges.last_mut() {
                Some(previous) if start - previous.end < CLIP_EDGE_SNAP => {
                    previous.end = previous.end.max(end);
                }
                _ => ranges.push(start..end),
            }
        }
        plan.extend(ranges.into_iter().map(|range| PlannedTypingSpeedup {
            clip_index,
            start: range.start,
            end: range.end,
        }));
    }
    plan
}
