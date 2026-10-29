use super::*;
use crate::{
    CaptionSegment, CaptionWord, ClipTransition, ClipTransitionType, KeyPressEvent, TimelineSegment,
};

fn press(key: &str, time: f64, down: bool) -> KeyPressEvent {
    KeyPressEvent {
        key: key.to_string(),
        key_code: key.to_string(),
        time_ms: time * 1000.0,
        down,
    }
}

fn typed(keys: &[(&str, f64)]) -> KeyboardEvents {
    let mut presses = Vec::new();
    for (key, time) in keys {
        presses.push(press(key, *time, true));
        presses.push(press(key, time + 0.08, false));
    }
    KeyboardEvents { presses }
}

fn letters(start: f64, count: usize, interval: f64) -> Vec<(&'static str, f64)> {
    (0..count)
        .map(|index| ("a", start + index as f64 * interval))
        .collect()
}

fn settings() -> TypingRunSettings {
    TypingRunSettings::default()
}

fn assert_ranges(actual: &[Range<f64>], expected: &[(f64, f64)]) {
    assert_eq!(actual.len(), expected.len(), "{actual:?} vs {expected:?}");
    for (actual, (start, end)) in actual.iter().zip(expected) {
        assert!(
            (actual.start - start).abs() < 1e-9 && (actual.end - end).abs() < 1e-9,
            "{actual:?} vs {start}..{end}"
        );
    }
}

fn clip(recording_clip: u32, start: f64, end: f64, timescale: f64) -> TimelineSegment {
    TimelineSegment {
        recording_clip,
        timescale,
        start,
        end,
        ..Default::default()
    }
}

fn timeline(segments: Vec<TimelineSegment>) -> TimelineConfiguration {
    TimelineConfiguration {
        segments,
        transitions: Vec::new(),
        zoom_segments: Vec::new(),
        scene_segments: Vec::new(),
        mask_segments: Vec::new(),
        text_segments: Vec::new(),
        caption_segments: Vec::new(),
        keyboard_segments: Vec::new(),
        audio_segments: Vec::new(),
        style_segments: Vec::new(),
        image_segments: Vec::new(),
        camera3d_segments: Vec::new(),
    }
}

fn section(recording_clip: u32, start: f64, end: f64) -> TypingSection {
    TypingSection {
        recording_clip,
        start,
        end,
    }
}

#[test]
fn steady_typing_is_one_padded_run() {
    let events = typed(&letters(10.0, 11, 0.3));
    let runs = detect_typing_runs(&events, 0.0, &settings());
    assert_ranges(&runs, &[(9.75, 13.25)]);
}

#[test]
fn capture_offset_moves_runs_into_clip_time() {
    let events = typed(&letters(10.0, 11, 0.3));
    let runs = detect_typing_runs(&events, 4.0, &settings());
    assert_ranges(&runs, &[(5.75, 9.25)]);
}

#[test]
fn short_or_sparse_typing_is_ignored() {
    let short = typed(&letters(0.0, 6, 0.3));
    assert!(detect_typing_runs(&short, 0.0, &settings()).is_empty());

    let sparse = typed(&letters(0.0, 5, 1.0));
    assert!(detect_typing_runs(&sparse, 0.0, &settings()).is_empty());

    let spaced_out = typed(&letters(0.0, 8, 1.3));
    assert!(detect_typing_runs(&spaced_out, 0.0, &settings()).is_empty());
}

#[test]
fn a_long_pause_splits_runs_and_a_short_one_merges_them() {
    let mut keys = letters(0.0, 8, 0.3);
    keys.extend(letters(2.1 + 2.0, 8, 0.3));
    let merged = detect_typing_runs(&typed(&keys), 0.0, &settings());
    assert_ranges(&merged, &[(-0.25, 6.45)]);

    let mut keys = letters(0.0, 8, 0.3);
    keys.extend(letters(2.1 + 3.0, 8, 0.3));
    let split = detect_typing_runs(&typed(&keys), 0.0, &settings());
    assert_ranges(&split, &[(-0.25, 2.35), (4.85, 7.45)]);
}

#[test]
fn padding_that_closes_a_gap_joins_runs() {
    let mut keys = letters(0.0, 8, 0.3);
    keys.extend(letters(2.1 + 0.4 + 2.6, 8, 0.3));
    let runs = detect_typing_runs(
        &typed(&keys),
        0.0,
        &TypingRunSettings {
            merge_gap: 0.0,
            padding: 1.6,
            ..settings()
        },
    );
    assert_ranges(&runs, &[(-1.6, 8.8)]);
}

#[test]
fn shortcuts_break_runs_and_are_not_typing() {
    let mut events = typed(&letters(0.0, 8, 0.3));
    events.presses.push(press("LMeta", 2.5, true));
    events.presses.push(press("v", 2.6, true));
    events.presses.push(press("v", 2.65, false));
    events.presses.push(press("LMeta", 2.7, false));
    events.presses.extend(typed(&letters(3.0, 8, 0.3)).presses);

    let runs = detect_typing_runs(&events, 0.0, &settings());
    assert_ranges(&runs, &[(-0.25, 2.35), (2.75, 5.35)]);

    let mut shortcuts = KeyboardEvents::default();
    for index in 0..12 {
        let time = index as f64 * 0.3;
        shortcuts.presses.push(press("LControl", time, true));
        shortcuts.presses.push(press("c", time + 0.05, true));
        shortcuts.presses.push(press("LControl", time + 0.1, false));
    }
    assert!(detect_typing_runs(&shortcuts, 0.0, &settings()).is_empty());
}

#[test]
fn shift_caps_lock_and_arrows_neither_extend_nor_break_a_run() {
    let mut events = typed(&letters(0.0, 8, 0.3));
    events.presses.push(press("LShift", 0.5, true));
    events.presses.push(press("LShift", 0.7, false));
    events.presses.push(press("CapsLock", 1.0, true));
    events.presses.push(press("Left", 1.1, true));
    events.presses.push(press("Left", 2.2, true));
    events.presses.push(press("Left", 3.3, true));
    let runs = detect_typing_runs(&events, 0.0, &settings());
    assert_ranges(&runs, &[(-0.25, 2.35)]);
}

#[test]
fn escape_ends_a_run_and_blocks_merging() {
    let mut keys = letters(0.0, 8, 0.3);
    keys.push(("Escape", 2.5));
    keys.extend(letters(3.0, 8, 0.3));
    let runs = detect_typing_runs(&typed(&keys), 0.0, &settings());
    assert_ranges(&runs, &[(-0.25, 2.35), (2.75, 5.35)]);
}

#[test]
fn special_typing_keys_count_as_typing() {
    let keys: Vec<_> = [
        "h",
        "Space",
        "Backspace",
        "Enter",
        "Tab",
        "Delete",
        "é",
        ";",
    ]
    .into_iter()
    .enumerate()
    .map(|(index, key)| (key, index as f64 * 0.4))
    .collect();
    let runs = detect_typing_runs(&typed(&keys), 0.0, &settings());
    assert_ranges(&runs, &[(-0.25, 3.05)]);
    assert!(!is_typing_key("F5"));
    assert!(!is_typing_key("Escape"));
}

#[test]
fn a_modifier_whose_release_was_lost_stops_blocking_typing() {
    let mut events = KeyboardEvents::default();
    events.presses.push(press("LMeta", 0.0, true));
    events.presses.extend(typed(&letters(1.0, 30, 0.3)).presses);
    let runs = detect_typing_runs(&events, 0.0, &settings());
    assert_ranges(&runs, &[(4.95, 9.95)]);
}

#[test]
fn unordered_and_non_finite_events_are_tolerated() {
    let mut events = typed(&letters(0.0, 8, 0.3));
    events.presses.reverse();
    events.presses.push(press("a", f64::NAN, true));
    events.presses.push(press("a", f64::INFINITY, true));
    let runs = detect_typing_runs(&events, 0.0, &settings());
    assert_ranges(&runs, &[(-0.25, 2.35)]);
}

#[test]
fn held_keys_repeat_downs_are_typing_and_ups_are_ignored() {
    let mut events = KeyboardEvents::default();
    for index in 0..30 {
        events.presses.push(press("x", index as f64 * 0.1, true));
    }
    events.presses.push(press("x", 2.95, false));
    let runs = detect_typing_runs(&events, 0.0, &settings());
    assert_ranges(&runs, &[(-0.25, 3.15)]);
}

#[test]
fn subtracting_ranges_cuts_holes_and_trims_edges() {
    let ranges = [0.0..10.0, 20.0..30.0];
    assert_ranges(
        &subtract_ranges(&ranges, &[4.0..5.0], 0.0),
        &[(0.0, 4.0), (5.0, 10.0), (20.0, 30.0)],
    );
    assert_ranges(
        &subtract_ranges(&ranges, &[4.0..5.0], 0.5),
        &[(0.0, 3.5), (5.5, 10.0), (20.0, 30.0)],
    );
    assert_ranges(
        &subtract_ranges(&ranges, &[-1.0..1.0, 9.0..21.0, 29.5..40.0], 0.0),
        &[(1.0, 9.0), (21.0, 29.5)],
    );
    assert_ranges(
        &subtract_ranges(&ranges, &[6.0..8.0, 2.0..3.0, 2.5..4.0], 0.0),
        &[(0.0, 2.0), (4.0, 6.0), (8.0, 10.0), (20.0, 30.0)],
    );
    assert!(subtract_ranges(&ranges, &[-5.0..35.0], 0.0).is_empty());
    assert_ranges(
        &subtract_ranges(&ranges, &[], 1.0),
        &[(0.0, 10.0), (20.0, 30.0)],
    );
}

fn captions(source_timed: bool, segments: Vec<CaptionSegment>) -> CaptionsData {
    CaptionsData {
        segments,
        source_timed,
        ..Default::default()
    }
}

fn caption(start: f32, end: f32, words: &[(f32, f32)]) -> CaptionSegment {
    CaptionSegment {
        id: format!("{start}"),
        start,
        end,
        text: String::new(),
        words: words
            .iter()
            .map(|(start, end)| CaptionWord {
                text: "word".into(),
                start: *start,
                end: *end,
            })
            .collect(),
    }
}

#[test]
fn transcript_words_split_into_each_recording_clips_time() {
    let data = captions(
        true,
        vec![
            caption(1.0, 4.0, &[(1.0, 1.5), (3.0, 4.0)]),
            caption(9.0, 12.0, &[]),
            caption(15.0, 16.0, &[(15.0, 16.0)]),
        ],
    );
    let ranges = transcript_speech_ranges(&data, &[10.0, 4.0, 5.0]).unwrap();
    assert_eq!(ranges.len(), 3);
    assert_ranges(&ranges[0], &[(1.0, 1.5), (3.0, 4.0), (9.0, 10.0)]);
    assert_ranges(&ranges[1], &[(0.0, 2.0)]);
    assert_ranges(&ranges[2], &[(1.0, 2.0)]);

    assert!(transcript_speech_ranges(&captions(false, Vec::new()), &[10.0]).is_none());
}

fn frames(duration: f64, base: f32, loud: &[(f64, f64, f32)]) -> Vec<f32> {
    (0..(duration / SPEECH_FRAME_SECS).round() as usize)
        .map(|index| {
            let time = (index as f64 + 0.5) * SPEECH_FRAME_SECS;
            loud.iter()
                .rev()
                .find(|(start, end, _)| time >= *start && time < *end)
                .map_or(base, |(_, _, level)| *level)
        })
        .collect()
}

fn clicks(
    start: f64,
    count: usize,
    interval: f64,
    length: f64,
    level: f32,
) -> Vec<(f64, f64, f32)> {
    (0..count)
        .map(|index| {
            let time = start + index as f64 * interval;
            (time, time + length, level)
        })
        .collect()
}

#[test]
fn audio_frame_levels_are_mean_absolute_dbfs_per_10ms() {
    let mono = vec![0.5_f32; 960];
    let levels = audio_frame_levels([mono.as_slice()], 1, 48_000);
    assert_eq!(levels.len(), 2);
    assert!((levels[0] - 20.0 * 0.5_f32.log10()).abs() < 1e-4);

    let stereo: Vec<f32> = (0..960 * 2)
        .map(|index| if index % 2 == 0 { 0.5 } else { -0.5 })
        .collect();
    let (first, second) = stereo.split_at(333);
    let levels = audio_frame_levels([first, second], 2, 48_000);
    assert_eq!(levels.len(), 2);
    assert!((levels[1] - 20.0 * 0.5_f32.log10()).abs() < 1e-4);

    let partial = vec![0.0_f32; 500];
    assert_eq!(
        audio_frame_levels([partial.as_slice()], 1, 48_000),
        vec![-60.0, -60.0]
    );
    assert!(audio_frame_levels(std::iter::empty::<&[f32]>(), 1, 48_000).is_empty());
}

#[test]
fn microphone_speech_is_sustained_level_above_the_noise_floor() {
    let mic = frames(20.0, -58.0, &[(5.0, 6.5, -25.0)]);
    assert_ranges(&microphone_speech_ranges(&mic), &[(5.0, 6.5)]);

    let short = frames(20.0, -58.0, &[(5.0, 5.07, -25.0)]);
    assert!(microphone_speech_ranges(&short).is_empty());

    let quiet_room = frames(20.0, -58.0, &[(5.0, 6.0, -51.0)]);
    assert!(microphone_speech_ranges(&quiet_room).is_empty());

    assert!(microphone_speech_ranges(&[]).is_empty());
    assert!(microphone_speech_ranges(&[f32::NEG_INFINITY; 500]).is_empty());
    assert!(microphone_speech_ranges(&[f32::NAN; 500]).is_empty());
}

#[test]
fn microphone_speech_bridges_dips_and_short_pauses() {
    let mut loud = vec![(5.0, 6.0, -25.0)];
    loud.extend((0..20).map(|index| {
        let time = 5.02 + index as f64 * 0.05;
        (time, time + 0.01, -58.0)
    }));
    let dipping = frames(20.0, -58.0, &loud);
    assert_ranges(&microphone_speech_ranges(&dipping), &[(5.0, 6.0)]);

    let pause = frames(20.0, -58.0, &[(5.0, 5.5, -25.0), (5.75, 6.5, -25.0)]);
    assert_ranges(&microphone_speech_ranges(&pause), &[(5.0, 6.5)]);

    let sentences = frames(20.0, -58.0, &[(5.0, 5.5, -25.0), (6.0, 6.5, -25.0)]);
    assert_ranges(
        &microphone_speech_ranges(&sentences),
        &[(5.0, 5.5), (6.0, 6.5)],
    );
}

#[test]
fn keystroke_clicks_on_the_mic_are_not_speech() {
    let typing = frames(20.0, -58.0, &clicks(5.0, 40, 0.12, 0.04, -30.0));
    assert!(microphone_speech_ranges(&typing).is_empty());

    let mut loud = vec![(5.0, 11.0, -28.0)];
    loud.extend(clicks(5.0, 40, 0.15, 0.03, -20.0));
    let talking = frames(20.0, -58.0, &loud);
    let ranges = microphone_speech_ranges(&talking);
    assert_ranges(&ranges, &[(5.0, 11.0)]);
}

#[test]
fn synthesized_voice_and_clicks_through_the_whole_mic_path() {
    let rate = 48_000;
    let mut samples = vec![0.0005_f32; rate * 10];
    for (index, sample) in samples
        .iter_mut()
        .enumerate()
        .skip(rate * 2)
        .take(rate * 6 / 10)
    {
        *sample = 0.1 * (index as f32 * 2.0 * std::f32::consts::PI * 180.0 / rate as f32).sin();
    }
    for click in 0..30 {
        let first = rate * 5 + click * rate / 8;
        for sample in samples.iter_mut().skip(first).take(rate * 25 / 1000) {
            *sample = 0.2;
        }
    }
    let levels = audio_frame_levels([samples.as_slice()], 1, rate as u32);
    assert_ranges(&microphone_speech_ranges(&levels), &[(2.0, 2.6)]);
}

#[test]
fn project_speech_joins_the_transcript_and_the_microphone() {
    let transcript = captions(true, vec![caption(6.0, 7.0, &[]), caption(12.0, 13.0, &[])]);
    let mic = vec![Some(vec![2.0..3.0]), None];

    let (source, ranges) = project_speech_ranges(Some(&transcript), &[10.0, 5.0], &mic);
    assert_eq!(source, SpeechSource::TranscriptAndMicrophone);
    assert_ranges(&ranges[0], &[(2.0, 3.0), (6.0, 7.0)]);
    assert_ranges(&ranges[1], &[(2.0, 3.0)]);

    let (source, ranges) = project_speech_ranges(Some(&transcript), &[10.0, 5.0], &[None, None]);
    assert_eq!(source, SpeechSource::Transcript);
    assert_ranges(&ranges[0], &[(6.0, 7.0)]);

    let empty = captions(true, Vec::new());
    let (source, ranges) = project_speech_ranges(Some(&empty), &[10.0, 5.0], &mic);
    assert_eq!(source, SpeechSource::Microphone);
    assert_ranges(&ranges[0], &[(2.0, 3.0)]);
    assert!(ranges[1].is_empty());

    let legacy = captions(false, vec![caption(6.0, 7.0, &[])]);
    let (source, _) = project_speech_ranges(Some(&legacy), &[10.0, 5.0], &mic);
    assert_eq!(source, SpeechSource::Microphone);

    let silent_mic = vec![Some(Vec::new())];
    let (source, ranges) = project_speech_ranges(None, &[10.0], &silent_mic);
    assert_eq!(source, SpeechSource::Microphone);
    assert!(ranges[0].is_empty());

    let (source, ranges) = project_speech_ranges(None, &[10.0], &[None]);
    assert_eq!(source, SpeechSource::None);
    assert!(ranges.iter().all(Vec::is_empty));
}

#[test]
fn speech_inside_a_run_keeps_only_the_long_silent_parts() {
    let takes = vec![
        (typed(&letters(0.0, 41, 0.3)), 0.0),
        (typed(&letters(100.0, 11, 0.3)), 90.0),
    ];
    let speech = vec![vec![2.0..2.5, 9.0..9.5], vec![]];
    let sections = find_typing_sections(&takes, &speech, &settings());
    assert_eq!(
        sections,
        vec![
            section(0, 2.8, 8.7),
            section(0, 9.8, 12.25),
            section(1, 9.75, 13.25)
        ],
        "{sections:?}"
    );
}

#[test]
fn plan_follows_cuts_across_clips_of_one_recording() {
    let timeline = timeline(vec![
        clip(0, 0.0, 12.0, 1.0),
        clip(0, 15.0, 40.0, 1.0),
        clip(1, 0.0, 30.0, 1.0),
    ]);
    let sections = [
        section(0, 5.0, 20.0),
        section(0, 30.0, 31.0),
        section(1, 3.0, 8.0),
    ];
    let plan = plan_typing_speedup(&timeline, &sections, &settings());
    assert_eq!(
        plan,
        vec![
            PlannedTypingSpeedup {
                clip_index: 0,
                start: 5.0,
                end: 12.0
            },
            PlannedTypingSpeedup {
                clip_index: 1,
                start: 15.0,
                end: 20.0
            },
            PlannedTypingSpeedup {
                clip_index: 2,
                start: 3.0,
                end: 8.0
            },
        ]
    );
    assert!((typing_speedup_time_saved(&plan, 3.0) - 17.0 * 2.0 / 3.0).abs() < 1e-9);
    assert_eq!(typing_speedup_time_saved(&plan, 1.0), 0.0);
}

#[test]
fn plan_drops_parts_left_too_short_by_a_cut() {
    let timeline = timeline(vec![clip(0, 0.0, 10.0, 1.0), clip(0, 18.5, 30.0, 1.0)]);
    let plan = plan_typing_speedup(&timeline, &[section(0, 8.5, 20.0)], &settings());
    assert!(plan.is_empty(), "{plan:?}");
}

#[test]
fn plan_skips_clips_with_their_own_speed() {
    let timeline = timeline(vec![clip(0, 0.0, 10.0, 2.0), clip(0, 10.0, 20.0, 1.0)]);
    let plan = plan_typing_speedup(&timeline, &[section(0, 2.0, 18.0)], &settings());
    assert_eq!(
        plan,
        vec![PlannedTypingSpeedup {
            clip_index: 1,
            start: 10.0,
            end: 18.0
        }]
    );
}

#[test]
fn plan_snaps_to_clip_edges_and_stays_clear_of_transitions() {
    let mut timeline = timeline(vec![clip(0, 0.0, 20.0, 1.0), clip(0, 20.0, 40.0, 1.0)]);
    let sections = [
        section(0, 0.05, 10.0),
        section(0, 12.0, 19.95),
        section(0, 20.0, 39.97),
    ];
    let plan = plan_typing_speedup(&timeline, &sections, &settings());
    let ranges: Vec<_> = plan
        .iter()
        .map(|range| (range.clip_index, range.start, range.end))
        .collect();
    assert_eq!(
        ranges,
        vec![(0, 0.0, 10.0), (0, 12.0, 20.0), (1, 20.0, 40.0)]
    );

    timeline.transitions.push(ClipTransition {
        segment_index: 1,
        kind: ClipTransitionType::CrossFade,
        duration: 0.5,
    });
    let plan = plan_typing_speedup(&timeline, &sections, &settings());
    let ranges: Vec<_> = plan
        .iter()
        .map(|range| (range.clip_index, range.start, range.end))
        .collect();
    assert_eq!(
        ranges,
        vec![(0, 0.0, 10.0), (0, 12.0, 19.0), (1, 21.0, 40.0)]
    );
}

#[test]
fn plan_joins_ranges_a_sliver_apart() {
    let timeline = timeline(vec![clip(0, 0.0, 30.0, 1.0)]);
    let plan = plan_typing_speedup(
        &timeline,
        &[section(0, 2.0, 6.0), section(0, 6.05, 9.0)],
        &settings(),
    );
    assert_eq!(
        plan,
        vec![PlannedTypingSpeedup {
            clip_index: 0,
            start: 2.0,
            end: 9.0
        }]
    );
}
