use crate::caption_timing::{
    caption_token_attaches_to_previous, clamp_caption_segment_words, derive_caption_track_segments,
};
use crate::{CaptionWord, ProjectConfiguration};

const MAX_CUE_LINE_CHARS: usize = 42;
const MAX_CUE_DURATION_SECS: f64 = 7.0;
const MAX_CUE_WORD_GAP_SECS: f64 = 1.5;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CaptionExportFormat {
    Srt,
    Vtt,
}

impl CaptionExportFormat {
    pub fn extension(self) -> &'static str {
        match self {
            Self::Srt => "srt",
            Self::Vtt => "vtt",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            Self::Srt => "SRT",
            Self::Vtt => "VTT",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CaptionCue {
    pub start_ms: u64,
    pub end_ms: u64,
    pub lines: Vec<String>,
}

struct TimedCaption<'a> {
    start: f64,
    end: f64,
    text: &'a str,
    words: &'a [CaptionWord],
}

struct CueWord {
    text: String,
    start: f64,
    end: f64,
}

pub fn caption_export_cues(
    project: &ProjectConfiguration,
    recording_durations: &[f64],
) -> Vec<CaptionCue> {
    let Some(captions) = project.captions.as_ref() else {
        return Vec::new();
    };

    let Some(timeline) = project.timeline.as_ref() else {
        let clamped: Vec<_> = captions
            .segments
            .iter()
            .map(clamp_caption_segment_words)
            .collect();
        let timed = clamped.iter().map(|segment| TimedCaption {
            start: f64::from(segment.start),
            end: f64::from(segment.end),
            text: &segment.text,
            words: &segment.words,
        });
        return build_cues(timed, Some(recording_durations.iter().sum()));
    };

    let derived;
    let track = if captions.source_timed {
        derived = derive_caption_track_segments(&captions.segments, timeline, recording_durations);
        &derived
    } else {
        &timeline.caption_segments
    };

    let timed = track.iter().map(|segment| TimedCaption {
        start: segment.start,
        end: segment.end,
        text: &segment.text,
        words: &segment.words,
    });
    build_cues(timed, Some(timeline.duration()))
}

fn build_cues<'a>(
    captions: impl Iterator<Item = TimedCaption<'a>>,
    output_end: Option<f64>,
) -> Vec<CaptionCue> {
    let mut cues: Vec<CaptionCue> = captions
        .flat_map(|caption| group_cue_words(caption_cue_words(&caption)))
        .filter_map(|group| cue_from_words(&group))
        .collect();
    cues.sort_by_key(|cue| (cue.start_ms, cue.end_ms));

    let output_end_ms = output_end
        .filter(|end| end.is_finite() && *end > 0.0)
        .map(seconds_to_ms);

    let mut resolved: Vec<CaptionCue> = Vec::with_capacity(cues.len());
    for mut cue in cues {
        if let Some(end) = output_end_ms {
            cue.end_ms = cue.end_ms.min(end);
        }
        if cue.end_ms <= cue.start_ms {
            continue;
        }
        if let Some(previous) = resolved.last_mut()
            && previous.end_ms > cue.start_ms
        {
            if cue.start_ms > previous.start_ms {
                previous.end_ms = cue.start_ms;
            } else if cue.end_ms > previous.end_ms {
                cue.start_ms = previous.end_ms;
            } else if previous.end_ms - previous.start_ms >= 2 {
                cue.end_ms = previous.end_ms;
                cue.start_ms = previous.start_ms + (previous.end_ms - previous.start_ms) / 2;
                previous.end_ms = cue.start_ms;
            } else {
                let text = format!("{} {}", previous.lines.join(" "), cue.lines.join(" "));
                previous.lines = wrap_lines(&text).unwrap_or_else(|| vec![text]);
                continue;
            }
        }
        resolved.push(cue);
    }
    resolved
}

fn caption_cue_words(caption: &TimedCaption) -> Vec<CueWord> {
    let words: Vec<CueWord> = caption
        .words
        .iter()
        .filter_map(|word| {
            let text = clean_text(&word.text);
            let (start, end) = (f64::from(word.start), f64::from(word.end));
            (!text.is_empty() && start.is_finite() && end.is_finite() && end >= start)
                .then_some(CueWord { text, start, end })
        })
        .collect();
    if !words.is_empty() {
        return words;
    }

    if !caption.start.is_finite() || !caption.end.is_finite() || caption.end <= caption.start {
        return Vec::new();
    }
    let tokens: Vec<String> = caption
        .text
        .split_whitespace()
        .map(clean_text)
        .filter(|token| !token.is_empty())
        .collect();
    let step = (caption.end - caption.start) / tokens.len().max(1) as f64;
    tokens
        .into_iter()
        .enumerate()
        .map(|(index, text)| CueWord {
            text,
            start: caption.start + step * index as f64,
            end: caption.start + step * (index + 1) as f64,
        })
        .collect()
}

fn group_cue_words(words: Vec<CueWord>) -> Vec<Vec<CueWord>> {
    let mut groups: Vec<Vec<CueWord>> = Vec::new();
    let mut current: Vec<CueWord> = Vec::new();

    for word in words {
        if let (Some(first), Some(last)) = (current.first(), current.last()) {
            let gap = word.start - last.end;
            let duration = word.end - first.start;
            let fits = wrap_lines(&joined_text(current.iter().chain([&word]))).is_some();
            if gap > MAX_CUE_WORD_GAP_SECS || duration > MAX_CUE_DURATION_SECS || !fits {
                groups.push(std::mem::take(&mut current));
            }
        }
        current.push(word);
    }

    if !current.is_empty() {
        groups.push(current);
    }
    groups
}

fn joined_text<'a>(words: impl IntoIterator<Item = &'a CueWord>) -> String {
    let mut text = String::new();
    for word in words {
        if !text.is_empty() && !caption_token_attaches_to_previous(&word.text) {
            text.push(' ');
        }
        text.push_str(&word.text);
    }
    text
}

fn cue_from_words(words: &[CueWord]) -> Option<CaptionCue> {
    let first = words.first()?;
    let last = words.last()?;
    let text = joined_text(words);
    let lines = wrap_lines(&text).unwrap_or_else(|| vec![text]);
    let start_ms = seconds_to_ms(first.start);
    let end_ms = seconds_to_ms(last.end.min(first.start + MAX_CUE_DURATION_SECS)).max(start_ms + 1);
    Some(CaptionCue {
        start_ms,
        end_ms,
        lines,
    })
}

fn wrap_lines(text: &str) -> Option<Vec<String>> {
    if text.chars().count() <= MAX_CUE_LINE_CHARS {
        return Some(vec![text.to_string()]);
    }

    let words: Vec<&str> = text.split(' ').collect();
    let (split, _) = (1..words.len())
        .map(|split| {
            let first = words[..split].join(" ").chars().count();
            let second = words[split..].join(" ").chars().count();
            (split, first.max(second))
        })
        .filter(|(_, widest)| *widest <= MAX_CUE_LINE_CHARS)
        .min_by_key(|(_, widest)| *widest)?;
    Some(vec![words[..split].join(" "), words[split..].join(" ")])
}

fn clean_text(text: &str) -> String {
    text.chars()
        .map(|c| if c.is_control() { ' ' } else { c })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn seconds_to_ms(seconds: f64) -> u64 {
    (seconds.max(0.0) * 1000.0).round() as u64
}

fn format_timestamp(ms: u64, separator: char) -> String {
    format!(
        "{:02}:{:02}:{:02}{separator}{:03}",
        ms / 3_600_000,
        (ms % 3_600_000) / 60_000,
        (ms % 60_000) / 1000,
        ms % 1000
    )
}

fn escape_vtt(line: &str) -> String {
    line.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
}

pub fn format_caption_cues(cues: &[CaptionCue], format: CaptionExportFormat) -> String {
    let separator = match format {
        CaptionExportFormat::Srt => ',',
        CaptionExportFormat::Vtt => '.',
    };
    let blocks: Vec<String> = cues
        .iter()
        .enumerate()
        .map(|(index, cue)| {
            let text = match format {
                CaptionExportFormat::Srt => cue.lines.join("\n"),
                CaptionExportFormat::Vtt => cue
                    .lines
                    .iter()
                    .map(|line| escape_vtt(line))
                    .collect::<Vec<_>>()
                    .join("\n"),
            };
            format!(
                "{}\n{} --> {}\n{text}",
                index + 1,
                format_timestamp(cue.start_ms, separator),
                format_timestamp(cue.end_ms, separator),
            )
        })
        .collect();

    match (format, blocks.is_empty()) {
        (CaptionExportFormat::Srt, true) => String::new(),
        (CaptionExportFormat::Srt, false) => format!("{}\n", blocks.join("\n\n")),
        (CaptionExportFormat::Vtt, true) => "WEBVTT\n".to_string(),
        (CaptionExportFormat::Vtt, false) => format!("WEBVTT\n\n{}\n", blocks.join("\n\n")),
    }
}

pub fn caption_export_file_name(name: &str, format: CaptionExportFormat) -> String {
    let cleaned = name
        .chars()
        .map(|c| {
            if c.is_control() || matches!(c, '<' | '>' | ':' | '"' | '/' | '\\' | '|' | '?' | '*') {
                '-'
            } else {
                c
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ");
    let truncated: String = cleaned.trim_end_matches('.').chars().take(120).collect();
    let stem = match truncated.trim() {
        "" => "captions",
        stem => stem,
    };
    format!("{stem}.{}", format.extension())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CaptionSegment, CaptionsData};
    use serde_json::{Value, json};

    fn project(clips: Value, captions: Value) -> ProjectConfiguration {
        ProjectConfiguration {
            timeline: Some(
                serde_json::from_value(json!({ "segments": clips, "zoomSegments": [] })).unwrap(),
            ),
            captions: Some(CaptionsData {
                segments: serde_json::from_value::<Vec<CaptionSegment>>(captions).unwrap(),
                source_timed: true,
                ..Default::default()
            }),
            ..Default::default()
        }
    }

    fn clip(start: f64, end: f64) -> Value {
        json!({ "start": start, "end": end, "timescale": 1.0 })
    }

    fn caption(id: &str, words: &[(&str, f32, f32)]) -> Value {
        json!({
            "id": id,
            "start": words.first().map_or(0.0, |word| word.1),
            "end": words.last().map_or(0.0, |word| word.2),
            "text": words.iter().map(|word| word.0).collect::<Vec<_>>().join(" "),
            "words": words
                .iter()
                .map(|(text, start, end)| json!({ "text": text, "start": start, "end": end }))
                .collect::<Vec<_>>(),
        })
    }

    fn cue(start_ms: u64, end_ms: u64, text: &str) -> CaptionCue {
        CaptionCue {
            start_ms,
            end_ms,
            lines: text.split('\n').map(str::to_string).collect(),
        }
    }

    fn parse_srt_timestamp(stamp: &str) -> u64 {
        let (clock, millis) = stamp.split_once(',').unwrap();
        assert_eq!(millis.len(), 3, "{stamp}");
        let parts: Vec<u64> = clock
            .split(':')
            .map(|part| {
                assert!(part.len() >= 2, "{stamp}");
                part.parse().unwrap()
            })
            .collect();
        assert_eq!(parts.len(), 3, "{stamp}");
        assert!(parts[1] < 60 && parts[2] < 60, "{stamp}");
        ((parts[0] * 60 + parts[1]) * 60 + parts[2]) * 1000 + millis.parse::<u64>().unwrap()
    }

    fn assert_valid_srt(srt: &str) {
        let mut previous_end = 0;
        for (index, block) in srt.trim_end().split("\n\n").enumerate() {
            let mut lines = block.lines();
            assert_eq!(lines.next().unwrap(), (index + 1).to_string());
            let timing = lines.next().unwrap();
            let (start, end) = timing.split_once(" --> ").unwrap();
            let (start, end) = (parse_srt_timestamp(start), parse_srt_timestamp(end));
            assert!(end > start, "{timing}");
            assert!(start >= previous_end, "{timing} overlaps the previous cue");
            previous_end = end;
            let text: Vec<&str> = lines.collect();
            assert!(!text.is_empty() && text.len() <= 2, "{block}");
            assert!(text.iter().all(|line| !line.trim().is_empty()), "{block}");
        }
    }

    #[test]
    fn no_edits_keep_source_timing() {
        let project = project(
            json!([clip(0.0, 10.0)]),
            json!([caption(
                "a",
                &[
                    ("Hello", 1.0, 1.4),
                    ("there", 1.5, 1.9),
                    (",", 1.9, 1.95),
                    ("world.", 2.0, 2.5)
                ]
            )]),
        );
        let cues = caption_export_cues(&project, &[10.0]);
        assert_eq!(cues, vec![cue(1000, 2500, "Hello there, world.")]);
        assert_eq!(
            format_caption_cues(&cues, CaptionExportFormat::Srt),
            "1\n00:00:01,000 --> 00:00:02,500\nHello there, world.\n"
        );
    }

    #[test]
    fn missing_timeline_uses_source_timing() {
        let mut project = project(
            json!([clip(0.0, 10.0)]),
            json!([caption(
                "a",
                &[("Hello", 1.0, 1.4), ("stretched", 2.0, 9.0)]
            )]),
        );
        project.timeline = None;
        assert_eq!(
            caption_export_cues(&project, &[10.0]),
            vec![cue(1000, 4500, "Hello stretched")]
        );
    }

    #[test]
    fn one_cut_shifts_later_words_and_drops_removed_words() {
        let project = project(
            json!([clip(0.0, 5.0), clip(6.0, 10.0)]),
            json!([
                caption("a", &[("kept", 1.0, 1.5)]),
                caption("b", &[("removed", 5.2, 5.8)]),
                caption("c", &[("shifted", 7.0, 8.0)]),
            ]),
        );
        assert_eq!(
            caption_export_cues(&project, &[10.0]),
            vec![cue(1000, 1500, "kept"), cue(6000, 7000, "shifted")]
        );
    }

    #[test]
    fn many_cuts_keep_sequential_indexes_and_valid_timings() {
        let project = project(
            json!([
                clip(0.0, 2.0),
                clip(4.0, 6.0),
                clip(8.0, 10.0),
                clip(12.0, 14.0),
            ]),
            json!([
                caption("a", &[("one", 0.5, 1.0)]),
                caption("x", &[("gone", 2.5, 3.0)]),
                caption("b", &[("two", 4.5, 5.0)]),
                caption("y", &[("gone", 6.5, 7.5)]),
                caption("c", &[("three", 8.5, 9.0)]),
                caption("z", &[("gone", 10.5, 11.0)]),
                caption("d", &[("four", 12.5, 13.0)]),
            ]),
        );
        let cues = caption_export_cues(&project, &[14.0]);
        assert_eq!(
            cues,
            vec![
                cue(500, 1000, "one"),
                cue(2500, 3000, "two"),
                cue(4500, 5000, "three"),
                cue(6500, 7000, "four"),
            ]
        );
        let srt = format_caption_cues(&cues, CaptionExportFormat::Srt);
        assert!(!srt.contains("gone"));
        assert!(srt.starts_with("1\n") && srt.contains("\n\n4\n00:00:06,500"));
        assert_valid_srt(&srt);
    }

    #[test]
    fn speed_change_scales_cue_times() {
        let project = project(
            json!([
                clip(0.0, 4.0),
                { "start": 4.0, "end": 12.0, "timescale": 2.0 },
                { "start": 12.0, "end": 13.0, "timescale": 0.5 },
            ]),
            json!([
                caption("a", &[("normal", 1.0, 2.0)]),
                caption("b", &[("fast", 8.0, 10.0)]),
                caption("c", &[("slow", 12.25, 12.75)]),
            ]),
        );
        assert_eq!(
            caption_export_cues(&project, &[13.0]),
            vec![
                cue(1000, 2000, "normal"),
                cue(6000, 7000, "fast"),
                cue(8500, 9500, "slow"),
            ]
        );
    }

    #[test]
    fn cue_straddling_a_cut_is_split_at_the_cut() {
        let project = project(
            json!([clip(0.0, 3.0), clip(5.0, 9.0)]),
            json!([caption(
                "a",
                &[
                    ("before", 2.0, 2.4),
                    ("edge", 2.6, 3.4),
                    ("removed", 3.5, 4.5),
                    ("after", 5.2, 5.6),
                ]
            )]),
        );
        let cues = caption_export_cues(&project, &[9.0]);
        assert_eq!(
            cues,
            vec![cue(2000, 3000, "before edge"), cue(3200, 3600, "after")]
        );
    }

    #[test]
    fn empty_transcript_exports_empty_files() {
        let project = project(json!([clip(0.0, 10.0)]), json!([]));
        let cues = caption_export_cues(&project, &[10.0]);
        assert!(cues.is_empty());
        assert_eq!(format_caption_cues(&cues, CaptionExportFormat::Srt), "");
        assert_eq!(
            format_caption_cues(&cues, CaptionExportFormat::Vtt),
            "WEBVTT\n"
        );

        let mut without_captions = project;
        without_captions.captions = None;
        assert!(caption_export_cues(&without_captions, &[10.0]).is_empty());
    }

    #[test]
    fn hour_plus_timestamps_are_formatted() {
        let project = project(
            json!([clip(0.0, 3600.0), clip(3700.0, 4000.0)]),
            json!([caption("a", &[("late", 3725.5, 3726.25)])]),
        );
        let cues = caption_export_cues(&project, &[4000.0]);
        assert_eq!(
            format_caption_cues(&cues, CaptionExportFormat::Srt),
            "1\n01:00:25,500 --> 01:00:26,250\nlate\n"
        );
        assert_eq!(
            format_caption_cues(&cues, CaptionExportFormat::Vtt),
            "WEBVTT\n\n1\n01:00:25.500 --> 01:00:26.250\nlate\n"
        );
        assert_eq!(format_timestamp(100 * 3_600_000 + 1, ','), "100:00:00,001");
    }

    #[test]
    fn long_captions_wrap_to_two_balanced_lines_and_split_into_cues() {
        let words: Vec<(String, f32, f32)> = (0..40)
            .map(|index| {
                let start = index as f32 * 0.3;
                (format!("word{index:02}"), start, start + 0.25)
            })
            .collect();
        let borrowed: Vec<(&str, f32, f32)> = words
            .iter()
            .map(|(text, start, end)| (text.as_str(), *start, *end))
            .collect();
        let project = project(json!([clip(0.0, 20.0)]), json!([caption("a", &borrowed)]));
        let cues = caption_export_cues(&project, &[20.0]);
        assert!(cues.len() > 1);
        assert!(cues.iter().all(|cue| {
            cue.lines.len() <= 2
                && cue
                    .lines
                    .iter()
                    .all(|line| line.chars().count() <= MAX_CUE_LINE_CHARS)
        }));
        let exported: Vec<String> = cues
            .iter()
            .flat_map(|cue| cue.lines.iter().flat_map(|line| line.split(' ')))
            .map(str::to_string)
            .collect();
        let expected: Vec<String> = words.into_iter().map(|word| word.0).collect();
        assert_eq!(exported, expected);
        assert_eq!(
            wrap_lines("aaaa bbbb cccc dddd eeee ffff gggg hhhh iiii jjjj"),
            Some(vec![
                "aaaa bbbb cccc dddd eeee".to_string(),
                "ffff gggg hhhh iiii jjjj".to_string(),
            ])
        );
        assert_valid_srt(&format_caption_cues(&cues, CaptionExportFormat::Srt));
    }

    #[test]
    fn cues_respect_maximum_duration_and_split_on_long_pauses() {
        let mut words: Vec<(&str, f32, f32)> = (0..12)
            .map(|index| ("ok", index as f32, index as f32 + 0.9))
            .collect();
        words.push(("later", 20.0, 20.5));
        let project = project(json!([clip(0.0, 30.0)]), json!([caption("a", &words)]));
        let cues = caption_export_cues(&project, &[30.0]);
        assert_eq!(
            cues,
            vec![
                cue(0, 6900, "ok ok ok ok ok ok ok"),
                cue(7000, 11900, "ok ok ok ok ok"),
                cue(20000, 20500, "later"),
            ]
        );
    }

    #[test]
    fn wordless_captions_are_split_evenly_and_capped() {
        let project = project(
            json!([clip(0.0, 30.0)]),
            json!([{ "id": "a", "start": 2.0, "end": 22.0, "text": "slow\u{7} words", "words": [] }]),
        );
        assert_eq!(
            caption_export_cues(&project, &[30.0]),
            vec![cue(2000, 9000, "slow"), cue(12000, 19000, "words")]
        );
    }

    #[test]
    fn transitions_never_produce_overlapping_cues() {
        let mut project = project(
            json!([clip(0.0, 4.0), clip(6.0, 10.0)]),
            json!([
                caption("a", &[("outgoing", 2.5, 3.5)]),
                caption("b", &[("incoming", 6.2, 7.0)]),
                caption("c", &[("tail", 9.0, 9.5)]),
            ]),
        );
        project.timeline.as_mut().unwrap().transitions = serde_json::from_value(json!([
            { "segmentIndex": 1, "type": "cross-fade", "duration": 1.0 }
        ]))
        .unwrap();
        let cues = caption_export_cues(&project, &[10.0]);
        assert_eq!(
            cues,
            vec![
                cue(2500, 3200, "outgoing"),
                cue(3200, 4000, "incoming"),
                cue(6000, 6500, "tail"),
            ]
        );
        assert_valid_srt(&format_caption_cues(&cues, CaptionExportFormat::Srt));
    }

    #[test]
    fn coincident_cues_are_serialized_instead_of_overlapping() {
        let mut project = project(
            json!([clip(0.0, 4.0), clip(6.0, 10.0)]),
            json!([
                caption("a", &[("outgoing", 3.0, 3.5)]),
                caption("b", &[("incoming", 6.0, 7.0)]),
            ]),
        );
        project.timeline.as_mut().unwrap().transitions = serde_json::from_value(json!([
            { "segmentIndex": 1, "type": "cross-fade", "duration": 1.0 }
        ]))
        .unwrap();
        assert_eq!(
            caption_export_cues(&project, &[10.0]),
            vec![cue(3000, 3500, "outgoing"), cue(3500, 4000, "incoming")]
        );
    }

    fn crossfade_project(captions: Value) -> ProjectConfiguration {
        let mut project = project(json!([clip(0.0, 4.0), clip(6.0, 10.0)]), captions);
        project.timeline.as_mut().unwrap().transitions = serde_json::from_value(json!([
            { "segmentIndex": 1, "type": "cross-fade", "duration": 1.0 }
        ]))
        .unwrap();
        project
    }

    fn exported_text(cues: &[CaptionCue]) -> String {
        cues.iter()
            .flat_map(|cue| cue.lines.iter().map(String::as_str))
            .collect::<Vec<_>>()
            .join(" ")
    }

    #[test]
    fn coincident_cues_with_equal_ends_share_the_interval() {
        let project = crossfade_project(json!([
            caption("a", &[("outgoing", 3.0, 3.5)]),
            caption("b", &[("incoming", 6.0, 6.5)]),
        ]));
        let cues = caption_export_cues(&project, &[10.0]);
        assert_eq!(
            cues,
            vec![cue(3000, 3250, "outgoing"), cue(3250, 3500, "incoming")]
        );
        assert_valid_srt(&format_caption_cues(&cues, CaptionExportFormat::Srt));
    }

    #[test]
    fn coincident_cues_clamped_to_the_same_end_keep_both_texts() {
        let mut project = project(json!([clip(0.0, 10.0)]), json!([]));
        project.captions.as_mut().unwrap().source_timed = false;
        project.timeline.as_mut().unwrap().caption_segments = serde_json::from_value(json!([
            { "id": "a", "start": 9.0, "end": 11.0, "text": "first" },
            { "id": "b", "start": 9.0, "end": 12.0, "text": "second" },
        ]))
        .unwrap();
        assert_eq!(
            caption_export_cues(&project, &[10.0]),
            vec![cue(9000, 9500, "first"), cue(9500, 10000, "second")]
        );
    }

    #[test]
    fn cue_overlapping_a_serialized_cue_keeps_its_text() {
        let project = crossfade_project(json!([
            caption("a", &[("outgoing", 3.0, 3.5)]),
            caption("b", &[("incoming", 6.0, 7.0)]),
            caption("c", &[("late", 3.2, 3.4)]),
        ]));
        let cues = caption_export_cues(&project, &[10.0]);
        assert_eq!(
            cues,
            vec![
                cue(3000, 3500, "outgoing"),
                cue(3500, 3750, "incoming"),
                cue(3750, 4000, "late"),
            ]
        );
        assert_valid_srt(&format_caption_cues(&cues, CaptionExportFormat::Srt));
    }

    #[test]
    fn coincident_cues_too_short_to_divide_are_merged() {
        let project = crossfade_project(json!([
            caption("a", &[("outgoing", 3.0, 3.001)]),
            caption("b", &[("incoming", 6.0, 6.001)]),
        ]));
        assert_eq!(
            caption_export_cues(&project, &[10.0]),
            vec![cue(3000, 3001, "outgoing incoming")]
        );
    }

    #[test]
    fn overlapping_cues_never_lose_text() {
        let words = ["one", "two", "three", "four", "five", "six"];
        let offsets = [0.0_f32, 0.001, 0.2, 0.4];
        let lengths = [0.001_f32, 0.3, 0.5, 1.0];
        let mut cases = 0;
        for outgoing_offset in offsets {
            for outgoing_length in lengths {
                for incoming_offset in offsets {
                    for incoming_length in lengths {
                        let captions: Vec<Value> = words
                            .chunks(2)
                            .enumerate()
                            .flat_map(|(index, pair)| {
                                let outgoing = 3.0 + outgoing_offset * index as f32;
                                let incoming = 6.0 + incoming_offset * index as f32;
                                [
                                    caption(
                                        &format!("out{index}"),
                                        &[(pair[0], outgoing, outgoing + outgoing_length)],
                                    ),
                                    caption(
                                        &format!("in{index}"),
                                        &[(pair[1], incoming, incoming + incoming_length)],
                                    ),
                                ]
                            })
                            .collect();
                        let project = crossfade_project(Value::Array(captions));
                        let cues = caption_export_cues(&project, &[10.0]);
                        let mut exported: Vec<String> = exported_text(&cues)
                            .split(' ')
                            .map(str::to_string)
                            .collect();
                        exported.sort();
                        let mut expected: Vec<String> =
                            words.iter().map(|word| word.to_string()).collect();
                        expected.sort();
                        assert_eq!(
                            exported, expected,
                            "{outgoing_offset} {outgoing_length} {incoming_offset} {incoming_length}"
                        );
                        assert_valid_srt(&format_caption_cues(&cues, CaptionExportFormat::Srt));
                        cases += 1;
                    }
                }
            }
        }
        assert_eq!(cases, 256);
    }

    #[test]
    fn fullscreen_holds_push_later_cues_back() {
        let mut project = project(
            json!([clip(0.0, 10.0)]),
            json!([caption("a", &[("after", 5.0, 6.0)])]),
        );
        project.timeline.as_mut().unwrap().text_segments.push(
            serde_json::from_value(json!({ "start": 2.0, "end": 4.0, "layout": "fullscreen" }))
                .unwrap(),
        );
        assert_eq!(
            caption_export_cues(&project, &[10.0]),
            vec![cue(7000, 8000, "after")]
        );
    }

    #[test]
    fn multiple_recording_clips_use_cumulative_offsets() {
        let project = project(
            json!([{ "recordingSegment": 1, "start": 1.0, "end": 3.0, "timescale": 1.0 }]),
            json!([
                caption("a", &[("first", 1.5, 2.0)]),
                caption("b", &[("second", 6.5, 7.0)]),
            ]),
        );
        assert_eq!(
            caption_export_cues(&project, &[5.0, 5.0]),
            vec![cue(500, 1000, "second")]
        );
    }

    #[test]
    fn legacy_tracks_are_exported_as_rendered() {
        let mut project = project(json!([clip(0.0, 10.0)]), json!([]));
        project.captions.as_mut().unwrap().source_timed = false;
        project.timeline.as_mut().unwrap().caption_segments = serde_json::from_value(json!([
            { "id": "a", "start": 1.0, "end": 2.0, "text": "legacy track" },
            { "id": "b", "start": 9.5, "end": 12.0, "text": "clamped" },
        ]))
        .unwrap();
        assert_eq!(
            caption_export_cues(&project, &[10.0]),
            vec![cue(1000, 2000, "legacy track"), cue(9500, 10000, "clamped")]
        );
    }

    #[test]
    fn vtt_escapes_markup_and_uses_dot_separator() {
        let cues = vec![cue(0, 1250, "Cap <Rend> & fast\nsecond line")];
        assert_eq!(
            format_caption_cues(&cues, CaptionExportFormat::Vtt),
            "WEBVTT\n\n1\n00:00:00.000 --> 00:00:01.250\nCap &lt;Rend&gt; &amp; fast\nsecond line\n"
        );
        assert_eq!(
            format_caption_cues(&cues, CaptionExportFormat::Srt),
            "1\n00:00:00,000 --> 00:00:01,250\nCap <Rend> & fast\nsecond line\n"
        );
    }

    #[test]
    fn file_names_are_sanitized() {
        assert_eq!(
            caption_export_file_name("My: \"Demo\"/take", CaptionExportFormat::Srt),
            "My- -Demo--take.srt"
        );
        assert_eq!(
            caption_export_file_name("...", CaptionExportFormat::Vtt),
            "captions.vtt"
        );
        assert_eq!(
            caption_export_file_name("Cap 2026-11-18 at 10.30", CaptionExportFormat::Srt),
            "Cap 2026-11-18 at 10.30.srt"
        );
    }
}
