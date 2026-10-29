use std::{ops::Range, path::Path, sync::Arc};

use cap_audio::{AudioData, DecodedAudio};
use cap_project::{
    ClipSpeedAudioMode, KeyboardEvents, PlannedTypingSpeedup, ProjectConfiguration, RecordingMeta,
    SpeechSource, TimelineConfiguration, TypingRunSettings,
};
use gpui::{AnyElement, Context, Hsla, IntoElement, ParentElement, Styled, div, px};

use crate::{editor_edits, editor_timeline::TrackKind, editor_window::EditorWindow, i18n::tr, ui};

const SPEEDS: [f64; 6] = [1.5, 2.0, 3.0, 4.0, 6.0, 8.0];

/// What detection needs from the recording itself, read once in the
/// background when the editor's audio has loaded.
pub(crate) struct TypingSpeedupInputs {
    takes: Result<Vec<(KeyboardEvents, f64)>, String>,
    mic_speech: Vec<Option<Vec<Range<f64>>>>,
}

impl TypingSpeedupInputs {
    pub(crate) fn load(project_path: &Path, mic: &[Option<Arc<DecodedAudio>>]) -> Self {
        let takes = RecordingMeta::load_for_project(project_path)
            .map_err(|error| error.to_string())
            .and_then(|meta| cap_project::load_project_keyboard_events(&meta));
        let mic_speech = mic
            .iter()
            .map(|audio| {
                audio.as_ref().map(|audio| {
                    cap_project::microphone_speech_ranges(&cap_project::audio_frame_levels(
                        audio.sample_slices(),
                        audio.channels(),
                        AudioData::SAMPLE_RATE,
                    ))
                })
            })
            .collect();
        Self { takes, mic_speech }
    }
}

pub(crate) struct TypingSpeedupState {
    pub(crate) inputs: Option<Arc<TypingSpeedupInputs>>,
    pub(crate) speed: f64,
}

impl Default for TypingSpeedupState {
    fn default() -> Self {
        Self {
            inputs: None,
            speed: cap_project::DEFAULT_TYPING_SPEED,
        }
    }
}

#[derive(Debug, PartialEq)]
pub(crate) enum TypingSpeedupPreview {
    Failed(String),
    NoKeyboard,
    Ready {
        plan: Vec<PlannedTypingSpeedup>,
        speech: SpeechSource,
    },
}

pub(crate) fn typing_speedup_preview(
    project: &ProjectConfiguration,
    recording_durations: &[f64],
    inputs: &TypingSpeedupInputs,
) -> TypingSpeedupPreview {
    let takes = match &inputs.takes {
        Ok(takes) => takes,
        Err(error) => return TypingSpeedupPreview::Failed(error.clone()),
    };
    if takes.iter().all(|(events, _)| events.presses.is_empty()) {
        return TypingSpeedupPreview::NoKeyboard;
    }
    let Some(timeline) = project.timeline.as_ref() else {
        return TypingSpeedupPreview::Ready {
            plan: Vec::new(),
            speech: SpeechSource::None,
        };
    };
    let settings = TypingRunSettings::default();
    let (speech, spoken) = cap_project::project_speech_ranges(
        project.captions.as_ref(),
        recording_durations,
        &inputs.mic_speech,
    );
    let sections = cap_project::find_typing_sections(takes, &spoken, &settings);
    TypingSpeedupPreview::Ready {
        plan: cap_project::plan_typing_speedup(timeline, &sections, &settings),
        speech,
    }
}

fn format_speed(speed: f64) -> String {
    if (speed - speed.round()).abs() < 1e-6 {
        format!("{}x", speed.round() as i32)
    } else {
        format!("{speed}x")
    }
}

impl EditorWindow {
    fn current_typing_preview(&self) -> Option<TypingSpeedupPreview> {
        let inputs = self.typing_speedup.inputs.as_ref()?;
        Some(typing_speedup_preview(
            &self.project,
            &self.clip_display_durations,
            inputs,
        ))
    }

    fn apply_typing_speedup_clicked(&mut self, window: &mut gpui::Window, cx: &mut Context<Self>) {
        let Some(TypingSpeedupPreview::Ready { plan, .. }) = self.current_typing_preview() else {
            return;
        };
        if plan.is_empty() {
            return;
        }
        let speed = self.typing_speedup.speed;
        let mut applied = 0;
        if self.edit(
            |timeline| {
                applied = apply_typing_speedup(timeline, &plan, speed);
                applied > 0
            },
            window,
            cx,
        ) {
            tracing::info!(applied, speed, "typing sped up");
            self.note_edit("typing-speedup", Some(TrackKind::Clip));
        }
    }

    pub(crate) fn render_typing_speedup(&self, cx: &mut Context<Self>) -> AnyElement {
        let theme = self.theme;
        let speed = self.typing_speedup.speed;
        let preview = self.current_typing_preview();
        let muted = |text: String| {
            div()
                .text_size(px(12.))
                .text_color(Hsla::from(theme.gray_10))
                .child(text)
        };
        let (status, plan, speech) = match preview {
            None => (
                tr::editor::tabs::keyboard::typing_analyzing().to_string(),
                None,
                None,
            ),
            Some(TypingSpeedupPreview::Failed(error)) => {
                (tr::editor::tabs::keyboard::load_failed(&error), None, None)
            }
            Some(TypingSpeedupPreview::NoKeyboard) => (
                tr::editor::tabs::keyboard::typing_no_keyboard().to_string(),
                None,
                None,
            ),
            Some(TypingSpeedupPreview::Ready { plan, speech }) if plan.is_empty() => (
                tr::editor::tabs::keyboard::typing_none_found().to_string(),
                None,
                Some(speech),
            ),
            Some(TypingSpeedupPreview::Ready { plan, speech }) => {
                let saved = cap_project::typing_speedup_time_saved(&plan, speed);
                (
                    tr::editor::tabs::keyboard::typing_found(
                        plan.len(),
                        crate::editor_clips::format_clip_duration(saved),
                    ),
                    Some(plan),
                    Some(speech),
                )
            }
        };

        let body = div()
            .flex()
            .flex_col()
            .gap(px(12.))
            .child(muted(
                tr::editor::tabs::keyboard::typing_description().to_string(),
            ))
            .child(
                ui::Field::inline(&theme, tr::editor::tabs::keyboard::typing_speed()).child(
                    ui::SegmentedControl::editor(
                        &theme,
                        "typing-speedup-speed",
                        SPEEDS
                            .iter()
                            .map(|value| {
                                ui::SegmentOption::new(
                                    format_speed(*value),
                                    (speed - value).abs() < 1e-9,
                                )
                            })
                            .collect(),
                    )
                    .on_select(cx.listener(|this, index: &usize, _, cx| {
                        if let Some(speed) = ui::option_at(&SPEEDS, *index) {
                            this.typing_speedup.speed = speed;
                            cx.notify();
                        }
                    })),
                ),
            )
            .child(muted(status))
            .children(
                (speech == Some(SpeechSource::None)).then(|| {
                    muted(tr::editor::tabs::keyboard::typing_no_speech_check().to_string())
                }),
            )
            .child(
                ui::Button::plain(
                    &theme,
                    "typing-speedup-apply",
                    ui::ButtonVariant::Primary,
                    ui::ButtonSize::Md,
                )
                .label(tr::editor::tabs::keyboard::typing_apply())
                .full_width()
                .disabled(plan.is_none())
                .on_click(cx.listener(|this, _, window, cx| {
                    this.apply_typing_speedup_clicked(window, cx);
                })),
            );

        ui::Field::section(&theme, tr::editor::tabs::keyboard::typing_title())
            .child(body)
            .into_any_element()
    }
}

/// Clip edges this close to a planned range need no split.
const SPLIT_EPSILON: f64 = 1e-6;

/// Speeds up every planned range: the clip is split at the range's edges and
/// the middle piece gets `speed` through the clip-speed edit, which ripples
/// every other track exactly as changing a clip's speed by hand does. The
/// sped-up piece is muted: nothing in it is speech, and keyboard clatter at
/// several times its speed is noise. A range whose clip no longer matches the
/// plan (it was edited, or already has a speed) is skipped. Returns how many
/// ranges were applied.
pub fn apply_typing_speedup(
    timeline: &mut TimelineConfiguration,
    plan: &[PlannedTypingSpeedup],
    speed: f64,
) -> usize {
    if !(cap_project::MIN_TYPING_SPEED..=cap_project::MAX_TYPING_SPEED).contains(&speed) {
        return 0;
    }
    let mut ordered: Vec<&PlannedTypingSpeedup> = plan.iter().collect();
    ordered.sort_by(|a, b| {
        a.clip_index
            .cmp(&b.clip_index)
            .then(a.start.total_cmp(&b.start))
    });
    let mut applied = 0;
    for range in ordered.into_iter().rev() {
        let mut index = range.clip_index;
        let Some(clip) = timeline.segments.get(index) else {
            continue;
        };
        if (clip.timescale - 1.0).abs() > 1e-9
            || range.start < clip.start - SPLIT_EPSILON
            || range.end > clip.end + SPLIT_EPSILON
            || range.end - range.start <= SPLIT_EPSILON
        {
            continue;
        }
        let (clip_start, clip_end) = (clip.start, clip.end);
        if range.end < clip_end - SPLIT_EPSILON
            && !editor_edits::split_clip_at(timeline, index, range.end - clip_start)
        {
            continue;
        }
        if range.start > clip_start + SPLIT_EPSILON {
            if !editor_edits::split_clip_at(timeline, index, range.start - clip_start) {
                continue;
            }
            index += 1;
        }
        if editor_edits::set_clip_segment_timescale(timeline, index, speed) {
            timeline.segments[index].speed_audio_mode = Some(ClipSpeedAudioMode::Mute);
            applied += 1;
        }
    }
    applied
}

#[cfg(test)]
mod tests {
    use super::*;
    use cap_project::{
        CaptionsData, ProjectConfiguration, TypingRunSettings, TypingSection, plan_typing_speedup,
        synchronize_captions,
    };

    fn timeline(json: serde_json::Value) -> TimelineConfiguration {
        serde_json::from_value(json).expect("fixture parses")
    }

    fn planned(clip_index: usize, start: f64, end: f64) -> PlannedTypingSpeedup {
        PlannedTypingSpeedup {
            clip_index,
            start,
            end,
        }
    }

    fn section(recording_clip: u32, start: f64, end: f64) -> TypingSection {
        TypingSection {
            recording_clip,
            start,
            end,
        }
    }

    fn clips(timeline: &TimelineConfiguration) -> Vec<(u32, f64, f64, f64)> {
        timeline
            .segments
            .iter()
            .map(|clip| {
                (
                    clip.recording_clip,
                    (clip.start * 1e6).round() / 1e6,
                    (clip.end * 1e6).round() / 1e6,
                    clip.timescale,
                )
            })
            .collect()
    }

    fn close(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-6
    }

    fn source_at(timeline: &TimelineConfiguration, output: f64) -> (f64, u32) {
        let (time, clip) = timeline.get_segment_time(output).expect("time is inside");
        (time, clip.recording_clip)
    }

    #[test]
    fn a_range_inside_a_clip_becomes_its_own_sped_up_muted_clip() {
        let mut timeline = timeline(serde_json::json!({
            "segments": [{ "recordingSegment": 0, "timescale": 1.0, "start": 0.0, "end": 60.0, "volume": 0.5 }],
            "zoomSegments": []
        }));
        assert_eq!(
            apply_typing_speedup(&mut timeline, &[planned(0, 10.0, 22.0)], 3.0),
            1
        );
        assert_eq!(
            clips(&timeline),
            vec![
                (0, 0.0, 10.0, 1.0),
                (0, 10.0, 22.0, 3.0),
                (0, 22.0, 60.0, 1.0)
            ]
        );
        assert_eq!(
            timeline.segments[1].speed_audio_mode,
            Some(ClipSpeedAudioMode::Mute)
        );
        assert_eq!(timeline.segments[0].speed_audio_mode, None);
        assert!(
            timeline
                .segments
                .iter()
                .all(|clip| clip.volume == Some(0.5))
        );
        assert!(close(timeline.duration(), 60.0 - 8.0));

        let (time, _) = source_at(&timeline, 10.0 + 1.0);
        assert!(close(time, 13.0), "{time}");
        let (time, _) = source_at(&timeline, 13.999);
        assert!(close(time, 21.997), "{time}");
        let (time, _) = source_at(&timeline, 14.0 + 5.0);
        assert!(close(time, 27.0), "{time}");
    }

    #[test]
    fn ranges_on_clip_edges_need_no_split() {
        let mut timeline = timeline(serde_json::json!({
            "segments": [
                { "recordingSegment": 0, "timescale": 1.0, "start": 0.0, "end": 10.0 },
                { "recordingSegment": 0, "timescale": 1.0, "start": 10.0, "end": 20.0 }
            ],
            "zoomSegments": []
        }));
        let plan = [planned(0, 0.0, 4.0), planned(1, 15.0, 20.0)];
        assert_eq!(apply_typing_speedup(&mut timeline, &plan, 2.0), 2);
        assert_eq!(
            clips(&timeline),
            vec![
                (0, 0.0, 4.0, 2.0),
                (0, 4.0, 10.0, 1.0),
                (0, 10.0, 15.0, 1.0),
                (0, 15.0, 20.0, 2.0)
            ]
        );
        assert!(close(timeline.duration(), 20.0 - 2.0 - 2.5));
    }

    #[test]
    fn other_tracks_stay_on_the_source_moments_they_covered() {
        let mut timeline = timeline(serde_json::json!({
            "segments": [{ "recordingSegment": 0, "timescale": 1.0, "start": 0.0, "end": 60.0 }],
            "zoomSegments": [
                { "start": 5.0, "end": 8.0, "amount": 1.5, "mode": "auto" },
                { "start": 12.0, "end": 18.0, "amount": 1.5, "mode": "auto" },
                { "start": 30.0, "end": 35.0, "amount": 2.0, "mode": "auto" }
            ],
            "captionSegments": [
                { "id": "a", "start": 40.0, "end": 41.0, "text": "after" }
            ]
        }));
        let before: Vec<_> = timeline
            .zoom_segments
            .iter()
            .map(|zoom| {
                (
                    source_at(&timeline, zoom.start).0,
                    source_at(&timeline, zoom.end - 1e-9).0,
                )
            })
            .collect();
        apply_typing_speedup(&mut timeline, &[planned(0, 10.0, 20.0)], 4.0);
        let zooms: Vec<_> = timeline
            .zoom_segments
            .iter()
            .map(|zoom| (zoom.start, zoom.end))
            .collect();
        assert!(
            close(zooms[0].0, 5.0) && close(zooms[0].1, 8.0),
            "{zooms:?}"
        );
        assert!(
            close(zooms[1].0, 10.5) && close(zooms[1].1, 12.0),
            "{zooms:?}"
        );
        assert!(
            close(zooms[2].0, 22.5) && close(zooms[2].1, 27.5),
            "{zooms:?}"
        );
        for (zoom, (start, end)) in timeline.zoom_segments.iter().zip(before) {
            assert!(close(source_at(&timeline, zoom.start).0, start));
            assert!((source_at(&timeline, zoom.end - 1e-9).0 - end).abs() < 1e-6);
        }
        let caption = &timeline.caption_segments[0];
        assert!(close(caption.start, 32.5) && close(caption.end, 33.5));
    }

    #[test]
    fn source_timed_captions_reproject_onto_the_sped_up_timeline() {
        let mut project: ProjectConfiguration = serde_json::from_value(serde_json::json!({
            "timeline": {
                "segments": [{ "recordingSegment": 0, "timescale": 1.0, "start": 0.0, "end": 60.0 }],
                "zoomSegments": []
            }
        }))
        .unwrap();
        project.captions = Some(CaptionsData {
            segments: vec![cap_project::CaptionSegment {
                id: "spoken".into(),
                start: 40.0,
                end: 42.0,
                text: "hello there".into(),
                words: vec![
                    cap_project::CaptionWord {
                        text: "hello".into(),
                        start: 40.0,
                        end: 41.0,
                    },
                    cap_project::CaptionWord {
                        text: "there".into(),
                        start: 41.0,
                        end: 42.0,
                    },
                ],
            }],
            source_timed: true,
            ..Default::default()
        });
        let timeline = project.timeline.as_mut().unwrap();
        apply_typing_speedup(timeline, &[planned(0, 10.0, 25.0)], 3.0);
        synchronize_captions(&mut project, &[60.0]);
        let caption = &project.timeline.as_ref().unwrap().caption_segments[0];
        assert!(close(caption.start, 30.0), "{caption:?}");
        assert!(close(caption.end, 32.0), "{caption:?}");
    }

    #[test]
    fn cuts_multiple_recordings_and_existing_speeds_through_plan_and_apply() {
        let mut timeline = timeline(serde_json::json!({
            "segments": [
                { "recordingSegment": 0, "timescale": 1.0, "start": 0.0, "end": 12.0 },
                { "recordingSegment": 0, "timescale": 1.0, "start": 15.0, "end": 40.0 },
                { "recordingSegment": 0, "timescale": 2.0, "start": 40.0, "end": 50.0 },
                { "recordingSegment": 1, "timescale": 1.0, "start": 0.0, "end": 30.0 }
            ],
            "zoomSegments": []
        }));
        let settings = TypingRunSettings::default();
        let sections = [section(0, 5.0, 45.0), section(1, 3.0, 8.0)];
        let plan = plan_typing_speedup(&timeline, &sections, &settings);
        let before = timeline.duration();
        let saved = cap_project::typing_speedup_time_saved(&plan, 3.0);
        assert_eq!(apply_typing_speedup(&mut timeline, &plan, 3.0), 3);
        assert_eq!(
            clips(&timeline),
            vec![
                (0, 0.0, 5.0, 1.0),
                (0, 5.0, 12.0, 3.0),
                (0, 15.0, 40.0, 3.0),
                (0, 40.0, 50.0, 2.0),
                (1, 0.0, 3.0, 1.0),
                (1, 3.0, 8.0, 3.0),
                (1, 8.0, 30.0, 1.0)
            ]
        );
        assert!(close(before - timeline.duration(), saved));
        assert!(close(saved, (7.0 + 25.0 + 5.0) * 2.0 / 3.0));

        assert!(plan_typing_speedup(&timeline, &sections, &settings).is_empty());
    }

    #[test]
    fn transitions_keep_their_boundaries_and_durations() {
        let mut timeline = timeline(serde_json::json!({
            "segments": [
                { "recordingSegment": 0, "timescale": 1.0, "start": 0.0, "end": 20.0 },
                { "recordingSegment": 0, "timescale": 1.0, "start": 25.0, "end": 45.0 },
                { "recordingSegment": 0, "timescale": 1.0, "start": 50.0, "end": 70.0 }
            ],
            "transitions": [
                { "segmentIndex": 1, "type": "cross-fade", "duration": 0.5 },
                { "segmentIndex": 2, "type": "fade-through-black", "duration": 1.0 }
            ],
            "zoomSegments": []
        }));
        let sections = [section(0, 10.0, 70.0)];
        let plan = plan_typing_speedup(&timeline, &sections, &TypingRunSettings::default());
        assert_eq!(
            plan,
            vec![
                planned(0, 10.0, 19.0),
                planned(1, 26.0, 43.0),
                planned(2, 52.0, 70.0)
            ]
        );
        assert_eq!(apply_typing_speedup(&mut timeline, &plan, 2.0), 3);
        assert_eq!(
            clips(&timeline),
            vec![
                (0, 0.0, 10.0, 1.0),
                (0, 10.0, 19.0, 2.0),
                (0, 19.0, 20.0, 1.0),
                (0, 25.0, 26.0, 1.0),
                (0, 26.0, 43.0, 2.0),
                (0, 43.0, 45.0, 1.0),
                (0, 50.0, 52.0, 1.0),
                (0, 52.0, 70.0, 2.0)
            ]
        );
        let transitions: Vec<_> = (0..timeline.segments.len())
            .filter_map(|index| {
                timeline
                    .effective_transition(index)
                    .map(|transition| (index, transition.duration))
            })
            .collect();
        assert_eq!(transitions, vec![(3, 0.5), (6, 1.0)]);
    }

    #[test]
    fn stale_or_invalid_ranges_are_skipped() {
        let original = timeline(serde_json::json!({
            "segments": [
                { "recordingSegment": 0, "timescale": 1.5, "start": 0.0, "end": 10.0 },
                { "recordingSegment": 0, "timescale": 1.0, "start": 10.0, "end": 20.0 }
            ],
            "zoomSegments": []
        }));
        let mut timeline = original.clone();
        let plan = [
            planned(0, 2.0, 6.0),
            planned(1, 5.0, 15.0),
            planned(1, 12.0, 12.0),
            planned(7, 1.0, 5.0),
        ];
        assert_eq!(apply_typing_speedup(&mut timeline, &plan, 3.0), 0);
        assert_eq!(clips(&timeline), clips(&original));
        assert_eq!(
            apply_typing_speedup(&mut timeline, &[planned(1, 12.0, 18.0)], 9.0),
            0
        );
        assert_eq!(clips(&timeline), clips(&original));
    }
}
