use cap_project::{MaskKeyframes, MaskKind, MaskSegment, MaskVectorKeyframe, XY};

use crate::{Rect, SensitiveKind, track::Track};

/// The shortest mask the editor's mask track lets a user make.
pub const MIN_MASK_DURATION: f64 = 1.0;
/// Extra time on both ends, so a blur never trails the text by a frame.
const TIME_PADDING: f64 = 0.1;
/// Positions that a straight line between their neighbours reproduces this
/// closely (a share of the output frame) are dropped from the keyframes.
const KEYFRAME_TOLERANCE: f64 = 0.0015;
/// Recognition boxes of still text wobble by a few pixels between frames;
/// movement within this share of the (padded) box height is not followed,
/// the padding already covers it.
const STILL_JITTER: f64 = 0.12;

pub struct MaskPlan<'a> {
    /// Source frame size in pixels, for padding boxes evenly on both axes.
    pub source_size: (u32, u32),
    pub total_duration: f64,
    /// Style (blur amount, feather) the generated masks copy.
    pub template: &'a MaskSegment,
    /// How often the output placement is probed between detections, so a
    /// mask follows a zoom or layout change that moves the screen.
    pub step: f64,
}

/// The box a mask covers: a little larger than the text, so anti-aliased
/// edges and OCR's tight boxes never leave a sliver readable.
pub fn padded_bounds(kind: SensitiveKind, bounds: Rect, source_size: (u32, u32)) -> Rect {
    let (width, height) = (
        f64::from(source_size.0.max(1)),
        f64::from(source_size.1.max(1)),
    );
    if kind == SensitiveKind::Face {
        return bounds.expand(bounds.width * 0.15, bounds.height * 0.2);
    }
    let pad_px = bounds.height * height * 0.3 + 3.0;
    bounds.expand(pad_px / width, pad_px / height)
}

/// The masks for one track: one per stretch of time the item is inside the
/// output frame. `map` places a source-frame rectangle in the output frame
/// at a timeline time, or `None` while the screen is hidden. An item zoomed
/// or scrolled out of the frame gets no mask there: the renderer would pin
/// the mask to the frame edge, over other content.
pub fn track_to_masks(
    track: &Track,
    plan: &MaskPlan,
    map: &mut dyn FnMut(f64, Rect) -> Option<Rect>,
) -> Vec<MaskSegment> {
    if track.points.is_empty() || plan.total_duration <= 0.0 {
        return Vec::new();
    }
    let start = (track.start - TIME_PADDING).max(0.0);
    let end = (track.end + TIME_PADDING)
        .min(plan.total_duration)
        .max(start);

    let mut times = vec![start, end];
    times.extend(track.points.iter().map(|point| point.time));
    if plan.step > 0.0 {
        let mut grid = (start / plan.step).ceil() as i64;
        loop {
            let time = grid as f64 * plan.step;
            if time >= end {
                break;
            }
            times.push(time);
            grid += 1;
        }
    }
    times.retain(|time| (start..=end).contains(time));
    times.sort_by(f64::total_cmp);
    times.dedup_by(|a, b| (*a - *b).abs() < 1e-6);

    let samples = times
        .into_iter()
        .map(|time| {
            let source = padded_bounds(track.kind, track.cover_at(time), plan.source_size);
            let placed = map(time, source).filter(|rect| rect.intersect(Rect::UNIT).is_some());
            (time, placed)
        })
        .collect::<Vec<_>>();

    let mut masks = Vec::new();
    let mut index = 0;
    while index < samples.len() {
        if samples[index].1.is_none() {
            index += 1;
            continue;
        }
        let first = index;
        while index < samples.len() && samples[index].1.is_some() {
            index += 1;
        }
        let run_start = if first == 0 {
            start
        } else {
            samples[first - 1].0
        };
        let run_end = samples.get(index).map_or(end, |(time, _)| *time);
        let visible = samples[first..index]
            .iter()
            .filter_map(|(time, rect)| rect.map(|rect| (*time, rect)))
            .collect::<Vec<_>>();
        masks.push(mask_from_samples(
            &visible,
            min_length(run_start, run_end, plan.total_duration),
            plan.template,
        ));
    }
    masks
}

fn mask_from_samples(
    samples: &[(f64, Rect)],
    (start, end): (f64, f64),
    template: &MaskSegment,
) -> MaskSegment {
    let tolerance = samples
        .iter()
        .map(|(_, rect)| rect.height * STILL_JITTER)
        .fold(f64::INFINITY, f64::min)
        .max(KEYFRAME_TOLERANCE);
    let samples = simplify(samples, tolerance);

    let (_, first) = samples[0];
    let mut segment = template.clone();
    segment.start = start;
    segment.end = end;
    segment.enabled = true;
    segment.mask_type = MaskKind::Sensitive;
    segment.center = XY::new(first.x + first.width / 2.0, first.y + first.height / 2.0);
    segment.size = XY::new(first.width, first.height);
    segment.keyframes = MaskKeyframes::default();
    if samples.len() > 1 {
        for (time, rect) in &samples {
            let relative = time - start;
            segment.keyframes.position.push(MaskVectorKeyframe {
                time: relative,
                x: rect.x + rect.width / 2.0,
                y: rect.y + rect.height / 2.0,
            });
            segment.keyframes.size.push(MaskVectorKeyframe {
                time: relative,
                x: rect.width,
                y: rect.height,
            });
        }
    }
    segment
}

fn min_length(start: f64, end: f64, total: f64) -> (f64, f64) {
    let (mut start, mut end) = (start, end);
    let missing = MIN_MASK_DURATION.min(total) - (end - start);
    if missing > 0.0 {
        start = (start - missing / 2.0).max(0.0);
        end = (start + MIN_MASK_DURATION).min(total);
        start = (end - MIN_MASK_DURATION).max(0.0);
    }
    (start, end)
}

fn rect_distance(a: Rect, b: Rect) -> f64 {
    (a.x - b.x)
        .abs()
        .max((a.y - b.y).abs())
        .max((a.right() - b.right()).abs())
        .max((a.bottom() - b.bottom()).abs())
}

/// Ramer-Douglas-Peucker over time: keeps the keyframes that linear
/// interpolation between the kept ones needs to stay within `tolerance`.
fn simplify(samples: &[(f64, Rect)], tolerance: f64) -> Vec<(f64, Rect)> {
    let still = samples
        .iter()
        .all(|(_, rect)| rect_distance(*rect, samples[0].1) <= tolerance);
    if still {
        let union = samples
            .iter()
            .fold(samples[0].1, |union, (_, rect)| union.union(*rect));
        return vec![(samples[0].0, union)];
    }
    if samples.len() == 2 {
        return samples.to_vec();
    }
    let mut keep = vec![false; samples.len()];
    keep[0] = true;
    keep[samples.len() - 1] = true;
    let mut stack = vec![(0, samples.len() - 1)];
    while let Some((first, last)) = stack.pop() {
        if last <= first + 1 {
            continue;
        }
        let (t0, r0) = samples[first];
        let (t1, r1) = samples[last];
        let span = (t1 - t0).max(1e-9);
        let mut worst = (0.0, first);
        for (index, (time, rect)) in samples.iter().enumerate().take(last).skip(first + 1) {
            let expected = r0.lerp(r1, (time - t0) / span);
            let error = rect_distance(expected, *rect);
            if error > worst.0 {
                worst = (error, index);
            }
        }
        if worst.0 > tolerance {
            keep[worst.1] = true;
            stack.push((first, worst.1));
            stack.push((worst.1, last));
        }
    }
    samples
        .iter()
        .zip(keep)
        .filter_map(|(sample, keep)| keep.then_some(*sample))
        .collect()
}

fn interpolate(base: XY<f64>, keys: &[MaskVectorKeyframe], time: f64) -> XY<f64> {
    let Some(first) = keys.first() else {
        return base;
    };
    if time <= first.time {
        return XY::new(first.x, first.y);
    }
    for pair in keys.windows(2) {
        let (a, b) = (&pair[0], &pair[1]);
        if time <= b.time {
            let t = ((time - a.time) / (b.time - a.time).max(1e-6)).clamp(0.0, 1.0);
            return XY::new(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t);
        }
    }
    let last = &keys[keys.len() - 1];
    XY::new(last.x, last.y)
}

/// Where a mask sits at a timeline time, as the renderer places it.
pub fn mask_rect_at(segment: &MaskSegment, time: f64) -> Rect {
    let relative = (time - segment.start).max(0.0);
    let center = interpolate(segment.center, &segment.keyframes.position, relative);
    let size = interpolate(segment.size, &segment.keyframes.size, relative);
    Rect::new(
        center.x - size.x / 2.0,
        center.y - size.y / 2.0,
        size.x,
        size.y,
    )
}

fn duplicates(existing: &MaskSegment, new: &MaskSegment) -> bool {
    if !existing.enabled || !matches!(existing.mask_type, MaskKind::Sensitive) {
        return false;
    }
    let overlap = new.end.min(existing.end) - new.start.max(existing.start);
    let duration = (new.end - new.start).max(1e-6);
    if overlap / duration < 0.8 {
        return false;
    }
    let from = new.start.max(existing.start);
    let to = new.end.min(existing.end);
    (0..=4).all(|step| {
        let time = from + (to - from) * f64::from(step) / 4.0;
        mask_rect_at(existing, time).overlap_of_smaller(mask_rect_at(new, time)) >= 0.7
    })
}

fn overlaps(a: &MaskSegment, b: &MaskSegment) -> bool {
    a.start < b.end && b.start < a.end
}

/// How many mask lanes a scan may add. Every lane is a row on the timeline
/// and a blur pass per frame, so past this, items that share the screen
/// share a mask.
pub const MAX_GENERATED_LANES: u32 = 8;

/// Gives each new mask the lowest mask lane it fits on without overlapping
/// another mask there, and drops masks an earlier scan already made. When
/// more items are on screen at once than [`MAX_GENERATED_LANES`] allows,
/// the extra ones join the placed mask that grows least by covering them.
pub fn place_masks(existing: &[MaskSegment], new: Vec<MaskSegment>) -> Vec<MaskSegment> {
    let mut new = new;
    new.sort_by(|a, b| a.start.total_cmp(&b.start));
    let lane_limit = existing
        .iter()
        .map(|mask| mask.track + 1)
        .max()
        .unwrap_or(0)
        + MAX_GENERATED_LANES;
    let mut placed: Vec<MaskSegment> = Vec::with_capacity(new.len());
    for mut mask in new {
        if existing.iter().any(|existing| duplicates(existing, &mask)) {
            continue;
        }
        let lane = free_lane(existing, &placed, &mask, None);
        if lane >= lane_limit
            && let Some((index, merged)) = best_merge(existing, &placed, &mask)
        {
            placed[index] = merged;
            continue;
        }
        mask.track = lane;
        placed.push(mask);
    }
    placed
}

fn free_lane(
    existing: &[MaskSegment],
    placed: &[MaskSegment],
    mask: &MaskSegment,
    skip: Option<usize>,
) -> u32 {
    let mut lane = 0;
    while existing
        .iter()
        .chain(
            placed
                .iter()
                .enumerate()
                .filter_map(|(index, other)| (Some(index) != skip).then_some(other)),
        )
        .any(|other| other.track == lane && overlaps(other, mask))
    {
        lane += 1;
    }
    lane
}

fn best_merge(
    existing: &[MaskSegment],
    placed: &[MaskSegment],
    mask: &MaskSegment,
) -> Option<(usize, MaskSegment)> {
    placed
        .iter()
        .enumerate()
        .filter(|(_, other)| overlaps(other, mask))
        .filter_map(|(index, other)| {
            let mut merged = merge_masks(other, mask);
            merged.track = other.track;
            let lane_clear = existing
                .iter()
                .chain(
                    placed
                        .iter()
                        .enumerate()
                        .filter_map(|(other_index, other)| (other_index != index).then_some(other)),
                )
                .all(|other| other.track != merged.track || !overlaps(other, &merged));
            lane_clear.then(|| {
                let probe = (mask.start.max(other.start) + mask.end.min(other.end)) / 2.0;
                let growth =
                    mask_rect_at(&merged, probe).area() - mask_rect_at(other, probe).area();
                (growth, index, merged)
            })
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, index, merged)| (index, merged))
}

/// One mask covering both masks for as long as either is shown.
fn merge_masks(a: &MaskSegment, b: &MaskSegment) -> MaskSegment {
    let start = a.start.min(b.start);
    let end = a.end.max(b.end);
    let mut times = vec![a.start, a.end, b.start, b.end];
    for mask in [a, b] {
        times.extend(
            mask.keyframes
                .position
                .iter()
                .chain(&mask.keyframes.size)
                .map(|key| mask.start + key.time),
        );
    }
    times.retain(|time| (start..=end).contains(time));
    times.sort_by(f64::total_cmp);
    times.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    let samples = times
        .into_iter()
        .map(|time| {
            let shown = [a, b]
                .into_iter()
                .filter(|mask| (mask.start - 1e-9..=mask.end + 1e-9).contains(&time))
                .map(|mask| mask_rect_at(mask, time))
                .reduce(|union, rect| union.union(rect))
                .unwrap_or_else(|| mask_rect_at(a, time).union(mask_rect_at(b, time)));
            (time, shown)
        })
        .collect::<Vec<_>>();
    mask_from_samples(&samples, (start, end), a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::track::TrackPoint;

    fn template() -> MaskSegment {
        MaskSegment {
            start: 0.0,
            end: 1.0,
            track: 0,
            enabled: true,
            mask_type: MaskKind::Highlight,
            center: XY::new(0.5, 0.5),
            size: XY::new(0.3, 0.3),
            feather: 0.1,
            opacity: 1.0,
            pixelation: 42.0,
            darkness: 0.5,
            fade_duration: 0.0,
            keyframes: MaskKeyframes::default(),
        }
    }

    fn track(points: &[(f64, f64)], start: f64, end: f64) -> Track {
        Track {
            kind: SensitiveKind::Email,
            key: "a@b.co".into(),
            points: points
                .iter()
                .map(|(time, y)| TrackPoint {
                    time: *time,
                    bounds: Rect::new(0.1, *y, 0.2, 0.02),
                })
                .collect(),
            start,
            end,
        }
    }

    fn plan(template: &MaskSegment) -> MaskPlan<'_> {
        MaskPlan {
            source_size: (1920, 1080),
            total_duration: 10.0,
            template,
            step: 0.1,
        }
    }

    fn identity(_: f64, rect: Rect) -> Option<Rect> {
        Some(rect)
    }

    fn only(mut masks: Vec<MaskSegment>) -> MaskSegment {
        assert_eq!(masks.len(), 1, "{masks:?}");
        masks.remove(0)
    }

    #[test]
    fn a_still_item_becomes_one_padded_static_mask() {
        let template = template();
        let mask = only(track_to_masks(
            &track(&[(2.0, 0.5), (3.0, 0.5)], 1.9, 3.4),
            &plan(&template),
            &mut identity,
        ));
        assert!(matches!(mask.mask_type, MaskKind::Sensitive));
        assert_eq!(mask.pixelation, 42.0);
        assert!((mask.start - 1.8).abs() < 1e-9 && (mask.end - 3.5).abs() < 1e-9);
        assert!(mask.keyframes.position.is_empty() && mask.keyframes.size.is_empty());
        let rect = mask_rect_at(&mask, 2.5);
        let text = Rect::new(0.1, 0.5, 0.2, 0.02);
        assert!(rect.x < text.x && rect.right() > text.right());
        assert!(rect.y < text.y && rect.bottom() > text.bottom());
        let pad_px = (rect.y - text.y).abs() * 1080.0;
        assert!((pad_px - (0.02 * 1080.0 * 0.3 + 3.0)).abs() < 1e-6);
    }

    #[test]
    fn short_items_get_the_minimum_length_inside_the_timeline() {
        let template = template();
        let mask = only(track_to_masks(
            &track(&[(5.0, 0.5)], 5.0, 5.0),
            &plan(&template),
            &mut identity,
        ));
        assert!((mask.end - mask.start - MIN_MASK_DURATION).abs() < 1e-9);
        assert!(mask.start < 5.0 && mask.end > 5.0);

        let at_end = only(track_to_masks(
            &track(&[(9.9, 0.5)], 9.9, 10.0),
            &plan(&template),
            &mut identity,
        ));
        assert_eq!(at_end.end, 10.0);
        assert!((at_end.start - 9.0).abs() < 1e-9);

        let at_start = only(track_to_masks(
            &track(&[(0.0, 0.5)], 0.0, 0.1),
            &plan(&template),
            &mut identity,
        ));
        assert_eq!(at_start.start, 0.0);
        assert!((at_start.end - 1.0).abs() < 1e-9);
    }

    #[test]
    fn a_moving_item_gets_keyframes_that_follow_it() {
        let template = template();
        let mask = only(track_to_masks(
            &track(&[(1.0, 0.8), (2.0, 0.6), (3.0, 0.6)], 1.0, 3.0),
            &plan(&template),
            &mut identity,
        ));
        assert!(mask.keyframes.position.len() >= 3);
        assert!(
            mask.keyframes.position.len() <= 5,
            "{:?}",
            mask.keyframes.position
        );
        for time in [1.0, 1.25, 1.5, 2.0, 2.5, 3.0] {
            let expected = padded_bounds(
                SensitiveKind::Email,
                track(&[(1.0, 0.8), (2.0, 0.6), (3.0, 0.6)], 1.0, 3.0).bounds_at(time),
                (1920, 1080),
            );
            let actual = mask_rect_at(&mask, time);
            assert!(rect_distance(actual, expected) < 1e-3, "{time}: {actual:?}");
        }
    }

    #[test]
    fn masks_follow_the_output_mapping_and_skip_hidden_screens() {
        let template = template();
        let mut zoomed = |time: f64, rect: Rect| {
            if time > 4.0 {
                return None;
            }
            let scale = if time < 2.0 { 1.0 } else { 2.0 };
            Some(Rect::new(
                rect.x * scale,
                rect.y * scale,
                rect.width * scale,
                rect.height * scale,
            ))
        };
        let mask = only(track_to_masks(
            &track(&[(1.0, 0.2), (3.0, 0.2)], 1.0, 3.0),
            &plan(&template),
            &mut zoomed,
        ));
        let before = mask_rect_at(&mask, 1.5);
        let after = mask_rect_at(&mask, 2.5);
        assert!((after.width / before.width - 2.0).abs() < 1e-3);

        let hidden = track_to_masks(
            &track(&[(5.0, 0.2), (6.0, 0.2)], 5.0, 6.0),
            &plan(&template),
            &mut zoomed,
        );
        assert!(hidden.is_empty());

        let mut leaves = |time: f64, rect: Rect| {
            let shift = if (2.0..3.0).contains(&time) { 1.5 } else { 0.0 };
            Some(Rect::new(rect.x + shift, rect.y, rect.width, rect.height))
        };
        let split = track_to_masks(
            &track(&[(0.5, 0.2), (4.5, 0.2)], 0.5, 4.5),
            &plan(&template),
            &mut leaves,
        );
        assert_eq!(split.len(), 2);
        assert!((split[0].end - 2.0).abs() < 1e-6, "{:?}", split[0].end);
        assert!((split[1].start - 2.9).abs() < 1e-6, "{:?}", split[1].start);
        assert!(split.iter().all(|mask| mask.keyframes.position.is_empty()));
    }

    #[test]
    fn lanes_avoid_overlaps_and_reruns_skip_existing_masks() {
        let template = template();
        let make = |start: f64, end: f64, y: f64| {
            only(track_to_masks(
                &track(&[(start, y)], start, end),
                &plan(&template),
                &mut identity,
            ))
        };
        let mut user_mask = template.clone();
        user_mask.mask_type = MaskKind::Sensitive;
        user_mask.start = 0.0;
        user_mask.end = 2.0;
        user_mask.track = 0;
        let first = place_masks(
            std::slice::from_ref(&user_mask),
            vec![
                make(1.0, 4.0, 0.2),
                make(1.5, 3.0, 0.6),
                make(5.0, 6.0, 0.6),
            ],
        );
        assert_eq!(
            first.iter().map(|mask| mask.track).collect::<Vec<_>>(),
            [1, 2, 0]
        );

        let mut all = vec![user_mask];
        all.extend(first.clone());
        let again = place_masks(
            &all,
            vec![
                make(1.0, 4.0, 0.2),
                make(1.5, 3.0, 0.6),
                make(5.0, 6.0, 0.6),
            ],
        );
        assert!(again.is_empty(), "{again:?}");

        let moved = place_masks(&all, vec![make(1.0, 4.0, 0.9)]);
        assert_eq!(moved.len(), 1);
    }

    #[test]
    fn many_items_on_screen_at_once_share_a_bounded_number_of_lanes() {
        let template = template();
        let rows = (0..40)
            .map(|row| {
                let y = 0.02 + f64::from(row) * 0.024;
                only(track_to_masks(
                    &track(&[(1.0, y), (5.0, y)], 1.0, 5.0),
                    &plan(&template),
                    &mut identity,
                ))
            })
            .collect::<Vec<_>>();
        let placed = place_masks(&[], rows.clone());
        let lanes = placed.iter().map(|mask| mask.track).max().unwrap() + 1;
        assert!(lanes <= MAX_GENERATED_LANES, "{lanes}");
        assert!(placed.len() < rows.len());
        for lane in 0..lanes {
            let on_lane = placed
                .iter()
                .filter(|mask| mask.track == lane)
                .collect::<Vec<_>>();
            for (index, a) in on_lane.iter().enumerate() {
                for b in &on_lane[index + 1..] {
                    assert!(!overlaps(a, b));
                }
            }
        }
        for row in &rows {
            let wanted = mask_rect_at(row, 3.0);
            assert!(
                placed.iter().any(|mask| {
                    mask.start <= row.start
                        && mask.end >= row.end
                        && mask_rect_at(mask, 3.0).overlap_of_smaller(wanted) > 0.999
                }),
                "{wanted:?}"
            );
        }

        let sequential = (0..40)
            .map(|step| {
                let start = f64::from(step) * 2.0;
                only(track_to_masks(
                    &track(&[(start, 0.5)], start, start + 1.0),
                    &MaskPlan {
                        total_duration: 100.0,
                        ..plan(&template)
                    },
                    &mut identity,
                ))
            })
            .collect::<Vec<_>>();
        let placed = place_masks(&[], sequential);
        assert_eq!(placed.len(), 40);
        assert!(placed.iter().all(|mask| mask.track == 0));
    }

    #[test]
    fn simplify_keeps_corners_and_drops_straight_runs() {
        let samples = (0..=20)
            .map(|step| {
                let time = f64::from(step) * 0.1;
                let y = if time <= 1.0 { time * 0.1 } else { 0.1 };
                (time, Rect::new(0.0, y, 0.1, 0.1))
            })
            .collect::<Vec<_>>();
        let kept = simplify(&samples, KEYFRAME_TOLERANCE);
        assert_eq!(
            kept.iter().map(|(time, _)| *time).collect::<Vec<_>>(),
            [0.0, 1.0, 2.0]
        );
        let still = vec![(0.0, Rect::UNIT), (1.0, Rect::UNIT), (2.0, Rect::UNIT)];
        assert_eq!(simplify(&still, KEYFRAME_TOLERANCE).len(), 1);
    }
}
