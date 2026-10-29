use crate::{Detection, Rect, SampleWindow, SensitiveKind};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TrackPoint {
    pub time: f64,
    pub bounds: Rect,
}

/// One sensitive item followed over time.
#[derive(Clone, Debug, PartialEq)]
pub struct Track {
    pub kind: SensitiveKind,
    pub key: String,
    pub points: Vec<TrackPoint>,
    /// The item may already be on screen from here: the last moment it was
    /// known to be absent.
    pub start: f64,
    /// The item may still be on screen until here.
    pub end: f64,
}

impl Track {
    pub fn bounds_at(&self, time: f64) -> Rect {
        let first = self.points[0];
        if time <= first.time {
            return first.bounds;
        }
        for pair in self.points.windows(2) {
            let (a, b) = (pair[0], pair[1]);
            if time <= b.time {
                let span = (b.time - a.time).max(1e-9);
                return a.bounds.lerp(b.bounds, (time - a.time) / span);
            }
        }
        self.points[self.points.len() - 1].bounds
    }
}

#[derive(Clone, Copy, Debug)]
pub struct TrackerConfig {
    /// How long an item may go unseen (an OCR miss, a tooltip over it)
    /// before its track ends.
    pub max_gap: f64,
    /// The furthest an item may move between two passes and still be the
    /// same item, as a share of the frame.
    pub max_jump: f64,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        Self {
            max_gap: 1.5,
            max_jump: 0.6,
        }
    }
}

struct ActiveTrack {
    track: Track,
    last_seen: f64,
    vanished_by: Option<f64>,
}

/// Links the detections of successive text-recognition passes into tracks
/// by kind, text and position.
pub struct Tracker {
    config: TrackerConfig,
    active: Vec<ActiveTrack>,
    done: Vec<Track>,
    last_motion: f64,
}

impl Tracker {
    pub fn new(config: TrackerConfig) -> Self {
        Self {
            config,
            active: Vec::new(),
            done: Vec::new(),
            last_motion: 0.0,
        }
    }

    pub fn active_len(&self) -> usize {
        self.active.len()
    }

    /// The furthest any item moved in the last pass, as a share of the
    /// frame. Scrolling content needs denser passes to be followed.
    pub fn last_motion(&self) -> f64 {
        self.last_motion
    }

    pub fn push(&mut self, window: &SampleWindow, detections: &[Detection]) {
        let time = window.end_time();
        let detections = dedupe_detections(detections);

        let mut pairs = Vec::new();
        for (track_index, active) in self.active.iter().enumerate() {
            let last = active.track.points[active.track.points.len() - 1].bounds;
            for (detection_index, detection) in detections.iter().enumerate() {
                if let Some(score) = self.match_score(&active.track, last, detection) {
                    pairs.push((score, track_index, detection_index));
                }
            }
        }
        pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

        self.last_motion = 0.0;
        let mut track_taken = vec![false; self.active.len()];
        let mut detection_taken = vec![false; detections.len()];
        for (_, track_index, detection_index) in pairs {
            if track_taken[track_index] || detection_taken[detection_index] {
                continue;
            }
            track_taken[track_index] = true;
            detection_taken[detection_index] = true;
            let active = &mut self.active[track_index];
            let (ax, ay) = active.track.points[active.track.points.len() - 1]
                .bounds
                .center();
            let (bx, by) = detections[detection_index].bounds.center();
            self.last_motion = self.last_motion.max((ax - bx).abs().max((ay - by).abs()));
            active.track.points.push(TrackPoint {
                time,
                bounds: detections[detection_index].bounds,
            });
            active.last_seen = time;
            active.vanished_by = None;
        }

        let max_gap = self.config.max_gap;
        let mut keep = track_taken.into_iter();
        self.active.retain_mut(|active| {
            if keep.next().unwrap_or(false) {
                return true;
            }
            if active.vanished_by.is_none() {
                let last = active.track.points[active.track.points.len() - 1].bounds;
                let vanished = if active.last_seen >= window.start_time() {
                    window.vanished_by(last)
                } else {
                    time
                };
                active.vanished_by = Some(vanished);
            }
            if time - active.last_seen > max_gap {
                let mut track = active.track.clone();
                track.end = active.vanished_by.unwrap_or(time);
                self.done.push(track);
                return false;
            }
            true
        });

        for (detection, taken) in detections.iter().zip(detection_taken) {
            if taken {
                continue;
            }
            self.active.push(ActiveTrack {
                track: Track {
                    kind: detection.kind,
                    key: detection.key.clone(),
                    points: vec![TrackPoint {
                        time,
                        bounds: detection.bounds,
                    }],
                    start: window.appeared_after(detection.bounds).min(time),
                    end: time,
                },
                last_seen: time,
                vanished_by: None,
            });
        }
    }

    fn match_score(&self, track: &Track, last: Rect, detection: &Detection) -> Option<f64> {
        if track.kind != detection.kind {
            return None;
        }
        let (ax, ay) = last.center();
        let (bx, by) = detection.bounds.center();
        let distance = ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt();
        if track.kind == SensitiveKind::Face {
            let reach = (last.width.max(last.height) * 1.5).max(0.15);
            return (distance <= reach).then_some(distance);
        }
        if distance > self.config.max_jump {
            return None;
        }
        if track.key == detection.key {
            return Some(distance);
        }
        let near = distance <= last.height.max(detection.bounds.height) * 2.0;
        (near && keys_similar(&track.key, &detection.key)).then_some(distance + 1.0)
    }

    pub fn finish(mut self, end_time: f64) -> Vec<Track> {
        for active in self.active.drain(..) {
            let mut track = active.track;
            track.end = active.vanished_by.unwrap_or(end_time).max(track.end);
            self.done.push(track);
        }
        merge_tracks(self.done, self.config)
    }
}

/// OCR of the same text in consecutive frames can differ by a character or
/// two; long keys that close are the same item.
pub fn keys_similar(a: &str, b: &str) -> bool {
    let shorter = a.len().min(b.len());
    if shorter < 12 || a.len().abs_diff(b.len()) > 2 {
        return false;
    }
    bounded_edit_distance(a.as_bytes(), b.as_bytes(), 2).is_some()
}

fn bounded_edit_distance(a: &[u8], b: &[u8], limit: usize) -> Option<usize> {
    let mut previous = (0..=b.len()).collect::<Vec<_>>();
    let mut current = vec![0; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        current[0] = i + 1;
        let mut row_min = current[0];
        for (j, cb) in b.iter().enumerate() {
            let substitution = previous[j] + usize::from(ca != cb);
            current[j + 1] = substitution.min(previous[j + 1] + 1).min(current[j] + 1);
            row_min = row_min.min(current[j + 1]);
        }
        if row_min > limit {
            return None;
        }
        std::mem::swap(&mut previous, &mut current);
    }
    (previous[b.len()] <= limit).then_some(previous[b.len()])
}

fn kind_rank(kind: SensitiveKind) -> u8 {
    match kind {
        SensitiveKind::ApiKey => 6,
        SensitiveKind::Token => 5,
        SensitiveKind::Secret => 4,
        SensitiveKind::CardNumber => 3,
        SensitiveKind::Email => 2,
        SensitiveKind::Phone => 1,
        SensitiveKind::Face => 0,
    }
}

/// Recognized lines can overlap (Vision sometimes reports a line and a
/// fragment of it), which would put two masks on one item.
pub fn dedupe_detections(detections: &[Detection]) -> Vec<Detection> {
    let mut kept: Vec<Detection> = Vec::with_capacity(detections.len());
    for detection in detections {
        let duplicate = kept.iter_mut().find(|existing| {
            (existing.kind == SensitiveKind::Face) == (detection.kind == SensitiveKind::Face)
                && existing.bounds.overlap_of_smaller(detection.bounds) >= 0.6
        });
        match duplicate {
            Some(existing) => {
                if kind_rank(detection.kind) > kind_rank(existing.kind)
                    || detection.bounds.area() > existing.bounds.area()
                        && kind_rank(detection.kind) == kind_rank(existing.kind)
                {
                    existing.kind = detection.kind;
                    existing.key = detection.key.clone();
                }
                existing.bounds = existing.bounds.union(detection.bounds);
            }
            None => kept.push(detection.clone()),
        }
    }
    kept
}

/// Joins tracks of one item that a long miss split in two, and tracks that
/// cover the same item at the same time.
pub fn merge_tracks(mut tracks: Vec<Track>, config: TrackerConfig) -> Vec<Track> {
    tracks.sort_by(|a, b| a.start.total_cmp(&b.start));
    let mut merged: Vec<Track> = Vec::with_capacity(tracks.len());
    'next: for track in tracks {
        for existing in merged.iter_mut() {
            if should_merge(existing, &track, config) {
                existing.points.extend(track.points.iter().copied());
                existing.points.sort_by(|a, b| a.time.total_cmp(&b.time));
                existing.points.dedup_by(|b, a| {
                    if (a.time - b.time).abs() < 1e-9 {
                        a.bounds = a.bounds.union(b.bounds);
                        true
                    } else {
                        false
                    }
                });
                existing.start = existing.start.min(track.start);
                existing.end = existing.end.max(track.end);
                if kind_rank(track.kind) > kind_rank(existing.kind) {
                    existing.kind = track.kind;
                }
                continue 'next;
            }
        }
        merged.push(track);
    }
    merged
}

fn should_merge(a: &Track, b: &Track, config: TrackerConfig) -> bool {
    let same_item = a.kind == b.kind && (a.key == b.key || keys_similar(&a.key, &b.key));
    let overlap_start = a.start.max(b.start);
    let overlap_end = a.end.min(b.end);
    if overlap_end >= overlap_start {
        let probe = (overlap_start + overlap_end) / 2.0;
        return a.bounds_at(probe).overlap_of_smaller(b.bounds_at(probe)) >= 0.6;
    }
    let (earlier, later) = if a.end <= b.start { (a, b) } else { (b, a) };
    if !same_item || later.start - earlier.end > config.max_gap {
        return false;
    }
    let from = earlier.points[earlier.points.len() - 1].bounds.center();
    let to = later.points[0].bounds.center();
    let distance = ((from.0 - to.0).powi(2) + (from.1 - to.1).powi(2)).sqrt();
    distance <= config.max_jump
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::Thumbnail;

    fn blank() -> Arc<Thumbnail> {
        Arc::new(Thumbnail {
            width: 10,
            height: 10,
            cells: vec![0; 100],
        })
    }

    fn window(from: f64, to: f64) -> SampleWindow {
        SampleWindow {
            frames: vec![(from, blank()), (to, blank())],
        }
    }

    fn email(key: &str, x: f64, y: f64) -> Detection {
        Detection {
            kind: SensitiveKind::Email,
            key: key.into(),
            bounds: Rect::new(x, y, 0.2, 0.03),
        }
    }

    fn run(samples: &[(f64, Vec<Detection>)], end: f64) -> Vec<Track> {
        let mut tracker = Tracker::new(TrackerConfig::default());
        let mut previous = 0.0;
        for (time, detections) in samples {
            tracker.push(&window(previous, *time), detections);
            previous = *time;
        }
        tracker.finish(end)
    }

    #[test]
    fn follows_an_item_that_scrolls() {
        let samples = (0..=10)
            .map(|step| {
                let time = f64::from(step) * 0.5;
                (time, vec![email("a@b.co", 0.1, 0.8 - time * 0.1)])
            })
            .collect::<Vec<_>>();
        let tracks = run(&samples, 5.0);
        assert_eq!(tracks.len(), 1);
        let track = &tracks[0];
        assert_eq!(track.points.len(), 11);
        assert_eq!(track.end, 5.0);
        let middle = track.bounds_at(1.25);
        assert!((middle.y - (0.8 - 0.125)).abs() < 1e-9, "{middle:?}");
    }

    #[test]
    fn keeps_two_copies_of_the_same_text_apart() {
        let samples = (0..4)
            .map(|step| {
                let time = f64::from(step) * 0.5;
                (
                    time,
                    vec![email("a@b.co", 0.1, 0.2), email("a@b.co", 0.1, 0.6)],
                )
            })
            .collect::<Vec<_>>();
        let tracks = run(&samples, 2.0);
        assert_eq!(tracks.len(), 2);
        for track in &tracks {
            let y = track.points[0].bounds.y;
            assert!(track.points.iter().all(|point| point.bounds.y == y));
        }
    }

    #[test]
    fn bridges_a_short_ocr_miss_and_splits_on_a_long_absence() {
        let present = |time: f64| (time, vec![email("a@b.co", 0.1, 0.2)]);
        let absent = |time: f64| (time, Vec::new());
        let tracks = run(
            &[present(0.0), absent(0.5), present(1.0), present(1.5)],
            2.0,
        );
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].start, 0.0);
        assert_eq!(tracks[0].end, 2.0);

        let mut samples = vec![present(0.0), present(0.5)];
        samples.extend((2..=10).map(|step| absent(f64::from(step) * 0.5)));
        samples.push(present(5.5));
        let tracks = run(&samples, 6.0);
        assert_eq!(tracks.len(), 2);
        assert_eq!(tracks[0].end, 1.0);
        assert_eq!(tracks[1].start, 5.0);
    }

    #[test]
    fn tolerates_ocr_noise_in_long_keys() {
        let key = concat!("sk", "-proj-Xr7kQ2mN9vB4tL8wZ1cY6pH3");
        let noisy = concat!("sk", "-proj-Xr7kQ2mN9vB4tL8wZlcY6pH3");
        let detection = |key: &str| Detection {
            kind: SensitiveKind::ApiKey,
            key: key.into(),
            bounds: Rect::new(0.1, 0.5, 0.3, 0.03),
        };
        let tracks = run(
            &[
                (0.0, vec![detection(key)]),
                (0.5, vec![detection(noisy)]),
                (1.0, vec![detection(key)]),
            ],
            1.0,
        );
        assert_eq!(tracks.len(), 1);
        assert!(keys_similar(key, noisy));
        assert!(!keys_similar("a@b.co", "a@c.co"));
        assert!(!keys_similar(
            key,
            concat!("sk", "-proj-AAAAAAAAAAAAAAAAAAAAAAAA")
        ));
    }

    #[test]
    fn different_text_at_the_same_place_shares_one_mask() {
        let tracks = run(
            &[
                (0.0, vec![email("first@example.com", 0.1, 0.2)]),
                (0.5, vec![email("other@example.com", 0.1, 0.2)]),
            ],
            1.0,
        );
        assert_eq!(tracks.len(), 1);
        assert_eq!((tracks[0].start, tracks[0].end), (0.0, 1.0));

        let tracks = run(
            &[
                (0.0, vec![email("first@example.com", 0.1, 0.2)]),
                (0.5, vec![email("other@example.com", 0.1, 0.6)]),
            ],
            1.0,
        );
        assert_eq!(tracks.len(), 2);
        let first = tracks.iter().find(|t| t.key.starts_with("first")).unwrap();
        assert_eq!(first.end, 0.5);
    }

    #[test]
    fn narrows_appearance_and_disappearance_to_the_changed_frame() {
        let filled = Arc::new(Thumbnail {
            width: 10,
            height: 10,
            cells: vec![200; 100],
        });
        let item = email("a@b.co", 0.1, 0.2);
        let mut tracker = Tracker::new(TrackerConfig::default());
        tracker.push(&SampleWindow::single(0.0, blank()), &[]);
        tracker.push(
            &SampleWindow {
                frames: vec![
                    (0.0, blank()),
                    (0.1, blank()),
                    (0.2, blank()),
                    (0.3, filled.clone()),
                    (0.4, filled.clone()),
                    (0.5, filled.clone()),
                ],
            },
            std::slice::from_ref(&item),
        );
        tracker.push(
            &SampleWindow {
                frames: vec![
                    (0.5, filled.clone()),
                    (0.6, filled.clone()),
                    (0.7, blank()),
                    (0.8, blank()),
                    (0.9, blank()),
                    (1.0, blank()),
                ],
            },
            &[],
        );
        let tracks = tracker.finish(3.0);
        assert_eq!(tracks.len(), 1);
        assert_eq!(tracks[0].start, 0.2);
        assert_eq!(tracks[0].end, 0.7);
    }

    #[test]
    fn duplicate_detections_in_one_frame_collapse() {
        let wide = email("jane@example.com", 0.1, 0.2);
        let mut fragment = email("jane@example.com", 0.12, 0.2);
        fragment.bounds.width = 0.1;
        let face = Detection {
            kind: SensitiveKind::Face,
            key: String::new(),
            bounds: Rect::new(0.1, 0.2, 0.2, 0.2),
        };
        let kept = dedupe_detections(&[wide.clone(), fragment, face]);
        assert_eq!(kept.len(), 2);
        assert!((kept[0].bounds.right() - wide.bounds.right()).abs() < 1e-9);
        assert_eq!(kept[0].bounds.x, wide.bounds.x);
    }

    #[test]
    fn faces_track_by_position() {
        let face = |x: f64| Detection {
            kind: SensitiveKind::Face,
            key: String::new(),
            bounds: Rect::new(x, 0.3, 0.1, 0.15),
        };
        let tracks = run(
            &[
                (0.0, vec![face(0.1), face(0.7)]),
                (0.5, vec![face(0.12), face(0.68)]),
            ],
            1.0,
        );
        assert_eq!(tracks.len(), 2);
        assert!(tracks.iter().all(|track| track.points.len() == 2));
    }

    #[test]
    fn merging_joins_split_tracks_of_one_item_only() {
        let track = |key: &str, start: f64, end: f64, x: f64| Track {
            kind: SensitiveKind::Email,
            key: key.into(),
            points: vec![
                TrackPoint {
                    time: start,
                    bounds: Rect::new(x, 0.2, 0.2, 0.03),
                },
                TrackPoint {
                    time: end,
                    bounds: Rect::new(x, 0.2, 0.2, 0.03),
                },
            ],
            start,
            end,
        };
        let config = TrackerConfig::default();
        let merged = merge_tracks(
            vec![
                track("a@b.co", 0.0, 2.0, 0.1),
                track("a@b.co", 2.5, 4.0, 0.1),
            ],
            config,
        );
        assert_eq!(merged.len(), 1);
        assert_eq!((merged[0].start, merged[0].end), (0.0, 4.0));

        let apart = merge_tracks(
            vec![
                track("a@b.co", 0.0, 2.0, 0.1),
                track("a@b.co", 5.0, 6.0, 0.1),
            ],
            config,
        );
        assert_eq!(apart.len(), 2);

        let different = merge_tracks(
            vec![
                track("a@b.co", 0.0, 2.0, 0.1),
                track("c@d.co", 2.5, 4.0, 0.1),
            ],
            config,
        );
        assert_eq!(different.len(), 2);

        let stacked = merge_tracks(
            vec![
                track("a@b.co", 0.0, 2.0, 0.1),
                track("a@b.co", 1.0, 3.0, 0.1),
            ],
            config,
        );
        assert_eq!(stacked.len(), 1);
    }
}
