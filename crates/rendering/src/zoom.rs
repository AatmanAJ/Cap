use cap_project::{LoupeShape, XY};

/// Bounds of the (possibly zoomed) display rect, expressed as a scale of the
/// unzoomed display: `(0,0)-(1,1)` means no zoom, while a zoomed-in view
/// expands the rect beyond the unit square so the visible viewport becomes a
/// sub-region of the source frame.
#[derive(Debug, PartialEq, Clone, Copy)]
pub struct SegmentBounds {
    pub top_left: XY<f64>,
    pub bottom_right: XY<f64>,
}

impl SegmentBounds {
    pub fn new(top_left: XY<f64>, bottom_right: XY<f64>) -> Self {
        Self {
            top_left,
            bottom_right,
        }
    }

    pub fn default() -> Self {
        SegmentBounds::new(XY::new(0.0, 0.0), XY::new(1.0, 1.0))
    }

    /// Build bounds from a zoom `amount` (>= 1) and a normalized center in
    /// [0, 1] "travel space": 0 pins the visible viewport flush to the
    /// left/top edge, 1 flush to the right/bottom edge, 0.5 centers it.
    /// Inputs are clamped so the resulting viewport is always contained in
    /// the source frame (the geometric in-bounds guarantee that replaced the
    /// old `ensure_cursor_visible*` post-correction).
    pub fn from_amount_center(amount: f64, center: XY<f64>) -> Self {
        let amount = if amount.is_finite() {
            amount.max(1.0)
        } else {
            1.0
        };
        let center = XY::new(clamp_unit(center.x), clamp_unit(center.y));

        // Same formula the easing-based `from_segment_with_cursor_constraint`
        // used: scale about the origin, then translate so `center` maps across
        // the valid travel range.
        let center_diff = XY::new(center.x * amount - center.x, center.y * amount - center.y);

        SegmentBounds::new(
            XY::new(0.0 - center_diff.x, 0.0 - center_diff.y),
            XY::new(amount - center_diff.x, amount - center_diff.y),
        )
    }

    /// Remaps a scalar in [0,1] so values inside the outer `snap_ratio` band
    /// stick to the edges: [r, 1-r] -> [0, 1], clamped.
    pub(crate) fn snap_to_edges(scalar: f64, snap_ratio: f64) -> f64 {
        if snap_ratio <= 0.0 {
            return scalar;
        }
        let lo = snap_ratio;
        let hi = 1.0 - snap_ratio + 0.0001;
        if hi <= lo {
            return 0.5;
        }
        ((scalar - lo) / (hi - lo)).clamp(0.0, 1.0)
    }

    /// Maps a focus position (UV space) to a `from_amount_center`-space center
    /// for auto zooms: edge-snap each axis, then clamp to [0, 1].
    ///
    /// `from_amount_center` places the center scalar PROPORTIONALLY across the
    /// frame (s = 0 is top-left-flush, s = 1 is bottom-right-flush), so the
    /// full [0, 1] range is exactly the set of in-bounds framings and a focus
    /// snapped to 1.0 puts the frame corner-flush with the content corner —
    /// no centered-viewport band, which would make corners unreachable.
    pub(crate) fn calculate_follow_center(
        focus_pos: (f64, f64),
        edge_snap_ratio: f64,
    ) -> (f64, f64) {
        (
            Self::snap_to_edges(focus_pos.0, edge_snap_ratio).clamp(0.0, 1.0),
            Self::snap_to_edges(focus_pos.1, edge_snap_ratio).clamp(0.0, 1.0),
        )
    }
}

fn clamp_unit(v: f64) -> f64 {
    if v.is_finite() {
        v.clamp(0.0, 1.0)
    } else {
        0.5
    }
}

/// A sampled zoom transform. Produced by `ZoomTransformTimeline::sample`.
#[derive(Debug, Clone, Copy)]
pub struct InterpolatedZoom {
    /// Smoothed 0..1 "zoom activity": the spring-driven response to whether
    /// any zoom segment is active. Reaches 1 once a zoom has fully engaged
    /// and returns to 0 at rest. Consumers (e.g. camera scale-during-zoom)
    /// must not assume monotonicity within a segment; it is always clamped
    /// to [0, 1].
    pub t: f64,
    pub bounds: SegmentBounds,
}

impl InterpolatedZoom {
    pub fn display_amount(&self) -> f64 {
        (self.bounds.bottom_right - self.bounds.top_left).x
    }
}

impl Default for InterpolatedZoom {
    /// The resting transform: no zoom engaged.
    fn default() -> Self {
        Self {
            t: 0.0,
            bounds: SegmentBounds::default(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoupeSample {
    pub presence: f64,
    pub magnification: f64,
    pub center: XY<f64>,
    pub size: f64,
    pub follow_cursor: f64,
    pub shape: LoupeShape,
}

const LOUPE_RECT_HALF_EXTENT: (f64, f64) = (1.3, 0.8);
const LOUPE_RECT_CORNER: f64 = 0.45;
const LOUPE_EDGE_MARGIN: f64 = 0.06;
const LOUPE_MAX_GROW: f64 = 1.1;
const LOUPE_MAX_FOCUS_REACH: f64 = 0.9;

impl LoupeSample {
    pub fn effective_magnification(&self) -> f64 {
        1.0 + (self.magnification.max(1.0) - 1.0) * self.presence.clamp(0.0, 1.0)
    }

    pub fn focus(&self, cursor: Option<XY<f64>>) -> XY<f64> {
        let follow = self.follow_cursor.clamp(0.0, 1.0);
        match cursor.filter(|cursor| cursor.x.is_finite() && cursor.y.is_finite()) {
            Some(cursor) if follow > 0.0 => self.center + (cursor - self.center) * follow,
            _ => self.center,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LoupeGeometry {
    pub center: XY<f64>,
    pub anchor: XY<f64>,
    pub half_extent: XY<f64>,
    pub corner_radius: f64,
    pub magnification: f64,
}

impl LoupeGeometry {
    pub fn resolve(sample: &LoupeSample, focus: XY<f64>, card_size: XY<f64>) -> Option<Self> {
        if !(card_size.x >= 1.0 && card_size.y >= 1.0) {
            return None;
        }
        let focus = XY::new(clamp_unit(focus.x), clamp_unit(focus.y)) * card_size;
        let grow = sample.presence.clamp(0.0, LOUPE_MAX_GROW);
        let size = if sample.size.is_finite() {
            sample.size.max(0.0)
        } else {
            0.0
        };
        let radius = size * card_size.x.min(card_size.y) * grow;
        let (mut half_extent, mut corner_radius) = match sample.shape {
            LoupeShape::Circle => (XY::new(radius, radius), radius),
            LoupeShape::RoundedRect => (
                XY::new(
                    radius * LOUPE_RECT_HALF_EXTENT.0,
                    radius * LOUPE_RECT_HALF_EXTENT.1,
                ),
                radius * LOUPE_RECT_HALF_EXTENT.1 * LOUPE_RECT_CORNER,
            ),
        };

        let margin = radius * LOUPE_EDGE_MARGIN;
        let fit = ((card_size.x * 0.5 - margin) / half_extent.x.max(f64::EPSILON))
            .min((card_size.y * 0.5 - margin) / half_extent.y.max(f64::EPSILON));
        if fit < 1.0 {
            let fit = fit.max(0.0);
            half_extent = half_extent * fit;
            corner_radius *= fit;
        }
        if half_extent.x.min(half_extent.y) < 0.5 {
            return None;
        }

        let magnification = sample.effective_magnification();
        let clamp_axis = |value: f64, inset: f64, extent: f64| {
            if extent - inset <= inset {
                extent * 0.5
            } else {
                value.clamp(inset, extent - inset)
            }
        };
        let center = XY::new(
            clamp_axis(focus.x, half_extent.x + margin, card_size.x),
            clamp_axis(focus.y, half_extent.y + margin, card_size.y),
        );
        let mut anchor = XY::new(
            clamp_axis(focus.x, half_extent.x / magnification, card_size.x),
            clamp_axis(focus.y, half_extent.y / magnification, card_size.y),
        );
        // In a card corner the per-axis clamp would push the focus out past
        // the lens's curved rim. Keeping the focus visible wins there, at the
        // cost of the lens corner reaching slightly past the content edge.
        let shown = (focus - anchor) * magnification / half_extent;
        let reach = shown.x.hypot(shown.y);
        if reach > LOUPE_MAX_FOCUS_REACH {
            anchor = focus - (focus - anchor) * (LOUPE_MAX_FOCUS_REACH / reach);
        }

        Some(Self {
            center,
            anchor,
            half_extent,
            corner_radius,
            magnification,
        })
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // The old easing-based tests (segment ramps, ensure_cursor_visible*) were
    // removed together with the state machine they exercised; the spring
    // timeline that replaced it is covered in `zoom_spring.rs`. These tests
    // pin down the pure geometry that survived the rework.

    macro_rules! assert_f64_near {
        ($left:expr, $right:expr, $label:literal) => {
            let left = $left;
            let right = $right;
            assert!(
                (left - right).abs() < 1e-9,
                "{}: left `{:?}` != right `{:?}`",
                $label,
                left,
                right
            )
        };
    }

    #[test]
    fn from_amount_center_identity_at_amount_one() {
        let bounds = SegmentBounds::from_amount_center(1.0, XY::new(0.3, 0.9));
        assert_f64_near!(bounds.top_left.x, 0.0, "top_left.x");
        assert_f64_near!(bounds.top_left.y, 0.0, "top_left.y");
        assert_f64_near!(bounds.bottom_right.x, 1.0, "bottom_right.x");
        assert_f64_near!(bounds.bottom_right.y, 1.0, "bottom_right.y");
    }

    #[test]
    fn from_amount_center_centered() {
        let bounds = SegmentBounds::from_amount_center(2.0, XY::new(0.5, 0.5));
        assert_f64_near!(bounds.top_left.x, -0.5, "top_left.x");
        assert_f64_near!(bounds.top_left.y, -0.5, "top_left.y");
        assert_f64_near!(bounds.bottom_right.x, 1.5, "bottom_right.x");
        assert_f64_near!(bounds.bottom_right.y, 1.5, "bottom_right.y");
    }

    #[test]
    fn from_amount_center_edges_stay_in_bounds() {
        // center 0 -> viewport flush left/top, center 1 -> flush right/bottom.
        for amount in [1.0, 1.5, 2.0, 3.0, 4.5] {
            for c in [0.0, 0.25, 0.5, 0.75, 1.0] {
                let bounds = SegmentBounds::from_amount_center(amount, XY::new(c, c));
                // Display must cover the whole output: top_left <= 0 and
                // bottom_right >= 1 on both axes.
                assert!(bounds.top_left.x <= 1e-9, "tl.x for a={amount} c={c}");
                assert!(bounds.top_left.y <= 1e-9, "tl.y for a={amount} c={c}");
                assert!(
                    bounds.bottom_right.x >= 1.0 - 1e-9,
                    "br.x for a={amount} c={c}"
                );
                assert!(
                    bounds.bottom_right.y >= 1.0 - 1e-9,
                    "br.y for a={amount} c={c}"
                );
            }
        }
    }

    #[test]
    fn from_amount_center_clamps_bad_inputs() {
        let bounds = SegmentBounds::from_amount_center(0.25, XY::new(-4.0, 7.0));
        // amount below 1 clamps to identity scale; centers clamp into [0,1].
        assert_f64_near!(bounds.bottom_right.x - bounds.top_left.x, 1.0, "width");
        let nan_bounds = SegmentBounds::from_amount_center(f64::NAN, XY::new(f64::NAN, 0.5));
        assert!(nan_bounds.top_left.x.is_finite());
        assert!(nan_bounds.bottom_right.y.is_finite());
    }

    #[test]
    fn follow_center_maps_focus_proportionally() {
        // Center focus stays centered.
        let (cx, cy) = SegmentBounds::calculate_follow_center((0.5, 0.5), 0.0);
        assert!((cx - 0.5).abs() < 1e-6);
        assert!((cy - 0.5).abs() < 1e-6);

        // Extreme focus reaches the corner-flush framings so edge/corner
        // content is actually reachable (proportional placement semantics).
        let (left, _) = SegmentBounds::calculate_follow_center((0.0, 0.5), 0.0);
        assert!(left.abs() < 1e-6);
        let (right, _) = SegmentBounds::calculate_follow_center((1.0, 0.5), 0.0);
        assert!((right - 1.0).abs() < 1e-6);

        // A corner-flush framing keeps the frame fully covered at any amount.
        let bounds = SegmentBounds::from_amount_center(2.0, XY::new(1.0, 1.0));
        assert!(bounds.top_left.x <= 0.0 && bounds.bottom_right.x >= 1.0);
    }

    #[test]
    fn display_amount_is_zoom_scale() {
        let zoom = InterpolatedZoom {
            t: 1.0,
            bounds: SegmentBounds::from_amount_center(2.5, XY::new(0.4, 0.6)),
        };
        assert!((zoom.display_amount() - 2.5).abs() < 1e-9);
    }

    fn loupe(presence: f64, magnification: f64, size: f64, shape: LoupeShape) -> LoupeSample {
        LoupeSample {
            presence,
            magnification,
            center: XY::new(0.5, 0.5),
            size,
            follow_cursor: 0.0,
            shape,
        }
    }

    fn magnified_position(geometry: &LoupeGeometry, content: XY<f64>) -> Option<XY<f64>> {
        let shown = geometry.center + (content - geometry.anchor) * geometry.magnification;
        let offset = shown - geometry.center;
        let inside = (offset.x / geometry.half_extent.x).powi(2)
            + (offset.y / geometry.half_extent.y).powi(2)
            <= 1.0 + 1e-9;
        inside.then_some(shown)
    }

    const CARD: XY<f64> = XY {
        x: 1920.0,
        y: 1080.0,
    };

    #[test]
    fn loupe_magnification_and_size_grow_with_presence() {
        let sample = loupe(0.0, 3.0, 0.2, LoupeShape::Circle);
        assert_f64_near!(sample.effective_magnification(), 1.0, "hidden");
        assert!(LoupeGeometry::resolve(&sample, XY::new(0.5, 0.5), CARD).is_none());

        let half = LoupeGeometry::resolve(
            &LoupeSample {
                presence: 0.5,
                ..sample
            },
            XY::new(0.5, 0.5),
            CARD,
        )
        .unwrap();
        assert_f64_near!(half.magnification, 2.0, "half-way magnification");
        assert_f64_near!(half.half_extent.x, 0.2 * 1080.0 * 0.5, "half-way radius");

        let full = LoupeGeometry::resolve(
            &LoupeSample {
                presence: 1.0,
                ..sample
            },
            XY::new(0.5, 0.5),
            CARD,
        )
        .unwrap();
        assert_f64_near!(full.magnification, 3.0, "settled magnification");
        assert_f64_near!(full.half_extent.x, 0.2 * 1080.0, "settled radius");
        assert_f64_near!(full.corner_radius, full.half_extent.x, "circle corner");

        let overshoot = LoupeSample {
            presence: 1.04,
            ..sample
        };
        assert_f64_near!(
            overshoot.effective_magnification(),
            3.0,
            "overshoot magnification"
        );
        let settled_bounce = LoupeGeometry::resolve(&overshoot, XY::new(0.5, 0.5), CARD).unwrap();
        assert!(settled_bounce.half_extent.x > full.half_extent.x);
    }

    #[test]
    fn loupe_grows_from_its_focus() {
        for presence in [0.02, 0.2, 0.6, 1.0] {
            let geometry = LoupeGeometry::resolve(
                &loupe(presence, 2.0, 0.22, LoupeShape::Circle),
                XY::new(0.3, 0.6),
                CARD,
            )
            .unwrap();
            let focus = XY::new(0.3 * CARD.x, 0.6 * CARD.y);
            assert_f64_near!(geometry.center.x, focus.x, "center.x");
            assert_f64_near!(geometry.center.y, focus.y, "center.y");
            assert_f64_near!(geometry.anchor.x, focus.x, "anchor.x");
            assert_f64_near!(geometry.anchor.y, focus.y, "anchor.y");
        }
    }

    #[test]
    fn loupe_stays_on_the_card_and_keeps_its_focus_visible_at_edges() {
        for shape in [LoupeShape::Circle, LoupeShape::RoundedRect] {
            for focus in [
                XY::new(0.0, 0.0),
                XY::new(1.0, 0.0),
                XY::new(0.02, 0.5),
                XY::new(1.0, 1.0),
                XY::new(0.5, 0.99),
            ] {
                let geometry =
                    LoupeGeometry::resolve(&loupe(1.0, 2.5, 0.3, shape), focus, CARD).unwrap();
                let h = geometry.half_extent;
                assert!(geometry.center.x - h.x >= 0.0 && geometry.center.x + h.x <= CARD.x);
                assert!(geometry.center.y - h.y >= 0.0 && geometry.center.y + h.y <= CARD.y);

                let corner =
                    (focus.x == 0.0 || focus.x == 1.0) && (focus.y == 0.0 || focus.y == 1.0);
                if !corner {
                    let reach = h / geometry.magnification;
                    let slack = reach * (1.0 - LOUPE_MAX_FOCUS_REACH) + XY::new(1e-9, 1e-9);
                    assert!(geometry.anchor.x - reach.x >= -slack.x);
                    assert!(geometry.anchor.x + reach.x <= CARD.x + slack.x);
                    assert!(geometry.anchor.y - reach.y >= -slack.y);
                    assert!(geometry.anchor.y + reach.y <= CARD.y + slack.y);
                }

                let focus_px = focus * CARD;
                assert!(
                    magnified_position(&geometry, focus_px).is_some(),
                    "{shape:?} focus {focus:?} must show inside the lens: {geometry:?}"
                );
            }
        }
    }

    #[test]
    fn loupe_shrinks_to_fit_a_small_card() {
        let card = XY::new(300.0, 120.0);
        let geometry = LoupeGeometry::resolve(
            &loupe(1.0, 2.0, 0.45, LoupeShape::RoundedRect),
            XY::new(0.5, 0.5),
            card,
        )
        .unwrap();
        assert!(geometry.half_extent.x > geometry.half_extent.y);
        assert!(geometry.half_extent.y * 2.0 <= card.y);
        assert!(geometry.half_extent.x * 2.0 <= card.x);
        assert!(geometry.corner_radius < geometry.half_extent.y);

        assert!(
            LoupeGeometry::resolve(
                &loupe(1.0, 2.0, 0.3, LoupeShape::Circle),
                XY::new(0.5, 0.5),
                XY::new(0.0, 0.0)
            )
            .is_none()
        );
    }

    #[test]
    fn loupe_focus_blends_toward_the_cursor() {
        let mut sample = loupe(1.0, 2.0, 0.2, LoupeShape::Circle);
        sample.center = XY::new(0.2, 0.2);
        assert_eq!(sample.focus(Some(XY::new(0.8, 0.6))), sample.center);
        sample.follow_cursor = 1.0;
        assert_eq!(sample.focus(Some(XY::new(0.8, 0.6))), XY::new(0.8, 0.6));
        assert_eq!(sample.focus(None), sample.center);
        assert_eq!(sample.focus(Some(XY::new(f64::NAN, 0.6))), sample.center);
        sample.follow_cursor = 0.5;
        let half = sample.focus(Some(XY::new(0.8, 0.6)));
        assert_f64_near!(half.x, 0.5, "x");
        assert_f64_near!(half.y, 0.4, "y");
    }
}
