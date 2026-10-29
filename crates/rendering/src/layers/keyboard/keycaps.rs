use bytemuck::{Pod, Zeroable};

pub(super) const MAX_KEYCAPS: usize = 8;
const APPEAR_SECONDS: f64 = 0.16;
const PRESS_DOWN_SECONDS: f64 = 0.035;
const PRESS_UP_SECONDS: f64 = 0.12;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
pub(super) struct KeycapInstance {
    pub rect: [f32; 4],
    pub params: [f32; 4],
}

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
pub(super) struct KeycapUniforms {
    pub face_color: [f32; 4],
    pub edge_color: [f32; 4],
    pub globals: [f32; 4],
    pub keycaps: [KeycapInstance; MAX_KEYCAPS],
}

impl Default for KeycapUniforms {
    fn default() -> Self {
        Self::zeroed()
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub(super) struct KeycapMetrics {
    pub label_size: f32,
    pub face_height: f32,
    pub depth: f32,
    pub padding_x: f32,
    pub gap: f32,
    pub radius: f32,
    pub shadow_blur: f32,
}

impl KeycapMetrics {
    pub fn new(font_size: f32) -> Self {
        Self {
            label_size: font_size * 0.72,
            face_height: font_size * 1.36,
            depth: (font_size * 0.1).max(1.0),
            padding_x: font_size * 0.38,
            gap: font_size * 0.16,
            radius: font_size * 0.26,
            shadow_blur: (font_size * 0.22).max(1.0),
        }
    }

    pub fn height(&self) -> f32 {
        self.face_height + self.depth
    }
}

pub(super) fn layout_keycaps(text_widths: &[f32], metrics: KeycapMetrics) -> (Vec<[f32; 4]>, f32) {
    let mut x = 0.0;
    let mut rects = Vec::with_capacity(text_widths.len());
    for (index, text_width) in text_widths.iter().enumerate() {
        if index > 0 {
            x += metrics.gap;
        }
        let width = (text_width + metrics.padding_x * 2.0).max(metrics.face_height);
        rects.push([x, 0.0, width, metrics.face_height]);
        x += width;
    }
    (rects, x)
}

pub(super) fn appear_scale(seconds_since_appear: f64) -> f32 {
    let t = (seconds_since_appear / APPEAR_SECONDS).clamp(0.0, 1.0);
    let overshoot = 1.70158;
    let shifted = t - 1.0;
    let eased = 1.0 + (overshoot + 1.0) * shifted.powi(3) + overshoot * shifted.powi(2);
    (0.82 + 0.18 * eased) as f32
}

pub(super) fn press_amount(seconds_since_press: f64) -> f32 {
    if !(0.0..PRESS_DOWN_SECONDS + PRESS_UP_SECONDS).contains(&seconds_since_press) {
        return 0.0;
    }
    if seconds_since_press < PRESS_DOWN_SECONDS {
        return (seconds_since_press / PRESS_DOWN_SECONDS) as f32;
    }
    let release = (seconds_since_press - PRESS_DOWN_SECONDS) / PRESS_UP_SECONDS;
    (1.0 - release).powi(2) as f32
}

pub(super) fn keycap_colors(face_rgb: [f32; 3], alpha: f32) -> ([f32; 4], [f32; 4]) {
    let luminance = 0.2126 * face_rgb[0] + 0.7152 * face_rgb[1] + 0.0722 * face_rgb[2];
    let lift = (0.16 - luminance).max(0.0);
    let face = face_rgb.map(|channel| channel + (1.0 - channel) * lift);
    let edge = face.map(|channel| channel * 0.55);
    (
        [face[0], face[1], face[2], alpha],
        [edge[0], edge[1], edge[2], alpha],
    )
}

pub(super) fn scale_rect(rect: [f32; 4], origin: [f32; 2], scale: f32) -> [f32; 4] {
    [
        origin[0] + (rect[0] - origin[0]) * scale,
        origin[1] + (rect[1] - origin[1]) * scale,
        rect[2] * scale,
        rect[3] * scale,
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keycaps_are_at_least_square_and_joined_by_one_gap() {
        let metrics = KeycapMetrics::new(50.0);
        let (rects, width) = layout_keycaps(&[20.0, 20.0, 150.0], metrics);
        assert_eq!(rects[0][2], metrics.face_height);
        assert_eq!(rects[1][0], metrics.face_height + metrics.gap);
        assert_eq!(rects[2][2], 150.0 + metrics.padding_x * 2.0);
        assert_eq!(width, rects[2][0] + rects[2][2]);
        assert!(rects.iter().all(|rect| rect[3] == metrics.face_height));
        assert_eq!(layout_keycaps(&[], metrics).1, 0.0);
    }

    #[test]
    fn metrics_scale_with_output_resolution() {
        let base = KeycapMetrics::new(50.0);
        let double = KeycapMetrics::new(100.0);
        assert_eq!(double.face_height, base.face_height * 2.0);
        assert_eq!(double.gap, base.gap * 2.0);
        assert_eq!(KeycapMetrics::new(2.0).depth, 1.0);
    }

    #[test]
    fn appear_overshoots_then_settles() {
        assert!((appear_scale(0.0) - 0.82).abs() < 1e-6);
        let peak = (1..16)
            .map(|step| appear_scale(step as f64 * 0.01))
            .fold(0.0_f32, f32::max);
        assert!(peak > 1.0 && peak < 1.05);
        assert_eq!(appear_scale(APPEAR_SECONDS), 1.0);
        assert_eq!(appear_scale(5.0), 1.0);
        assert!((appear_scale(-1.0) - 0.82).abs() < 1e-6);
    }

    #[test]
    fn press_goes_down_fast_and_returns_to_rest() {
        assert_eq!(press_amount(-0.01), 0.0);
        assert_eq!(press_amount(0.0), 0.0);
        assert_eq!(press_amount(PRESS_DOWN_SECONDS), 1.0);
        assert!(press_amount(PRESS_DOWN_SECONDS + PRESS_UP_SECONDS / 2.0) < 0.5);
        assert_eq!(press_amount(PRESS_DOWN_SECONDS + PRESS_UP_SECONDS), 0.0);
        assert_eq!(press_amount(1.0), 0.0);
    }

    #[test]
    fn dark_faces_are_lifted_so_the_edge_reads_darker() {
        let (face, edge) = keycap_colors([0.0, 0.0, 0.0], 0.9);
        assert!(face[0] > 0.1);
        assert!(edge[0] < face[0]);
        assert_eq!(face[3], 0.9);
        let (face, edge) = keycap_colors([1.0, 1.0, 1.0], 1.0);
        assert_eq!(face, [1.0, 1.0, 1.0, 1.0]);
        assert!(edge[0] < 0.6);
    }

    #[test]
    fn scaling_keeps_the_origin_fixed() {
        assert_eq!(
            scale_rect([100.0, 50.0, 40.0, 20.0], [100.0, 50.0], 0.5),
            [100.0, 50.0, 20.0, 10.0]
        );
        assert_eq!(
            scale_rect([0.0, 0.0, 40.0, 20.0], [20.0, 10.0], 0.5),
            [10.0, 5.0, 20.0, 10.0]
        );
    }

    #[test]
    fn uniforms_match_the_shader_layout() {
        assert_eq!(std::mem::size_of::<KeycapUniforms>(), 48 + 32 * MAX_KEYCAPS);
    }
}
