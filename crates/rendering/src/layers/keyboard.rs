use bytemuck::{Pod, Zeroable};
use cap_project::{KeyboardStyle, XY, keycap_labels};
use glyphon::{
    Attrs, Buffer, Cache, Color, Family, FontSystem, Metrics, Resolution, Shaping, SwashCache,
    TextArea, TextAtlas, TextBounds, TextRenderer, Viewport, Weight,
};
use log::warn;
use wgpu::{Device, Queue, include_wgsl, util::DeviceExt};

use super::captions::{CaptionOverlayLayout, CaptionPosition};
use crate::{DecodedSegmentFrames, ProjectUniforms, RenderVideoConstants, parse_color_component};

mod keycaps;

use keycaps::{
    KeycapInstance, KeycapMetrics, KeycapUniforms, MAX_KEYCAPS, appear_scale, keycap_colors,
    layout_keycaps, press_amount, scale_rect,
};

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug)]
struct KeyboardBackgroundUniforms {
    rect: [f32; 4],
    color: [f32; 4],
    radius: f32,
    _padding: [f32; 3],
    _padding2: [f32; 4],
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum KeyboardPosition {
    AboveCaptions,
    TopLeft,
    TopCenter,
    TopRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
}

impl KeyboardPosition {
    fn from_str(s: &str) -> Self {
        match s {
            "top-left" => Self::TopLeft,
            "top-center" | "top" => Self::TopCenter,
            "top-right" => Self::TopRight,
            "bottom-left" => Self::BottomLeft,
            "bottom-right" => Self::BottomRight,
            "bottom-center" | "bottom" => Self::BottomCenter,
            _ => Self::AboveCaptions,
        }
    }

    fn y_factor(&self) -> f32 {
        match self {
            Self::TopLeft | Self::TopCenter | Self::TopRight => 0.08,
            Self::AboveCaptions => 0.75,
            Self::BottomLeft | Self::BottomCenter | Self::BottomRight => 0.85,
        }
    }
}

const BOUNCE_OFFSET_PIXELS: f32 = 6.0;
const MIN_OVERLAY_GAP: f32 = 15.0;

fn shares_caption_lane(
    caption_position: CaptionPosition,
    keyboard_position: KeyboardPosition,
) -> bool {
    matches!(
        (caption_position, keyboard_position),
        (CaptionPosition::TopLeft, KeyboardPosition::TopLeft)
            | (CaptionPosition::TopCenter, KeyboardPosition::TopCenter)
            | (CaptionPosition::TopRight, KeyboardPosition::TopRight)
            | (CaptionPosition::BottomLeft, KeyboardPosition::BottomLeft)
            | (
                CaptionPosition::BottomCenter,
                KeyboardPosition::AboveCaptions
            )
            | (
                CaptionPosition::BottomCenter,
                KeyboardPosition::BottomCenter
            )
            | (CaptionPosition::BottomRight, KeyboardPosition::BottomRight)
    )
}

fn prefers_above_caption(keyboard_position: KeyboardPosition) -> bool {
    matches!(
        keyboard_position,
        KeyboardPosition::AboveCaptions
            | KeyboardPosition::BottomLeft
            | KeyboardPosition::BottomCenter
            | KeyboardPosition::BottomRight
    )
}

fn resolve_keyboard_position(
    keyboard_position: KeyboardPosition,
    caption_layout: Option<CaptionOverlayLayout>,
) -> KeyboardPosition {
    if keyboard_position == KeyboardPosition::AboveCaptions && caption_layout.is_none() {
        KeyboardPosition::BottomCenter
    } else {
        keyboard_position
    }
}

fn resolve_background_top(
    current_top: f32,
    box_height: f32,
    frame_height: f32,
    keyboard_position: KeyboardPosition,
    caption_layout: Option<CaptionOverlayLayout>,
) -> f32 {
    let Some(caption_layout) = caption_layout else {
        return current_top;
    };

    if !shares_caption_lane(caption_layout.position, keyboard_position) {
        return current_top;
    }

    let max_top = (frame_height - box_height).max(0.0);
    let caption_top = caption_layout.rect[1];
    let caption_bottom = caption_layout.rect[1] + caption_layout.rect[3];
    let current_bottom = current_top + box_height;

    if current_bottom + MIN_OVERLAY_GAP <= caption_top
        || current_top >= caption_bottom + MIN_OVERLAY_GAP
    {
        return current_top.clamp(0.0, max_top);
    }

    let above_candidate = caption_top - MIN_OVERLAY_GAP - box_height;
    let below_candidate = caption_bottom + MIN_OVERLAY_GAP;
    let above_valid = above_candidate >= 0.0;
    let below_valid = below_candidate <= max_top;

    let adjusted_top = if prefers_above_caption(keyboard_position) {
        if above_valid {
            above_candidate
        } else if below_valid {
            below_candidate
        } else {
            current_top
        }
    } else if below_valid {
        below_candidate
    } else if above_valid {
        above_candidate
    } else {
        current_top
    };

    adjusted_top.clamp(0.0, max_top)
}

#[derive(PartialEq)]
struct ShapedKeyboardText {
    text: String,
    font_size: u32,
    wrap_width: u32,
    family: Family<'static>,
    weight: Weight,
}

pub struct KeyboardLayer {
    font_system: FontSystem,
    swash_cache: SwashCache,
    text_atlas: TextAtlas,
    text_renderer: TextRenderer,
    text_buffer: Buffer,
    shaped: Option<ShapedKeyboardText>,
    viewport: Viewport,
    background_pipeline: wgpu::RenderPipeline,
    background_bind_group: wgpu::BindGroup,
    background_uniform_buffer: wgpu::Buffer,
    background_scissor: Option<[u32; 4]>,
    keycap_pipeline: wgpu::RenderPipeline,
    keycap_bind_group: wgpu::BindGroup,
    keycap_uniform_buffer: wgpu::Buffer,
    keycap_buffers: Vec<(ShapedKeyboardText, Buffer)>,
    draws_keycaps: bool,
    output_size: (u32, u32),
    has_content: bool,
}

impl KeyboardLayer {
    pub fn new(device: &Device, queue: &Queue) -> Self {
        let font_system = super::new_font_system();
        let glyph_phase = crate::readiness::Phase::start("glyph.keyboard.gpu_resources");
        let swash_cache = SwashCache::new();
        let cache = Cache::new(device);
        let viewport = Viewport::new(device, &cache);
        let mut text_atlas = TextAtlas::new(device, queue, &cache, wgpu::TextureFormat::Rgba8Unorm);
        let text_renderer = TextRenderer::new(
            &mut text_atlas,
            device,
            wgpu::MultisampleState::default(),
            None,
        );
        glyph_phase.finish("returned");

        let metrics = Metrics::new(28.0, 28.0 * 1.2);
        let text_buffer = Buffer::new_empty(metrics);

        let background_uniforms = KeyboardBackgroundUniforms {
            rect: [0.0; 4],
            color: [0.0; 4],
            radius: 0.0,
            _padding: [0.0; 3],
            _padding2: [0.0; 4],
        };

        let background_uniform_buffer =
            device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Keyboard Background Uniform Buffer"),
                contents: bytemuck::bytes_of(&background_uniforms),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        let (background_pipeline, background_bind_group) = overlay_pipeline(
            device,
            "Keyboard Background",
            &device.create_shader_module(include_wgsl!("../shaders/caption_bg.wgsl")),
            &background_uniform_buffer,
        );

        let keycap_uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Keyboard Keycap Uniform Buffer"),
            contents: bytemuck::bytes_of(&KeycapUniforms::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        let (keycap_pipeline, keycap_bind_group) = overlay_pipeline(
            device,
            "Keyboard Keycap",
            &device.create_shader_module(include_wgsl!("../shaders/keycap.wgsl")),
            &keycap_uniform_buffer,
        );

        Self {
            font_system,
            swash_cache,
            text_atlas,
            text_renderer,
            text_buffer,
            shaped: None,
            viewport,
            background_pipeline,
            background_bind_group,
            background_uniform_buffer,
            background_scissor: None,
            keycap_pipeline,
            keycap_bind_group,
            keycap_uniform_buffer,
            keycap_buffers: Vec::new(),
            draws_keycaps: false,
            output_size: (0, 0),
            has_content: false,
        }
    }

    pub fn prepare(
        &mut self,
        uniforms: &ProjectUniforms,
        _segment_frames: &DecodedSegmentFrames,
        output_size: XY<u32>,
        constants: &RenderVideoConstants,
        caption_layout: Option<CaptionOverlayLayout>,
    ) {
        self.has_content = false;
        self.background_scissor = None;
        self.draws_keycaps = false;
        self.output_size = (output_size.x, output_size.y);
        self.text_atlas.trim();

        let Some(keyboard_data) = &uniforms.project.keyboard else {
            return;
        };

        if !keyboard_data.settings.enabled {
            return;
        }

        let timeline = match &uniforms.project.timeline {
            Some(t) => t,
            None => return,
        };

        if timeline.keyboard_segments.is_empty() {
            return;
        }

        let current_time = uniforms.frame_number as f64 / uniforms.frame_rate as f64;
        let settings = &keyboard_data.settings;

        let active_segment = find_active_keyboard_segment(
            current_time,
            &timeline.keyboard_segments,
            settings.fade_duration,
        );

        let Some(active) = active_segment else {
            return;
        };

        let keycap_style = settings.style == KeyboardStyle::Keycaps;
        let (repeat_count, appear_start) = if keycap_style || settings.shortcuts_only {
            repeat_chain(&timeline.keyboard_segments, active.index)
        } else {
            (1, active.segment.start)
        };

        let segment_fade = active
            .segment
            .fade_duration_override
            .unwrap_or(settings.fade_duration) as f64;

        let raw_text = build_visible_text(active.segment, current_time);

        if raw_text.is_empty() {
            return;
        }

        let should_uppercase = active
            .segment
            .uppercase_override
            .unwrap_or(settings.uppercase);
        let mut visible_text = if should_uppercase {
            raw_text.to_uppercase()
        } else {
            raw_text.clone()
        };
        if repeat_count > 1 {
            visible_text.push_str(&format!(" {}", repeat_label(repeat_count)));
        }

        let fade_opacity =
            calculate_fade(current_time, appear_start, active.segment.end, segment_fade)
                * uniforms.takeover_overlay_fade();

        if fade_opacity <= 0.0 {
            return;
        }

        let bounce_offset =
            calculate_keyboard_bounce(current_time, appear_start, active.segment.end, segment_fade);

        let (width, height) = (output_size.x, output_size.y);
        let device = &constants.device;
        let queue = &constants.queue;

        let position = active
            .segment
            .position_override
            .as_deref()
            .map(KeyboardPosition::from_str)
            .unwrap_or_else(|| KeyboardPosition::from_str(&settings.position));

        let margin = width as f32 * 0.05;

        let color_hex = active
            .segment
            .color_override
            .as_deref()
            .unwrap_or(&settings.color);
        let text_color = [
            parse_color_component(color_hex, 0),
            parse_color_component(color_hex, 1),
            parse_color_component(color_hex, 2),
        ];

        let bg_color_hex = active
            .segment
            .background_color_override
            .as_deref()
            .unwrap_or(&settings.background_color);
        let background_color_rgb = [
            parse_color_component(bg_color_hex, 0),
            parse_color_component(bg_color_hex, 1),
            parse_color_component(bg_color_hex, 2),
        ];

        let background_alpha =
            ((settings.background_opacity as f32 / 100.0) * fade_opacity).clamp(0.0, 1.0);

        let font_size_base = active.segment.font_size_override.unwrap_or(settings.size) as f32;
        let font_size = font_size_base * (height as f32 / 1080.0);
        let wrap_width = (width as f32 - margin * 2.0).max(font_size);

        let font_family = match settings.font.as_str() {
            "System Serif" => Family::Serif,
            "System Monospace" => Family::Monospace,
            _ => Family::SansSerif,
        };

        let weight = if settings.font_weight >= 700 {
            Weight::BOLD
        } else if settings.font_weight >= 500 {
            Weight::MEDIUM
        } else {
            Weight::NORMAL
        };

        let text_alpha = fade_opacity.clamp(0.0, 1.0);
        let color = Color::rgba(
            (text_color[0] * 255.0) as u8,
            (text_color[1] * 255.0) as u8,
            (text_color[2] * 255.0) as u8,
            (text_alpha * 255.0) as u8,
        );

        if keycap_style {
            let press_seconds = latest_press_time(active.segment, current_time)
                .map_or(f64::INFINITY, |press_time| current_time - press_time);
            let appear_seconds = if active.segment.keys.is_empty() {
                f64::INFINITY
            } else {
                current_time - appear_start
            };
            self.prepare_keycaps(
                constants,
                KeycapFrame {
                    raw_text: &raw_text,
                    uppercase: should_uppercase,
                    repeat_count,
                    appear_seconds,
                    press_seconds,
                    opacity: fade_opacity.clamp(0.0, 1.0),
                    text_color,
                    face_color: background_color_rgb,
                    face_alpha: (settings.background_opacity as f32 / 100.0).clamp(0.0, 1.0),
                    font_size,
                    family: font_family,
                    weight,
                    output_size: [width, height],
                    margin,
                    position,
                    caption_layout,
                },
            );
            return;
        }

        let shaped = ShapedKeyboardText {
            text: visible_text,
            font_size: font_size.to_bits(),
            wrap_width: wrap_width.to_bits(),
            family: font_family,
            weight,
        };
        if self.shaped.as_ref() != Some(&shaped) {
            let metrics = Metrics::new(font_size, font_size * 1.2);
            let mut buffer = Buffer::new(&mut self.font_system, metrics);
            buffer.set_size(&mut self.font_system, Some(wrap_width), None);
            buffer.set_text(
                &mut self.font_system,
                &shaped.text,
                &Attrs::new().family(font_family).weight(weight),
                Shaping::Advanced,
            );
            self.text_buffer = buffer;
            self.shaped = Some(shaped);
        }

        let mut layout_width: f32 = 0.0;
        let mut layout_height: f32 = 0.0;
        for run in glyphon::cosmic_text::LayoutRunIter::new(&self.text_buffer) {
            layout_width = layout_width.max(run.line_w);
            layout_height = layout_height.max(run.line_top + run.line_height);
        }

        if layout_height == 0.0 {
            layout_height = font_size * 1.2;
            layout_width = layout_width.max(font_size);
        }

        let available_width = (width as f32 - margin * 2.0).max(1.0);
        let padding = font_size * 0.45;
        let corner_radius = font_size * 0.5;
        let text_width = layout_width.min(available_width);
        let text_height = layout_height;
        let box_width = (text_width + padding * 2.0).min(available_width).max(1.0);
        let box_height = (text_height + padding * 2.0).min(height as f32).max(1.0);

        let [background_left, background_top] = place_overlay(
            [box_width, box_height],
            [width as f32, height as f32],
            margin,
            bounce_offset as f32,
            position,
            caption_layout,
        );

        let text_left = background_left + padding;
        let text_top = background_top + padding;

        let bounds = TextBounds {
            left: (text_left - 2.0).floor() as i32,
            top: (text_top - 2.0).floor() as i32,
            right: (text_left + text_width + 2.0).ceil() as i32,
            bottom: (text_top + text_height + 2.0).ceil() as i32,
        };

        self.viewport.update(queue, Resolution { width, height });

        let text_areas = vec![TextArea {
            buffer: &self.text_buffer,
            left: text_left,
            top: text_top,
            scale: 1.0,
            bounds,
            default_color: color,
            custom_glyphs: &[],
        }];

        match self.text_renderer.prepare(
            device,
            queue,
            &mut self.font_system,
            &mut self.text_atlas,
            &self.viewport,
            text_areas,
            &mut self.swash_cache,
        ) {
            Ok(_) => {}
            Err(e) => warn!("Error preparing keyboard text: {e:?}"),
        }

        let rect = KeyboardBackgroundUniforms {
            rect: [
                background_left.max(0.0),
                background_top.max(0.0),
                box_width,
                box_height,
            ],
            color: [
                background_color_rgb[0],
                background_color_rgb[1],
                background_color_rgb[2],
                background_alpha,
            ],
            radius: corner_radius.min(box_width / 2.0).min(box_height / 2.0),
            _padding: [0.0; 3],
            _padding2: [0.0; 4],
        };

        queue.write_buffer(
            &self.background_uniform_buffer,
            0,
            bytemuck::bytes_of(&rect),
        );

        let scissor_padding = 4.0;
        let scissor_x = (background_left - scissor_padding).max(0.0).floor() as u32;
        let scissor_y = (background_top - scissor_padding).max(0.0).floor() as u32;
        let max_width = width.saturating_sub(scissor_x);
        let max_height = height.saturating_sub(scissor_y);

        if max_width == 0 || max_height == 0 {
            self.has_content = false;
            return;
        }

        let scissor_width = (box_width + scissor_padding * 2.0)
            .ceil()
            .max(1.0)
            .min(max_width as f32) as u32;
        let scissor_height = (box_height + scissor_padding * 2.0)
            .ceil()
            .max(1.0)
            .min(max_height as f32) as u32;

        if scissor_width == 0 || scissor_height == 0 {
            self.has_content = false;
            return;
        }

        self.background_scissor = Some([scissor_x, scissor_y, scissor_width, scissor_height]);
        self.has_content = true;
    }

    pub fn has_content(&self) -> bool {
        self.has_content
    }

    pub fn render(&self, pass: &mut wgpu::RenderPass<'_>) {
        if !self.has_content {
            return;
        }

        if let Some([x, y, width, height]) = self.background_scissor {
            pass.set_scissor_rect(x, y, width, height);
            if self.draws_keycaps {
                pass.set_pipeline(&self.keycap_pipeline);
                pass.set_bind_group(0, &self.keycap_bind_group, &[]);
            } else {
                pass.set_pipeline(&self.background_pipeline);
                pass.set_bind_group(0, &self.background_bind_group, &[]);
            }
            pass.draw(0..6, 0..1);
            pass.set_scissor_rect(x, y, width, height);
        } else if self.output_size.0 > 0 && self.output_size.1 > 0 {
            pass.set_scissor_rect(0, 0, self.output_size.0, self.output_size.1);
        }

        match self
            .text_renderer
            .render(&self.text_atlas, &self.viewport, pass)
        {
            Ok(_) => {}
            Err(e) => warn!("Error rendering keyboard text: {e:?}"),
        }

        if self.output_size.0 > 0 && self.output_size.1 > 0 {
            pass.set_scissor_rect(0, 0, self.output_size.0, self.output_size.1);
        }
    }
}

struct KeycapFrame<'a> {
    raw_text: &'a str,
    uppercase: bool,
    repeat_count: usize,
    appear_seconds: f64,
    press_seconds: f64,
    opacity: f32,
    text_color: [f32; 3],
    face_color: [f32; 3],
    face_alpha: f32,
    font_size: f32,
    family: Family<'static>,
    weight: Weight,
    output_size: [u32; 2],
    margin: f32,
    position: KeyboardPosition,
    caption_layout: Option<CaptionOverlayLayout>,
}

impl KeyboardLayer {
    fn prepare_keycaps(&mut self, constants: &RenderVideoConstants, frame: KeycapFrame<'_>) {
        let [width, height] = frame.output_size;
        let metrics = KeycapMetrics::new(frame.font_size);
        let labels = keycap_labels(frame.raw_text);
        let mut texts: Vec<(String, f32)> = labels
            .iter()
            .map(|label| {
                let text = if frame.uppercase && !label.modifier {
                    label.text.to_uppercase()
                } else {
                    label.text.to_string()
                };
                let size = if labels.len() > 1 && label.text.chars().count() > 1 {
                    metrics.label_size * 0.78
                } else {
                    metrics.label_size
                };
                (text, size)
            })
            .collect();
        if frame.repeat_count > 1 {
            texts.push((repeat_label(frame.repeat_count), metrics.label_size * 0.78));
        }
        texts.truncate(MAX_KEYCAPS);
        if texts.is_empty() {
            return;
        }

        self.keycap_buffers.truncate(texts.len());
        for (index, (text, size)) in texts.into_iter().enumerate() {
            let shaped = ShapedKeyboardText {
                text,
                font_size: size.to_bits(),
                wrap_width: 0,
                family: frame.family,
                weight: frame.weight,
            };
            if self
                .keycap_buffers
                .get(index)
                .is_some_and(|(cached, _)| *cached == shaped)
            {
                continue;
            }
            let mut buffer = Buffer::new(&mut self.font_system, Metrics::new(size, size * 1.2));
            buffer.set_size(&mut self.font_system, None, None);
            buffer.set_text(
                &mut self.font_system,
                &shaped.text,
                &Attrs::new().family(frame.family).weight(frame.weight),
                Shaping::Advanced,
            );
            if index < self.keycap_buffers.len() {
                self.keycap_buffers[index] = (shaped, buffer);
            } else {
                self.keycap_buffers.push((shaped, buffer));
            }
        }

        let text_layouts: Vec<(f32, f32)> = self
            .keycap_buffers
            .iter()
            .map(|(_, buffer)| {
                glyphon::cosmic_text::LayoutRunIter::new(buffer)
                    .next()
                    .map_or((0.0, 0.0), |run| (run.line_w, run.line_y))
            })
            .collect();
        let widths: Vec<f32> = text_layouts.iter().map(|(width, _)| *width).collect();
        let (rects, group_width) = layout_keycaps(&widths, metrics);

        let available_width = (width as f32 - frame.margin * 2.0).max(1.0);
        let fit = (available_width / group_width.max(1.0)).min(1.0);
        let scale = appear_scale(frame.appear_seconds) * fit;
        let box_size = [group_width * fit, metrics.height() * fit];
        let [left, top] = place_overlay(
            box_size,
            [width as f32, height as f32],
            frame.margin,
            0.0,
            frame.position,
            frame.caption_layout,
        );
        let origin = [left + box_size[0] / 2.0, top + box_size[1] / 2.0];
        let group_origin = [
            origin[0] - group_width / 2.0,
            origin[1] - metrics.height() / 2.0,
        ];

        let press = press_amount(frame.press_seconds) * metrics.depth * 0.8 * scale;
        let (face_color, edge_color) = keycap_colors(frame.face_color, frame.face_alpha);
        let mut keycap_uniforms = KeycapUniforms {
            face_color,
            edge_color,
            globals: [
                frame.opacity,
                0.35 * frame.face_alpha,
                metrics.shadow_blur * scale,
                rects.len() as f32,
            ],
            ..Default::default()
        };
        let text_color = Color::rgba(
            (frame.text_color[0] * 255.0) as u8,
            (frame.text_color[1] * 255.0) as u8,
            (frame.text_color[2] * 255.0) as u8,
            (frame.opacity * 255.0) as u8,
        );

        let mut bounds = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
        let mut text_areas = Vec::with_capacity(rects.len());
        for (index, rect) in rects.iter().enumerate() {
            let placed = scale_rect(
                [
                    group_origin[0] + rect[0],
                    group_origin[1] + rect[1],
                    rect[2],
                    rect[3],
                ],
                origin,
                scale,
            );
            keycap_uniforms.keycaps[index] = KeycapInstance {
                rect: placed,
                params: [press, metrics.depth * scale, metrics.radius * scale, 0.0],
            };
            let (text_width, baseline) = text_layouts[index];
            let label_size = f32::from_bits(self.keycap_buffers[index].0.font_size);
            let text_left = placed[0] + (placed[2] - text_width * scale) / 2.0;
            let text_top =
                placed[1] + press + placed[3] / 2.0 + label_size * 0.36 * scale - baseline * scale;
            text_areas.push(TextArea {
                buffer: &self.keycap_buffers[index].1,
                left: text_left,
                top: text_top,
                scale,
                bounds: TextBounds {
                    left: placed[0].floor() as i32,
                    top: (placed[1] + press).floor() as i32,
                    right: (placed[0] + placed[2]).ceil() as i32,
                    bottom: (placed[1] + press + placed[3]).ceil() as i32,
                },
                default_color: text_color,
                custom_glyphs: &[],
            });
            let shadow_reach = metrics.depth * scale + metrics.shadow_blur * scale * 1.5;
            bounds[0] = bounds[0].min(placed[0] - shadow_reach);
            bounds[1] = bounds[1].min(placed[1] - shadow_reach);
            bounds[2] = bounds[2].max(placed[0] + placed[2] + shadow_reach);
            bounds[3] = bounds[3].max(placed[1] + placed[3] + shadow_reach * 2.0);
        }

        let queue = &constants.queue;
        self.viewport.update(queue, Resolution { width, height });
        if let Err(error) = self.text_renderer.prepare(
            &constants.device,
            queue,
            &mut self.font_system,
            &mut self.text_atlas,
            &self.viewport,
            text_areas,
            &mut self.swash_cache,
        ) {
            warn!("Error preparing keyboard keycap text: {error:?}");
        }
        queue.write_buffer(
            &self.keycap_uniform_buffer,
            0,
            bytemuck::bytes_of(&keycap_uniforms),
        );

        let left = bounds[0].max(0.0).floor() as u32;
        let top = bounds[1].max(0.0).floor() as u32;
        let right = (bounds[2].ceil().max(0.0) as u32).min(width);
        let bottom = (bounds[3].ceil().max(0.0) as u32).min(height);
        if right <= left || bottom <= top {
            return;
        }
        self.background_scissor = Some([left, top, right - left, bottom - top]);
        self.draws_keycaps = true;
        self.has_content = true;
    }
}

fn place_overlay(
    box_size: [f32; 2],
    output_size: [f32; 2],
    margin: f32,
    vertical_offset: f32,
    position: KeyboardPosition,
    caption_layout: Option<CaptionOverlayLayout>,
) -> [f32; 2] {
    let [box_width, box_height] = box_size;
    let [width, height] = output_size;
    let resolved_position = resolve_keyboard_position(position, caption_layout);

    let left = match resolved_position {
        KeyboardPosition::TopLeft | KeyboardPosition::BottomLeft => margin,
        KeyboardPosition::TopRight | KeyboardPosition::BottomRight => {
            (width - margin - box_width).max(0.0)
        }
        _ => ((width - box_width) / 2.0).max(0.0),
    };

    let center_y = height * resolved_position.y_factor();
    let max_top = (height - box_height).max(0.0);
    let base_top = (center_y - box_height / 2.0).clamp(0.0, max_top);
    let top = resolve_background_top(
        (base_top + vertical_offset).clamp(0.0, max_top),
        box_height,
        height,
        position,
        caption_layout,
    );
    [left, top]
}

fn repeat_label(count: usize) -> String {
    format!("×{count}")
}

fn repeat_chain(segments: &[cap_project::KeyboardTrackSegment], index: usize) -> (usize, f64) {
    let mut current = &segments[index];
    let mut count = 1;
    if current.keys.len() != 1 {
        return (count, current.start);
    }
    for previous in segments[..index].iter().rev() {
        if previous.display_text != current.display_text
            || previous.keys.len() != 1
            || previous.start > current.start
            || previous.end < current.start
        {
            break;
        }
        count += 1;
        current = previous;
    }
    (count, current.start)
}

fn latest_press_time(
    segment: &cap_project::KeyboardTrackSegment,
    current_time: f64,
) -> Option<f64> {
    let elapsed_ms = (current_time - segment.start) * 1000.0;
    segment
        .keys
        .iter()
        .map(|key| key.time_offset)
        .filter(|offset| *offset <= elapsed_ms)
        .reduce(f64::max)
        .map(|offset| segment.start + offset / 1000.0)
}

fn overlay_pipeline(
    device: &Device,
    label: &str,
    shader: &wgpu::ShaderModule,
    uniform_buffer: &wgpu::Buffer,
) -> (wgpu::RenderPipeline, wgpu::BindGroup) {
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(&format!("{label} Bind Group Layout")),
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        }],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&format!("{label} Bind Group")),
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: uniform_buffer.as_entire_binding(),
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(&format!("{label} Pipeline Layout")),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(&format!("{label} Pipeline")),
        layout: Some(&pipeline_layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: Some("vs_main"),
            buffers: &[],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: Some("fs_main"),
            targets: &[Some(wgpu::ColorTargetState {
                format: wgpu::TextureFormat::Rgba8Unorm,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    });

    (pipeline, bind_group)
}

struct ActiveKeyboardSegment<'a> {
    segment: &'a cap_project::KeyboardTrackSegment,
    index: usize,
}

fn find_active_keyboard_segment<'a>(
    time: f64,
    segments: &'a [cap_project::KeyboardTrackSegment],
    default_fade_duration: f32,
) -> Option<ActiveKeyboardSegment<'a>> {
    segments
        .iter()
        .enumerate()
        .filter(|(_, segment)| time >= segment.start && time < segment.end)
        .max_by(|(_, a), (_, b)| a.start.total_cmp(&b.start))
        .or_else(|| {
            segments
                .iter()
                .enumerate()
                .filter(|(_, segment)| {
                    let fade = segment
                        .fade_duration_override
                        .unwrap_or(default_fade_duration) as f64;
                    time >= segment.end && time < segment.end + fade
                })
                .max_by(|(_, a), (_, b)| a.end.total_cmp(&b.end))
        })
        .map(|(index, segment)| ActiveKeyboardSegment { segment, index })
}

fn build_visible_text(segment: &cap_project::KeyboardTrackSegment, current_time: f64) -> String {
    if segment.keys.is_empty() {
        return segment.display_text.clone();
    }

    let time_offset_from_start = (current_time - segment.start) * 1000.0;
    let chars: Vec<char> = segment.display_text.chars().collect();

    if segment.keys.len() != chars.len() {
        if segment
            .keys
            .iter()
            .any(|key| time_offset_from_start >= key.time_offset)
        {
            return segment.display_text.clone();
        }

        return String::new();
    }

    let mut visible = String::new();

    for (i, key) in segment.keys.iter().enumerate() {
        if time_offset_from_start >= key.time_offset && i < chars.len() {
            visible.push(chars[i]);
        }
    }

    if visible.is_empty() && !segment.display_text.is_empty() {
        visible.push(chars[0]);
    }

    visible
}

fn calculate_fade(current_time: f64, start: f64, end: f64, fade_duration: f64) -> f32 {
    if fade_duration <= 0.0 {
        if current_time >= start && current_time < end {
            return 1.0;
        }
        return 0.0;
    }

    let time_from_start = current_time - start;
    let time_to_end = end - current_time;

    let fade_in = (time_from_start / fade_duration).clamp(0.0, 1.0) as f32;

    let fade_out = if time_to_end >= 0.0 {
        1.0
    } else {
        let past_end = -time_to_end;
        (1.0 - past_end / fade_duration).clamp(0.0, 1.0) as f32
    };

    fade_in.min(fade_out)
}

fn calculate_keyboard_bounce(current_time: f64, start: f64, end: f64, fade_duration: f64) -> f64 {
    if fade_duration <= 0.0 {
        return 0.0;
    }

    let time_from_start = current_time - start;
    let time_to_end = end - current_time;

    let fade_in_progress = (time_from_start / fade_duration).clamp(0.0, 1.0);
    let fade_out_progress = (time_to_end / fade_duration).clamp(0.0, 1.0);

    if fade_in_progress < 1.0 {
        let ease = 1.0 - fade_in_progress;
        -(ease * ease) * BOUNCE_OFFSET_PIXELS as f64
    } else if fade_out_progress < 1.0 {
        let ease = 1.0 - fade_out_progress;
        (ease * ease) * BOUNCE_OFFSET_PIXELS as f64
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::{
        KeyboardPosition, build_visible_text, find_active_keyboard_segment, latest_press_time,
        repeat_chain, resolve_background_top, resolve_keyboard_position,
    };
    use crate::layers::{CaptionOverlayLayout, CaptionPosition};

    #[test]
    fn new_typing_is_not_hidden_by_the_previous_words_linger() {
        let segment = |id: &str, start: f64, end: f64| -> cap_project::KeyboardTrackSegment {
            serde_json::from_value(
                serde_json::json!({"id":id,"start":start,"end":end,"displayText":id}),
            )
            .unwrap()
        };
        let segments = [segment("previous", 7.0, 7.8), segment("current", 7.2, 8.0)];
        let active = find_active_keyboard_segment(7.5, &segments, 0.15).unwrap();
        assert_eq!(build_visible_text(active.segment, 7.5), "current");
        assert_eq!(
            find_active_keyboard_segment(8.1, &segments, 0.15)
                .unwrap()
                .segment
                .id,
            "current"
        );
    }

    #[test]
    fn shortcut_segments_show_the_full_combo() {
        let segment = cap_project::KeyboardTrackSegment {
            id: "kb-1".to_string(),
            start: 1.0,
            end: 1.8,
            display_text: "⌘T".to_string(),
            keys: vec![cap_project::KeyPressDisplay {
                key: "t".to_string(),
                time_offset: 0.0,
            }],
            fade_duration_override: None,
            position_override: None,
            color_override: None,
            background_color_override: None,
            font_size_override: None,
            uppercase_override: None,
        };

        assert_eq!(build_visible_text(&segment, 1.0), "⌘T");
    }

    fn single_key(id: &str, start: f64, end: f64, text: &str) -> cap_project::KeyboardTrackSegment {
        serde_json::from_value(serde_json::json!({
            "id": id, "start": start, "end": end, "displayText": text,
            "keys": [{"key": "z", "timeOffset": 0.0}]
        }))
        .unwrap()
    }

    #[test]
    fn repeats_chain_only_while_the_previous_press_is_on_screen() {
        let mut segments = vec![
            single_key("a", 1.0, 1.8, "⌘Z"),
            single_key("b", 1.3, 2.1, "⌘Z"),
            single_key("c", 1.6, 2.4, "⌘Z"),
            single_key("d", 3.0, 3.8, "⌘Z"),
            single_key("e", 3.2, 4.0, "⌘C"),
        ];
        assert_eq!(repeat_chain(&segments, 0), (1, 1.0));
        assert_eq!(repeat_chain(&segments, 2), (3, 1.0));
        assert_eq!(repeat_chain(&segments, 3), (1, 3.0));
        assert_eq!(repeat_chain(&segments, 4), (1, 3.2));

        let (head, tail) = segments[2].split_at(2.0).unwrap();
        assert!(tail.keys.is_empty());
        segments.splice(2..3, [head, tail]);
        assert_eq!(repeat_chain(&segments, 2), (3, 1.0));
        assert_eq!(repeat_chain(&segments, 3), (1, 2.0));
    }

    #[test]
    fn press_time_follows_the_latest_revealed_key() {
        let mut segment = single_key("a", 2.0, 3.0, "ab");
        segment.keys.push(cap_project::KeyPressDisplay {
            key: "b".to_string(),
            time_offset: 300.0,
        });
        assert_eq!(latest_press_time(&segment, 2.1), Some(2.0));
        assert_eq!(latest_press_time(&segment, 2.5), Some(2.3));
        assert_eq!(latest_press_time(&segment, 1.9), None);
        segment.keys.clear();
        assert_eq!(latest_press_time(&segment, 2.5), None);
    }

    #[test]
    fn above_captions_falls_back_to_bottom_center_without_caption_layout() {
        assert_eq!(
            resolve_keyboard_position(KeyboardPosition::AboveCaptions, None),
            KeyboardPosition::BottomCenter
        );
    }

    #[test]
    fn bottom_keyboard_moves_above_captions_with_gap() {
        let caption_layout = CaptionOverlayLayout {
            rect: [100.0, 760.0, 400.0, 140.0],
            position: CaptionPosition::BottomCenter,
        };

        let resolved_top = resolve_background_top(
            720.0,
            120.0,
            1080.0,
            KeyboardPosition::AboveCaptions,
            Some(caption_layout),
        );

        assert_eq!(resolved_top, 625.0);
    }

    #[test]
    fn top_keyboard_moves_below_captions_with_gap() {
        let caption_layout = CaptionOverlayLayout {
            rect: [100.0, 60.0, 400.0, 100.0],
            position: CaptionPosition::TopCenter,
        };

        let resolved_top = resolve_background_top(
            90.0,
            80.0,
            1080.0,
            KeyboardPosition::TopCenter,
            Some(caption_layout),
        );

        assert_eq!(resolved_top, 175.0);
    }

    #[test]
    fn different_lanes_do_not_shift_keyboard() {
        let caption_layout = CaptionOverlayLayout {
            rect: [100.0, 760.0, 400.0, 140.0],
            position: CaptionPosition::BottomLeft,
        };

        let resolved_top = resolve_background_top(
            720.0,
            120.0,
            1080.0,
            KeyboardPosition::BottomRight,
            Some(caption_layout),
        );

        assert_eq!(resolved_top, 720.0);
    }
}

#[cfg(test)]
mod gpu_tests {
    use cap_project::{
        CursorEvents, KeyPressDisplay, KeyboardData, KeyboardSettings, KeyboardTrackSegment,
        ProjectConfiguration, RecordingMeta, RecordingMetaInner, StudioRecordingMeta,
        TimelineConfiguration, XY,
    };

    use super::KeyboardLayer;
    use crate::{
        DecodedSegmentFrames, ProjectUniforms, RenderOptions, RenderVideoConstants,
        ZoomTransformTimeline,
    };

    const FPS: u32 = 60;

    fn constants(size: XY<u32>) -> RenderVideoConstants {
        let meta: StudioRecordingMeta = serde_json::from_value(serde_json::json!({
            "display": { "path": "synthetic.mp4", "fps": FPS }
        }))
        .unwrap();
        let recording = RecordingMeta {
            platform: None,
            project_path: Default::default(),
            pretty_name: "Keyboard overlay test".into(),
            sharing: None,
            inner: RecordingMetaInner::Studio(Box::new(meta.clone())),
            upload: None,
        };
        pollster::block_on(RenderVideoConstants::new_with_options(
            RenderOptions {
                screen_size: size,
                camera_size: None,
                preserve_screen_alpha: false,
            },
            recording,
            meta,
        ))
        .unwrap()
    }

    fn typed_segment(id: &str, start: f64, text: &str, key_gap_ms: f64) -> KeyboardTrackSegment {
        let keys = text
            .chars()
            .enumerate()
            .map(|(index, key)| KeyPressDisplay {
                key: key.to_string(),
                time_offset: index as f64 * key_gap_ms,
            })
            .collect::<Vec<_>>();
        KeyboardTrackSegment {
            id: id.to_string(),
            start,
            end: start + keys.len() as f64 * key_gap_ms / 1000.0 + 0.8,
            display_text: text.to_string(),
            keys,
            fade_duration_override: None,
            position_override: None,
            color_override: None,
            background_color_override: None,
            font_size_override: None,
            uppercase_override: None,
        }
    }

    fn uniforms(
        constants: &RenderVideoConstants,
        segments: &[KeyboardTrackSegment],
        size: XY<u32>,
    ) -> ProjectUniforms {
        let project = ProjectConfiguration::default();
        let cursor = CursorEvents::default();
        let frames = frames(size);
        let mut zoom = ZoomTransformTimeline::from_project(&project, &cursor, 60.0, size);
        zoom.ensure_precomputed_until(60.0);
        let mut uniforms = ProjectUniforms::new(
            constants,
            &std::sync::Arc::new(project),
            0,
            FPS,
            size,
            &cursor,
            &frames,
            60.0,
            &zoom,
        );
        uniforms.output_size = (size.x, size.y);
        std::sync::Arc::make_mut(&mut uniforms.project).keyboard = Some(KeyboardData {
            settings: KeyboardSettings {
                enabled: true,
                ..Default::default()
            },
        });
        std::sync::Arc::make_mut(&mut uniforms.project).timeline = Some(TimelineConfiguration {
            segments: vec![],
            transitions: vec![],
            zoom_segments: vec![],
            scene_segments: vec![],
            mask_segments: vec![],
            text_segments: vec![],
            caption_segments: vec![],
            keyboard_segments: segments.to_vec(),
            audio_segments: vec![],
            style_segments: vec![],
            image_segments: vec![],
            camera3d_segments: vec![],
        });
        uniforms
    }

    fn frames(size: XY<u32>) -> DecodedSegmentFrames {
        DecodedSegmentFrames {
            screen_size: size,
            screen_frame: None,
            camera_frame: None,
            segment_time: 0.0,
            recording_time: 0.0,
            segment_has_camera: false,
        }
    }

    fn render_pixels(
        constants: &RenderVideoConstants,
        layer: &KeyboardLayer,
        size: XY<u32>,
    ) -> Vec<u8> {
        let device = &constants.device;
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Keyboard Test Target"),
            size: wgpu::Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = target.create_view(&Default::default());
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Keyboard Test Readback"),
            size: u64::from(size.x * size.y * 4),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Keyboard Test Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            layer.render(&mut pass);
        }
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &readback,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(size.x * 4),
                    rows_per_image: Some(size.y),
                },
            },
            wgpu::Extent3d {
                width: size.x,
                height: size.y,
                depth_or_array_layers: 1,
            },
        );
        constants.queue.submit([encoder.finish()]);
        readback.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::PollType::Wait).unwrap();
        let pixels = readback.slice(..).get_mapped_range().to_vec();
        readback.unmap();
        pixels
    }

    #[test]
    #[ignore = "requires a working wgpu adapter"]
    fn every_frame_of_a_typed_segment_renders_like_a_fresh_layer() {
        let size = XY::new(512, 288);
        let constants = constants(size);
        let segments = [typed_segment("typed", 0.2, "git push", 90.0)];
        let mut uniforms = uniforms(&constants, &segments, size);
        let frames = frames(size);
        let mut reused = KeyboardLayer::new(&constants.device, &constants.queue);
        let mut drawn_frames = 0;
        for frame in 0..((segments[0].end + 0.2) * FPS as f64) as u32 {
            uniforms.frame_number = frame;
            reused.prepare(&uniforms, &frames, size, &constants, None);
            let mut fresh = KeyboardLayer::new(&constants.device, &constants.queue);
            fresh.prepare(&uniforms, &frames, size, &constants, None);
            assert_eq!(reused.has_content(), fresh.has_content(), "frame {frame}");
            let pixels = render_pixels(&constants, &reused, size);
            assert_eq!(
                pixels,
                render_pixels(&constants, &fresh, size),
                "frame {frame}"
            );
            drawn_frames += usize::from(pixels.chunks_exact(4).any(|pixel| pixel[3] != 0));
        }
        assert!(drawn_frames > 40);
    }

    #[test]
    #[ignore = "requires a working wgpu adapter"]
    fn reused_layer_follows_font_and_output_size_changes() {
        type Edit = (&'static str, XY<u32>, fn(&mut KeyboardSettings));
        let base_size = XY::new(512, 288);
        let constants = constants(base_size);
        let segments = [typed_segment("typed", 0.2, "git push origin main", 60.0)];
        let frames = frames(base_size);
        let prepare =
            |layer: &mut KeyboardLayer, size: XY<u32>, edit: fn(&mut KeyboardSettings)| {
                let mut uniforms = uniforms(&constants, &segments, size);
                uniforms.frame_number = 96;
                edit(
                    &mut std::sync::Arc::make_mut(&mut uniforms.project)
                        .keyboard
                        .as_mut()
                        .unwrap()
                        .settings,
                );
                layer.prepare(&uniforms, &frames, size, &constants, None);
                assert!(layer.has_content());
                render_pixels(&constants, layer, size)
            };
        let unchanged: fn(&mut KeyboardSettings) = |_| {};
        let edits: [Edit; 5] = [
            ("font size", base_size, |settings| settings.size = 70),
            ("font family", base_size, |settings| {
                settings.font = "System Monospace".to_string()
            }),
            ("font weight", base_size, |settings| {
                settings.font_weight = 700
            }),
            ("output height", XY::new(512, 216), unchanged),
            ("output width", XY::new(128, 288), unchanged),
        ];
        let base_pixels = prepare(
            &mut KeyboardLayer::new(&constants.device, &constants.queue),
            base_size,
            unchanged,
        );

        for (name, size, edit) in edits {
            let expected = prepare(
                &mut KeyboardLayer::new(&constants.device, &constants.queue),
                size,
                edit,
            );
            assert!(
                expected != base_pixels,
                "{name} edit should change the output"
            );

            let mut reused = KeyboardLayer::new(&constants.device, &constants.queue);
            prepare(&mut reused, base_size, unchanged);
            assert!(
                prepare(&mut reused, size, edit) == expected,
                "{name} edit reused a stale layout"
            );
        }
    }

    fn shortcut_segment(id: &str, start: f64, text: &str) -> KeyboardTrackSegment {
        KeyboardTrackSegment {
            id: id.to_string(),
            start,
            end: start + 0.8,
            display_text: text.to_string(),
            keys: vec![KeyPressDisplay {
                key: "t".to_string(),
                time_offset: 0.0,
            }],
            fade_duration_override: None,
            position_override: None,
            color_override: None,
            background_color_override: None,
            font_size_override: None,
            uppercase_override: None,
        }
    }

    fn set_style(uniforms: &mut ProjectUniforms, edit: impl FnOnce(&mut KeyboardSettings)) {
        edit(
            &mut std::sync::Arc::make_mut(&mut uniforms.project)
                .keyboard
                .as_mut()
                .unwrap()
                .settings,
        );
    }

    #[test]
    #[ignore = "requires a working wgpu adapter"]
    fn keycap_frames_render_like_a_fresh_layer_and_animate() {
        let size = XY::new(512, 288);
        let constants = constants(size);
        let segments = [
            shortcut_segment("a", 0.2, "⌘⇧T"),
            shortcut_segment("b", 0.6, "⌘⇧T"),
            typed_segment("typed", 1.8, "git push", 60.0),
        ];
        let mut uniforms = uniforms(&constants, &segments, size);
        set_style(&mut uniforms, |settings| {
            settings.style = cap_project::KeyboardStyle::Keycaps
        });
        let frames = frames(size);
        let mut reused = KeyboardLayer::new(&constants.device, &constants.queue);
        let mut drawn = Vec::new();
        for frame in 0..((segments[2].end + 0.2) * FPS as f64) as u32 {
            uniforms.frame_number = frame;
            reused.prepare(&uniforms, &frames, size, &constants, None);
            let mut fresh = KeyboardLayer::new(&constants.device, &constants.queue);
            fresh.prepare(&uniforms, &frames, size, &constants, None);
            assert_eq!(reused.has_content(), fresh.has_content(), "frame {frame}");
            let pixels = render_pixels(&constants, &reused, size);
            assert_eq!(
                pixels,
                render_pixels(&constants, &fresh, size),
                "frame {frame}"
            );
            drawn.push(pixels);
        }
        let visible = drawn
            .iter()
            .filter(|pixels| pixels.chunks_exact(4).any(|pixel| pixel[3] != 0))
            .count();
        assert!(visible > 100, "{visible}");
        let settled = 30;
        assert_ne!(drawn[14], drawn[settled], "appear animation");
        assert_ne!(drawn[settled], drawn[37], "repeat press and badge");
        assert_eq!(drawn[46], drawn[50], "settled repeat");
    }

    #[test]
    #[ignore = "requires a working wgpu adapter"]
    fn keycap_style_draws_one_cap_per_key_and_text_style_is_untouched_by_it() {
        let size = XY::new(640, 360);
        let constants = constants(size);
        let segments = [shortcut_segment("a", 0.0, "⌃⌥⇧⌘K")];
        let mut uniforms = uniforms(&constants, &segments, size);
        uniforms.frame_number = 30;
        let frames = frames(size);
        let mut layer = KeyboardLayer::new(&constants.device, &constants.queue);
        layer.prepare(&uniforms, &frames, size, &constants, None);
        let text = render_pixels(&constants, &layer, size);
        set_style(&mut uniforms, |settings| {
            settings.style = cap_project::KeyboardStyle::Keycaps
        });
        layer.prepare(&uniforms, &frames, size, &constants, None);
        let keycaps = render_pixels(&constants, &layer, size);
        assert_ne!(text, keycaps);

        let row = (0..size.y)
            .max_by_key(|y| {
                (0..size.x)
                    .filter(|x| keycaps[((y * size.x + x) * 4 + 3) as usize] > 200)
                    .count()
            })
            .unwrap();
        let mut runs = 0;
        let mut inside = false;
        for x in 0..size.x {
            let opaque = keycaps[((row * size.x + x) * 4 + 3) as usize] > 200;
            if opaque && !inside {
                runs += 1;
            }
            inside = opaque;
        }
        assert_eq!(runs, 5, "row {row}");

        set_style(&mut uniforms, |settings| {
            settings.style = cap_project::KeyboardStyle::Text
        });
        layer.prepare(&uniforms, &frames, size, &constants, None);
        assert_eq!(render_pixels(&constants, &layer, size), text);
    }

    #[test]
    #[ignore = "benchmark; requires a working wgpu adapter"]
    fn benchmark_keyboard_layer_prepare() {
        let size = XY::new(1920, 1080);
        let constants = constants(size);
        let mut segments = Vec::new();
        let mut start = 0.5;
        for (index, text) in ["cargo test --release", "git status", "npm run dev"]
            .iter()
            .enumerate()
        {
            let segment = typed_segment(&format!("typed-{index}"), start, text, 120.0);
            start = segment.end + 0.5;
            segments.push(segment);
            for repeat in 0..3 {
                segments.push(shortcut_segment(
                    &format!("shortcut-{index}-{repeat}"),
                    start + repeat as f64 * 0.25,
                    "⌘⇧Z",
                ));
            }
            start += 1.8;
        }
        let frames = frames(size);
        let total_frames = (start * FPS as f64) as u32;

        for style in [
            cap_project::KeyboardStyle::Text,
            cap_project::KeyboardStyle::Keycaps,
        ] {
            let mut uniforms = uniforms(&constants, &segments, size);
            set_style(&mut uniforms, |settings| settings.style = style);
            let mut layer = KeyboardLayer::new(&constants.device, &constants.queue);
            let mut samples = Vec::new();
            for pass in 0..4 {
                for frame in 0..total_frames {
                    uniforms.frame_number = frame;
                    let started = std::time::Instant::now();
                    layer.prepare(&uniforms, &frames, size, &constants, None);
                    let elapsed = started.elapsed().as_secs_f64() * 1e6;
                    if pass > 0 && layer.has_content() {
                        samples.push(elapsed);
                    }
                }
            }
            constants.device.poll(wgpu::PollType::Wait).unwrap();
            samples.sort_by(f64::total_cmp);
            println!(
                "keyboard layer prepare, {style:?} ({} visible frames): p50 {:.1} us, p95 {:.1} us, mean {:.1} us",
                samples.len(),
                samples[samples.len() / 2],
                samples[samples.len() * 95 / 100],
                samples.iter().sum::<f64>() / samples.len() as f64,
            );
        }
    }
}
