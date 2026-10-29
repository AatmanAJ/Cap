use cap_project::{
    BackgroundSource, CursorEvents, LoupeShape, ProjectConfiguration, RecordingMeta, XY,
};
use cap_rendering::{
    DecodedFrame, DecodedSegmentFrames, FrameRenderer, LoupeGeometry, ProjectUniforms,
    RenderOptions, RenderVideoConstants, RenderedFrame, RendererLayers, ZoomTransformTimeline,
};

const FPS: u32 = 30;
const DURATION: f64 = 10.0;
const SETTLED_FRAME: u32 = 90;

struct Harness {
    constants: RenderVideoConstants,
    source: XY<u32>,
}

impl Harness {
    async fn new(source: XY<u32>) -> Option<Self> {
        let instance = cap_rendering::create_wgpu_instance_sync();
        if instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .is_err()
        {
            eprintln!("no GPU adapter available, skipping loupe rendering test");
            return None;
        }
        let recording_meta: RecordingMeta = serde_json::from_value(serde_json::json!({
            "pretty_name": "loupe",
            "display": { "path": "display.mp4", "fps": FPS },
            "camera": null,
            "audio": null,
            "cursor": null
        }))
        .unwrap();
        let meta = recording_meta.studio_meta().unwrap().clone();
        let options = RenderOptions {
            screen_size: source,
            camera_size: None,
            preserve_screen_alpha: false,
        };
        let constants = RenderVideoConstants::new_with_options(options, recording_meta, meta)
            .await
            .unwrap();
        Some(Self { constants, source })
    }

    async fn render(
        &self,
        project: &ProjectConfiguration,
        pixels: &[u8],
        output: XY<u32>,
        frame_number: u32,
    ) -> Image {
        let cursor = CursorEvents::default();
        let mut zoom = ZoomTransformTimeline::from_project(project, &cursor, DURATION, self.source);
        zoom.precompute();
        let frames = DecodedSegmentFrames {
            screen_size: self.source,
            screen_frame: Some(DecodedFrame::new(
                pixels.to_vec(),
                self.source.x,
                self.source.y,
            )),
            camera_frame: None,
            segment_time: frame_number as f32 / FPS as f32,
            recording_time: frame_number as f32 / FPS as f32,
            segment_has_camera: false,
        };
        let shared = std::sync::Arc::new(project.clone());
        let uniforms = ProjectUniforms::new(
            &self.constants,
            &shared,
            frame_number,
            FPS,
            output,
            &cursor,
            &frames,
            DURATION,
            &zoom,
        );
        let mut renderer = FrameRenderer::new(&self.constants);
        let mut layers = RendererLayers::new(&self.constants.device, &self.constants.queue);
        let rendered = renderer
            .render_immediate(frames, uniforms, &cursor, true, &mut layers)
            .await
            .unwrap();
        Image::from_rendered(&rendered)
    }
}

struct Image {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
}

impl Image {
    fn from_rendered(frame: &RenderedFrame) -> Self {
        let rgba = frame
            .data
            .chunks(frame.padded_bytes_per_row as usize)
            .take(frame.height as usize)
            .flat_map(|row| row[..(frame.width * 4) as usize].to_vec())
            .collect();
        Self {
            width: frame.width,
            height: frame.height,
            rgba,
        }
    }

    fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let offset = ((y * self.width + x) * 4) as usize;
        self.rgba[offset..offset + 4].try_into().unwrap()
    }
}

fn plain_project(zoom_segments: serde_json::Value) -> ProjectConfiguration {
    let mut project = ProjectConfiguration {
        timeline: Some(
            serde_json::from_value(serde_json::json!({
                "segments": [{ "recordingClip": 0, "start": 0.0, "end": DURATION, "timescale": 1.0 }],
                "zoomSegments": zoom_segments
            }))
            .unwrap(),
        ),
        ..ProjectConfiguration::default()
    };
    project.background.source = BackgroundSource::Color {
        value: [0, 0, 0],
        alpha: 255,
    };
    project.background.padding = 0.0;
    project.background.rounding = 0.0;
    project.background.shadow = 0.0;
    project.cursor.hide = true;
    project.screen_motion_blur = 0.0;
    project
}

/// R encodes source x and G source y, so a rendered pixel says which source
/// point it shows.
fn coordinate_pattern(size: XY<u32>) -> Vec<u8> {
    let mut pixels = Vec::with_capacity((size.x * size.y * 4) as usize);
    for y in 0..size.y {
        for x in 0..size.x {
            pixels.extend_from_slice(&[
                (x * 255 / (size.x - 1)) as u8,
                (y * 255 / (size.y - 1)) as u8,
                0,
                255,
            ]);
        }
    }
    pixels
}

#[tokio::test]
async fn loupe_magnifies_its_focus_and_leaves_the_rest_of_the_frame_alone() {
    let source = XY::new(512, 288);
    let output = XY::new(1024, 576);
    let Some(harness) = Harness::new(source).await else {
        return;
    };
    let pixels = coordinate_pattern(source);

    let magnification = 2.0;
    let loupe = plain_project(serde_json::json!([{
        "start": 0.5, "end": 8.0, "amount": magnification,
        "mode": { "manual": { "x": 0.5, "y": 0.5 } },
        "style": "loupe", "loupeSize": 0.3
    }]));
    let rendered = harness.render(&loupe, &pixels, output, SETTLED_FRAME).await;
    let plain = harness
        .render(
            &plain_project(serde_json::json!([])),
            &pixels,
            output,
            SETTLED_FRAME,
        )
        .await;
    assert_eq!((rendered.width, rendered.height), (output.x, output.y));

    let mut timeline =
        ZoomTransformTimeline::from_project(&loupe, &CursorEvents::default(), DURATION, source);
    timeline.precompute();
    let sample = timeline
        .sample_loupe(SETTLED_FRAME as f32 / FPS as f32)
        .expect("lens is shown inside its segment");
    assert!((sample.presence - 1.0).abs() < 0.01, "{sample:?}");
    let card = XY::new(output.x as f64, output.y as f64);
    let geometry = LoupeGeometry::resolve(&sample, XY::new(0.5, 0.5), card).unwrap();
    assert!((geometry.center.x - 512.0).abs() < 1.0 && (geometry.center.y - 288.0).abs() < 1.0);

    let source_value = |card_px: f64, card_extent: f64| card_px / card_extent * 255.0;
    for dx in [-0.5, -0.25, 0.0, 0.25, 0.5] {
        let offset = dx * geometry.half_extent.x;
        let x = (geometry.center.x + offset).round() as u32;
        let shown = rendered.pixel(x, geometry.center.y as u32);
        let expected = source_value(geometry.anchor.x + offset / magnification, card.x);
        assert!(
            (f64::from(shown[0]) - expected).abs() <= 2.5,
            "lens offset {dx}: R={} expected {expected:.1}",
            shown[0]
        );
        let unmagnified = plain.pixel(x, geometry.center.y as u32);
        if dx != 0.0 {
            assert!(
                (f64::from(unmagnified[0]) - expected).abs() > 4.0,
                "lens offset {dx} must differ from the unmagnified frame"
            );
        }
    }

    let rim_y = (geometry.center.y - geometry.half_extent.y + 1.0) as u32;
    let rim = rendered.pixel(geometry.center.x as u32, rim_y);
    let under_rim = plain.pixel(geometry.center.x as u32, rim_y);
    assert!(
        u32::from(rim[0]) + u32::from(rim[1]) > u32::from(under_rim[0]) + u32::from(under_rim[1]),
        "the lit top rim brightens the content: {rim:?} vs {under_rim:?}"
    );
    let shadow_y = (geometry.center.y + geometry.half_extent.y + 6.0) as u32;
    let shadow = rendered.pixel(geometry.center.x as u32, shadow_y);
    let unshadowed = plain.pixel(geometry.center.x as u32, shadow_y);
    assert!(
        shadow[1] < unshadowed[1],
        "the drop shadow darkens below the lens: {shadow:?} vs {unshadowed:?}"
    );

    let reach = geometry.half_extent.x * 1.6;
    for (x, y) in [(10, 10), (1010, 20), (20, 560), (1000, 550)] {
        let distance = ((x as f64 - geometry.center.x).powi(2)
            + (y as f64 - geometry.center.y).powi(2))
        .sqrt();
        assert!(distance > reach);
        assert_eq!(rendered.pixel(x, y), plain.pixel(x, y), "{x},{y}");
    }
}

#[tokio::test]
async fn hidden_loupe_and_unzoomed_frames_render_identically() {
    let source = XY::new(512, 288);
    let output = XY::new(1024, 576);
    let Some(harness) = Harness::new(source).await else {
        return;
    };
    let pixels = coordinate_pattern(source);
    let plain = harness
        .render(&plain_project(serde_json::json!([])), &pixels, output, 15)
        .await;
    for shape in ["circle", "roundedRect"] {
        let later_loupe = plain_project(serde_json::json!([{
            "start": 6.0, "end": 8.0, "amount": 3.0, "mode": "auto",
            "style": "loupe", "loupeShape": shape
        }]));
        let rendered = harness.render(&later_loupe, &pixels, output, 15).await;
        assert!(rendered.rgba == plain.rgba, "{shape}");
    }
}

#[tokio::test]
async fn rounded_rect_loupe_is_wider_than_tall() {
    let source = XY::new(512, 288);
    let output = XY::new(1024, 576);
    let Some(harness) = Harness::new(source).await else {
        return;
    };
    let pixels = vec![200u8; (source.x * source.y * 4) as usize];
    let project = plain_project(serde_json::json!([{
        "start": 0.5, "end": 8.0, "amount": 2.0,
        "mode": { "manual": { "x": 0.5, "y": 0.5 } },
        "style": "loupe", "loupeShape": "roundedRect", "loupeSize": 0.25
    }]));
    let rendered = harness
        .render(&project, &pixels, output, SETTLED_FRAME)
        .await;
    let mut timeline =
        ZoomTransformTimeline::from_project(&project, &CursorEvents::default(), DURATION, source);
    timeline.precompute();
    let sample = timeline
        .sample_loupe(SETTLED_FRAME as f32 / FPS as f32)
        .unwrap();
    assert_eq!(sample.shape, LoupeShape::RoundedRect);
    let geometry = LoupeGeometry::resolve(
        &sample,
        XY::new(0.5, 0.5),
        XY::new(output.x as f64, output.y as f64),
    )
    .unwrap();
    assert!(geometry.half_extent.x > geometry.half_extent.y * 1.5);

    let plain_gray = rendered.pixel(5, 5);
    let differs = |x: f64, y: f64| rendered.pixel(x as u32, y as u32) != plain_gray;
    let (cx, cy) = (geometry.center.x, geometry.center.y);
    let (hx, hy) = (geometry.half_extent.x, geometry.half_extent.y);
    assert!(differs(cx + hx - 2.0, cy), "right rim is drawn");
    assert!(
        !differs(cx + hx + 40.0, cy - hy - 40.0),
        "outside stays untouched"
    );
}

/// A 2x zoom of a hard 64 -> 192 edge, upscaled 4x overall. Bilinear spreads
/// the step evenly across 4 output px (max step 32); the Catmull-Rom + CAS
/// path must be visibly steeper while never leaving the [64, 192] range
/// (no dark or bright halo on either side of the edge).
#[tokio::test]
async fn zoomed_edges_are_sharper_without_halos() {
    let source = XY::new(512, 288);
    let output = XY::new(1024, 576);
    let Some(harness) = Harness::new(source).await else {
        return;
    };
    let mut pixels = Vec::with_capacity((source.x * source.y * 4) as usize);
    for _ in 0..source.y {
        for x in 0..source.x {
            let value = if (x / 37) % 2 == 0 { 64 } else { 192 };
            pixels.extend_from_slice(&[value, value, value, 255]);
        }
    }
    let project = plain_project(serde_json::json!([{
        "start": 0.5, "end": 8.0, "amount": 2.0,
        "mode": { "manual": { "x": 0.5, "y": 0.5 } }
    }]));
    let rendered = harness
        .render(&project, &pixels, output, SETTLED_FRAME)
        .await;

    let row: Vec<i32> = (0..output.x)
        .map(|x| i32::from(rendered.pixel(x, output.y / 2)[0]))
        .collect();
    let (lo, hi) = (*row.iter().min().unwrap(), *row.iter().max().unwrap());
    assert!(lo >= 63 && hi <= 193, "halo: row spans {lo}..{hi}");
    let max_step = row
        .windows(2)
        .map(|pair| (pair[1] - pair[0]).abs())
        .max()
        .unwrap();
    assert!(
        max_step >= 40,
        "edge is no sharper than bilinear: max step {max_step}"
    );
}

#[tokio::test]
#[ignore]
async fn benchmark_zoom_and_loupe_render_cost() {
    use std::time::Instant;

    let source = XY::new(3840, 2160);
    let Some(harness) = Harness::new(source).await else {
        return;
    };
    let pixels = coordinate_pattern(source)
        .chunks(4)
        .enumerate()
        .flat_map(|(index, pixel)| {
            let noise = ((index as u32).wrapping_mul(2_654_435_761) >> 24) as u8;
            [pixel[0], pixel[1], noise, 255]
        })
        .collect::<Vec<_>>();
    let cases = [
        ("1x", serde_json::json!([])),
        (
            "2x zoom",
            serde_json::json!([{ "start": 0.5, "end": 8.0, "amount": 2.0,
                "mode": { "manual": { "x": 0.4, "y": 0.6 } } }]),
        ),
        (
            "loupe 2x",
            serde_json::json!([{ "start": 0.5, "end": 8.0, "amount": 2.0,
                "mode": { "manual": { "x": 0.4, "y": 0.6 } },
                "style": "loupe", "loupeSize": 0.3 }]),
        ),
    ];
    let cursor = CursorEvents::default();
    let mut renderer = FrameRenderer::new(&harness.constants);
    let mut layers = RendererLayers::new(&harness.constants.device, &harness.constants.queue);
    for output in [XY::new(1920, 1080), XY::new(3840, 2160)] {
        for (label, segments) in &cases {
            let project = plain_project(segments.clone());
            let mut zoom = ZoomTransformTimeline::from_project(&project, &cursor, DURATION, source);
            zoom.precompute();
            let shared = std::sync::Arc::new(project);
            let render = async |renderer: &mut FrameRenderer<'_>, layers: &mut RendererLayers| {
                let frames = DecodedSegmentFrames {
                    screen_size: source,
                    screen_frame: Some(DecodedFrame::new(pixels.clone(), source.x, source.y)),
                    camera_frame: None,
                    segment_time: 3.0,
                    recording_time: 3.0,
                    segment_has_camera: false,
                };
                let uniforms = ProjectUniforms::new(
                    &harness.constants,
                    &shared,
                    SETTLED_FRAME,
                    FPS,
                    output,
                    &cursor,
                    &frames,
                    DURATION,
                    &zoom,
                );
                renderer
                    .render_immediate(frames, uniforms, &cursor, true, layers)
                    .await
                    .unwrap();
            };
            for _ in 0..10 {
                render(&mut renderer, &mut layers).await;
            }
            let runs = 60;
            let start = Instant::now();
            for _ in 0..runs {
                render(&mut renderer, &mut layers).await;
            }
            println!(
                "{}x{} {label}: {:.2} ms/frame",
                output.x,
                output.y,
                start.elapsed().as_secs_f64() * 1000.0 / runs as f64
            );
        }
    }
}

/// Isolates the display shader: one 4K source drawn 200 times per submission
/// into a 1080p or 4K target, so the timing is GPU-bound and excludes upload,
/// readback and the rest of the frame.
#[tokio::test]
#[ignore]
async fn benchmark_display_shader_gpu_cost() {
    use cap_rendering::composite_frame::{
        CompositeVideoFramePipeline, CompositeVideoFrameUniforms,
    };
    use std::time::Instant;

    let instance = cap_rendering::create_wgpu_instance_sync();
    let Ok(adapter) = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
    else {
        return;
    };
    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor::default())
        .await
        .unwrap();
    let source = XY::new(3840u32, 2160u32);
    let texture = CompositeVideoFramePipeline::create_frame_texture(&device, source.x, source.y);
    let pixels = coordinate_pattern(source)
        .chunks(4)
        .enumerate()
        .flat_map(|(index, pixel)| {
            let noise = ((index as u32).wrapping_mul(2_654_435_761) >> 24) as u8;
            [pixel[0], pixel[1], noise, 255]
        })
        .collect::<Vec<_>>();
    queue.write_texture(
        texture.as_image_copy(),
        &pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(source.x * 4),
            rows_per_image: None,
        },
        wgpu::Extent3d {
            width: source.x,
            height: source.y,
            depth_or_array_layers: 1,
        },
    );
    let view = texture.create_view(&Default::default());
    let pipeline = CompositeVideoFramePipeline::new(&device);

    for output in [XY::new(1920u32, 1080u32), XY::new(3840, 2160)] {
        let target = device.create_texture(&wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: output.x,
                height: output.y,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let target_view = target.create_view(&Default::default());
        let (w, h) = (output.x as f32, output.y as f32);
        let base = CompositeVideoFrameUniforms {
            output_size: [w, h],
            frame_size: [source.x as f32, source.y as f32],
            crop_bounds: [0.0, 0.0, source.x as f32, source.y as f32],
            target_bounds: [0.0, 0.0, w, h],
            target_size: [w, h],
            sampling_params: [1.0, 0.0, 0.0, 0.0],
            ..Default::default()
        };
        let zoomed = CompositeVideoFrameUniforms {
            target_bounds: [-0.4 * w, -0.6 * h, 1.6 * w, 1.4 * h],
            target_size: [2.0 * w, 2.0 * h],
            sampling_params: [2.0, 0.0, 0.0, 0.0],
            ..base
        };
        let radius = 0.3 * h;
        let loupe = CompositeVideoFrameUniforms {
            loupe_position: [0.4, 0.6, 0.4, 0.6],
            loupe_shape: [radius, radius, radius, 2.0],
            ..base
        };
        for (label, uniforms) in [("1x", base), ("2x zoom", zoomed), ("loupe 2x", loupe)] {
            let buffer = uniforms.to_buffer(&device);
            let bind_group = pipeline.bind_group(&device, &buffer, &view);
            let run = |draws: u32| {
                let mut encoder = device.create_command_encoder(&Default::default());
                for _ in 0..draws {
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: None,
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &target_view,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        timestamp_writes: None,
                        occlusion_query_set: None,
                    });
                    pass.set_pipeline(&pipeline.render_pipeline);
                    pass.set_bind_group(0, &bind_group, &[]);
                    pass.draw(0..3, 0..1);
                }
                queue.submit([encoder.finish()]);
                device.poll(wgpu::PollType::Wait).unwrap();
            };
            run(100);
            let draws = 200;
            let best = (0..15)
                .map(|_| {
                    let start = Instant::now();
                    run(draws);
                    start.elapsed().as_secs_f64() * 1000.0 / f64::from(draws)
                })
                .fold(f64::INFINITY, f64::min);
            println!("{}x{} {label}: {best:.3} ms/draw", output.x, output.y);
        }
    }
}
