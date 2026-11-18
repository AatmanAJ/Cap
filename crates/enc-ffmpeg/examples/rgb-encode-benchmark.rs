use cap_enc_ffmpeg::{
    convert::RgbToYuv420,
    h264::{H264Encoder, H264Preset},
    segmented_stream::{SegmentedVideoEncoder, SegmentedVideoEncoderConfig},
};
use cap_media_info::VideoInfo;
use ffmpeg::{format::Pixel, frame, software::scaling};
use std::{
    sync::mpsc::{Receiver, RecvTimeoutError, SyncSender, TrySendError, sync_channel},
    thread,
    time::{Duration, Instant},
};

const RESOLUTIONS: [(&str, u32, u32); 3] = [
    ("1080p", 1920, 1080),
    ("1440p", 2560, 1440),
    ("4K", 3840, 2160),
];
const SOURCE_FRAMES: usize = 8;
const CHANNEL_CAPACITY: usize = 30;
const STALL_BUDGET: Duration = Duration::from_millis(50);

#[derive(Clone, Copy, PartialEq, Eq)]
enum Path {
    Swscale,
    Simd,
}

impl Path {
    fn label(self) -> &'static str {
        match self {
            Path::Swscale => "before (swscale)",
            Path::Simd => "after (SIMD)",
        }
    }
}

struct Options {
    encode_frames: usize,
    realtime_seconds: u64,
    resolutions: Vec<(&'static str, u32, u32)>,
    trace_dir: Option<std::path::PathBuf>,
}

fn options() -> Options {
    let mut options = Options {
        encode_frames: 240,
        realtime_seconds: 10,
        resolutions: RESOLUTIONS.to_vec(),
        trace_dir: None,
    };
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut index = 0;
    while index < args.len() {
        let value = args.get(index + 1);
        match (args[index].as_str(), value) {
            ("--frames", Some(value)) => options.encode_frames = value.parse().expect("--frames"),
            ("--seconds", Some(value)) => {
                options.realtime_seconds = value.parse().expect("--seconds")
            }
            ("--only", Some(value)) => {
                options.resolutions = RESOLUTIONS
                    .into_iter()
                    .filter(|(label, _, _)| label.eq_ignore_ascii_case(value))
                    .collect()
            }
            ("--trace-dir", Some(value)) => {
                std::fs::create_dir_all(value).expect("--trace-dir");
                options.trace_dir = Some(value.into())
            }
            (flag, _) => panic!("unknown argument {flag}"),
        }
        index += 2;
    }
    options
}

fn write_trace(options: &Options, name: &str, header: &str, rows: &[(f64, f64)]) {
    let Some(directory) = &options.trace_dir else {
        return;
    };
    let mut csv = format!("{header}\n");
    for (x, y) in rows {
        csv.push_str(&format!("{x:.3},{y:.3}\n"));
    }
    std::fs::write(directory.join(format!("{name}.csv")), csv).expect("write trace");
}

fn trace_name(path: Path) -> &'static str {
    match path {
        Path::Swscale => "before",
        Path::Simd => "after",
    }
}

fn process_cpu_seconds() -> Option<f64> {
    let stat = std::fs::read_to_string("/proc/self/stat").ok()?;
    let fields: Vec<&str> = stat.rsplit_once(')')?.1.split_whitespace().collect();
    let user: f64 = fields.get(11)?.parse().ok()?;
    let system: f64 = fields.get(12)?.parse().ok()?;
    Some((user + system) / 100.0)
}

fn screen_frames(width: u32, height: u32) -> Vec<frame::Video> {
    (0..SOURCE_FRAMES)
        .map(|index| {
            let mut frame = frame::Video::new(Pixel::BGRZ, width, height);
            let stride = frame.stride(0);
            let scroll = index * 24;
            let data = frame.data_mut(0);
            for y in 0..height as usize {
                let content_row = y + scroll;
                for x in 0..width as usize {
                    let offset = y * stride + x * 4;
                    let window = x > width as usize / 8 && x < width as usize * 7 / 8 && y > 64;
                    let pixel = if !window {
                        [
                            (40 + x * 60 / width as usize) as u8,
                            (24 + y * 40 / height as usize) as u8,
                            90,
                        ]
                    } else if (content_row / 18) % 3 != 2
                        && (x / 9 + content_row / 18 * 7) % 11 < 8
                        && (content_row % 18) > 4
                        && ((x * 13 + content_row * 7) % 5) < 3
                    {
                        [30, 30, 36]
                    } else if (content_row / 240) % 4 == 1 && x < width as usize / 2 {
                        [
                            (x * 255 / width as usize) as u8,
                            (content_row % 255) as u8,
                            200,
                        ]
                    } else {
                        [250, 250, 248]
                    };
                    data[offset..offset + 4].copy_from_slice(&[pixel[0], pixel[1], pixel[2], 255]);
                }
            }
            frame
        })
        .collect()
}

fn swscale_converter(width: u32, height: u32) -> scaling::Context {
    let mut context = scaling::Context::get(
        Pixel::BGRZ,
        width,
        height,
        Pixel::YUV420P,
        width,
        height,
        scaling::Flags::FAST_BILINEAR,
    )
    .expect("swscale context");
    let result = unsafe {
        let coefficients = ffmpeg::ffi::sws_getCoefficients(ffmpeg::ffi::SWS_CS_ITU709);
        ffmpeg::ffi::sws_setColorspaceDetails(
            context.as_mut_ptr(),
            coefficients,
            1,
            coefficients,
            0,
            0,
            1 << 16,
            1 << 16,
        )
    };
    assert!(result >= 0, "swscale colorspace configuration failed");
    context
}

struct Stats {
    mean: f64,
    p50: f64,
    p95: f64,
}

fn stats(mut samples: Vec<f64>) -> Stats {
    samples.sort_by(f64::total_cmp);
    let percentile = |p: f64| samples[((samples.len() - 1) as f64 * p).round() as usize];
    Stats {
        mean: samples.iter().sum::<f64>() / samples.len() as f64,
        p50: percentile(0.5),
        p95: percentile(0.95),
    }
}

fn time_conversion(iterations: usize, mut convert: impl FnMut(usize)) -> Stats {
    for index in 0..8 {
        convert(index);
    }
    stats(
        (0..iterations)
            .map(|index| {
                let start = Instant::now();
                convert(index);
                start.elapsed().as_secs_f64() * 1000.0
            })
            .collect(),
    )
}

fn conversion_section(options: &Options) {
    println!("## Conversion only: BGRZ -> YUV420P BT.709 limited, one thread\n");
    println!("| resolution | path | mean ms | p50 ms | p95 ms |");
    println!("|---|---|---|---|---|");
    for &(label, width, height) in &options.resolutions {
        let frames = screen_frames(width, height);
        let mut output = frame::Video::new(Pixel::YUV420P, width, height);
        let iterations = 200;

        let mut swscale = swscale_converter(width, height);
        let before = time_conversion(iterations, |index| {
            swscale
                .run(&frames[index % SOURCE_FRAMES], &mut output)
                .expect("swscale run")
        });
        let simd = RgbToYuv420::new(Pixel::BGRZ, Pixel::YUV420P, width, height).expect("converter");
        let after = time_conversion(iterations, |index| {
            simd.run(&frames[index % SOURCE_FRAMES], &mut output)
                .expect("simd run")
        });
        let portable = simd.without_simd();
        let fallback = time_conversion(iterations, |index| {
            portable
                .run(&frames[index % SOURCE_FRAMES], &mut output)
                .expect("portable run")
        });
        for (name, result) in [
            ("before: swscale", before),
            ("after: SIMD", after),
            ("after: portable fallback", fallback),
        ] {
            println!(
                "| {label} | {name} | {:.2} | {:.2} | {:.2} |",
                result.mean, result.p50, result.p95
            );
        }
    }
    println!();
}

struct Encoder {
    encoder: H264Encoder,
    output: ffmpeg::format::context::Output,
    swscale: Option<(scaling::Context, frame::Video)>,
    _directory: tempfile::TempDir,
}

impl Encoder {
    fn new(path: Path, width: u32, height: u32, fps: u32) -> Self {
        let directory = tempfile::tempdir().expect("tempdir");
        let mut output =
            ffmpeg::format::output(&directory.path().join("bench.mp4")).expect("output");
        let input = match path {
            Path::Swscale => Pixel::YUV420P,
            Path::Simd => Pixel::BGRZ,
        };
        let encoder = H264Encoder::builder(VideoInfo::from_raw_ffmpeg(input, width, height, fps))
            .with_preset(H264Preset::Ultrafast)
            .with_encoder_priority_override(&["libx264"])
            .build(&mut output)
            .expect("encoder");
        output.write_header().expect("header");
        Self {
            encoder,
            output,
            swscale: (path == Path::Swscale).then(|| {
                (
                    swscale_converter(width, height),
                    frame::Video::new(Pixel::YUV420P, width, height),
                )
            }),
            _directory: directory,
        }
    }

    fn queue(&mut self, frame: frame::Video, timestamp: Duration) {
        let frame = match &mut self.swscale {
            Some((context, converted)) => {
                context.run(&frame, converted).expect("swscale run");
                converted.clone()
            }
            None => frame,
        };
        self.encoder
            .queue_frame(frame, timestamp, &mut self.output)
            .expect("queue frame");
    }

    fn finish(mut self) {
        self.encoder.flush(&mut self.output).expect("flush");
        self.output.write_trailer().expect("trailer");
    }
}

fn encode_section(options: &Options) {
    println!(
        "## Encoder hot path: capture frame in, libx264 ultrafast packet out, {} frames, unthrottled\n",
        options.encode_frames
    );
    println!(
        "| resolution | path | wall ms/frame (mean) | p95 ms | CPU ms/frame | max sustained fps |"
    );
    println!("|---|---|---|---|---|---|");
    for &(label, width, height) in &options.resolutions {
        let frames = screen_frames(width, height);
        for path in [Path::Swscale, Path::Simd] {
            let mut encoder = Encoder::new(path, width, height, 30);
            for (index, frame) in frames.iter().enumerate() {
                encoder.queue(frame.clone(), Duration::from_millis(index as u64 * 33));
            }
            let cpu_start = process_cpu_seconds();
            let wall_start = Instant::now();
            let samples: Vec<f64> = (SOURCE_FRAMES..SOURCE_FRAMES + options.encode_frames)
                .map(|index| {
                    let start = Instant::now();
                    encoder.queue(
                        frames[index % SOURCE_FRAMES].clone(),
                        Duration::from_millis(index as u64 * 33),
                    );
                    start.elapsed().as_secs_f64() * 1000.0
                })
                .collect();
            write_trace(
                options,
                &format!("encode-{label}-{}", trace_name(path)),
                "frame,wall_ms",
                &samples
                    .iter()
                    .enumerate()
                    .map(|(index, ms)| (index as f64, *ms))
                    .collect::<Vec<_>>(),
            );
            let wall = wall_start.elapsed().as_secs_f64();
            let cpu = process_cpu_seconds()
                .zip(cpu_start)
                .map(|(end, start)| {
                    format!(
                        "{:.2}",
                        (end - start) * 1000.0 / options.encode_frames as f64
                    )
                })
                .unwrap_or_else(|| "n/a".to_string());
            encoder.finish();
            let result = stats(samples);
            println!(
                "| {label} | {} | {:.2} | {:.2} | {cpu} | {:.1} |",
                path.label(),
                result.mean,
                result.p95,
                options.encode_frames as f64 / wall
            );
        }
    }
    println!();
}

type Queued = Option<(frame::Video, Duration)>;

fn produce(
    tx: SyncSender<Queued>,
    frames: Vec<frame::Video>,
    fps: u32,
    seconds: u64,
    started: Instant,
) {
    let interval = Duration::from_secs_f64(1.0 / fps as f64);
    let duration = Duration::from_secs(seconds);
    let mut next_capture = started;
    let (mut sent, mut dropped) = (0u64, 0u64);
    while started.elapsed() < duration {
        let index = (sent + dropped) as usize;
        let mut item: Queued = Some((frames[index % SOURCE_FRAMES].clone(), started.elapsed()));
        let deadline = Instant::now() + STALL_BUDGET;
        loop {
            match tx.try_send(item) {
                Ok(()) => {
                    sent += 1;
                    break;
                }
                Err(TrySendError::Full(rejected)) if Instant::now() < deadline => {
                    item = rejected;
                    thread::sleep(Duration::from_millis(1));
                }
                Err(TrySendError::Full(_)) => {
                    dropped += 1;
                    break;
                }
                Err(TrySendError::Disconnected(_)) => return,
            }
        }
        next_capture += interval;
        let now = Instant::now();
        match next_capture.checked_duration_since(now) {
            Some(wait) => thread::sleep(wait),
            None => next_capture = now,
        }
    }
    let _ = tx.send(None);
}

fn consume(
    rx: Receiver<Queued>,
    path: Path,
    width: u32,
    height: u32,
    fps: u32,
    started: Instant,
) -> (u64, Duration, Vec<(f64, f64)>) {
    let directory = tempfile::tempdir().expect("tempdir");
    let input = match path {
        Path::Swscale => Pixel::YUV420P,
        Path::Simd => Pixel::BGRZ,
    };
    let mut encoder = SegmentedVideoEncoder::init(
        directory.path().join("display"),
        VideoInfo::from_raw_ffmpeg(input, width, height, fps),
        SegmentedVideoEncoderConfig {
            segment_duration: Duration::from_secs(2),
            preset: H264Preset::Ultrafast,
            ..Default::default()
        },
    )
    .expect("segmented encoder");
    let mut swscale = (path == Path::Swscale).then(|| {
        (
            swscale_converter(width, height),
            frame::Video::new(Pixel::YUV420P, width, height),
        )
    });
    let mut encoded = 0;
    let mut max_latency = Duration::ZERO;
    let mut trace = Vec::new();
    loop {
        match rx.recv_timeout(Duration::from_secs(5)) {
            Ok(Some((frame, timestamp))) => {
                let frame = match &mut swscale {
                    Some((context, converted)) => {
                        context.run(&frame, converted).expect("swscale run");
                        converted.clone()
                    }
                    None => frame,
                };
                encoder.queue_frame(frame, timestamp).expect("queue frame");
                encoded += 1;
                let latency = started.elapsed().saturating_sub(timestamp);
                max_latency = max_latency.max(latency);
                trace.push((
                    timestamp.as_secs_f64() * 1000.0,
                    latency.as_secs_f64() * 1000.0,
                ));
            }
            Ok(None) | Err(RecvTimeoutError::Disconnected) => break,
            Err(RecvTimeoutError::Timeout) => panic!("producer stalled"),
        }
    }
    encoder.finish().expect("finish");
    (encoded, max_latency, trace)
}

fn realtime_section(options: &Options) {
    println!(
        "## Real-time capture load: X11-style paced capture into SegmentedVideoEncoder (Linux studio muxer) for {} s, queue of {CHANNEL_CAPACITY}, {} ms stall budget\n",
        options.realtime_seconds,
        STALL_BUDGET.as_millis()
    );
    println!(
        "| resolution | target fps | path | target frames | encoded | lost | lost % | max capture-to-encode ms | process CPU % |"
    );
    println!("|---|---|---|---|---|---|---|---|---|");
    for &(label, width, height) in &options.resolutions {
        let frames = screen_frames(width, height);
        for fps in [30, 60] {
            for path in [Path::Swscale, Path::Simd] {
                let (tx, rx) = sync_channel::<Queued>(CHANNEL_CAPACITY);
                let cpu_start = process_cpu_seconds();
                let wall_start = Instant::now();
                let consumer =
                    thread::spawn(move || consume(rx, path, width, height, fps, wall_start));
                produce(
                    tx,
                    frames.clone(),
                    fps,
                    options.realtime_seconds,
                    wall_start,
                );
                let (encoded, max_latency, trace) = consumer.join().expect("consumer");
                write_trace(
                    options,
                    &format!("realtime-{label}-{fps}fps-{}", trace_name(path)),
                    "capture_ms,capture_to_encode_ms",
                    &trace,
                );
                let wall = wall_start.elapsed().as_secs_f64();
                let cpu = process_cpu_seconds()
                    .zip(cpu_start)
                    .map(|(end, start)| format!("{:.0}", (end - start) * 100.0 / wall))
                    .unwrap_or_else(|| "n/a".to_string());
                let target = fps as u64 * options.realtime_seconds;
                let lost = target.saturating_sub(encoded);
                println!(
                    "| {label} | {fps} | {} | {target} | {encoded} | {lost} | {:.1} | {} | {cpu} |",
                    path.label(),
                    lost as f64 * 100.0 / target as f64,
                    max_latency.as_millis()
                );
            }
        }
    }
    println!();
}

fn main() {
    ffmpeg::init().expect("ffmpeg init");
    ffmpeg::util::log::set_level(ffmpeg::util::log::Level::Error);
    let options = options();
    println!(
        "# RGB capture -> H.264 encode benchmark ({} logical CPUs, AVX2: {})\n",
        thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1),
        simd_label()
    );
    conversion_section(&options);
    encode_section(&options);
    if options.realtime_seconds > 0 {
        realtime_section(&options);
    }
}

#[cfg(target_arch = "x86_64")]
fn simd_label() -> bool {
    std::arch::is_x86_feature_detected!("avx2")
}

#[cfg(not(target_arch = "x86_64"))]
fn simd_label() -> bool {
    false
}
