use cap_rendering::decoder::{AsyncVideoDecoderHandle, spawn_decoder};
use std::{
    path::{Path, PathBuf},
    process::Command,
    sync::Arc,
    time::{Duration, Instant},
};

struct Args {
    video: Option<PathBuf>,
    fps: u32,
    playback_frames: u32,
    scrub_samples: u32,
    force_ffmpeg: bool,
    trace: Option<PathBuf>,
}

fn parse_args() -> Args {
    let args: Vec<String> = std::env::args().collect();
    let value = |name: &str| {
        args.iter()
            .position(|arg| arg == name)
            .and_then(|index| args.get(index + 1))
            .cloned()
    };
    Args {
        video: value("--video").map(PathBuf::from),
        fps: value("--fps").and_then(|v| v.parse().ok()).unwrap_or(60),
        playback_frames: value("--playback-frames")
            .and_then(|v| v.parse().ok())
            .unwrap_or(600),
        scrub_samples: value("--scrub-samples")
            .and_then(|v| v.parse().ok())
            .unwrap_or(60),
        force_ffmpeg: args.iter().any(|arg| arg == "--force-ffmpeg"),
        trace: value("--trace").map(PathBuf::from),
    }
}

fn generate_sample(directory: &Path, fps: u32) -> PathBuf {
    let output = directory.join("display.mp4");
    let status = Command::new("ffmpeg")
        .args([
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            &format!("testsrc2=size=1920x1080:rate={fps}:duration=30"),
            "-f",
            "lavfi",
            "-i",
            &format!("mandelbrot=size=640x360:rate={fps}"),
            "-filter_complex",
            "[0:v][1:v]overlay=x='mod(t*200,1280)':y=200:shortest=1",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-g",
            &(fps * 2).to_string(),
            "-pix_fmt",
            "yuv420p",
        ])
        .arg(&output)
        .status()
        .expect("ffmpeg must be on PATH to generate the sample recording");
    assert!(status.success(), "ffmpeg failed to generate the sample");
    output
}

fn video_duration(path: &Path) -> f64 {
    let input = ffmpeg::format::input(path).expect("open sample");
    input.duration() as f64 / f64::from(ffmpeg::ffi::AV_TIME_BASE)
}

#[derive(Default)]
struct Stats {
    values_ms: Vec<f64>,
}

impl Stats {
    fn push(&mut self, started: Instant) {
        self.values_ms
            .push(started.elapsed().as_secs_f64() * 1000.0);
    }

    fn percentile(&self, p: f64) -> f64 {
        let mut sorted = self.values_ms.clone();
        sorted.sort_by(f64::total_cmp);
        if sorted.is_empty() {
            return 0.0;
        }
        let index = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
        sorted[index.min(sorted.len() - 1)]
    }

    fn total(&self) -> f64 {
        self.values_ms.iter().sum()
    }

    fn avg(&self) -> f64 {
        if self.values_ms.is_empty() {
            return 0.0;
        }
        self.total() / self.values_ms.len() as f64
    }

    fn print(&self, label: &str) {
        println!(
            "  {label:<22} avg={:>8.2}ms p50={:>8.2}ms p95={:>8.2}ms max={:>8.2}ms n={}",
            self.avg(),
            self.percentile(50.0),
            self.percentile(95.0),
            self.percentile(100.0),
            self.values_ms.len()
        );
    }
}

async fn timed_frames(
    decoder: &AsyncVideoDecoderHandle,
    times: impl IntoIterator<Item = f32>,
) -> (Stats, usize) {
    let mut stats = Stats::default();
    let mut misses = 0;
    for time in times {
        let started = Instant::now();
        if decoder.get_frame(time).await.is_none() {
            misses += 1;
        }
        stats.push(started);
    }
    (stats, misses)
}

struct DragResult {
    delivered: usize,
    distinct_shown: usize,
    lag_avg_s: f64,
    lag_max_s: f64,
    ticks: Vec<String>,
}

async fn drag_gesture(
    decoder: Arc<AsyncVideoDecoderHandle>,
    duration: f64,
    fps: u32,
) -> DragResult {
    const TICK_HZ: f64 = 60.0;
    const GESTURE_SECS: f64 = 4.0;
    const SETTLE_SECS: f64 = 1.0;
    let gesture_ticks = (GESTURE_SECS * TICK_HZ) as usize;
    let total_ticks = ((GESTURE_SECS + SETTLE_SECS) * TICK_HZ) as usize;
    let span = (duration - 0.5).max(0.0);
    let mut interval = tokio::time::interval(Duration::from_secs_f64(1.0 / TICK_HZ));
    let mut in_flight: Option<tokio::task::JoinHandle<Option<f64>>> = None;
    let mut shown = 0.0_f64;
    let mut delivered = 0;
    let mut shown_in_gesture = std::collections::BTreeSet::new();
    let mut lags = Vec::with_capacity(gesture_ticks);
    let mut ticks = Vec::with_capacity(total_ticks);
    let started = Instant::now();

    for tick in 0..total_ticks {
        interval.tick().await;
        let phase = (tick as f64 / (GESTURE_SECS * TICK_HZ)).min(1.0);
        let eased = phase * phase * (3.0 - 2.0 * phase);
        let target = (eased * span * f64::from(fps)).floor() / f64::from(fps);

        if let Some(handle) = in_flight.take_if(|handle| handle.is_finished())
            && let Ok(Some(time)) = handle.await
        {
            shown = time;
            delivered += 1;
        }
        if in_flight.is_none() {
            let decoder = decoder.clone();
            in_flight = Some(tokio::spawn(async move {
                decoder.get_frame(target as f32).await.map(|_| target)
            }));
        }

        if tick < gesture_ticks {
            shown_in_gesture.insert((shown * f64::from(fps)).round() as u32);
            lags.push((target - shown).abs());
        }
        ticks.push(format!(
            "{{\"tick\":{tick},\"elapsed\":{:.4},\"target\":{target:.4},\"shown\":{shown:.4},\"delivered\":{delivered}}}",
            started.elapsed().as_secs_f64()
        ));
    }

    DragResult {
        delivered,
        distinct_shown: shown_in_gesture.len(),
        lag_avg_s: lags.iter().sum::<f64>() / lags.len().max(1) as f64,
        lag_max_s: lags.iter().copied().fold(0.0, f64::max),
        ticks,
    }
}

#[tokio::main]
async fn main() {
    ffmpeg::init().expect("ffmpeg init");
    let args = parse_args();
    let scratch = tempfile::tempdir().expect("tempdir");
    let video = args
        .video
        .clone()
        .unwrap_or_else(|| generate_sample(scratch.path(), args.fps));
    let duration = video_duration(&video);
    let fps = args.fps;

    let opened = Instant::now();
    let decoder = Arc::new(
        spawn_decoder("bench", video.clone(), fps, 0.0, args.force_ffmpeg)
            .await
            .expect("spawn decoder"),
    );
    let open_ms = opened.elapsed().as_secs_f64() * 1000.0;
    let (width, height) = decoder.video_dimensions();

    println!("video: {}", video.display());
    println!(
        "decoder: {:?} {width}x{height} duration={duration:.2}s fps={fps} cpus={}",
        decoder.decoder_type(),
        std::thread::available_parallelism().map_or(0, |n| n.get())
    );

    let playback_frames = args
        .playback_frames
        .min(((duration * f64::from(fps)) as u32).saturating_sub(1));
    let (playback, playback_misses) = timed_frames(
        &decoder,
        (0..playback_frames).map(|frame| frame as f32 / fps as f32),
    )
    .await;
    let playback_fps = playback.values_ms.len() as f64 / (playback.total() / 1000.0);

    let mut seed = 0x2545_f491_4f6c_dd1d_u64;
    let max_time = (duration - 0.1).max(0.0);
    let scrub_times: Vec<f32> = (0..args.scrub_samples)
        .map(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            ((seed % 10_000) as f64 / 10_000.0 * max_time) as f32
        })
        .collect();
    let (scrub, scrub_misses) = timed_frames(&decoder, scrub_times).await;

    decoder.get_frame(0.0).await;
    let drag = drag_gesture(decoder.clone(), duration, fps).await;
    if let Some(trace) = &args.trace {
        std::fs::write(trace, format!("[{}]", drag.ticks.join(","))).expect("write trace");
    }

    println!("  open                   {open_ms:.2}ms");
    println!("  playback throughput    {playback_fps:.1} fps");
    playback.print("playback frame");
    scrub.print("random scrub");
    println!(
        "  4s drag gesture        frames delivered={} distinct frames shown={} lag avg={:.3}s max={:.3}s",
        drag.delivered, drag.distinct_shown, drag.lag_avg_s, drag.lag_max_s
    );
    println!("  misses                 playback={playback_misses} scrub={scrub_misses}");
    println!(
        "{{\"open_ms\":{open_ms:.2},\"playback_fps\":{playback_fps:.1},\"playback_p95_ms\":{:.2},\"scrub_avg_ms\":{:.2},\"scrub_p50_ms\":{:.2},\"scrub_p95_ms\":{:.2},\"drag_delivered\":{},\"drag_distinct_shown\":{},\"drag_lag_avg_s\":{:.3}}}",
        playback.percentile(95.0),
        scrub.avg(),
        scrub.percentile(50.0),
        scrub.percentile(95.0),
        drag.delivered,
        drag.distinct_shown,
        drag.lag_avg_s,
    );
}
