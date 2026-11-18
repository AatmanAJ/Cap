use cap_rendering::decoder::{AsyncVideoDecoderHandle, DecodedFrame, pts_to_frame, spawn_decoder};
use std::{
    collections::{BTreeMap, BTreeSet},
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

fn sample_frames(path: &Path, fps: u32) -> Vec<u32> {
    let mut input = ffmpeg::format::input(path).expect("open sample");
    let stream = input
        .streams()
        .best(ffmpeg::media::Type::Video)
        .expect("video stream");
    let stream_index = stream.index();
    let time_base = stream.time_base();
    let start_time = match stream.start_time() {
        ffmpeg::ffi::AV_NOPTS_VALUE => 0,
        start_time => start_time,
    };
    let frames: BTreeSet<u32> = input
        .packets()
        .filter(|(stream, _)| stream.index() == stream_index)
        .filter_map(|(_, packet)| packet.pts())
        .map(|pts| pts_to_frame(pts.saturating_sub(start_time).max(0), time_base, fps))
        .collect();
    frames.into_iter().collect()
}

fn held_sample(samples: &[u32], requested_frame: u32) -> Option<u32> {
    let index = samples.partition_point(|&sample| sample <= requested_frame);
    samples.get(index.saturating_sub(1)).copied()
}

fn fingerprint(frame: &DecodedFrame) -> u64 {
    let data = frame.data();
    let mut hash = 0x9e37_79b9_7f4a_7c15_u64 ^ data.len() as u64;
    let mut chunks = data.chunks_exact(8);
    for chunk in &mut chunks {
        let word = u64::from_le_bytes(chunk.try_into().expect("eight byte chunk"));
        hash = (hash.rotate_left(5) ^ word).wrapping_mul(0x0000_0100_0000_01b3);
    }
    for &byte in chunks.remainder() {
        hash = (hash.rotate_left(5) ^ u64::from(byte)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

fn resolve_shown(
    reference: &BTreeMap<u32, u64>,
    picture: u64,
    requested_frame: u32,
) -> Option<u32> {
    reference
        .iter()
        .filter(|&(_, &known)| known == picture)
        .map(|(&sample, _)| sample)
        .min_by_key(|&sample| sample.abs_diff(requested_frame))
}

#[derive(Clone, Copy, PartialEq, Debug)]
struct Delivery {
    requested_frame: u32,
    picture: u64,
}

#[derive(Clone, Copy)]
struct Tick {
    elapsed: f64,
    target: f64,
    shown: Delivery,
    delivered: usize,
}

struct DragMetrics {
    distinct_shown: usize,
    unresolved_ticks: usize,
    lag_avg_s: f64,
    lag_max_s: f64,
    shown_times: Vec<f64>,
}

fn drag_metrics(
    ticks: &[Tick],
    gesture_ticks: usize,
    reference: &BTreeMap<u32, u64>,
    fps: u32,
) -> DragMetrics {
    let mut pictures = BTreeSet::new();
    let mut lags = Vec::with_capacity(gesture_ticks);
    let mut shown_times = Vec::with_capacity(ticks.len());
    let mut unresolved_ticks = 0;
    let mut shown_time = 0.0;

    for (index, tick) in ticks.iter().enumerate() {
        match resolve_shown(reference, tick.shown.picture, tick.shown.requested_frame) {
            Some(sample) => shown_time = f64::from(sample) / f64::from(fps),
            None => unresolved_ticks += 1,
        }
        shown_times.push(shown_time);
        if index < gesture_ticks {
            pictures.insert(tick.shown.picture);
            lags.push((tick.target - shown_time).abs());
        }
    }

    DragMetrics {
        distinct_shown: pictures.len(),
        unresolved_ticks,
        lag_avg_s: lags.iter().sum::<f64>() / lags.len().max(1) as f64,
        lag_max_s: lags.iter().copied().fold(0.0, f64::max),
        shown_times,
    }
}

struct DragResult {
    delivered: usize,
    metrics: DragMetrics,
    ticks: Vec<String>,
}

async fn drag_gesture(
    decoder: Arc<AsyncVideoDecoderHandle>,
    duration: f64,
    fps: u32,
    samples: &[u32],
) -> DragResult {
    const TICK_HZ: f64 = 60.0;
    const GESTURE_SECS: f64 = 4.0;
    const SETTLE_SECS: f64 = 1.0;
    let gesture_ticks = (GESTURE_SECS * TICK_HZ) as usize;
    let total_ticks = ((GESTURE_SECS + SETTLE_SECS) * TICK_HZ) as usize;
    let span = (duration - 0.5).max(0.0);
    let initial = decoder
        .get_frame(0.0)
        .await
        .expect("decode the first frame before the drag");
    let mut shown = Delivery {
        requested_frame: 0,
        picture: fingerprint(&initial),
    };
    drop(initial);
    let mut interval = tokio::time::interval(Duration::from_secs_f64(1.0 / TICK_HZ));
    let mut in_flight: Option<tokio::task::JoinHandle<Option<Delivery>>> = None;
    let mut delivered = 0;
    let mut ticks = Vec::with_capacity(total_ticks);
    let started = Instant::now();

    for tick in 0..total_ticks {
        interval.tick().await;
        let phase = (tick as f64 / (GESTURE_SECS * TICK_HZ)).min(1.0);
        let eased = phase * phase * (3.0 - 2.0 * phase);
        let requested_frame = (eased * span * f64::from(fps)).floor() as u32;
        let target = f64::from(requested_frame) / f64::from(fps);

        if let Some(handle) = in_flight.take_if(|handle| handle.is_finished())
            && let Ok(Some(delivery)) = handle.await
        {
            shown = delivery;
            delivered += 1;
        }
        if in_flight.is_none() {
            let decoder = decoder.clone();
            in_flight = Some(tokio::spawn(async move {
                let frame = decoder.get_frame(target as f32).await?;
                Some(Delivery {
                    requested_frame,
                    picture: fingerprint(&frame),
                })
            }));
        }

        ticks.push(Tick {
            elapsed: started.elapsed().as_secs_f64(),
            target,
            shown,
            delivered,
        });
    }

    if let Some(handle) = in_flight {
        handle.await.ok();
    }

    let mut reference = BTreeMap::new();
    let wanted: BTreeSet<u32> = ticks
        .iter()
        .filter_map(|tick| held_sample(samples, tick.shown.requested_frame))
        .collect();
    for sample in wanted {
        if let Some(frame) = decoder.get_frame(sample as f32 / fps as f32).await {
            reference.insert(sample, fingerprint(&frame));
        }
    }

    let metrics = drag_metrics(&ticks, gesture_ticks, &reference, fps);
    let ticks = ticks
        .iter()
        .zip(&metrics.shown_times)
        .enumerate()
        .map(|(index, (tick, shown_time))| {
            format!(
                "{{\"tick\":{index},\"elapsed\":{:.4},\"target\":{:.4},\"shown\":{shown_time:.4},\"picture\":\"{:016x}\",\"delivered\":{}}}",
                tick.elapsed, tick.target, tick.shown.picture, tick.delivered
            )
        })
        .collect();

    DragResult {
        delivered,
        metrics,
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

    let samples = sample_frames(&video, fps);
    let drag = drag_gesture(decoder.clone(), duration, fps, &samples).await;
    if let Some(trace) = &args.trace {
        std::fs::write(trace, format!("[{}]", drag.ticks.join(","))).expect("write trace");
    }

    println!("  open                   {open_ms:.2}ms");
    println!("  playback throughput    {playback_fps:.1} fps");
    playback.print("playback frame");
    scrub.print("random scrub");
    println!(
        "  4s drag gesture        frames delivered={} distinct pictures shown={} lag avg={:.3}s max={:.3}s unresolved ticks={}",
        drag.delivered,
        drag.metrics.distinct_shown,
        drag.metrics.lag_avg_s,
        drag.metrics.lag_max_s,
        drag.metrics.unresolved_ticks
    );
    println!("  misses                 playback={playback_misses} scrub={scrub_misses}");
    println!(
        "{{\"open_ms\":{open_ms:.2},\"playback_fps\":{playback_fps:.1},\"playback_p95_ms\":{:.2},\"scrub_avg_ms\":{:.2},\"scrub_p50_ms\":{:.2},\"scrub_p95_ms\":{:.2},\"drag_delivered\":{},\"drag_distinct_shown\":{},\"drag_lag_avg_s\":{:.3},\"drag_unresolved_ticks\":{}}}",
        playback.percentile(95.0),
        scrub.avg(),
        scrub.percentile(50.0),
        scrub.percentile(95.0),
        drag.delivered,
        drag.metrics.distinct_shown,
        drag.metrics.lag_avg_s,
        drag.metrics.unresolved_ticks,
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tick(target_frame: u32, requested_frame: u32, picture: u64) -> Tick {
        Tick {
            elapsed: 0.0,
            target: f64::from(target_frame) / 60.0,
            shown: Delivery {
                requested_frame,
                picture,
            },
            delivered: 0,
        }
    }

    #[test]
    fn held_sample_is_the_newest_sample_at_or_before_the_request() {
        let samples = [0, 1, 2, 60, 61];
        assert_eq!(held_sample(&samples, 0), Some(0));
        assert_eq!(held_sample(&samples, 2), Some(2));
        assert_eq!(held_sample(&samples, 30), Some(2));
        assert_eq!(held_sample(&samples, 59), Some(2));
        assert_eq!(held_sample(&samples, 60), Some(60));
        assert_eq!(held_sample(&samples, 900), Some(61));
        assert_eq!(held_sample(&[5, 6], 2), Some(5));
        assert_eq!(held_sample(&[], 2), None);
    }

    #[test]
    fn held_frame_counts_once_and_lag_tracks_the_returned_frame() {
        let reference = BTreeMap::from([(0, 0xa), (2, 0xb), (60, 0xc)]);
        let ticks: Vec<Tick> = [10, 20, 30, 40, 50]
            .into_iter()
            .map(|frame| tick(frame, frame, 0xb))
            .collect();
        let metrics = drag_metrics(&ticks, ticks.len(), &reference, 60);
        assert_eq!(metrics.distinct_shown, 1);
        assert_eq!(metrics.unresolved_ticks, 0);
        assert!(metrics.shown_times.iter().all(|&time| time == 2.0 / 60.0));
        assert!((metrics.lag_max_s - 48.0 / 60.0).abs() < 1e-9);
        assert!((metrics.lag_avg_s - 28.0 / 60.0).abs() < 1e-9);
    }

    #[test]
    fn exact_frames_have_zero_lag_and_count_separately() {
        let reference = BTreeMap::from([(0, 0xa), (1, 0xb), (2, 0xc)]);
        let ticks = [tick(0, 0, 0xa), tick(1, 1, 0xb), tick(2, 2, 0xc)];
        let metrics = drag_metrics(&ticks, 3, &reference, 60);
        assert_eq!(metrics.distinct_shown, 3);
        assert_eq!(metrics.lag_max_s, 0.0);
    }

    #[test]
    fn identical_pictures_resolve_to_the_sample_nearest_the_request() {
        let reference = BTreeMap::from([(0, 0xa), (30, 0xa), (60, 0xa)]);
        assert_eq!(resolve_shown(&reference, 0xa, 31), Some(30));
        assert_eq!(resolve_shown(&reference, 0xa, 58), Some(60));
        assert_eq!(resolve_shown(&reference, 0xf, 58), None);
    }

    #[test]
    fn unknown_picture_keeps_the_previous_position_and_is_reported() {
        let reference = BTreeMap::from([(0, 0xa), (30, 0xb)]);
        let ticks = [tick(30, 30, 0xb), tick(40, 40, 0xf), tick(90, 90, 0xf)];
        let metrics = drag_metrics(&ticks, 2, &reference, 60);
        assert_eq!(metrics.unresolved_ticks, 2);
        assert_eq!(metrics.shown_times, vec![0.5, 0.5, 0.5]);
        assert_eq!(metrics.distinct_shown, 2);
        assert!((metrics.lag_max_s - 10.0 / 60.0).abs() < 1e-9);
    }
}
