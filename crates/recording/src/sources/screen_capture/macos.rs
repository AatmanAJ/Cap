use super::*;
use crate::{
    ChannelAudioSourceConfig,
    output_pipeline::{
        self, AudioFrame, ChannelAudioSource, ChannelVideoSource, ChannelVideoSourceConfig,
        SetupCtx,
    },
};
use anyhow::{Context, anyhow};
use cap_timestamp::Timestamp;
use cidre::*;
use futures::{FutureExt as _, channel::mpsc, future::BoxFuture};
use std::{
    ptr,
    sync::{
        Arc, Mutex,
        atomic::{self, AtomicBool, AtomicU32, AtomicU64},
    },
    time::{Duration, Instant},
};
use tokio::{select, sync::broadcast};
use tokio_util::{
    future::FutureExt as _,
    sync::{CancellationToken, DropGuard},
};
use tracing::{debug, error, info, warn};

mod app_audio;

pub use app_audio::{AudioApp, app_audio_supported, list_audio_apps};

const MAX_CAPTURE_RESTARTS: u32 = 3;
const RESTART_DELAY: Duration = Duration::from_secs(2);
const APP_AUDIO_REFRESH_INTERVAL: Duration = Duration::from_secs(2);
const APP_AUDIO_TIMEOUT: Duration = Duration::from_secs(5);

// The display filter is what hides excluded windows, so an excluded selection must stay on
// the display-crop path instead of being captured on its own.
fn capturable_window<'a>(
    config: &'a Config,
    excluded_windows: &[WindowId],
    capture_phase: &'static str,
) -> Option<&'a WindowId> {
    let window_id = config.window.as_ref()?;

    if excluded_windows.contains(window_id) {
        warn!(
            %window_id,
            capture_phase,
            "Capture window is excluded from recordings, using display crop with exclusions"
        );
        return None;
    }

    Some(window_id)
}

fn window_content_filter(
    window_id: &WindowId,
    content: &sc::ShareableContent,
    capture_phase: &'static str,
) -> Option<arc::R<sc::ContentFilter>> {
    let Ok(native_id) = window_id.to_string().parse::<u32>() else {
        warn!(%window_id, capture_phase, "Capture window has an invalid native id");
        return None;
    };

    let windows = content.windows();
    let Some(window) = windows.iter().find(|window| window.id() == native_id) else {
        warn!(
            %window_id,
            capture_phase,
            "Capture window missing from ScreenCaptureKit shareable content"
        );
        return None;
    };

    Some(sc::ContentFilter::with_desktop_independent_window(window))
}

/// A single window at the display's full pixel density, filling the frame.
/// Left to themselves, window captures pick their resolution automatically,
/// which can read a Retina window at 1x and upscale it, and do not scale the
/// window to the output, leaving it in a corner of a larger frame.
fn configure_window_capture(settings: &mut sc::StreamCfg) {
    settings.set_scales_to_fit(true);
    // Both arrived in macOS 14; Cap runs from 12.3.
    if cidre::api::macos_available("14.0") {
        unsafe {
            settings.set_capture_resolution(sc::CaptureResolution::Best);
            settings.set_preserves_aspect_ratio(true);
        }
    }
}

/// What the stream reads. A display capture crops to `crop_bounds`; a window
/// capture reads the window itself, inside its rounded corners, where
/// `crop_bounds` is the window already trimmed by `WINDOW_CORNER_TRIM`.
fn source_rect(captures_window: bool, crop_bounds: super::CropBounds) -> cg::Rect {
    let (x, y) = if captures_window {
        (super::WINDOW_CORNER_TRIM, super::WINDOW_CORNER_TRIM)
    } else {
        (crop_bounds.position().x(), crop_bounds.position().y())
    };
    cg::Rect::new(
        x,
        y,
        crop_bounds.size().width(),
        crop_bounds.size().height(),
    )
}

fn excluded_shareable_windows(
    excluded_windows: &[WindowId],
    content: &sc::ShareableContent,
    capture_phase: &'static str,
) -> Vec<arc::R<sc::Window>> {
    if excluded_windows.is_empty() {
        return Vec::new();
    }

    let windows = content.windows();
    let mut collected = Vec::new();
    for window_id in excluded_windows {
        let Ok(native_id) = window_id.to_string().parse::<u32>() else {
            warn!(%window_id, capture_phase, "Excluded window has an invalid native id");
            continue;
        };

        // New or protected panels can be absent from the CG on-screen list
        // while still belonging to the snapshot used to build this filter.
        if let Some(window) = windows.iter().find(|window| window.id() == native_id) {
            collected.push(window.retained());
        } else {
            warn!(
                %window_id,
                capture_phase,
                "Excluded window missing from ScreenCaptureKit shareable content"
            );
        }
    }

    info!(
        capture_phase,
        configured_excluded_windows = excluded_windows.len(),
        mapped_excluded_windows = collected.len(),
        "Mapped ScreenCaptureKit excluded windows"
    );
    collected
}

struct FrameScaler {
    session: arc::R<cidre::vt::PixelTransferSession>,
    pool: arc::R<cv::PixelBufPool>,
}

unsafe impl Send for FrameScaler {}

impl FrameScaler {
    fn new(expected_width: usize, expected_height: usize) -> Option<Self> {
        let mut session = cidre::vt::PixelTransferSession::new().ok()?;
        session.set_scaling_letter_box().ok()?;
        session.set_realtime(true).ok()?;
        let pool = create_pixel_buffer_pool(expected_width, expected_height)?;

        Some(Self { session, pool })
    }

    fn scale_frame(&self, src_sample_buf: &cm::SampleBuf) -> Option<arc::R<cm::SampleBuf>> {
        let src_image_buf = src_sample_buf.image_buf()?;
        let dst_buf = self.pool.pixel_buf().ok()?;

        self.session.transfer(src_image_buf, &dst_buf).ok()?;

        let format_desc = cm::VideoFormatDesc::with_image_buf(&dst_buf).ok()?;
        let timing = src_sample_buf.timing_info(0).ok()?;

        cm::SampleBuf::with_image_buf(&dst_buf, true, None, ptr::null(), &format_desc, &timing).ok()
    }
}

fn get_pixel_buffer_pool_size() -> usize {
    std::env::var("CAP_PIXEL_BUFFER_POOL_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(20)
}

fn create_pixel_buffer_pool(width: usize, height: usize) -> Option<arc::R<cv::PixelBufPool>> {
    let min_count = get_pixel_buffer_pool_size();

    let min_count_num = cf::Number::from_usize(min_count);
    let width_num = cf::Number::from_usize(width);
    let height_num = cf::Number::from_usize(height);
    let io_props = cf::Dictionary::new();

    let pool_attr_keys: [&cf::Type; 1] =
        [cv::pixel_buffer_pool::keys::minimum_buffer_count().as_ref()];
    let pool_attr_values: [&cf::Type; 1] = [min_count_num.as_ref()];
    let pool_attrs = cf::Dictionary::with_keys_values(&pool_attr_keys, &pool_attr_values)?;

    let pixel_buf_attr_keys: [&cf::Type; 5] = [
        cv::pixel_buffer::keys::pixel_format().as_ref(),
        cv::pixel_buffer::keys::width().as_ref(),
        cv::pixel_buffer::keys::height().as_ref(),
        cv::pixel_buffer::keys::io_surf_props().as_ref(),
        cv::pixel_buffer::keys::metal_compatibility().as_ref(),
    ];
    let pixel_buf_attr_values: [&cf::Type; 5] = [
        cv::PixelFormat::_420V.to_cf_number().as_ref(),
        width_num.as_ref(),
        height_num.as_ref(),
        io_props.as_ref(),
        cf::Boolean::value_true().as_ref(),
    ];
    let pixel_buf_attrs =
        cf::Dictionary::with_keys_values(&pixel_buf_attr_keys, &pixel_buf_attr_values)?;

    debug!(min_count, width, height, "Pixel buffer pool initialized");
    cv::PixelBufPool::new(Some(pool_attrs.as_ref()), Some(pixel_buf_attrs.as_ref())).ok()
}

fn get_screen_buffer_size() -> usize {
    std::env::var("CAP_SCREEN_BUFFER_SIZE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(15)
}

fn get_max_queue_depth() -> isize {
    std::env::var("CAP_MAX_QUEUE_DEPTH")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8)
}

fn system_audio_frame(sample_buffer: &cm::SampleBuf) -> Option<ffmpeg::frame::Audio> {
    let Ok(buf_list) = sample_buffer.audio_buf_list::<2>() else {
        warn!("Failed to extract audio buffer list from sample, dropping audio chunk");
        return None;
    };
    let Ok(slice) = buf_list.block().as_slice() else {
        warn!("Failed to get audio buffer slice, dropping audio chunk");
        return None;
    };

    let mut frame = ffmpeg::frame::Audio::new(
        ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Planar),
        sample_buffer.num_samples() as usize,
        ffmpeg::ChannelLayout::STEREO,
    );
    frame.set_rate(48_000);
    let data_bytes_size = buf_list.list().buffers[0].data_bytes_size as usize;
    for i in 0..frame.planes() {
        let start = i.saturating_mul(data_bytes_size);
        let end = start.saturating_add(data_bytes_size);
        let Some(source) = slice.get(start..end) else {
            warn!("Audio buffer slice too small for plane data, dropping audio chunk");
            return None;
        };
        let destination = frame.data_mut(i);
        if destination.len() != source.len() {
            warn!("Audio frame plane size mismatch, dropping audio chunk");
            return None;
        }
        destination.copy_from_slice(source);
    }

    Some(frame)
}

#[derive(Debug)]
pub struct CMSampleBufferCapture;

impl ScreenCaptureFormat for CMSampleBufferCapture {
    type VideoFormat = cidre::arc::R<cidre::cm::SampleBuf>;

    fn pixel_format() -> ffmpeg::format::Pixel {
        ffmpeg::format::Pixel::NV12
    }

    fn audio_info() -> AudioInfo {
        AudioInfo::new(
            ffmpeg::format::Sample::F32(ffmpeg::format::sample::Type::Planar),
            48_000,
            2,
        )
        .expect("static F32/48kHz/stereo audio config")
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SourceError {
    #[error("NoDisplay: Id '{0}'")]
    NoDisplay(DisplayId),
    #[error("AsContentFilter")]
    AsContentFilter,
    /// A window-only capture whose window is gone. A display crop would
    /// record whatever now covers the window's rectangle.
    #[error("WindowGone: Id '{0}'")]
    WindowGone(WindowId),
}

pub struct VideoFrame {
    pub sample_buf: arc::R<cm::SampleBuf>,
    pub timestamp: Timestamp,
}

impl output_pipeline::VideoFrame for VideoFrame {
    fn timestamp(&self) -> Timestamp {
        self.timestamp
    }

    fn duplicate(&self) -> Option<Self> {
        Some(Self {
            sample_buf: self.sample_buf.clone(),
            timestamp: self.timestamp,
        })
    }
}

impl ScreenCaptureConfig<CMSampleBufferCapture> {
    pub async fn to_sources(
        &self,
    ) -> anyhow::Result<(VideoSourceConfig, Option<SystemAudioSourceConfig>)> {
        let (error_tx, error_rx) = broadcast::channel(1);
        let buffer_size = get_screen_buffer_size();
        debug!(buffer_size = buffer_size, "Screen capture buffer size");
        let (video_tx, video_rx) = flume::bounded(buffer_size);
        let drop_counter: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
        let (system_audio_tx, audio_rx) = if self.system_audio {
            let (tx, rx) = mpsc::channel(128);
            (Some(tx), Some(rx))
        } else {
            (None, None)
        };
        let system_audio_drop_counter: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));
        let system_audio_frame_counter: Arc<AtomicU64> = Arc::new(AtomicU64::new(0));

        // Only the audio follows the app selection: the screen stream keeps its
        // own filter and a second, audio-only stream carries the chosen apps.
        let (mut audio_tx, app_audio_tx) = if !self.system_audio_apps.is_empty() {
            (None, system_audio_tx)
        } else {
            (system_audio_tx, None)
        };
        let stream_captures_audio = audio_tx.is_some();

        let display = Display::from_id(&self.config.display)
            .ok_or_else(|| SourceError::NoDisplay(self.config.display.clone()))?;

        let window_filter = capturable_window(&self.config, &self.excluded_windows, "initial")
            .map(|id| {
                window_content_filter(id, &self.shareable_content, "initial")
                    .ok_or_else(|| SourceError::WindowGone(id.clone()))
            })
            .transpose()?;
        let captures_window = window_filter.is_some();

        let content_filter = match window_filter {
            Some(filter) => filter,
            None => {
                let excluded_sc_windows = excluded_shareable_windows(
                    &self.excluded_windows,
                    &self.shareable_content,
                    "initial",
                );

                display
                    .raw_handle()
                    .as_content_filter_excluding_windows(
                        self.shareable_content.clone(),
                        excluded_sc_windows,
                    )
                    .ok_or(SourceError::AsContentFilter)?
            }
        };

        debug!("SCK content filter: {:?}", content_filter);

        let size = PhysicalSize::new(self.video_info.width as f64, self.video_info.height as f64);

        debug!("size: {:?}", size);

        let max_queue_depth = get_max_queue_depth();
        let queue_depth =
            ((self.config.fps as f32 / 30.0 * 5.0).ceil() as isize).clamp(3, max_queue_depth);
        debug!(
            queue_depth = queue_depth,
            max_queue_depth = max_queue_depth,
            "Screen capture queue depth"
        );

        let mut settings = scap_screencapturekit::StreamCfgBuilder::default()
            .with_width(size.width() as usize)
            .with_height(size.height() as usize)
            .with_fps(self.config.fps as f32)
            .with_shows_cursor(self.config.show_cursor)
            .with_captures_audio(stream_captures_audio)
            .with_queue_depth(queue_depth)
            .build();

        settings.set_pixel_format(cv::PixelFormat::_420V);
        settings.set_color_space_name(cg::color_space::names::srgb());
        if captures_window {
            configure_window_capture(&mut settings);
        }

        if let Some(crop_bounds) = self.config.crop_bounds {
            debug!("crop bounds: {:?}", crop_bounds);
            settings.set_src_rect(source_rect(captures_window, crop_bounds));
        }
        cap_fail::fail_err!(
            "macos::ScreenCaptureActor::new",
            ns::Error::with_domain(ns::ErrorDomain::os_status(), 69420, None)
        );

        let video_frame_counter: Arc<AtomicU32> = Arc::new(AtomicU32::new(0));

        let expected_width = self.video_info.width as usize;
        let expected_height = self.video_info.height as usize;
        let frame_scaler: Arc<Mutex<Option<FrameScaler>>> = Arc::new(Mutex::new(None));
        let scaling_logged = Arc::new(AtomicBool::new(false));
        let scaled_frame_count = Arc::new(AtomicU64::new(0));

        let (stall_health_tx, stall_health_rx) =
            tokio::sync::mpsc::channel::<output_pipeline::PipelineHealthEvent>(32);

        let rebuild_params = Arc::new(CapturerRebuildParams {
            display_id: self.config.display.clone(),
            config: self.config.clone(),
            system_audio: stream_captures_audio,
            excluded_windows: self.excluded_windows.clone(),
            video_info: self.video_info,
            video_tx: video_tx.clone(),
            audio_tx: audio_tx.clone(),
            error_tx: error_tx.clone(),
            video_frame_counter: video_frame_counter.clone(),
            drop_counter: drop_counter.clone(),
            system_audio_frame_counter: system_audio_frame_counter.clone(),
            system_audio_drop_counter: system_audio_drop_counter.clone(),
            frame_scaler: frame_scaler.clone(),
            scaling_logged: scaling_logged.clone(),
            scaled_frame_count: scaled_frame_count.clone(),
            stall_health_tx: stall_health_tx.clone(),
        });

        let builder = scap_screencapturekit::Capturer::builder(content_filter, settings)
            .with_output_sample_buf_cb({
                let video_frame_count = video_frame_counter.clone();
                let drop_counter = drop_counter.clone();
                let sys_audio_drop_counter = system_audio_drop_counter.clone();
                let sys_audio_frame_counter = system_audio_frame_counter.clone();
                let stall_health_tx = stall_health_tx.clone();
                move |frame| {
                    let sample_buffer = frame.sample_buf();

                    let mach_timestamp =
                        cm::Clock::convert_host_time_to_sys_units(sample_buffer.pts());
                    let timestamp = Timestamp::MachAbsoluteTime(
                        cap_timestamp::MachAbsoluteTimestamp::new(mach_timestamp),
                    );

                    match &frame {
                        scap_screencapturekit::Frame::Screen(frame) => {
                            let Some(image_buf) = frame.image_buf() else {
                                return;
                            };
                            if image_buf.height() == 0 || image_buf.width() == 0 {
                                return;
                            }

                            let frame_width = image_buf.width();
                            let frame_height = image_buf.height();

                            let final_sample_buf =
                                if frame_width != expected_width || frame_height != expected_height
                                {
                                    let Ok(mut scaler_guard) = frame_scaler.lock() else {
                                        drop_counter.fetch_add(1, atomic::Ordering::Relaxed);
                                        return;
                                    };

                                    if scaler_guard.is_none() {
                                        *scaler_guard =
                                            FrameScaler::new(expected_width, expected_height);

                                        if scaler_guard.is_some() {
                                            info!(
                                                expected_width,
                                                expected_height,
                                                frame_width,
                                                frame_height,
                                                "Display configuration changed, scaling frames to match original dimensions"
                                            );
                                        } else {
                                            warn!(
                                                "Failed to create frame scaler, dropping mismatched frames"
                                            );
                                        }
                                        scaling_logged.store(true, atomic::Ordering::Relaxed);
                                    }

                                    let Some(scaler) = scaler_guard.as_ref() else {
                                        drop_counter.fetch_add(1, atomic::Ordering::Relaxed);
                                        return;
                                    };

                                    let Some(scaled) = scaler.scale_frame(sample_buffer) else {
                                        drop_counter.fetch_add(1, atomic::Ordering::Relaxed);
                                        return;
                                    };

                                    let count =
                                        scaled_frame_count.fetch_add(1, atomic::Ordering::Relaxed)
                                            + 1;
                                    if count.is_multiple_of(300) {
                                        debug!(scaled_frames = count, "Scaling frames");
                                    }

                                    scaled
                                } else {
                                    if scaling_logged.swap(false, atomic::Ordering::Relaxed) {
                                        let count =
                                            scaled_frame_count.swap(0, atomic::Ordering::Relaxed);
                                        info!(
                                            scaled_frames = count,
                                            "Display dimensions restored, resuming direct capture"
                                        );
                                        if let Ok(mut guard) = frame_scaler.lock() {
                                            *guard = None;
                                        }
                                    }

                                    sample_buffer.retained()
                                };

                            cap_fail::fail_ret!("screen_capture video frame skip");

                            video_frame_count.fetch_add(1, atomic::Ordering::Relaxed);

                            match output_pipeline::send_with_stall_budget_flume(
                                &video_tx,
                                VideoFrame {
                                    sample_buf: final_sample_buf,
                                    timestamp,
                                },
                                "screen-video",
                                &stall_health_tx,
                            ) {
                                output_pipeline::StallSendOutcome::Sent => {}
                                output_pipeline::StallSendOutcome::StalledAndDropped { .. }
                                | output_pipeline::StallSendOutcome::Disconnected => {
                                    drop_counter.fetch_add(1, atomic::Ordering::Relaxed);
                                }
                            }
                        }
                        scap_screencapturekit::Frame::Audio(_) => {
                            cap_fail::fail_ret!("screen_capture audio frame skip");

                            let Some(audio_tx) = &mut audio_tx else {
                                return;
                            };

                            let Some(frame) = system_audio_frame(sample_buffer) else {
                                return;
                            };

                            match output_pipeline::send_with_stall_budget_futures(
                                audio_tx,
                                AudioFrame::new(frame, timestamp),
                                "screen-system-audio",
                                &stall_health_tx,
                            ) {
                                output_pipeline::StallSendOutcome::Sent => {
                                    sys_audio_frame_counter
                                        .fetch_add(1, atomic::Ordering::Relaxed);
                                }
                                output_pipeline::StallSendOutcome::StalledAndDropped { .. }
                                | output_pipeline::StallSendOutcome::Disconnected => {
                                    sys_audio_drop_counter
                                        .fetch_add(1, atomic::Ordering::Relaxed);
                                }
                            }
                        }
                        _ => {}
                    }
                }
            })
            .with_stop_with_err_cb({
                let video_frame_count = video_frame_counter.clone();
                let error_tx = error_tx.clone();
                move |_, err| {
                    debug!(
                        "Capturer stopping after creating {} video frames",
                        video_frame_count.load(atomic::Ordering::Relaxed)
                    );

                    let _ = error_tx.send(err.retained());
                }
            });

        let cancel_token = CancellationToken::new();
        let capturer = Capturer::new(Arc::new(builder.build()?));

        let app_audio = app_audio_tx.map(|audio_tx| {
            let (app_error_tx, app_error_rx) = broadcast::channel(1);
            let app_audio = AppAudio::new(AppAudioParams {
                display_id: self.config.display.clone(),
                bundle_ids: self.system_audio_apps.clone(),
                audio_tx,
                error_tx: app_error_tx,
                frame_counter: system_audio_frame_counter.clone(),
                drop_counter: system_audio_drop_counter.clone(),
                stall_health_tx: stall_health_tx.clone(),
            });
            (app_audio, app_error_rx)
        });

        Ok((
            VideoSourceConfig {
                inner: ChannelVideoSourceConfig::new(self.video_info, video_rx),
                capturer: capturer.clone(),
                error_rx: error_rx.resubscribe(),
                video_frame_counter: video_frame_counter.clone(),
                drop_counter,
                cancel_token: cancel_token.clone(),
                drop_guard: cancel_token.drop_guard(),
                rebuild_params,
                stall_health_rx: Some(stall_health_rx),
            },
            audio_rx.map(|rx| SystemAudioSourceConfig {
                channel: ChannelAudioSourceConfig::new(self.audio_info(), rx),
                capturer: match app_audio {
                    Some((app_audio, error_rx)) => SystemAudioCapturer::Apps(app_audio, error_rx),
                    None => SystemAudioCapturer::Screen(capturer),
                },
                screen_error_rx: error_rx,
                frame_counter: system_audio_frame_counter,
                drop_counter: system_audio_drop_counter,
            }),
        ))
    }
}

// Public

pub struct StartCapturing;

// External

pub struct NewFrame(pub scap_screencapturekit::Frame);

// Internal

pub struct CaptureError(pub arc::R<ns::Error>);

struct Capturer {
    started: Arc<AtomicBool>,
    capturer: Arc<scap_screencapturekit::Capturer>,
}

impl Clone for Capturer {
    fn clone(&self) -> Self {
        Self {
            started: self.started.clone(),
            capturer: self.capturer.clone(),
            // error_rx: self.error_rx.resubscribe(),
        }
    }
}

impl Capturer {
    fn new(capturer: Arc<scap_screencapturekit::Capturer>) -> Self {
        Self {
            started: Arc::new(AtomicBool::new(false)),
            capturer,
        }
    }

    fn mark_stopped(&self) {
        self.started.store(false, atomic::Ordering::Relaxed);
    }

    async fn start(&self) -> anyhow::Result<()> {
        if self
            .started
            .compare_exchange(
                false,
                true,
                atomic::Ordering::Relaxed,
                atomic::Ordering::Relaxed,
            )
            .is_ok()
        {
            self.capturer
                .start()
                .await
                .map_err(|err| anyhow!(format!("{err}")))?;
        }

        Ok(())
    }

    async fn stop(&self) -> anyhow::Result<()> {
        if self
            .started
            .compare_exchange(
                true,
                false,
                atomic::Ordering::Relaxed,
                atomic::Ordering::Relaxed,
            )
            .is_ok()
        {
            self.capturer.stop().await.context("capturer_stop")?;
        }

        Ok(())
    }
}

pub struct VideoSourceConfig {
    inner: ChannelVideoSourceConfig<VideoFrame>,
    capturer: Capturer,
    error_rx: broadcast::Receiver<arc::R<ns::Error>>,
    cancel_token: CancellationToken,
    drop_guard: DropGuard,
    video_frame_counter: Arc<AtomicU32>,
    drop_counter: Arc<AtomicU64>,
    rebuild_params: Arc<CapturerRebuildParams>,
    stall_health_rx: Option<tokio::sync::mpsc::Receiver<output_pipeline::PipelineHealthEvent>>,
}
pub struct VideoSource {
    inner: ChannelVideoSource<VideoFrame>,
    capturer: Capturer,
    cancel_token: CancellationToken,
    video_frame_counter: Arc<AtomicU32>,
    drop_counter: Arc<AtomicU64>,
    _drop_guard: DropGuard,
    health_tx: output_pipeline::HealthSender,
    active_capturer: Arc<Mutex<Option<Capturer>>>,
}

impl output_pipeline::VideoSource for VideoSource {
    type Config = VideoSourceConfig;
    type Frame = VideoFrame;

    async fn setup(
        config: Self::Config,
        video_tx: mpsc::Sender<Self::Frame>,
        ctx: &mut SetupCtx,
    ) -> anyhow::Result<Self>
    where
        Self: Sized,
    {
        let VideoSourceConfig {
            inner,
            capturer,
            mut error_rx,
            cancel_token,
            drop_guard,
            video_frame_counter,
            drop_counter,
            rebuild_params,
            stall_health_rx,
        } = config;

        let monitor_cancel = cancel_token.clone();
        let pipeline_cancel = ctx.stop_token();
        let stop_signal = ctx.stop_signal();
        let health_tx = ctx.health_tx().clone();

        if let Some(mut stall_rx) = stall_health_rx {
            let forward_tx = ctx.health_tx().clone();
            let forward_cancel = cancel_token.clone();
            ctx.tasks()
                .spawn("screen-capture-stall-forwarder", async move {
                    loop {
                        tokio::select! {
                            _ = forward_cancel.cancelled() => break Ok(()),
                            event = stall_rx.recv() => match event {
                                Some(ev) => output_pipeline::emit_health(&forward_tx, ev),
                                None => break Ok(()),
                            }
                        }
                    }
                });
        }
        let active_capturer: Arc<Mutex<Option<Capturer>>> = Arc::new(Mutex::new(None));
        let active_capturer_for_stop = active_capturer.clone();
        let original_capturer = capturer.clone();
        ctx.tasks().spawn("screen-capture-monitor", async move {
            let mut restart_count = 0u32;

            loop {
                select! {
                    _ = monitor_cancel.cancelled() => break Ok(()),
                    recv = error_rx.recv() => {
                        let err = match recv {
                            Ok(err) => err,
                            Err(broadcast::error::RecvError::Closed) => break Ok(()),
                            Err(broadcast::error::RecvError::Lagged(_)) => {
                                warn!("Screen capture error channel lagged; continuing");
                                continue;
                            }
                        };

                        if is_user_stop_error(err.as_ref()) {
                            if let Ok(guard) = active_capturer.lock() {
                                match guard.as_ref() {
                                    Some(c) => c.mark_stopped(),
                                    None => original_capturer.mark_stopped(),
                                }
                            } else {
                                original_capturer.mark_stopped();
                            }

                            info!(
                                "Screen capture stream stopped from macOS sharing controls"
                            );
                            stop_signal.mark_user_stopped();
                            pipeline_cancel.cancel();
                            break Ok(());
                        }

                        if is_system_stop_error(err.as_ref()) {
                            if monitor_cancel.is_cancelled() {
                                break Ok(());
                            }

                            if let Ok(guard) = active_capturer.lock() {
                                match guard.as_ref() {
                                    Some(c) => c.mark_stopped(),
                                    None => original_capturer.mark_stopped(),
                                }
                            } else {
                                original_capturer.mark_stopped();
                            }

                            if restart_count >= MAX_CAPTURE_RESTARTS {
                                error!(
                                    restart_count,
                                    max_restarts = MAX_CAPTURE_RESTARTS,
                                    "Screen capture failed after max restart attempts"
                                );
                                return Err(anyhow!(system_stop_message()));
                            }

                            restart_count += 1;
                            warn!(
                                restart_count,
                                max_restarts = MAX_CAPTURE_RESTARTS,
                                "Screen capture stream stopped by system, attempting restart"
                            );

                            output_pipeline::emit_health(
                                &health_tx,
                                output_pipeline::PipelineHealthEvent::SourceRestarting,
                            );

                            tokio::time::sleep(RESTART_DELAY).await;

                            if monitor_cancel.is_cancelled() {
                                break Ok(());
                            }

                            if Display::from_id(&rebuild_params.display_id).is_none() {
                                error!(
                                    display_id = ?rebuild_params.display_id,
                                    "Capture target display is gone; aborting restart"
                                );
                                output_pipeline::emit_health(
                                    &health_tx,
                                    output_pipeline::PipelineHealthEvent::CaptureTargetLost {
                                        target: "display".to_string(),
                                    },
                                );
                                return Err(anyhow!(
                                    "Capture target display disappeared: {:?}",
                                    rebuild_params.display_id
                                ));
                            }

                            match rebuild_capturer(&rebuild_params).await {
                                Ok(new_capturer) => {
                                    if let Ok(mut guard) = active_capturer.lock() {
                                        *guard = Some(new_capturer);
                                    }
                                    info!(
                                        restart_count,
                                        "macOS screen capture restarted successfully"
                                    );
                                    output_pipeline::emit_health(
                                        &health_tx,
                                        output_pipeline::PipelineHealthEvent::SourceRestarted,
                                    );
                                    continue;
                                }
                                Err(e) => {
                                    error!(
                                        restart_count,
                                        error = %e,
                                        "Failed to restart macOS screen capture"
                                    );
                                    return Err(anyhow!(
                                        "Screen capture restart failed after {} attempts: {e}",
                                        restart_count
                                    ));
                                }
                            }
                        }

                        return Err(anyhow!(format!("{err}")));
                    }
                }
            }
        });

        let stats_health_tx = ctx.health_tx().clone();
        ChannelVideoSource::setup(inner, video_tx, ctx)
            .await
            .map(|source| Self {
                inner: source,
                capturer,
                cancel_token,
                _drop_guard: drop_guard,
                video_frame_counter,
                drop_counter,
                health_tx: stats_health_tx,
                active_capturer: active_capturer_for_stop,
            })
    }

    fn start(&mut self) -> BoxFuture<'_, anyhow::Result<()>> {
        async move {
            self.capturer.start().await?;

            tokio::spawn({
                let video_frame_count = self.video_frame_counter.clone();
                let drop_counter = self.drop_counter.clone();
                let health_tx = self.health_tx.clone();
                async move {
                    let mut prev_frames = 0u32;
                    let mut prev_drops = 0u64;
                    loop {
                        tokio::time::sleep(Duration::from_secs(5)).await;
                        let current_frames = video_frame_count.load(atomic::Ordering::Relaxed);
                        let current_drops = drop_counter.load(atomic::Ordering::Relaxed);

                        let frame_delta = current_frames.saturating_sub(prev_frames);
                        let drop_delta = current_drops.saturating_sub(prev_drops);

                        if frame_delta > 0 {
                            let drop_rate = 100.0 * drop_delta as f64
                                / (frame_delta as f64 + drop_delta as f64);
                            if drop_rate > 5.0 {
                                warn!(
                                    frames = frame_delta,
                                    drops = drop_delta,
                                    drop_rate_pct = format!("{:.1}%", drop_rate),
                                    total_frames = current_frames,
                                    total_drops = current_drops,
                                    "Screen capture frame drop rate exceeds 5% threshold"
                                );
                                output_pipeline::emit_health(
                                    &health_tx,
                                    output_pipeline::PipelineHealthEvent::FrameDropRateHigh {
                                        source: "screen-video".to_string(),
                                        rate_pct: drop_rate,
                                    },
                                );
                            } else {
                                debug!(
                                    frames = frame_delta,
                                    drops = drop_delta,
                                    drop_rate_pct = format!("{:.1}%", drop_rate),
                                    total_frames = current_frames,
                                    "Screen capture stats"
                                );
                            }
                        }

                        prev_frames = current_frames;
                        prev_drops = current_drops;
                    }
                }
                .with_cancellation_token_owned(self.cancel_token.clone())
                .in_current_span()
            });

            Ok(())
        }
        .boxed()
    }

    fn stop(&mut self) -> BoxFuture<'_, anyhow::Result<()>> {
        async move {
            debug!(
                "Capturer stopping after creating {} video frames",
                self.video_frame_counter.load(atomic::Ordering::Relaxed)
            );

            self.cancel_token.cancel();

            let capturer_to_stop = self
                .active_capturer
                .lock()
                .ok()
                .and_then(|mut guard| guard.take());

            let target = capturer_to_stop.as_ref().unwrap_or(&self.capturer);

            let stop_result =
                tokio::time::timeout(std::time::Duration::from_secs(5), target.stop()).await;

            match stop_result {
                Ok(result) => result,
                Err(_) => {
                    error!("Screen capturer stop timed out after 5s");
                    Err(anyhow::anyhow!(
                        "Screen capturer stop timed out after 5s — native resources may not be fully released"
                    ))
                }
            }
        }
        .boxed()
    }

    fn video_info(&self) -> VideoInfo {
        self.inner.video_info()
    }
}

fn is_system_stop_error(err: &ns::Error) -> bool {
    err.code() == sc::error::code::SYSTEM_STOPPED_STREAM as ns::Integer
        && err.domain().to_string() == sc::error::domain().to_string()
}

fn is_user_stop_error(err: &ns::Error) -> bool {
    err.code() == sc::error::code::USER_STOPPED as ns::Integer
        && err.domain().to_string() == sc::error::domain().to_string()
}

fn system_stop_message() -> &'static str {
    "Screen capture stopped because macOS made the display unavailable. This commonly happens when the lid is closed or the display sleeps."
}

struct CapturerRebuildParams {
    display_id: DisplayId,
    config: Config,
    system_audio: bool,
    excluded_windows: Vec<WindowId>,
    video_info: VideoInfo,
    video_tx: flume::Sender<VideoFrame>,
    audio_tx: Option<mpsc::Sender<AudioFrame>>,
    error_tx: broadcast::Sender<arc::R<ns::Error>>,
    video_frame_counter: Arc<AtomicU32>,
    drop_counter: Arc<AtomicU64>,
    system_audio_frame_counter: Arc<AtomicU64>,
    system_audio_drop_counter: Arc<AtomicU64>,
    frame_scaler: Arc<Mutex<Option<FrameScaler>>>,
    scaling_logged: Arc<AtomicBool>,
    scaled_frame_count: Arc<AtomicU64>,
    stall_health_tx: output_pipeline::HealthSender,
}

unsafe impl Send for CapturerRebuildParams {}
unsafe impl Sync for CapturerRebuildParams {}

async fn rebuild_capturer(params: &CapturerRebuildParams) -> anyhow::Result<Capturer> {
    let shareable_content = sc::ShareableContent::current()
        .await
        .map_err(|e| anyhow!("Failed to get shareable content during restart: {e}"))?;

    let display = Display::from_id(&params.display_id)
        .ok_or_else(|| anyhow!("Display not found during restart: {:?}", params.display_id))?;

    let window_filter = capturable_window(&params.config, &params.excluded_windows, "restart")
        .map(|id| {
            window_content_filter(id, &shareable_content, "restart").ok_or_else(|| {
                anyhow!("Capture window {id} is gone; not falling back to the display")
            })
        })
        .transpose()?;
    let captures_window = window_filter.is_some();

    let content_filter = match window_filter {
        Some(filter) => filter,
        None => {
            let excluded_sc_windows =
                excluded_shareable_windows(&params.excluded_windows, &shareable_content, "restart");

            display
                .raw_handle()
                .as_content_filter_excluding_windows(shareable_content, excluded_sc_windows)
                .ok_or_else(|| anyhow!("Failed to create content filter during restart"))?
        }
    };

    let size = PhysicalSize::new(
        params.video_info.width as f64,
        params.video_info.height as f64,
    );

    let max_queue_depth = get_max_queue_depth();
    let queue_depth =
        ((params.config.fps as f32 / 30.0 * 5.0).ceil() as isize).clamp(3, max_queue_depth);

    let mut settings = scap_screencapturekit::StreamCfgBuilder::default()
        .with_width(size.width() as usize)
        .with_height(size.height() as usize)
        .with_fps(params.config.fps as f32)
        .with_shows_cursor(params.config.show_cursor)
        .with_captures_audio(params.system_audio)
        .with_queue_depth(queue_depth)
        .build();

    settings.set_pixel_format(cv::PixelFormat::_420V);
    settings.set_color_space_name(cg::color_space::names::srgb());
    if captures_window {
        configure_window_capture(&mut settings);
    }

    if let Some(crop_bounds) = params.config.crop_bounds {
        settings.set_src_rect(source_rect(captures_window, crop_bounds));
    }

    let expected_width = params.video_info.width as usize;
    let expected_height = params.video_info.height as usize;

    let video_tx = params.video_tx.clone();
    let mut audio_tx = params.audio_tx.clone();
    let error_tx = params.error_tx.clone();

    let builder = scap_screencapturekit::Capturer::builder(content_filter, settings)
        .with_output_sample_buf_cb({
            let video_frame_count = params.video_frame_counter.clone();
            let drop_counter = params.drop_counter.clone();
            let frame_scaler = params.frame_scaler.clone();
            let scaling_logged = params.scaling_logged.clone();
            let scaled_frame_count = params.scaled_frame_count.clone();
            let sys_audio_drop_counter = params.system_audio_drop_counter.clone();
            let sys_audio_frame_counter = params.system_audio_frame_counter.clone();
            let stall_health_tx = params.stall_health_tx.clone();
            move |frame| {
                let sample_buffer = frame.sample_buf();

                let mach_timestamp =
                    cm::Clock::convert_host_time_to_sys_units(sample_buffer.pts());
                let timestamp = Timestamp::MachAbsoluteTime(
                    cap_timestamp::MachAbsoluteTimestamp::new(mach_timestamp),
                );

                match &frame {
                    scap_screencapturekit::Frame::Screen(frame) => {
                        let Some(image_buf) = frame.image_buf() else {
                            return;
                        };
                        if image_buf.height() == 0 || image_buf.width() == 0 {
                            return;
                        }

                        let frame_width = image_buf.width();
                        let frame_height = image_buf.height();

                        let final_sample_buf =
                            if frame_width != expected_width || frame_height != expected_height {
                                let Ok(mut scaler_guard) = frame_scaler.lock() else {
                                    drop_counter.fetch_add(1, atomic::Ordering::Relaxed);
                                    return;
                                };

                                if scaler_guard.is_none() {
                                    *scaler_guard =
                                        FrameScaler::new(expected_width, expected_height);

                                    if scaler_guard.is_some() {
                                        info!(
                                            expected_width,
                                            expected_height,
                                            frame_width,
                                            frame_height,
                                            "Display configuration changed, scaling frames to match original dimensions"
                                        );
                                    } else {
                                        warn!(
                                            "Failed to create frame scaler, dropping mismatched frames"
                                        );
                                    }
                                    scaling_logged.store(true, atomic::Ordering::Relaxed);
                                }

                                let Some(scaler) = scaler_guard.as_ref() else {
                                    drop_counter.fetch_add(1, atomic::Ordering::Relaxed);
                                    return;
                                };

                                let Some(scaled) = scaler.scale_frame(sample_buffer) else {
                                    drop_counter.fetch_add(1, atomic::Ordering::Relaxed);
                                    return;
                                };

                                let count =
                                    scaled_frame_count.fetch_add(1, atomic::Ordering::Relaxed) + 1;
                                if count.is_multiple_of(300) {
                                    debug!(scaled_frames = count, "Scaling frames");
                                }

                                scaled
                            } else {
                                if scaling_logged.swap(false, atomic::Ordering::Relaxed) {
                                    let count =
                                        scaled_frame_count.swap(0, atomic::Ordering::Relaxed);
                                    info!(
                                        scaled_frames = count,
                                        "Display dimensions restored, resuming direct capture"
                                    );
                                    if let Ok(mut guard) = frame_scaler.lock() {
                                        *guard = None;
                                    }
                                }

                                    sample_buffer.retained()
                            };

                        cap_fail::fail_ret!("screen_capture video frame skip");

                        video_frame_count.fetch_add(1, atomic::Ordering::Relaxed);

                        match output_pipeline::send_with_stall_budget_flume(
                            &video_tx,
                            VideoFrame {
                                sample_buf: final_sample_buf,
                                timestamp,
                            },
                            "screen-video",
                            &stall_health_tx,
                        ) {
                            output_pipeline::StallSendOutcome::Sent => {}
                            output_pipeline::StallSendOutcome::StalledAndDropped { .. }
                            | output_pipeline::StallSendOutcome::Disconnected => {
                                drop_counter.fetch_add(1, atomic::Ordering::Relaxed);
                            }
                        }
                    }
                    scap_screencapturekit::Frame::Audio(_) => {
                        cap_fail::fail_ret!("screen_capture audio frame skip");

                        let Some(audio_tx) = &mut audio_tx else {
                            return;
                        };

                        let Some(frame) = system_audio_frame(sample_buffer) else {
                            return;
                        };

                        match output_pipeline::send_with_stall_budget_futures(
                            audio_tx,
                            AudioFrame::new(frame, timestamp),
                            "screen-system-audio",
                            &stall_health_tx,
                        ) {
                            output_pipeline::StallSendOutcome::Sent => {
                                sys_audio_frame_counter
                                    .fetch_add(1, atomic::Ordering::Relaxed);
                            }
                            output_pipeline::StallSendOutcome::StalledAndDropped { .. }
                            | output_pipeline::StallSendOutcome::Disconnected => {
                                sys_audio_drop_counter
                                    .fetch_add(1, atomic::Ordering::Relaxed);
                            }
                        }
                    }
                    _ => {}
                }
            }
        })
        .with_stop_with_err_cb({
            let video_frame_count = params.video_frame_counter.clone();
            move |_, err| {
                debug!(
                    "Capturer stopping after creating {} video frames",
                    video_frame_count.load(atomic::Ordering::Relaxed)
                );
                let _ = error_tx.send(err.retained());
            }
        });

    let capturer = Capturer::new(Arc::new(builder.build()?));
    capturer.start().await?;
    Ok(capturer)
}

pub struct SystemAudioSourceConfig {
    channel: ChannelAudioSourceConfig,
    capturer: SystemAudioCapturer,
    screen_error_rx: broadcast::Receiver<arc::R<ns::Error>>,
    frame_counter: Arc<AtomicU64>,
    drop_counter: Arc<AtomicU64>,
}

enum SystemAudioCapturer {
    Screen(Capturer),
    Apps(AppAudio, broadcast::Receiver<arc::R<ns::Error>>),
}

#[derive(Clone)]
struct AppAudio {
    params: Arc<AppAudioParams>,
    state: Arc<tokio::sync::Mutex<AppAudioState>>,
}

#[derive(Default)]
struct AppAudioState {
    active: Option<ActiveAppAudio>,
    started: bool,
    disabled: bool,
}

struct ActiveAppAudio {
    capturer: Capturer,
    pids: Vec<i32>,
}

struct AppAudioParams {
    display_id: DisplayId,
    bundle_ids: Vec<String>,
    audio_tx: mpsc::Sender<AudioFrame>,
    error_tx: broadcast::Sender<arc::R<ns::Error>>,
    frame_counter: Arc<AtomicU64>,
    drop_counter: Arc<AtomicU64>,
    stall_health_tx: output_pipeline::HealthSender,
}

type ContentFilterUpdate =
    std::pin::Pin<Box<dyn Future<Output = Result<(), arc::R<ns::Error>>> + Send>>;

enum AppAudioPlan {
    Keep,
    Stop,
    Start {
        capturer: Capturer,
        pids: Vec<i32>,
    },
    Update {
        update: ContentFilterUpdate,
        pids: Vec<i32>,
    },
}

impl AppAudioParams {
    fn selected_apps(
        &self,
        content: &sc::ShareableContent,
    ) -> (Vec<arc::R<sc::RunningApp>>, Vec<i32>) {
        let apps = app_audio::selected_running_apps(content, &self.bundle_ids);
        let pids = app_audio::normalized_pids(apps.iter().map(|app| app.process_id()));
        (apps, pids)
    }

    fn content_filter(
        &self,
        content: &sc::ShareableContent,
        apps: &[arc::R<sc::RunningApp>],
    ) -> anyhow::Result<arc::R<sc::ContentFilter>> {
        // The filter needs a display, but app audio is not tied to one: fall
        // back to any display if the captured one has gone.
        let display = Display::from_id(&self.display_id)
            .and_then(|display| display.raw_handle().as_sc(content.retained()))
            .or_else(|| {
                content
                    .displays()
                    .iter()
                    .next()
                    .map(|display| display.retained())
            })
            .ok_or_else(|| anyhow!("No display for the system audio app filter"))?;

        app_audio::display_including_apps_filter(&display, apps)
            .ok_or_else(|| anyhow!("Failed to create the system audio app filter"))
    }

    fn build_capturer(
        &self,
        content: &sc::ShareableContent,
        apps: &[arc::R<sc::RunningApp>],
    ) -> anyhow::Result<Capturer> {
        let filter = self.content_filter(content, apps)?;

        let settings = scap_screencapturekit::StreamCfgBuilder::default()
            .with_width(2)
            .with_height(2)
            .with_fps(1.0)
            .with_shows_cursor(false)
            .with_captures_audio(true)
            .build();

        let mut audio_tx = self.audio_tx.clone();
        let error_tx = self.error_tx.clone();
        let frame_counter = self.frame_counter.clone();
        let drop_counter = self.drop_counter.clone();
        let stall_health_tx = self.stall_health_tx.clone();

        let builder = scap_screencapturekit::Capturer::builder(filter, settings)
            .with_output_sample_buf_cb(move |frame| {
                let scap_screencapturekit::Frame::Audio(_) = &frame else {
                    return;
                };

                cap_fail::fail_ret!("screen_capture audio frame skip");

                let sample_buffer = frame.sample_buf();
                let Some(audio) = system_audio_frame(sample_buffer) else {
                    return;
                };
                let timestamp =
                    Timestamp::MachAbsoluteTime(cap_timestamp::MachAbsoluteTimestamp::new(
                        cm::Clock::convert_host_time_to_sys_units(sample_buffer.pts()),
                    ));

                match output_pipeline::send_with_stall_budget_futures(
                    &mut audio_tx,
                    AudioFrame::new(audio, timestamp),
                    "screen-system-audio",
                    &stall_health_tx,
                ) {
                    output_pipeline::StallSendOutcome::Sent => {
                        frame_counter.fetch_add(1, atomic::Ordering::Relaxed);
                    }
                    output_pipeline::StallSendOutcome::StalledAndDropped { .. }
                    | output_pipeline::StallSendOutcome::Disconnected => {
                        drop_counter.fetch_add(1, atomic::Ordering::Relaxed);
                    }
                }
            })
            .with_stop_with_err_cb(move |_, err| {
                let _ = error_tx.send(err.retained());
            });

        Ok(Capturer::new(Arc::new(builder.build()?)))
    }

    /// Decides what the app stream needs for the apps running now. Built
    /// synchronously, so no ScreenCaptureKit object outlives it across an await.
    fn plan(
        &self,
        content: &sc::ShareableContent,
        state: &AppAudioState,
        rebuild: bool,
    ) -> anyhow::Result<AppAudioPlan> {
        let (apps, pids) = self.selected_apps(content);
        let active = state.active.as_ref();

        Ok(
            match app_audio::next_step(
                active.map(|active| active.pids.as_slice()),
                &pids,
                state.started,
                rebuild,
            ) {
                app_audio::AppAudioStep::Keep => AppAudioPlan::Keep,
                app_audio::AppAudioStep::Stop => AppAudioPlan::Stop,
                app_audio::AppAudioStep::Update => {
                    let filter = self.content_filter(content, &apps)?;
                    let update = active
                        .map(|active| active.capturer.capturer.update_content_filter(&filter))
                        .ok_or_else(|| anyhow!("No app audio stream to update"))?;
                    AppAudioPlan::Update {
                        update: Box::pin(update),
                        pids,
                    }
                }
                app_audio::AppAudioStep::Start => AppAudioPlan::Start {
                    capturer: self.build_capturer(content, &apps)?,
                    pids,
                },
            },
        )
    }
}

impl AppAudio {
    fn new(params: AppAudioParams) -> Self {
        Self {
            params: Arc::new(params),
            state: Arc::default(),
        }
    }

    async fn shareable_content() -> Option<arc::R<sc::ShareableContent>> {
        match tokio::time::timeout(
            APP_AUDIO_TIMEOUT,
            app_audio::shareable_content_with_offscreen_apps(),
        )
        .await
        {
            Ok(Ok(content)) => Some(content),
            Ok(Err(error)) => {
                warn!(%error, "Could not list apps for system audio");
                None
            }
            Err(_) => {
                warn!("Listing apps for system audio timed out");
                None
            }
        }
    }

    async fn sync(
        &self,
        content: arc::R<sc::ShareableContent>,
        rebuild: bool,
    ) -> anyhow::Result<()> {
        let mut state = self.state.lock().await;
        if state.disabled {
            return Ok(());
        }

        let plan = self.params.plan(&content, &state, rebuild)?;
        drop(content);

        match plan {
            AppAudioPlan::Keep => {}
            AppAudioPlan::Stop => {
                if let Some(active) = state.active.take() {
                    info!(previous = ?active.pids, "Selected apps closed, pausing app audio");
                    if let Err(error) = active.capturer.stop().await {
                        warn!(%error, "Failed to stop the app audio stream");
                    }
                }
            }
            AppAudioPlan::Start { capturer, pids } => {
                if state.started {
                    tokio::time::timeout(APP_AUDIO_TIMEOUT, capturer.start())
                        .await
                        .map_err(|_| anyhow!("Starting the app audio stream timed out"))??;
                }
                info!(
                    selected = ?self.params.bundle_ids,
                    ?pids,
                    "Capturing system audio from selected apps"
                );
                state.active = Some(ActiveAppAudio { capturer, pids });
            }
            AppAudioPlan::Update { update, pids } => {
                match tokio::time::timeout(APP_AUDIO_TIMEOUT, update).await {
                    Ok(Ok(())) => {
                        if let Some(active) = state.active.as_mut() {
                            info!(
                                previous = ?active.pids,
                                current = ?pids,
                                "Selected apps changed, updated system audio filter"
                            );
                            active.pids = pids;
                        }
                    }
                    Ok(Err(error)) => {
                        warn!(%error, "Failed to update the system audio app filter")
                    }
                    Err(_) => warn!("Updating the system audio app filter timed out"),
                }
            }
        }

        Ok(())
    }

    async fn refresh(self, cancel: CancellationToken) -> anyhow::Result<()> {
        loop {
            if let Some(content) = Self::shareable_content().await
                && let Err(error) = self.sync(content, false).await
            {
                warn!(%error, "Could not follow the selected apps for system audio");
            }

            tokio::select! {
                _ = cancel.cancelled() => return Ok(()),
                _ = tokio::time::sleep(APP_AUDIO_REFRESH_INTERVAL) => {}
            }
        }
    }

    async fn start(&self) {
        let mut state = self.state.lock().await;
        state.started = true;
        if let Some(active) = &state.active
            && let Err(error) = active.capturer.start().await
        {
            warn!(%error, "Failed to start the app audio stream, recording silence");
            state.active = None;
        }
    }

    async fn stop(&self) {
        let mut state = self.state.lock().await;
        state.started = false;
        state.disabled = true;
        if let Some(active) = state.active.take()
            && let Err(error) = active.capturer.stop().await
        {
            warn!(%error, "Failed to stop the app audio stream");
        }
    }

    /// App audio is optional: when it can't be kept running, the track records
    /// silence instead of failing the recording.
    async fn disable(&self) {
        let mut state = self.state.lock().await;
        state.disabled = true;
        if let Some(active) = state.active.take()
            && let Err(error) = active.capturer.stop().await
        {
            debug!(%error, "Failed to stop the app audio stream");
        }
    }

    async fn restart(&self, cancel: &CancellationToken, restart_count: &mut u32) -> bool {
        while *restart_count < MAX_CAPTURE_RESTARTS {
            *restart_count += 1;
            warn!(
                restart_count = *restart_count,
                max_restarts = MAX_CAPTURE_RESTARTS,
                "App audio stream stopped by system, attempting restart"
            );

            tokio::select! {
                _ = cancel.cancelled() => return true,
                _ = tokio::time::sleep(RESTART_DELAY) => {}
            }

            let Some(content) = Self::shareable_content().await else {
                continue;
            };
            match self.sync(content, true).await {
                Ok(()) => {
                    info!(restart_count = *restart_count, "App audio stream restarted");
                    return true;
                }
                Err(error) => warn!(%error, "Failed to restart the app audio stream"),
            }
        }

        false
    }

    async fn monitor(
        self,
        mut app_error_rx: broadcast::Receiver<arc::R<ns::Error>>,
        mut screen_error_rx: broadcast::Receiver<arc::R<ns::Error>>,
        cancel: CancellationToken,
        pipeline_cancel: CancellationToken,
        stop_signal: output_pipeline::PipelineStopSignal,
    ) -> anyhow::Result<()> {
        let mut screen_errors_open = true;
        let mut restart_count = 0u32;

        loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => break,
                result = screen_error_rx.recv(), if screen_errors_open => match result {
                    Ok(err) if is_user_stop_error(err.as_ref()) => {
                        info!("Screen capture stopped from macOS sharing controls, stopping app audio");
                        stop_signal.mark_user_stopped();
                        pipeline_cancel.cancel();
                        break;
                    }
                    Ok(_) | Err(broadcast::error::RecvError::Lagged(_)) => {}
                    Err(broadcast::error::RecvError::Closed) => screen_errors_open = false,
                },
                result = app_error_rx.recv() => {
                    let err = match result {
                        Ok(err) => err,
                        Err(broadcast::error::RecvError::Lagged(_)) => continue,
                        Err(broadcast::error::RecvError::Closed) => break,
                    };

                    if let Some(active) = &self.state.lock().await.active {
                        active.capturer.mark_stopped();
                    }

                    if is_user_stop_error(err.as_ref()) {
                        info!("App audio stream stopped from macOS sharing controls");
                        stop_signal.mark_user_stopped();
                        pipeline_cancel.cancel();
                        break;
                    }

                    if !is_system_stop_error(err.as_ref()) {
                        error!(%err, "App audio stream failed, recording silence");
                        self.disable().await;
                        continue;
                    }

                    if !self.restart(&cancel, &mut restart_count).await {
                        error!("App audio stream could not be restarted, recording silence");
                        self.disable().await;
                    }
                }
            }
        }

        Ok(())
    }
}

enum SystemAudioControl {
    Screen(Capturer),
    Apps(AppAudio),
}

pub struct SystemAudioSource {
    inner: ChannelAudioSource,
    capturer: SystemAudioControl,
    cancel_token: CancellationToken,
}

async fn monitor_screen_stream_audio(
    capturer: Capturer,
    mut error_rx: broadcast::Receiver<arc::R<ns::Error>>,
    cancel: CancellationToken,
    pipeline_cancel: CancellationToken,
    stop_signal: output_pipeline::PipelineStopSignal,
) -> anyhow::Result<()> {
    let mut system_stop_count = 0u32;

    loop {
        tokio::select! {
            biased;
            _ = cancel.cancelled() => break,
            result = error_rx.recv() => {
                match result {
                    Ok(err) => {
                        if is_user_stop_error(err.as_ref()) {
                            capturer.mark_stopped();
                            info!(
                                "Screen capture audio stream stopped from macOS sharing controls"
                            );
                            stop_signal.mark_user_stopped();
                            pipeline_cancel.cancel();
                            break;
                        }
                        if is_system_stop_error(err.as_ref()) {
                            system_stop_count += 1;
                            if system_stop_count > MAX_CAPTURE_RESTARTS {
                                warn!("Screen capture audio stream stopped by the system (exhausted restarts)");
                                return Err(anyhow!(system_stop_message()));
                            }
                            warn!(
                                system_stop_count,
                                "Screen capture audio stream stopped by system, waiting for restart"
                            );
                            tokio::time::sleep(RESTART_DELAY + Duration::from_secs(1)).await;
                            continue;
                        }
                        return Err(anyhow!("{err}"));
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                    Err(broadcast::error::RecvError::Lagged(_)) => continue,
                }
            }
        }
    }

    Ok(())
}

impl output_pipeline::AudioSource for SystemAudioSource {
    type Config = SystemAudioSourceConfig;

    fn setup(
        config: Self::Config,
        tx: mpsc::Sender<AudioFrame>,
        ctx: &mut SetupCtx,
    ) -> impl Future<Output = anyhow::Result<Self>> + 'static
    where
        Self: Sized,
    {
        let SystemAudioSourceConfig {
            channel,
            capturer,
            screen_error_rx,
            frame_counter,
            drop_counter,
        } = config;

        let cancel_token = CancellationToken::new();
        let pipeline_cancel = ctx.stop_token();
        let stop_signal = ctx.stop_signal();

        let capturer = match capturer {
            SystemAudioCapturer::Screen(capturer) => {
                ctx.tasks().spawn(
                    "system-audio",
                    monitor_screen_stream_audio(
                        capturer.clone(),
                        screen_error_rx,
                        cancel_token.child_token(),
                        pipeline_cancel,
                        stop_signal,
                    ),
                );
                SystemAudioControl::Screen(capturer)
            }
            SystemAudioCapturer::Apps(app_audio, app_error_rx) => {
                ctx.tasks().spawn(
                    "system-audio-apps",
                    app_audio.clone().refresh(cancel_token.child_token()),
                );
                ctx.tasks().spawn(
                    "system-audio",
                    app_audio.clone().monitor(
                        app_error_rx,
                        screen_error_rx,
                        cancel_token.child_token(),
                        pipeline_cancel,
                        stop_signal,
                    ),
                );
                SystemAudioControl::Apps(app_audio)
            }
        };

        let stats_cancel = cancel_token.clone();
        tokio::spawn(
            async move {
                let mut last_log = Instant::now();
                loop {
                    tokio::time::sleep(Duration::from_secs(5)).await;
                    let captured = frame_counter.load(atomic::Ordering::Relaxed);
                    let dropped = drop_counter.load(atomic::Ordering::Relaxed);

                    if dropped > 0 {
                        let total = captured + dropped;
                        let drop_pct = if total > 0 {
                            100.0 * dropped as f64 / total as f64
                        } else {
                            0.0
                        };

                        if last_log.elapsed() >= Duration::from_secs(5) {
                            warn!(
                                captured = captured,
                                dropped = dropped,
                                drop_pct = format!("{:.1}%", drop_pct),
                                "System audio dropping frames due to full channel"
                            );
                            last_log = Instant::now();
                        }
                    } else if captured > 0 {
                        debug!(captured = captured, "System audio frames captured");
                    }
                }
            }
            .with_cancellation_token_owned(stats_cancel)
            .in_current_span(),
        );

        ChannelAudioSource::setup(channel, tx, ctx).map({
            let cancel_token = cancel_token.clone();
            move |v| {
                v.map(|source| Self {
                    inner: source,
                    capturer,
                    cancel_token,
                })
            }
        })
    }

    async fn start(&mut self) -> anyhow::Result<()> {
        match &self.capturer {
            SystemAudioControl::Screen(capturer) => capturer.start().await,
            SystemAudioControl::Apps(app_audio) => {
                app_audio.start().await;
                Ok(())
            }
        }
    }

    async fn stop(&mut self) -> anyhow::Result<()> {
        self.cancel_token.cancel();
        match &self.capturer {
            SystemAudioControl::Screen(capturer) => capturer.stop().await,
            SystemAudioControl::Apps(app_audio) => {
                app_audio.stop().await;
                Ok(())
            }
        }
    }

    fn audio_info(&self) -> AudioInfo {
        self.inner.audio_info()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn window_id(id: &str) -> WindowId {
        id.parse().unwrap()
    }

    fn window_config(window: Option<WindowId>) -> Config {
        Config {
            display: "1".parse().unwrap(),
            crop_bounds: None,
            fps: 60,
            show_cursor: true,
            window,
        }
    }

    #[test]
    fn initial_capture_uses_window_filter_for_non_excluded_window() {
        let config = window_config(Some(window_id("247")));
        let excluded = vec![window_id("12"), window_id("99")];

        assert_eq!(
            capturable_window(&config, &excluded, "initial"),
            Some(&window_id("247"))
        );
    }

    #[test]
    fn initial_capture_keeps_excluded_window_on_display_path() {
        let config = window_config(Some(window_id("247")));
        let excluded = vec![window_id("12"), window_id("247")];

        assert_eq!(capturable_window(&config, &excluded, "initial"), None);
    }

    #[test]
    fn restart_capture_keeps_excluded_window_on_display_path() {
        let config = window_config(Some(window_id("247")));
        let excluded = vec![window_id("247")];
        let (restart_config, restart_excluded) = (config.clone(), excluded.clone());

        assert_eq!(
            capturable_window(&restart_config, &restart_excluded, "restart"),
            None
        );
    }

    #[test]
    fn restart_capture_uses_window_filter_for_non_excluded_window() {
        let config = window_config(Some(window_id("247")));
        let excluded = vec![window_id("12")];
        let (restart_config, restart_excluded) = (config.clone(), excluded.clone());

        assert_eq!(
            capturable_window(&restart_config, &restart_excluded, "restart"),
            Some(&window_id("247"))
        );
    }

    #[test]
    fn non_window_targets_never_use_window_filter() {
        let config = window_config(None);

        assert_eq!(capturable_window(&config, &[], "initial"), None);
        assert_eq!(capturable_window(&config, &[], "restart"), None);
    }
}
