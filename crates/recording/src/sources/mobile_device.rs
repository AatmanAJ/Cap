use crate::{
    feeds::camera::CameraFeedLock,
    output_pipeline::{NativeCameraFrame, SetupCtx, VideoSource},
    sources::NativeCamera,
};
use cap_media_info::VideoInfo;
use futures::{FutureExt, SinkExt, StreamExt, channel::mpsc, future::BoxFuture};
use std::sync::{Arc, Mutex};

/// The writer of a segment is sized once, so a device screen that rotates
/// mid-recording cannot keep going into the same file. The source tracks the
/// size it was opened with; the first frame of another size is recorded as
/// the device's new size and requests a segment cut, and frames are held back
/// from the old writer until the new segment takes over.
pub struct MobileDeviceScreenState {
    size: Mutex<(u32, u32)>,
    cut_tx: flume::Sender<()>,
}

impl MobileDeviceScreenState {
    pub fn new(initial_size: (u32, u32)) -> (Arc<Self>, flume::Receiver<()>) {
        let (cut_tx, cut_rx) = flume::unbounded();
        (
            Arc::new(Self {
                size: Mutex::new(initial_size),
                cut_tx,
            }),
            cut_rx,
        )
    }

    pub fn size(&self) -> (u32, u32) {
        *self.size.lock().unwrap_or_else(|error| error.into_inner())
    }

    fn resized(&self, size: (u32, u32)) {
        *self.size.lock().unwrap_or_else(|error| error.into_inner()) = size;
        let _ = self.cut_tx.send(());
    }
}

pub struct MobileDeviceScreenConfig {
    pub feed: Arc<CameraFeedLock>,
    pub state: Arc<MobileDeviceScreenState>,
}

pub struct MobileDeviceScreen {
    inner: NativeCamera,
    video_info: VideoInfo,
}

fn frame_size(frame: &NativeCameraFrame) -> Option<(u32, u32)> {
    let image = frame.sample_buf.image_buf()?;
    Some((image.width() as u32, image.height() as u32))
}

async fn forward_until_resized(
    mut rx: mpsc::Receiver<NativeCameraFrame>,
    mut video_tx: mpsc::Sender<NativeCameraFrame>,
    segment_size: (u32, u32),
    state: Arc<MobileDeviceScreenState>,
) {
    while let Some(frame) = rx.next().await {
        if let Some(size) = frame_size(&frame)
            && size != segment_size
        {
            tracing::info!(
                from = ?segment_size,
                to = ?size,
                "Device screen changed size, starting a new segment"
            );
            state.resized(size);
            while rx.next().await.is_some() {}
            return;
        }
        if video_tx.send(frame).await.is_err() {
            return;
        }
    }
}

impl VideoSource for MobileDeviceScreen {
    type Config = MobileDeviceScreenConfig;
    type Frame = NativeCameraFrame;

    async fn setup(
        config: Self::Config,
        video_tx: mpsc::Sender<Self::Frame>,
        ctx: &mut SetupCtx,
    ) -> anyhow::Result<Self> {
        let (width, height) = config.state.size();
        let video_info = VideoInfo {
            width,
            height,
            ..*config.feed.video_info()
        };
        let (inner_tx, inner_rx) = mpsc::channel(8);
        let inner = NativeCamera::setup(config.feed, inner_tx, ctx).await?;
        drop(tokio::spawn(forward_until_resized(
            inner_rx,
            video_tx,
            (width, height),
            config.state,
        )));
        Ok(Self { inner, video_info })
    }

    fn video_info(&self) -> VideoInfo {
        self.video_info
    }

    fn start(&mut self) -> BoxFuture<'_, anyhow::Result<()>> {
        self.inner.start()
    }

    fn stop(&mut self) -> BoxFuture<'_, anyhow::Result<()>> {
        self.inner.stop().boxed()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cap_timestamp::{MachAbsoluteTimestamp, Timestamp};
    use cidre::{cm, cv};

    fn frame(width: usize, height: usize) -> NativeCameraFrame {
        let buf = cv::PixelBuf::new(width, height, cv::PixelFormat::_420V, None).unwrap();
        let desc = cm::VideoFormatDesc::with_image_buf(&buf).unwrap();
        let timing = cm::SampleTimingInfo {
            duration: cm::Time::new(1, 30),
            pts: cm::Time::new(0, 30),
            dts: cm::Time::invalid(),
        };
        let sample_buf =
            cm::SampleBuf::with_image_buf(&buf, true, None, std::ptr::null(), &desc, &timing)
                .unwrap();
        NativeCameraFrame {
            sample_buf,
            timestamp: Timestamp::MachAbsoluteTime(MachAbsoluteTimestamp::now()),
        }
    }

    #[tokio::test]
    async fn a_rotated_frame_requests_a_cut_and_never_reaches_the_old_writer() {
        let (state, cut_rx) = MobileDeviceScreenState::new((60, 130));
        let (mut in_tx, in_rx) = mpsc::channel(8);
        let (out_tx, mut out_rx) = mpsc::channel(8);
        let task = tokio::spawn(forward_until_resized(
            in_rx,
            out_tx,
            (60, 130),
            state.clone(),
        ));

        in_tx.send(frame(60, 130)).await.unwrap();
        in_tx.send(frame(60, 130)).await.unwrap();
        in_tx.send(frame(130, 60)).await.unwrap();
        in_tx.send(frame(130, 60)).await.unwrap();
        drop(in_tx);
        task.await.unwrap();

        let mut forwarded = Vec::new();
        while let Some(frame) = out_rx.next().await {
            forwarded.push(frame_size(&frame).unwrap());
        }
        assert_eq!(forwarded, [(60, 130), (60, 130)]);
        assert_eq!(state.size(), (130, 60));
        assert_eq!(cut_rx.try_iter().count(), 1);
    }

    #[tokio::test]
    async fn same_size_frames_pass_through_without_a_cut() {
        let (state, cut_rx) = MobileDeviceScreenState::new((60, 130));
        let (mut in_tx, in_rx) = mpsc::channel(8);
        let (out_tx, mut out_rx) = mpsc::channel(8);
        let task = tokio::spawn(forward_until_resized(
            in_rx,
            out_tx,
            (60, 130),
            state.clone(),
        ));
        for _ in 0..4 {
            in_tx.send(frame(60, 130)).await.unwrap();
        }
        drop(in_tx);
        task.await.unwrap();
        let mut count = 0;
        while out_rx.next().await.is_some() {
            count += 1;
        }
        assert_eq!(count, 4);
        assert!(cut_rx.try_recv().is_err());
    }
}
