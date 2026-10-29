#![cfg(target_os = "macos")]

use std::{sync::Arc, time::Duration};

use cap_media_info::{RawVideoFormat, VideoInfo};
use cap_project::{RecordingMeta, StudioRecordingMeta};
use cap_recording::{
    NativeCameraFrame, feeds::camera::synthetic::SyntheticCamera,
    sources::screen_capture::ScreenCaptureTarget, studio_recording,
};
use cap_timestamp::{MachAbsoluteTimestamp, Timestamp};
use cidre::{cm, cv};

const WIDTH: usize = 590;
const HEIGHT: usize = 1278;
const FPS: u32 = 30;

fn device_frame(index: i64, width: usize, height: usize) -> NativeCameraFrame {
    let buf = cv::PixelBuf::new(width, height, cv::PixelFormat::_420V, None).unwrap();
    let desc = cm::VideoFormatDesc::with_image_buf(&buf).unwrap();
    let timing = cm::SampleTimingInfo {
        duration: cm::Time::new(1, FPS as i32),
        pts: cm::Time::new(index, FPS as i32),
        dts: cm::Time::invalid(),
    };
    let sample_buf =
        cm::SampleBuf::with_image_buf(&buf, true, None, std::ptr::null(), &desc, &timing).unwrap();
    NativeCameraFrame {
        sample_buf,
        timestamp: Timestamp::MachAbsoluteTime(MachAbsoluteTimestamp::now()),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn mobile_device_feed_is_recorded_as_the_screen_track() {
    record_device(false).await;
}

#[tokio::test(flavor = "multi_thread")]
async fn webcam_still_records_its_own_track_next_to_a_device_screen() {
    record_device(true).await;
}

async fn record_device(with_webcam: bool) {
    let dir = tempfile::tempdir().unwrap();
    let project = dir.path().join("device.cap");

    let device = SyntheticCamera::spawn(
        "00008110-000A1C2E3F40001E",
        VideoInfo::from_raw(RawVideoFormat::Nv12, WIDTH as u32, HEIGHT as u32, FPS),
    );
    let lock = Arc::new(device.lock().await.unwrap());
    let webcam = with_webcam.then(|| {
        SyntheticCamera::spawn(
            "FaceTime HD Camera",
            VideoInfo::from_raw(RawVideoFormat::Nv12, 320, 240, FPS),
        )
    });

    let mut builder = studio_recording::Actor::builder(
        project.clone(),
        ScreenCaptureTarget::MobileDevice {
            id: "00008110-000A1C2E3F40001E".to_string(),
        },
    )
    .with_mobile_device_feed(lock)
    .with_custom_cursor(true)
    .with_keyboard_capture(true);
    if let Some(webcam) = &webcam {
        builder = builder.with_camera_feed(Arc::new(webcam.lock().await.unwrap()));
    }
    let handle = builder
        .build(None)
        .await
        .expect("a device target needs no ScreenCaptureKit content");

    for index in 0..(FPS as i64) {
        device.push_native(device_frame(index, WIDTH, HEIGHT)).await;
        if let Some(webcam) = &webcam {
            webcam.push_native(device_frame(index, 320, 240)).await;
        }
        tokio::time::sleep(Duration::from_millis(1000 / u64::from(FPS))).await;
    }

    let completed = handle.stop().await.unwrap();
    let meta = RecordingMeta::load_for_project(&completed.project_path).unwrap();
    let Some(StudioRecordingMeta::MultipleSegments { inner }) = meta.studio_meta() else {
        panic!("expected a multi-segment Studio recording");
    };
    let segment = &inner.segments[0];
    assert!(
        segment.cursor.is_none(),
        "a device screen has no cursor data"
    );
    assert!(
        segment.keyboard.is_none(),
        "a device screen has no key presses"
    );
    assert_eq!(segment.camera.is_some(), with_webcam);

    let display = segment.display.path.to_path(&completed.project_path);
    let input = ffmpeg::format::input(&display).unwrap();
    let stream = input.streams().best(ffmpeg::media::Type::Video).unwrap();
    let decoder = ffmpeg::codec::Context::from_parameters(stream.parameters())
        .unwrap()
        .decoder()
        .video()
        .unwrap();
    assert_eq!(
        (decoder.width() as usize, decoder.height() as usize),
        (WIDTH, HEIGHT)
    );
    assert!(input.duration() > 0);
}
