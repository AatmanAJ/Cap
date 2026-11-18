use cap_enc_ffmpeg::segmented_stream::{SegmentedVideoEncoder, SegmentedVideoEncoderConfig};
use cap_rendering::decoder::{DecoderType, spawn_decoder};
use std::{path::Path, time::Duration};

const FPS: u32 = 30;

fn luma(index: u64) -> u8 {
    40 + (index * 60) as u8
}

fn encode(directory: &Path, count: u64) {
    ffmpeg::init().unwrap();
    let mut encoder = SegmentedVideoEncoder::init(
        directory.to_path_buf(),
        cap_media_info::VideoInfo {
            pixel_format: cap_media_info::Pixel::NV12,
            width: 160,
            height: 120,
            time_base: ffmpeg::Rational(1, 1_000_000),
            frame_rate: ffmpeg::Rational(FPS as i32, 1),
        },
        SegmentedVideoEncoderConfig::default(),
    )
    .unwrap();
    for index in 0..count {
        let mut frame = ffmpeg::frame::Video::new(ffmpeg::format::Pixel::NV12, 160, 120);
        frame.data_mut(0).fill(luma(index));
        frame.data_mut(1).fill(128);
        encoder
            .queue_frame(
                frame,
                Duration::from_micros(index * 1_000_000 / u64::from(FPS)),
            )
            .unwrap();
    }
    encoder.finish().unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn software_decoder_serves_every_frame_of_clips_shorter_than_the_thread_delay() {
    for count in [1_u64, 3] {
        let directory = tempfile::tempdir().unwrap();
        let source = directory.path().join("segments");
        encode(&source, count);
        let decoder = spawn_decoder("screen", source, FPS, 0.0, true)
            .await
            .unwrap();
        assert!(matches!(
            decoder.decoder_type(),
            DecoderType::FFmpegSoftware | DecoderType::FFmpegHardware
        ));
        for index in (0..count).chain((0..count).rev()) {
            let frame = decoder
                .get_frame(index as f32 / FPS as f32)
                .await
                .unwrap_or_else(|| panic!("frame {index} of {count}"));
            let actual = i32::from(frame.y_plane().unwrap()[0]);
            assert!(
                (actual - i32::from(luma(index))).abs() <= 8,
                "frame {index} of {count} decoded luma {actual}"
            );
        }
    }
}
