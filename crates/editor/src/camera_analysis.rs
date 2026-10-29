use cap_project::StudioRecordingMeta;
use cap_rendering::{
    CameraAnalysis, SegmentSources, SourceFingerprint, SpeechTrack, segment_sources,
};
use std::{
    path::Path,
    sync::atomic::{AtomicBool, Ordering},
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CameraAnalysisRequest {
    pub speech: bool,
    pub faces: bool,
}

impl CameraAnalysisRequest {
    pub fn for_project(project: &cap_project::ProjectConfiguration) -> Self {
        Self {
            speech: project.wants_speech_analysis(),
            faces: project.wants_face_analysis(),
        }
    }

    pub fn is_empty(self) -> bool {
        !self.speech && !self.faces
    }

    pub fn missing_from(self, analysis: &CameraAnalysis, sources: &[SegmentSources]) -> Self {
        Self {
            speech: self.speech && !analysis.has_speech(sources),
            faces: self.faces && !analysis.has_faces(sources),
        }
    }
}

/// Runs the speech and face passes a project's camera settings need, reusing
/// whatever `camera-analysis.json` already holds for unchanged sources, and
/// writes the result back next to the project.
pub fn analyze_camera(
    project_path: &Path,
    meta: &StudioRecordingMeta,
    request: CameraAnalysisRequest,
    cancelled: &AtomicBool,
) -> Result<CameraAnalysis, String> {
    let sources = segment_sources(project_path, meta);
    let mut analysis = CameraAnalysis::load(project_path, &sources);
    let missing = request.missing_from(&analysis, &sources);
    if missing.is_empty() {
        return Ok(analysis);
    }
    for (index, (segment, source)) in analysis.segments.iter_mut().zip(&sources).enumerate() {
        if missing.speech
            && segment.speech.is_none()
            && let Some(mic) = &source.mic
            && let Some(fingerprint) = SourceFingerprint::of(mic)
        {
            let started = std::time::Instant::now();
            let spans =
                match cap_audio::detect_speech_in_file(mic, || cancelled.load(Ordering::Relaxed)) {
                    Ok(spans) => spans,
                    Err(_) if cancelled.load(Ordering::Relaxed) => return Err("cancelled".into()),
                    Err(error) => {
                        tracing::warn!(%error, segment = index, "Speech analysis failed");
                        Vec::new()
                    }
                };
            tracing::info!(
                segment = index,
                spans = spans.len(),
                elapsed_ms = started.elapsed().as_millis() as u64,
                "Analysed camera-silence speech"
            );
            segment.speech = Some(SpeechTrack {
                source: fingerprint,
                spans,
            });
        }
        #[cfg(target_os = "macos")]
        if missing.faces
            && segment.faces.is_none()
            && let Some(camera) = &source.camera
            && let Some(fingerprint) = SourceFingerprint::of(camera)
        {
            let started = std::time::Instant::now();
            let samples = match faces::track(camera, cancelled) {
                Ok(samples) => samples,
                Err(_) if cancelled.load(Ordering::Relaxed) => return Err("cancelled".into()),
                Err(error) => {
                    tracing::warn!(%error, segment = index, "Face analysis failed");
                    Vec::new()
                }
            };
            tracing::info!(
                segment = index,
                samples = samples.len(),
                faces = samples.iter().flatten().count(),
                elapsed_ms = started.elapsed().as_millis() as u64,
                "Analysed camera face track"
            );
            segment.faces = Some(cap_rendering::FaceTrack {
                source: fingerprint,
                interval: cap_rendering::FACE_SAMPLE_INTERVAL_SECS,
                samples,
            });
        }
    }
    if let Err(error) = analysis.save(project_path) {
        tracing::warn!(%error, "Could not save camera analysis");
    }
    Ok(analysis)
}

#[cfg(target_os = "macos")]
mod faces {
    use cap_rendering::{FACE_SAMPLE_INTERVAL_SECS, FaceBox};
    use std::{
        path::Path,
        sync::atomic::{AtomicBool, Ordering},
    };

    const ANALYSIS_WIDTH: u32 = 480;

    pub(super) fn track(
        path: &Path,
        cancelled: &AtomicBool,
    ) -> Result<Vec<Option<FaceBox>>, String> {
        let mut input = ffmpeg::format::input(path).map_err(|e| format!("open camera / {e}"))?;
        let stream = input
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or("camera has no video stream")?;
        let stream_index = stream.index();
        let time_base = f64::from(stream.time_base());
        let start = match stream.start_time() {
            ffmpeg::ffi::AV_NOPTS_VALUE => 0,
            start => start,
        };
        let mut context = ffmpeg::codec::Context::from_parameters(stream.parameters())
            .map_err(|e| format!("camera decoder / {e}"))?;
        context.set_threading(ffmpeg::threading::Config {
            kind: ffmpeg::threading::Type::Frame,
            count: 4,
        });
        let mut decoder = context
            .decoder()
            .video()
            .map_err(|e| format!("camera decoder / {e}"))?;
        let width = ANALYSIS_WIDTH.min(decoder.width()).max(2) & !1;
        let height = ((u64::from(decoder.height()) * u64::from(width)
            / u64::from(decoder.width().max(1))) as u32)
            .max(2)
            & !1;
        let mut scaler = ffmpeg::software::scaling::Context::get(
            decoder.format(),
            decoder.width(),
            decoder.height(),
            ffmpeg::format::Pixel::BGRA,
            width,
            height,
            ffmpeg::software::scaling::Flags::BILINEAR,
        )
        .map_err(|e| format!("camera scaler / {e}"))?;

        let mut detector = cap_camera_effects::FaceDetector::new();
        let mut samples: Vec<Option<FaceBox>> = Vec::new();
        let mut decoded = ffmpeg::frame::Video::empty();
        let mut scaled = ffmpeg::frame::Video::empty();
        let mut pixels = Vec::new();
        let mut process = |decoder: &mut ffmpeg::decoder::Video| -> Result<(), String> {
            while decoder.receive_frame(&mut decoded).is_ok() {
                if cancelled.load(Ordering::Relaxed) {
                    return Err("cancelled".into());
                }
                let Some(pts) = decoded.timestamp().or_else(|| decoded.pts()) else {
                    continue;
                };
                let time = (pts - start) as f64 * time_base;
                let index = (time / FACE_SAMPLE_INTERVAL_SECS).round();
                if !(0.0..10_000_000.0).contains(&index) || (index as usize) < samples.len() {
                    continue;
                }
                scaler
                    .run(&decoded, &mut scaled)
                    .map_err(|e| format!("camera scale / {e}"))?;
                let stride = scaled.stride(0);
                let row = width as usize * 4;
                pixels.clear();
                for line in scaled.data(0).chunks(stride).take(height as usize) {
                    pixels.extend_from_slice(&line[..row]);
                }
                let face = detector
                    .detect(&pixels, width, height)?
                    .map(|face| FaceBox {
                        x: face.x,
                        y: face.y,
                        width: face.width,
                        height: face.height,
                    });
                let index = index as usize;
                samples.resize(index, None);
                samples.push(face);
            }
            Ok(())
        };
        for (stream, packet) in input.packets() {
            if stream.index() != stream_index {
                continue;
            }
            if decoder.send_packet(&packet).is_ok() {
                process(&mut decoder)?;
            }
        }
        decoder.send_eof().ok();
        process(&mut decoder)?;
        Ok(samples)
    }
}
