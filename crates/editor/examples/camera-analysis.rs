use cap_editor::{CameraAnalysisRequest, analyze_camera};
use cap_project::{ProjectConfiguration, RecordingMeta, SceneMode};
use cap_rendering::{CameraAnalysisStore, SmoothedFaceTrack};
use std::{path::PathBuf, sync::atomic::AtomicBool, time::Instant};

fn main() {
    let project_path = PathBuf::from(
        std::env::args()
            .nth(1)
            .expect("usage: camera-analysis <project.cap>"),
    );
    let project = ProjectConfiguration::load(&project_path).expect("project config");
    let recording = RecordingMeta::load_for_project(&project_path).expect("recording meta");
    let meta = recording.studio_meta().expect("studio recording");
    let request = CameraAnalysisRequest::for_project(&project);
    println!("request: {request:?}");

    let started = Instant::now();
    let analysis =
        analyze_camera(&project_path, meta, request, &AtomicBool::new(false)).expect("analysis");
    println!("analysed in {:?}", started.elapsed());

    for (index, segment) in analysis.segments.iter().enumerate() {
        if let Some(speech) = &segment.speech {
            println!("segment {index}: {} speech spans", speech.spans.len());
            for (start, end) in &speech.spans {
                println!("  speech {start:8.2} - {end:8.2}");
            }
        }
        if let Some(faces) = &segment.faces {
            let found = faces.samples.iter().flatten().count();
            println!(
                "segment {index}: {} face samples, {found} with a face",
                faces.samples.len()
            );
            if let Some(track) = SmoothedFaceTrack::new(faces) {
                for (sample, raw) in faces.samples.iter().enumerate().step_by(10) {
                    let time = sample as f64 * faces.interval;
                    let smooth = track.sample(time);
                    let raw = raw.map_or("none".to_string(), |face| {
                        format!(
                            "{:.3},{:.3}",
                            face.x + face.width / 2.0,
                            face.y + face.height / 2.0
                        )
                    });
                    println!(
                        "  face {time:6.1}s raw {raw:>13} smooth {:.3},{:.3}",
                        smooth[0], smooth[1]
                    );
                }
            }
        }
    }

    let duration = project.timeline.as_ref().map_or(0.0, |t| t.duration());
    let store = CameraAnalysisStore::default();
    store.set(analysis);
    let started = Instant::now();
    let scenes = store.scene_segments(&project, duration);
    println!(
        "silence windows for {duration:.1}s of timeline computed in {:?}",
        started.elapsed()
    );
    match scenes {
        Some(segments) => {
            for segment in segments.iter() {
                let kind = match segment.mode {
                    SceneMode::HideCamera => "hidden",
                    _ => "manual",
                };
                println!(
                    "  scene {kind:>6} {:8.2} - {:8.2} (fade out from {:.2}, fully back at {:.2})",
                    segment.start,
                    segment.end,
                    segment.start - segment.transition_in,
                    segment.end
                );
            }
        }
        None => println!("camera does not hide during silence"),
    }
}
