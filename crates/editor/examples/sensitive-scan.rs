use std::{
    path::PathBuf,
    sync::{Arc, atomic::AtomicBool},
};

use cap_editor::{SensitiveScanOptions, scan_for_sensitive_info};
use cap_project::{
    MaskKind, MaskSegment, ProjectConfiguration, RecordingMeta, XY, mask_effect_contract,
};
use cap_rendering::{ProjectRecordingsMeta, RenderVideoConstants};

fn mask_template() -> MaskSegment {
    let contract = mask_effect_contract();
    MaskSegment {
        start: 0.0,
        end: 1.0,
        track: 0,
        enabled: true,
        mask_type: MaskKind::Sensitive,
        center: XY::new(0.5, 0.5),
        size: XY::new(0.35, 0.35),
        feather: 0.1,
        opacity: 1.0,
        pixelation: contract.blur_encoding_offset + contract.default_amount,
        darkness: 0.5,
        fade_duration: 0.0,
        keyframes: Default::default(),
    }
}

#[tokio::main]
async fn main() -> Result<(), String> {
    let args = std::env::args().collect::<Vec<_>>();
    let Some(path) = args.get(1).map(PathBuf::from) else {
        return Err("usage: sensitive-scan <project.cap> [--write]".into());
    };
    let write = args.iter().any(|arg| arg == "--write");

    let recording_meta = RecordingMeta::load_for_project(&path).map_err(|e| e.to_string())?;
    let studio_meta = recording_meta
        .studio_meta()
        .ok_or("not a studio recording")?
        .clone();
    let recordings = ProjectRecordingsMeta::new(&recording_meta.project_path, &studio_meta)?;
    let mut project = ProjectConfiguration::load(&path).unwrap_or_default();
    if project.timeline.is_none() {
        let segments = recordings
            .segments
            .iter()
            .enumerate()
            .map(|(index, segment)| {
                serde_json::json!({
                    "start": 0.0,
                    "end": segment.display.duration,
                    "timescale": 1.0,
                    "recordingClip": index,
                })
            })
            .collect::<Vec<_>>();
        project.timeline = Some(
            serde_json::from_value(serde_json::json!({
                "segments": segments,
                "zoomSegments": [],
            }))
            .map_err(|e| e.to_string())?,
        );
    }

    let constants = Arc::new(
        RenderVideoConstants::new(&recordings.segments, recording_meta.clone(), studio_meta)
            .await
            .map_err(|e| e.to_string())?,
    );
    let project = Arc::new(project);
    let result = scan_for_sensitive_info(
        &recording_meta,
        constants,
        project.clone(),
        mask_template(),
        SensitiveScanOptions::default(),
        |_| {},
        Arc::new(AtomicBool::new(false)),
    )
    .await?;

    let existing = project
        .timeline
        .as_ref()
        .map(|timeline| timeline.mask_segments.clone())
        .unwrap_or_default();
    let placed = cap_redaction::masks::place_masks(&existing, result.masks);
    println!(
        "{} masks from {} probed frames, {} recognition passes, {:.2}s",
        placed.len(),
        result.probed_frames,
        result.recognition_passes,
        result.elapsed.as_secs_f64()
    );
    for mask in &placed {
        println!(
            "lane {} {:.2}-{:.2}s center ({:.3}, {:.3}) size ({:.3}, {:.3}) keyframes {}",
            mask.track,
            mask.start,
            mask.end,
            mask.center.x,
            mask.center.y,
            mask.size.x,
            mask.size.y,
            mask.keyframes.position.len()
        );
    }

    if write {
        let mut project = (*project).clone();
        if let Some(timeline) = project.timeline.as_mut() {
            timeline.mask_segments.extend(placed);
        }
        project.write(&path).map_err(|e| e.to_string())?;
        println!("wrote {}", path.join("project-config.json").display());
    }
    Ok(())
}
