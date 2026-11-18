use cap_project::{
    CaptionExportFormat, ProjectConfiguration, caption_export_cues, format_caption_cues,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let usage =
        "usage: caption-export <project-config.json> <srt|vtt> <recording-duration-secs>...";
    let config_path = args.next().ok_or(usage)?;
    let format = match args.next().as_deref() {
        Some("srt") => CaptionExportFormat::Srt,
        Some("vtt") => CaptionExportFormat::Vtt,
        _ => return Err(usage.into()),
    };
    let recording_durations = args
        .map(|value| value.parse::<f64>())
        .collect::<Result<Vec<_>, _>>()?;

    let project: ProjectConfiguration =
        serde_json::from_str(&std::fs::read_to_string(config_path)?)?;
    let cues = caption_export_cues(&project, &recording_durations);
    print!("{}", format_caption_cues(&cues, format));
    Ok(())
}
