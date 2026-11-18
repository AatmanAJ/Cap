mod animated_gradient;
mod caption_export;
mod caption_timing;
mod configuration;
pub mod cursor;
pub mod keyboard;
mod meta;

pub use animated_gradient::*;
pub use caption_export::{
    CaptionCue, CaptionExportFormat, caption_export_cues, caption_export_file_name,
    format_caption_cues,
};
pub use caption_timing::{derive_caption_track_segments, source_caption_id, synchronize_captions};
pub use configuration::*;
pub use cursor::*;
pub use keyboard::*;
pub use meta::*;

use serde::{Deserialize, Serialize};
use specta::Type;

#[derive(Debug, Clone, Serialize, Deserialize, Type)]
#[serde(rename_all = "camelCase")]
pub struct RecordingConfig {
    pub fps: u32,
    pub resolution: Resolution,
}

#[derive(Debug, Clone, Serialize, Deserialize, Type)]
pub struct Resolution {
    pub width: u32,
    pub height: u32,
}

impl Default for RecordingConfig {
    fn default() -> Self {
        Self {
            fps: 30,
            resolution: Resolution {
                width: 1920,
                height: 1080,
            },
        }
    }
}
