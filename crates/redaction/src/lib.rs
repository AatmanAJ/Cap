//! Finds sensitive information on screen (email addresses, payment cards,
//! credentials, phone numbers, faces) and turns it into blur masks that
//! follow it on the timeline.
//!
//! Detection and tracking are platform-neutral. Reading the screen needs
//! OCR, which is only implemented with Apple Vision; [`ocr_available`]
//! reports whether this build can scan at all.

pub mod detect;
pub mod frame;
pub mod masks;
pub mod track;
#[cfg(target_os = "macos")]
pub mod vision;

pub use detect::DetectorOptions;
pub use frame::{LumaFrame, SampleWindow, Thumbnail};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SensitiveKind {
    Email,
    CardNumber,
    ApiKey,
    Token,
    Secret,
    Phone,
    Face,
}

/// A rectangle in normalized coordinates with a top-left origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Rect {
    pub x: f64,
    pub y: f64,
    pub width: f64,
    pub height: f64,
}

impl Rect {
    pub const UNIT: Self = Self {
        x: 0.0,
        y: 0.0,
        width: 1.0,
        height: 1.0,
    };

    pub fn new(x: f64, y: f64, width: f64, height: f64) -> Self {
        Self {
            x,
            y,
            width,
            height,
        }
    }

    pub fn right(&self) -> f64 {
        self.x + self.width
    }

    pub fn bottom(&self) -> f64 {
        self.y + self.height
    }

    pub fn center(&self) -> (f64, f64) {
        (self.x + self.width / 2.0, self.y + self.height / 2.0)
    }

    pub fn area(&self) -> f64 {
        self.width.max(0.0) * self.height.max(0.0)
    }

    pub fn intersect(&self, other: Rect) -> Option<Rect> {
        let x = self.x.max(other.x);
        let y = self.y.max(other.y);
        let right = self.right().min(other.right());
        let bottom = self.bottom().min(other.bottom());
        (right > x && bottom > y).then(|| Rect::new(x, y, right - x, bottom - y))
    }

    pub fn union(&self, other: Rect) -> Rect {
        let x = self.x.min(other.x);
        let y = self.y.min(other.y);
        Rect::new(
            x,
            y,
            self.right().max(other.right()) - x,
            self.bottom().max(other.bottom()) - y,
        )
    }

    pub fn iou(&self, other: Rect) -> f64 {
        let Some(overlap) = self.intersect(other) else {
            return 0.0;
        };
        let union = self.area() + other.area() - overlap.area();
        if union <= 0.0 {
            0.0
        } else {
            overlap.area() / union
        }
    }

    /// How much of the smaller rectangle the two share.
    pub fn overlap_of_smaller(&self, other: Rect) -> f64 {
        let Some(overlap) = self.intersect(other) else {
            return 0.0;
        };
        let smaller = self.area().min(other.area());
        if smaller <= 0.0 {
            0.0
        } else {
            overlap.area() / smaller
        }
    }

    pub fn lerp(&self, other: Rect, t: f64) -> Rect {
        let mix = |a: f64, b: f64| a + (b - a) * t;
        Rect::new(
            mix(self.x, other.x),
            mix(self.y, other.y),
            mix(self.width, other.width),
            mix(self.height, other.height),
        )
    }

    pub fn expand(&self, dx: f64, dy: f64) -> Rect {
        Rect::new(
            self.x - dx,
            self.y - dy,
            self.width + dx * 2.0,
            self.height + dy * 2.0,
        )
    }
}

/// One sensitive item found in one frame, in source-frame coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct Detection {
    pub kind: SensitiveKind,
    /// Normalized text that identifies the item across frames; empty for
    /// faces.
    pub key: String,
    pub bounds: Rect,
}

pub fn ocr_available() -> bool {
    cfg!(target_os = "macos")
}
