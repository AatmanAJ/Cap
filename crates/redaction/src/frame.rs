use std::sync::Arc;

use crate::Rect;

/// One 8-bit luma plane. Text recognition and change detection only need
/// brightness, and every decoder hands out a Y plane for free.
#[derive(Clone, Debug)]
pub struct LumaFrame {
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub stride: usize,
}

impl LumaFrame {
    pub fn from_rgba(rgba: &[u8], width: u32, height: u32, stride: usize) -> Option<Self> {
        let row = width as usize * 4;
        if stride < row || rgba.len() < stride * height as usize {
            return None;
        }
        let mut data = Vec::with_capacity(width as usize * height as usize);
        for line in rgba.chunks(stride).take(height as usize) {
            for pixel in line[..row].chunks_exact(4) {
                let luma = (u32::from(pixel[0]) * 54
                    + u32::from(pixel[1]) * 183
                    + u32::from(pixel[2]) * 19)
                    >> 8;
                data.push(luma as u8);
            }
        }
        Some(Self {
            data,
            width,
            height,
            stride: width as usize,
        })
    }

    pub fn from_plane(plane: &[u8], width: u32, height: u32, stride: usize) -> Option<Self> {
        if stride < width as usize || plane.len() < stride * height as usize {
            return None;
        }
        Some(Self {
            data: plane[..stride * height as usize].to_vec(),
            width,
            height,
            stride,
        })
    }
}

impl LumaFrame {
    /// A 2x box-filtered copy. Retina captures carry text at twice the size
    /// recognition needs, and halving them quarters the recognition cost.
    pub fn half(&self) -> Self {
        let width = (self.width / 2).max(1);
        let height = (self.height / 2).max(1);
        let mut data = Vec::with_capacity(width as usize * height as usize);
        for y in 0..height as usize {
            let top = &self.data[(y * 2) * self.stride..];
            let bottom = &self.data[((y * 2 + 1).min(self.height as usize - 1)) * self.stride..];
            for x in 0..width as usize {
                let sum = u32::from(top[x * 2])
                    + u32::from(top[(x * 2 + 1).min(self.width as usize - 1)])
                    + u32::from(bottom[x * 2])
                    + u32::from(bottom[(x * 2 + 1).min(self.width as usize - 1)]);
                data.push(((sum + 2) / 4) as u8);
            }
        }
        Self {
            data,
            width,
            height,
            stride: width as usize,
        }
    }
}

const THUMBNAIL_WIDTH: usize = 320;
/// Brightness change a cell needs before it counts as changed; below it is
/// encoder noise on a static screen.
const CELL_CHANGE: u8 = 14;
/// Share of a region's cells that must change for the region to have
/// changed.
const REGION_CHANGE: f64 = 0.12;

/// A block-averaged, low-resolution copy of a frame for cheap change
/// detection.
#[derive(Clone, Debug, PartialEq)]
pub struct Thumbnail {
    pub width: usize,
    pub height: usize,
    pub cells: Vec<u8>,
}

impl Thumbnail {
    pub fn from_luma(frame: &LumaFrame) -> Self {
        let (width, height) = (frame.width as usize, frame.height as usize);
        let block = width.div_ceil(THUMBNAIL_WIDTH).max(1);
        let columns = width.div_ceil(block).max(1);
        let rows = height.div_ceil(block).max(1);
        let mut sums = vec![0u32; columns * rows];
        let mut counts = vec![0u32; columns * rows];
        for y in 0..height {
            let line = &frame.data[y * frame.stride..y * frame.stride + width];
            let row = (y / block) * columns;
            for (x, value) in line.iter().enumerate() {
                let cell = row + x / block;
                sums[cell] += u32::from(*value);
                counts[cell] += 1;
            }
        }
        let cells = sums
            .iter()
            .zip(&counts)
            .map(|(sum, count)| (sum / (*count).max(1)) as u8)
            .collect();
        Self {
            width: columns,
            height: rows,
            cells,
        }
    }

    /// Share of all cells that changed between the two frames.
    pub fn difference(&self, other: &Thumbnail) -> f64 {
        self.region_difference(other, Rect::UNIT)
    }

    /// Share of the cells under `rect` (normalized) that changed.
    pub fn region_difference(&self, other: &Thumbnail, rect: Rect) -> f64 {
        if self.width != other.width || self.height != other.height {
            return 1.0;
        }
        let to_cells = |start: f64, end: f64, size: usize| {
            let first = ((start * size as f64).floor().max(0.0) as usize).min(size - 1);
            let last = ((end * size as f64).ceil() as usize).clamp(first + 1, size);
            first..last
        };
        let columns = to_cells(rect.x, rect.right(), self.width);
        let rows = to_cells(rect.y, rect.bottom(), self.height);
        let mut changed = 0usize;
        let mut total = 0usize;
        for row in rows {
            for column in columns.clone() {
                let index = row * self.width + column;
                total += 1;
                if self.cells[index].abs_diff(other.cells[index]) > CELL_CHANGE {
                    changed += 1;
                }
            }
        }
        if total == 0 {
            0.0
        } else {
            changed as f64 / total as f64
        }
    }

    pub fn region_changed(&self, other: &Thumbnail, rect: Rect) -> bool {
        self.region_difference(other, rect) > REGION_CHANGE
    }
}

/// The frames between two text-recognition passes: the previous pass first,
/// the current one last. It narrows down when an item appeared or vanished
/// to the frame where its area actually changed.
#[derive(Clone, Debug, Default)]
pub struct SampleWindow {
    pub frames: Vec<(f64, Arc<Thumbnail>)>,
}

impl SampleWindow {
    pub fn single(time: f64, thumbnail: Arc<Thumbnail>) -> Self {
        Self {
            frames: vec![(time, thumbnail)],
        }
    }

    pub fn start_time(&self) -> f64 {
        self.frames.first().map_or(0.0, |(time, _)| *time)
    }

    pub fn end_time(&self) -> f64 {
        self.frames.last().map_or(0.0, |(time, _)| *time)
    }

    /// The latest frame before the current one where `rect` still looked
    /// different, which the item can only have appeared after. With no such
    /// frame it was likely already there at the previous pass.
    pub fn appeared_after(&self, rect: Rect) -> f64 {
        let Some((_, current)) = self.frames.last() else {
            return 0.0;
        };
        self.frames[..self.frames.len() - 1]
            .iter()
            .rev()
            .find(|(_, frame)| frame.region_changed(current, rect))
            .map_or(self.start_time(), |(time, _)| *time)
    }

    /// The first frame after the previous pass where `rect` changed, by
    /// which the item seen there had gone. With no such frame it lasted up
    /// to the current pass.
    pub fn vanished_by(&self, rect: Rect) -> f64 {
        let Some((_, previous)) = self.frames.first() else {
            return 0.0;
        };
        self.frames[1..]
            .iter()
            .find(|(_, frame)| frame.region_changed(previous, rect))
            .map_or(self.end_time(), |(time, _)| *time)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(width: u32, height: u32, fill: impl Fn(u32, u32) -> u8) -> LumaFrame {
        let mut data = Vec::new();
        for y in 0..height {
            for x in 0..width {
                data.push(fill(x, y));
            }
        }
        LumaFrame {
            data,
            width,
            height,
            stride: width as usize,
        }
    }

    #[test]
    fn thumbnails_see_region_changes_but_not_noise() {
        let base = frame(640, 360, |_, _| 240);
        let noisy = frame(640, 360, |x, y| 240 - ((x + y) % 5) as u8);
        let text = frame(640, 360, |x, y| {
            if (100..300).contains(&x) && (50..70).contains(&y) && x % 3 == 0 {
                0
            } else {
                240
            }
        });
        let base = Thumbnail::from_luma(&base);
        let noisy = Thumbnail::from_luma(&noisy);
        let text = Thumbnail::from_luma(&text);
        assert_eq!(base.width, 320);
        assert_eq!(base.difference(&noisy), 0.0);
        let line = Rect::new(100.0 / 640.0, 50.0 / 360.0, 200.0 / 640.0, 20.0 / 360.0);
        assert!(base.region_changed(&text, line));
        assert!(!base.region_changed(&text, Rect::new(0.6, 0.6, 0.3, 0.3)));
        assert!(base.difference(&text) < 0.1);
    }

    #[test]
    fn luma_conversions_keep_geometry() {
        let rgba = [255, 255, 255, 255, 0, 0, 0, 255, 0, 0, 0, 0, 255, 0, 0, 255];
        let luma = LumaFrame::from_rgba(&rgba, 2, 1, 16).unwrap();
        assert_eq!(luma.data.len(), 2);
        assert!(luma.data[0] >= 254 && luma.data[1] == 0);
        assert!(LumaFrame::from_rgba(&rgba, 4, 1, 8).is_none());

        let padded = [1, 2, 9, 9, 3, 4, 9, 9];
        let plane = LumaFrame::from_plane(&padded, 2, 2, 4).unwrap();
        let half = plane.half();
        assert_eq!(
            (half.width, half.height, half.data.as_slice()),
            (1, 1, &[3u8][..])
        );
    }

    #[test]
    fn windows_report_unknown_boundaries_conservatively() {
        let thumbnail = Arc::new(Thumbnail {
            width: 4,
            height: 4,
            cells: vec![10; 16],
        });
        let window = SampleWindow {
            frames: vec![(1.0, thumbnail.clone()), (1.5, thumbnail)],
        };
        assert_eq!(window.appeared_after(Rect::UNIT), 1.0);
        assert_eq!(window.vanished_by(Rect::UNIT), 1.5);
    }
}
