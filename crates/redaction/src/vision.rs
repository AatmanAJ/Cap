use cidre::{arc, cf, cv, ns, objc, vn};

use crate::{
    Detection, LumaFrame, Rect, SensitiveKind,
    detect::{self, DetectorOptions},
};

const MIN_TEXT_HEIGHT_PX: f32 = 7.0;
const MIN_FACE_CONFIDENCE: f32 = 0.75;
const MIN_FACE_SIZE: f64 = 0.02;

unsafe extern "C" {
    fn CVPixelBufferGetBaseAddress(pixel_buffer: &cv::PixelBuf) -> *mut u8;
    fn CVPixelBufferGetBytesPerRow(pixel_buffer: &cv::PixelBuf) -> usize;
}

/// Apple Vision text recognition and face detection over screen frames.
pub struct VisionAnalyzer {
    input: Option<arc::R<cv::PixelBuf>>,
    options: DetectorOptions,
}

impl VisionAnalyzer {
    pub fn new(options: DetectorOptions) -> Self {
        Self {
            input: None,
            options,
        }
    }

    pub fn analyze(&mut self, frame: &LumaFrame) -> Result<Vec<Detection>, String> {
        objc::ar_pool(|| self.analyze_inner(frame))
    }

    fn analyze_inner(&mut self, frame: &LumaFrame) -> Result<Vec<Detection>, String> {
        let (width, height) = (frame.width as usize, frame.height as usize);
        if width == 0 || height == 0 || frame.data.len() < frame.stride * height {
            return Err("invalid frame for text recognition".into());
        }
        let input = match &mut self.input {
            Some(input) if input.width() == width && input.height() == height => input,
            slot => slot.insert(create_input(width, height)?),
        };
        write_luma(input, frame)?;

        let mut text_request = vn::RecognizeTextRequest::new();
        text_request.set_recognition_level(vn::RequestTextRecognitionLevel::Accurate);
        // Language correction rewrites keys and addresses into dictionary
        // words, which is exactly the text this pass has to read verbatim.
        text_request.set_uses_lang_correction(false);
        text_request.set_min_text_height(MIN_TEXT_HEIGHT_PX / height as f32);
        let face_request = self
            .options
            .faces
            .then(vn::DetectFaceRectanglesRequest::new);

        let mut requests: Vec<&vn::Request> = vec![text_request.as_ref()];
        if let Some(face_request) = &face_request {
            requests.push(face_request.as_ref());
        }
        let requests = ns::Array::<vn::Request>::from_slice(&requests);
        let handler = vn::ImageRequestHandler::with_cv_pixel_buf(input, None)
            .ok_or("could not create a Vision request handler")?;
        handler
            .perform(&requests)
            .map_err(|error| format!("Vision text recognition failed: {error:?}"))?;

        let mut detections = Vec::new();
        if let Some(observations) = text_request.results() {
            for observation in observations.iter() {
                let candidates = observation.top_candidates(1);
                let Some(candidate) = candidates.first() else {
                    continue;
                };
                let text = candidate.string().to_string();
                let line_bounds = flip(observation.bounding_box());
                for found in detect::find_sensitive_text(&text, &self.options) {
                    // Vision boxes whole words, so a key glued to its label
                    // (`API_KEY=sk-...`) gets the word's box: slice the
                    // match out of it by character position instead.
                    let word = word_around(&text, found.range.clone());
                    let bounds = candidate
                        .bounding_box_for_range(utf16_range(&text, word.clone()))
                        .map(|rect| flip(rect.bounding_box()))
                        .ok()
                        .filter(|bounds| bounds.width > 0.0 && bounds.height > 0.0)
                        .map(|word_bounds| {
                            slice_bounds(word_bounds, &text[word.clone()], {
                                found.range.start - word.start..found.range.end - word.start
                            })
                        })
                        .unwrap_or_else(|| slice_bounds(line_bounds, &text, found.range.clone()));
                    detections.push(Detection {
                        kind: found.kind,
                        key: found.key,
                        bounds: bounds.intersect(Rect::UNIT).unwrap_or(bounds),
                    });
                }
            }
        }

        if let Some(face_request) = &face_request
            && let Some(faces) = face_request.results()
        {
            for face in faces.iter() {
                let bounds = flip(face.bounding_box());
                if face.confidence() < MIN_FACE_CONFIDENCE
                    || bounds.width.min(bounds.height) < MIN_FACE_SIZE
                {
                    continue;
                }
                detections.push(Detection {
                    kind: SensitiveKind::Face,
                    key: String::new(),
                    bounds,
                });
            }
        }

        Ok(detections)
    }
}

fn flip(rect: cidre::cg::Rect) -> Rect {
    Rect {
        x: rect.origin.x,
        y: 1.0 - rect.origin.y - rect.size.height,
        width: rect.size.width,
        height: rect.size.height,
    }
}

fn utf16_range(text: &str, range: std::ops::Range<usize>) -> ns::Range {
    let start = text[..range.start].encode_utf16().count();
    let len = text[range].encode_utf16().count();
    ns::Range::new(start, len)
}

/// The whitespace-delimited run of `text` that contains `range`.
fn word_around(text: &str, range: std::ops::Range<usize>) -> std::ops::Range<usize> {
    let start = text[..range.start]
        .rfind(char::is_whitespace)
        .map_or(0, |index| {
            index + text[index..].chars().next().map_or(1, char::len_utf8)
        });
    let end = text[range.end..]
        .find(char::is_whitespace)
        .map_or(text.len(), |index| range.end + index);
    start.min(range.start)..end.max(range.end)
}

/// The part of `bounds` that `range` of `text` covers, assuming even
/// character widths, widened by half a character on each side for
/// proportional fonts.
fn slice_bounds(bounds: Rect, text: &str, range: std::ops::Range<usize>) -> Rect {
    let total = text.chars().count().max(1) as f64;
    let start = text[..range.start].chars().count() as f64;
    let len = text[range].chars().count() as f64;
    if start == 0.0 && len >= total {
        return bounds;
    }
    let char_width = bounds.width / total;
    let x = (bounds.x + char_width * (start - 0.5)).max(bounds.x);
    let right = (bounds.x + char_width * (start + len + 0.5)).min(bounds.right());
    Rect {
        x,
        y: bounds.y,
        width: right - x,
        height: bounds.height,
    }
}

fn create_input(width: usize, height: usize) -> Result<arc::R<cv::PixelBuf>, String> {
    let properties = cf::Dictionary::new();
    let attributes = cf::Dictionary::with_keys_values(
        &[cv::pixel_buffer::keys::io_surf_props().as_ref()],
        &[properties.as_ref()],
    )
    .ok_or("text recognition input attributes")?;
    cv::PixelBuf::new(
        width,
        height,
        cv::PixelFormat::ONE_COMPONENT_8,
        Some(&attributes),
    )
    .map_err(|error| format!("text recognition input buffer: {error:?}"))
}

fn write_luma(buffer: &mut cv::PixelBuf, frame: &LumaFrame) -> Result<(), String> {
    let row = frame.width as usize;
    unsafe {
        buffer
            .lock_base_addr(cv::pixel_buffer::LockFlags::DEFAULT)
            .result()
            .map_err(|error| format!("lock text recognition input: {error:?}"))?;
    }
    let base = unsafe { CVPixelBufferGetBaseAddress(buffer) };
    let stride = unsafe { CVPixelBufferGetBytesPerRow(buffer) };
    let result = if base.is_null() || stride < row {
        Err("text recognition input is not addressable".to_string())
    } else {
        for (y, source) in frame
            .data
            .chunks(frame.stride)
            .take(frame.height as usize)
            .enumerate()
        {
            unsafe { std::ptr::copy_nonoverlapping(source.as_ptr(), base.add(y * stride), row) };
        }
        Ok(())
    };
    unsafe {
        buffer.unlock_lock_base_addr(cv::pixel_buffer::LockFlags::DEFAULT);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn words_and_slices_cover_the_match() {
        let text = "export API_KEY=sk-abc next";
        let start = text.find("sk-").unwrap();
        let range = start..start + "sk-abc".len();
        let word = word_around(text, range.clone());
        assert_eq!(&text[word.clone()], "API_KEY=sk-abc");

        let bounds = Rect::new(0.1, 0.2, 0.14, 0.02);
        let slice = slice_bounds(
            bounds,
            &text[word.clone()],
            range.start - word.start..range.end - word.start,
        );
        assert!((slice.x - (0.1 + 0.01 * 7.5)).abs() < 1e-9, "{slice:?}");
        assert!((slice.right() - 0.24).abs() < 1e-9, "{slice:?}");
        assert_eq!(slice_bounds(bounds, "whole", 0..5), bounds);
        assert_eq!(word_around("a@b.co", 0..6), 0..6);
        let accented = "Café ✓ x@y.co";
        let at = accented.find("x@").unwrap();
        assert_eq!(&accented[word_around(accented, at..at + 6)], "x@y.co");
    }
}
