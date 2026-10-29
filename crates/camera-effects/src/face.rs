use cidre::{arc, cv, ns, objc, vn};

use crate::vision::{create_input, write_bgra};

/// A face as fractions of the frame, origin at the top-left.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DetectedFace {
    pub x: f32,
    pub y: f32,
    pub width: f32,
    pub height: f32,
}

/// Apple Vision face rectangles (`VNDetectFaceRectanglesRequest`) over BGRA
/// frames. Keeps the largest face, which is the presenter in a webcam shot.
#[derive(Default)]
pub struct FaceDetector {
    input: Option<arc::R<cv::PixelBuf>>,
}

impl FaceDetector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn detect(
        &mut self,
        bgra: &[u8],
        width: u32,
        height: u32,
    ) -> Result<Option<DetectedFace>, String> {
        let (width, height) = (width as usize, height as usize);
        if width == 0 || height == 0 || bgra.len() < width * height * 4 {
            return Err("invalid face detection input".into());
        }
        let input = match &mut self.input {
            Some(input) if input.width() == width && input.height() == height => input,
            slot => slot.insert(create_input(width, height)?),
        };
        write_bgra(input, bgra, width, height)?;
        objc::ar_pool(|| detect_largest(input))
    }
}

fn detect_largest(input: &cv::PixelBuf) -> Result<Option<DetectedFace>, String> {
    let request = vn::DetectFaceRectanglesRequest::new();
    let handler = vn::ImageRequestHandler::with_cv_pixel_buf(input, None)
        .ok_or("Vision could not wrap the camera frame")?;
    let requests = ns::Array::<vn::Request>::from_slice(&[request.as_ref()]);
    handler
        .perform(&requests)
        .map_err(|error| format!("Vision face detection failed: {error:?}"))?;
    let Some(results) = request.results() else {
        return Ok(None);
    };
    Ok(results
        .iter()
        .filter(|face| face.confidence() >= 0.5)
        .map(|face| {
            let bounds = face.bounding_box();
            DetectedFace {
                x: bounds.origin.x as f32,
                y: (1.0 - bounds.origin.y - bounds.size.height) as f32,
                width: bounds.size.width as f32,
                height: bounds.size.height as f32,
            }
        })
        .max_by(|a, b| (a.width * a.height).total_cmp(&(b.width * b.height))))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blank_frames_have_no_face_and_bad_input_is_rejected() {
        let mut detector = FaceDetector::new();
        let frame = vec![96u8; 320 * 180 * 4];
        assert_eq!(detector.detect(&frame, 320, 180), Ok(None));
        assert_eq!(detector.detect(&frame, 320, 180), Ok(None));
        assert!(detector.detect(&frame, 640, 360).is_err());
        assert!(detector.detect(&frame, 0, 180).is_err());
    }
}
