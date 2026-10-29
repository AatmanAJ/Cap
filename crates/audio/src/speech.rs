use crate::AudioData;
use cap_rnnoise::{DenoiseState, FRAME_SIZE};
use std::path::Path;

pub const SPEECH_FRAME_SECS: f64 = FRAME_SIZE as f64 / AudioData::SAMPLE_RATE as f64;
const SPEECH_PROBABILITY: f32 = 0.6;
const SPEECH_FLOOR_DBFS: f32 = -55.0;
const SPEECH_MERGE_GAP_SECS: f64 = 0.1;
const DECODE_WINDOW_FRAMES: usize = 3_000;

#[derive(Default)]
pub struct SpeechDetector {
    state: DenoiseState,
    pending: Vec<f32>,
    probabilities: Vec<f32>,
}

impl SpeechDetector {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, samples: &[f32], channels: u16) {
        let channels = usize::from(channels.max(1));
        self.pending.extend(
            samples
                .chunks_exact(channels)
                .map(|frame| frame.iter().sum::<f32>() / channels as f32),
        );
        let complete = self.pending.len() / FRAME_SIZE * FRAME_SIZE;
        let mut input = [0.0; FRAME_SIZE];
        let mut output = [0.0; FRAME_SIZE];
        for frame in self.pending[..complete].chunks_exact(FRAME_SIZE) {
            let mut energy = 0.0;
            for (target, &sample) in input.iter_mut().zip(frame) {
                let sample = if sample.is_finite() {
                    sample.clamp(-1.0, 1.0)
                } else {
                    0.0
                };
                energy += sample * sample;
                *target = sample * 32_768.0;
            }
            let probability = self.state.process_frame(&mut output, &input);
            let dbfs = 10.0 * (energy / FRAME_SIZE as f32).max(1e-12).log10();
            self.probabilities.push(if dbfs < SPEECH_FLOOR_DBFS {
                0.0
            } else {
                probability
            });
        }
        self.pending.drain(..complete);
    }

    pub fn finish(self) -> Vec<(f64, f64)> {
        speech_spans(&self.probabilities, SPEECH_FRAME_SECS)
    }
}

pub fn speech_spans(probabilities: &[f32], frame_secs: f64) -> Vec<(f64, f64)> {
    let mut spans: Vec<(f64, f64)> = Vec::new();
    let mut start = None;
    for (index, &probability) in probabilities
        .iter()
        .chain(std::iter::once(&0.0))
        .enumerate()
    {
        let time = index as f64 * frame_secs;
        match (start, probability >= SPEECH_PROBABILITY) {
            (None, true) => start = Some(time),
            (Some(begin), false) => {
                start = None;
                match spans.last_mut() {
                    Some(last) if begin - last.1 < SPEECH_MERGE_GAP_SECS => last.1 = time,
                    _ => spans.push((begin, time)),
                }
            }
            _ => {}
        }
    }
    spans
}

pub fn detect_speech_in_file(
    path: &Path,
    mut cancelled: impl FnMut() -> bool,
) -> Result<Vec<(f64, f64)>, String> {
    let window = DECODE_WINDOW_FRAMES * FRAME_SIZE;
    let mut detector = SpeechDetector::new();
    let mut start = 0;
    loop {
        if cancelled() {
            return Err("cancelled".into());
        }
        let audio = AudioData::from_file_range(path, start, start + window)?;
        let channels = usize::from(audio.channels().max(1));
        let available = audio.sample_count().min(window);
        detector.push(&audio.samples()[..available * channels], audio.channels());
        if available < window {
            break;
        }
        start += window;
    }
    Ok(detector.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spans_follow_frames_above_the_threshold() {
        let mut probabilities = vec![0.0; 100];
        probabilities[10..30].fill(0.9);
        probabilities[60..61].fill(0.7);
        probabilities[80..100].fill(0.95);
        let spans = speech_spans(&probabilities, 0.01);
        assert_eq!(spans.len(), 3);
        assert!((spans[0].0 - 0.10).abs() < 1e-9 && (spans[0].1 - 0.30).abs() < 1e-9);
        assert!((spans[1].0 - 0.60).abs() < 1e-9 && (spans[1].1 - 0.61).abs() < 1e-9);
        assert!((spans[2].0 - 0.80).abs() < 1e-9 && (spans[2].1 - 1.00).abs() < 1e-9);
    }

    #[test]
    fn spans_bridge_syllable_gaps_but_not_pauses() {
        let mut probabilities = vec![0.0; 200];
        probabilities[0..20].fill(0.9);
        probabilities[25..40].fill(0.9);
        probabilities[100..120].fill(0.9);
        let spans = speech_spans(&probabilities, 0.01);
        assert_eq!(spans.len(), 2);
        assert!((spans[0].1 - 0.40).abs() < 1e-9);
        assert!((spans[1].0 - 1.00).abs() < 1e-9);
    }

    #[test]
    fn digital_silence_and_non_finite_input_are_never_speech() {
        let mut detector = SpeechDetector::new();
        let mut samples = vec![0.0; FRAME_SIZE * 2 * 50];
        samples[100] = f32::NAN;
        samples[200] = f32::INFINITY;
        detector.push(&samples, 2);
        assert_eq!(detector.probabilities.len(), 50);
        assert!(detector.finish().is_empty());
    }

    #[test]
    fn partial_frames_carry_over_between_pushes() {
        let mut detector = SpeechDetector::new();
        detector.push(&vec![0.0; FRAME_SIZE / 2], 1);
        assert!(detector.probabilities.is_empty());
        detector.push(&vec![0.0; FRAME_SIZE / 2 + 10], 1);
        assert_eq!(detector.probabilities.len(), 1);
        assert_eq!(detector.pending.len(), 10);
    }
}
