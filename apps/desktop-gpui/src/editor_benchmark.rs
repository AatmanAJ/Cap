use std::time::{Duration, Instant};

use gpui::{AsyncApp, FrameTiming, FrameTimingCollector, WindowHandle};
use serde::Serialize;

use crate::editor_window::EditorWindow;

const STEPS: u32 = 240;
const STEP_TIMEOUT: Duration = Duration::from_millis(250);

#[derive(Serialize)]
struct Scenario {
    name: &'static str,
    steps: u32,
    frames: usize,
    draw_ms_total: f64,
    draw_ms_per_step: f64,
    draw_ms_mean: f64,
    draw_ms_p50: f64,
    draw_ms_p95: f64,
    draw_ms_max: f64,
    input_to_draw_end_samples: usize,
    input_to_draw_end_ms_p50: f64,
    input_to_draw_end_ms_p95: f64,
    draw_ms_trace: Vec<f64>,
}

#[derive(Serialize)]
struct Report {
    window_active: bool,
    viewport: (f32, f32),
    total_duration: f64,
    scenarios: Vec<Scenario>,
}

#[derive(Clone, Copy)]
enum Input {
    Hover,
    Scrub,
}

pub async fn run(handle: WindowHandle<EditorWindow>, cx: &mut AsyncApp) {
    let Some(output) = std::env::var_os("CAP_EDITOR_BENCHMARK_OUTPUT") else {
        return;
    };
    let started = Instant::now();
    loop {
        let ready = handle
            .update(cx, |view, _, _| view.benchmark_ready())
            .unwrap_or(false);
        if ready || started.elapsed() > Duration::from_secs(60) {
            break;
        }
        cx.background_executor()
            .timer(Duration::from_millis(50))
            .await;
    }
    cx.background_executor()
        .timer(Duration::from_millis(1500))
        .await;

    gpui::set_frame_trace_enabled(true);
    let mut scenarios = Vec::new();
    for (name, input) in [("hover", Input::Hover), ("scrub", Input::Scrub)] {
        let Some(scenario) = measure(handle, name, input, cx).await else {
            return;
        };
        tracing::info!(
            name,
            per_step_ms = scenario.draw_ms_per_step,
            p95_ms = scenario.draw_ms_p95,
            "editor benchmark scenario"
        );
        scenarios.push(scenario);
    }
    gpui::set_frame_trace_enabled(false);

    let Ok((window_active, viewport, total_duration)) = handle.update(cx, |view, window, _| {
        let size = window.viewport_size();
        (
            window.is_window_active(),
            (f32::from(size.width), f32::from(size.height)),
            view.total_duration(),
        )
    }) else {
        return;
    };
    let report = Report {
        window_active,
        viewport,
        total_duration,
        scenarios,
    };
    match serde_json::to_vec_pretty(&report) {
        Ok(bytes) => {
            if let Err(error) = std::fs::write(&output, bytes) {
                tracing::error!(%error, "editor benchmark: write failed");
            }
        }
        Err(error) => tracing::error!(%error, "editor benchmark: encode failed"),
    }
    cx.update(|cx| cx.quit());
}

async fn measure(
    handle: WindowHandle<EditorWindow>,
    name: &'static str,
    input: Input,
    cx: &mut AsyncApp,
) -> Option<Scenario> {
    let window_id = handle.window_id();
    let mut collector = FrameTimingCollector::new();
    let mut inputs = Vec::new();
    for step in 0..STEPS {
        let fraction = 0.05 + 0.9 * f64::from(step) / f64::from(STEPS - 1);
        let (sender, receiver) = flume::bounded(1);
        inputs.push(Instant::now());
        handle
            .update(cx, |view, window, cx| {
                match input {
                    Input::Hover => view.hover_fraction(fraction, window, cx),
                    Input::Scrub => view.seek_fraction(fraction, window, cx),
                }
                cx.on_next_frame(window, move |_, _, _| {
                    let _ = sender.send(());
                });
            })
            .ok()?;
        let timeout = cx.background_executor().timer(STEP_TIMEOUT);
        futures_util::future::select(Box::pin(receiver.recv_async()), timeout).await;
        cx.background_executor()
            .timer(Duration::from_millis(16))
            .await;
    }
    if matches!(input, Input::Hover) {
        handle
            .update(cx, |view, window, cx| view.hover_fraction(-1.0, window, cx))
            .ok()?;
    }
    cx.background_executor()
        .timer(Duration::from_millis(100))
        .await;

    let frames: Vec<FrameTiming> = collector
        .collect_unseen()
        .into_iter()
        .filter(|frame| frame.window_id == window_id)
        .collect();
    let mut latencies = input_to_draw_end_ms(&inputs, &frames);
    let trace: Vec<f64> = frames
        .iter()
        .map(|frame| frame.draw_duration().as_secs_f64() * 1000.0)
        .collect();
    let mut draws = trace.clone();
    draws.sort_by(f64::total_cmp);
    latencies.sort_by(f64::total_cmp);
    let total: f64 = draws.iter().sum();
    Some(Scenario {
        name,
        steps: STEPS,
        frames: draws.len(),
        draw_ms_total: total,
        draw_ms_per_step: total / f64::from(STEPS),
        draw_ms_mean: if draws.is_empty() {
            0.0
        } else {
            total / draws.len() as f64
        },
        draw_ms_p50: percentile(&draws, 0.5),
        draw_ms_p95: percentile(&draws, 0.95),
        draw_ms_max: draws.last().copied().unwrap_or(0.0),
        input_to_draw_end_samples: latencies.len(),
        input_to_draw_end_ms_p50: percentile(&latencies, 0.5),
        input_to_draw_end_ms_p95: percentile(&latencies, 0.95),
        draw_ms_trace: trace,
    })
}

fn input_to_draw_end_ms(inputs: &[Instant], frames: &[FrameTiming]) -> Vec<f64> {
    inputs
        .iter()
        .enumerate()
        .filter_map(|(index, &sent)| {
            frames
                .iter()
                .find(|frame| frame.draw_start >= sent)
                .filter(|frame| {
                    inputs
                        .get(index + 1)
                        .is_none_or(|&next| frame.draw_start < next)
                })
                .map(|frame| frame.draw_end.duration_since(sent).as_secs_f64() * 1000.0)
        })
        .collect()
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f64 * quantile).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(origin: Instant, draw_start_ms: u64, draw_end_ms: u64) -> FrameTiming {
        FrameTiming {
            window_id: 1.into(),
            dirty_at: None,
            invalidations: 1,
            draw_start: origin + Duration::from_millis(draw_start_ms),
            draw_end: origin + Duration::from_millis(draw_end_ms),
        }
    }

    #[test]
    fn latency_runs_to_the_end_of_the_draw_that_follows_the_input() {
        let origin = Instant::now();
        let inputs = [origin, origin + Duration::from_millis(40)];
        let frames = [frame(origin, 8, 30), frame(origin, 48, 51)];

        assert_eq!(input_to_draw_end_ms(&inputs, &frames), vec![30.0, 11.0]);
    }

    #[test]
    fn slower_draw_raises_latency_when_the_frame_starts_at_the_same_time() {
        let origin = Instant::now();
        let fast = input_to_draw_end_ms(&[origin], &[frame(origin, 8, 10)]);
        let slow = input_to_draw_end_ms(&[origin], &[frame(origin, 8, 33)]);

        assert_eq!(fast, vec![10.0]);
        assert_eq!(slow, vec![33.0]);
    }

    #[test]
    fn frame_already_drawing_when_the_input_arrives_is_not_counted() {
        let origin = Instant::now();
        let inputs = [origin + Duration::from_millis(10)];
        let frames = [frame(origin, 5, 12), frame(origin, 20, 26)];

        assert_eq!(input_to_draw_end_ms(&inputs, &frames), vec![16.0]);
    }

    #[test]
    fn input_without_its_own_frame_yields_no_sample() {
        let origin = Instant::now();
        let inputs = [
            origin,
            origin + Duration::from_millis(20),
            origin + Duration::from_millis(40),
        ];
        let frames = [frame(origin, 25, 31)];

        assert_eq!(input_to_draw_end_ms(&inputs, &frames), vec![11.0]);
    }
}
