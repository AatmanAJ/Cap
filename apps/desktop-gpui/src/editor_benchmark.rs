use std::time::{Duration, Instant};

use gpui::{AsyncApp, FrameTimingCollector, WindowHandle};
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
    input_to_frame_ms_p50: f64,
    input_to_frame_ms_p95: f64,
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
    let mut latencies = Vec::new();
    for step in 0..STEPS {
        let fraction = 0.05 + 0.9 * f64::from(step) / f64::from(STEPS - 1);
        let (sender, receiver) = flume::bounded(1);
        let sent = Instant::now();
        handle
            .update(cx, |view, window, cx| {
                match input {
                    Input::Hover => view.hover_fraction(fraction, window, cx),
                    Input::Scrub => view.seek_fraction(fraction, window, cx),
                }
                cx.on_next_frame(window, move |_, _, _| {
                    let _ = sender.send(Instant::now());
                });
            })
            .ok()?;
        let timeout = cx.background_executor().timer(STEP_TIMEOUT);
        if let futures_util::future::Either::Left((Ok(drawn), _)) =
            futures_util::future::select(Box::pin(receiver.recv_async()), timeout).await
        {
            latencies.push(drawn.duration_since(sent).as_secs_f64() * 1000.0);
        }
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

    let trace: Vec<f64> = collector
        .collect_unseen()
        .into_iter()
        .filter(|frame| frame.window_id == window_id)
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
        input_to_frame_ms_p50: percentile(&latencies, 0.5),
        input_to_frame_ms_p95: percentile(&latencies, 0.95),
        draw_ms_trace: trace,
    })
}

fn percentile(sorted: &[f64], quantile: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let index = ((sorted.len() - 1) as f64 * quantile).round() as usize;
    sorted[index.min(sorted.len() - 1)]
}
