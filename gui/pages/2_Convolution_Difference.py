#!/usr/bin/env python3
"""GUI page: Student 2 — Convolution & Difference Equations."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "main.py").is_file()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from gui import _shared as ui

ui.setup_page("Convolution & Difference Equations", "🧮")
ui.sidebar_context("02", "Convolution & Difference Equations", "student2")
ui.header(
    "02",
    "Convolution & Difference Equations",
    "Verify that convolution in the time domain equals multiplication in the "
    "frequency domain, and explore a first-order recursive difference equation "
    "that behaves as a simple low-pass filter.",
    "student2",
)

with st.expander("Parameters", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        signal_length = st.slider("Signal length", 64, 512, 200, 16)
        window_size = st.slider("Moving-average window", 2, 25, 5, 1)
    with col2:
        feedback = st.slider("Feedback coefficient a", 0.0, 0.99, 0.90, 0.01)
        feedforward = st.slider("Feed-forward coefficient b", 0.1, 2.0, 1.0, 0.1)
    with col3:
        initial = st.slider("Initial condition y₀", -1.0, 1.0, 0.0, 0.1)
        noise_std = st.slider("Input noise σ", 0.0, 0.5, 0.10, 0.01)

run = st.button("▶ Run convolution & difference analysis", type="primary")

if not run:
    st.info("Adjust the parameters and press **Run** to compare the two domains.")
else:
    with st.spinner("Computing convolutions and the recursion…"):
        n = np.arange(signal_length)
        rng = np.random.default_rng(0)
        x = np.sin(2.0 * np.pi * 0.06 * n) + 0.5 * np.cos(2.0 * np.pi * 0.11 * n)
        x = x + noise_std * rng.standard_normal(signal_length)
        h = np.ones(window_size, dtype=np.float64) / float(window_size)

        y_time = np.convolve(x, h, mode="full")
        fft_length = len(x) + len(h) - 1
        y_freq = np.fft.ifft(np.fft.fft(x, fft_length) * np.fft.fft(h, fft_length)).real

        y_diff = np.empty(signal_length, dtype=np.float64)
        y_diff[0] = float(feedforward) * x[0] + float(feedback) * float(initial)
        for index in range(1, signal_length):
            y_diff[index] = float(feedforward) * x[index] + float(feedback) * y_diff[index - 1]

        max_error = float(np.max(np.abs(y_time - y_freq)))
        mean_error = float(np.mean(np.abs(y_time - y_freq)))

        source = "Reference demo"
        if ui.is_implemented("student2"):
            try:
                module = ui.load_student_module("student2")
                time_result = np.asarray(module.time_domain_convolution(x, h), dtype=np.float64)
                freq_result = np.asarray(module.frequency_domain_convolution(x, h), dtype=np.float64)
                diff_result = np.asarray(
                    module.difference_equation(x, a=feedback, b=feedforward, y0=initial),
                    dtype=np.float64,
                )
                y_time, y_freq, y_diff = time_result, freq_result, diff_result
                max_error = float(np.max(np.abs(y_time - y_freq)))
                mean_error = float(np.mean(np.abs(y_time - y_freq)))
                source = "Student 2 module"
            except Exception as exc:
                st.warning(
                    f"Student 2 module could not be used ({type(exc).__name__}: {exc}); "
                    "showing the reference demo instead."
                )

    st.caption(f"Results source: **{source}**")

    conv_tab, diff_tab, metric_tab = st.tabs(
        ["🔁 Convolution", "📉 Difference equation", "🔢 Metrics"]
    )

    with conv_tab:
        figure, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
        axes[0].plot(x, label="x[n]", color="#94a3b8", linewidth=1.0)
        axes[0].plot(h, label="h[n] (moving average)", color="#f59e0b", linewidth=2.0)
        axes[0].set_title("Input signal and impulse response")
        axes[0].legend(loc="upper right", fontsize=8)

        axes[1].plot(y_time, label="Time-domain convolution", color="#1d4ed8", linewidth=1.6)
        axes[1].plot(y_freq, "--", label="Frequency-domain result", color="#dc2626", linewidth=1.2)
        axes[1].set_title("The two methods overlap after numerical rounding")
        axes[1].set_xlabel("Sample index")
        axes[1].legend(loc="upper right", fontsize=8)

        for axis in axes:
            axis.grid(True, alpha=0.3)
        figure.tight_layout()
        ui.show_figure(figure)

    with diff_tab:
        figure, axis = plt.subplots(figsize=(10, 4))
        axis.plot(x, label="x[n] (input)", color="#94a3b8", linewidth=1.0)
        axis.plot(y_diff, label="y[n] (recursive output)", color="#0891b2", linewidth=1.5)
        axis.set_title(
            f"First-order recursion: y[n] = {feedforward:.2f}·x[n] + {feedback:.2f}·y[n-1]"
        )
        axis.set_xlabel("Sample index")
        axis.set_ylabel("Amplitude")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", fontsize=8)
        figure.tight_layout()
        ui.show_figure(figure)
        ui.card(
            "Interpreting the recursion",
            "With |a| < 1 the feedback term accumulates past inputs, so the output "
            "is smoothed. Larger a gives a longer memory and a narrower low-pass "
            "response; b controls the overall gain.",
        )

    with metric_tab:
        ui.metrics_grid(
            {
                "Signal length": signal_length,
                "Kernel length": window_size,
                "Max |time − freq|": f"{max_error:.2e}",
                "Mean |time − freq|": f"{mean_error:.2e}",
                "Feedback a": f"{feedback:.2f}",
                "Feed-forward b": f"{feedforward:.2f}",
                "Initial y₀": f"{initial:.2f}",
                "Output energy": f"{float(np.sum(y_diff**2)):.2f}",
            }
        )
        ui.card(
            "Why the error is tiny, not zero",
            "The FFT evaluates the same convolution sum, but floating-point "
            "round-off prevents an exact match. A difference of order 1e-14 "
            "confirms the convolution theorem numerically.",
        )

    if ui.is_implemented("student2"):
        with st.expander("Student 2 module output", expanded=False):
            _, log, figures = ui.run_student_main("student2")
            if log:
                st.code(log, language="text")
            for figure in figures:
                ui.show_figure(figure)
