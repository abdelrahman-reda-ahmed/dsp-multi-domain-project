#!/usr/bin/env python3
"""GUI page: Student 1 — Mathematical Signal & DTFT."""

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

ui.setup_page("Mathematical Signal & DTFT", "📐")
ui.sidebar_context("01", "Mathematical Signal & DTFT", "student1")
ui.header(
    "01",
    "Mathematical Signal & DTFT",
    "Generate a two-tone discrete-time signal with additive noise, then compute "
    "its Discrete-Time Fourier Transform directly on an omega grid and inspect "
    "the magnitude and phase spectra.",
    "student1",
)

with st.expander("Parameters", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        sample_rate = st.slider("Sample rate (Hz)", 200.0, 4000.0, 1000.0, 100.0)
        duration = st.slider("Duration (s)", 0.05, 1.0, 0.25, 0.05)
    with col2:
        f1 = st.slider("Frequency f₁ (Hz)", 10.0, 400.0, 50.0, 5.0)
        f2 = st.slider("Frequency f₂ (Hz)", 20.0, 600.0, 120.0, 5.0)
    with col3:
        amplitude1 = st.slider("Amplitude A₁", 0.0, 2.0, 1.0, 0.1)
        amplitude2 = st.slider("Amplitude A₂", 0.0, 2.0, 0.6, 0.1)
    noise_std = st.slider("Noise standard deviation", 0.0, 1.0, 0.3, 0.05)
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=0)

run = st.button("▶ Run DTFT analysis", type="primary")

if not run:
    st.info("Adjust the parameters and press **Run** to generate the signal and its DTFT.")
else:
    with st.spinner("Generating signal and evaluating the DTFT…"):
        n_samples = int(round(sample_rate * duration))
        n = np.arange(n_samples)
        rng = np.random.default_rng(int(seed))
        signal = (
            amplitude1 * np.sin(2.0 * np.pi * f1 * n / sample_rate)
            + amplitude2 * np.cos(2.0 * np.pi * f2 * n / sample_rate)
            + noise_std * rng.standard_normal(n_samples)
        )

        # Direct DTFT: X(omega) = sum x[n] * exp(-j*omega*n).
        omega = np.linspace(0.0, np.pi, 1024)
        spectrum = np.exp(-1j * np.outer(omega, n)) @ signal
        magnitude = np.abs(spectrum)
        phase = np.unwrap(np.angle(spectrum))

        peak_index = int(np.argmax(magnitude[1:])) + 1
        peak_omega = float(omega[peak_index])
        peak_hz = peak_omega * sample_rate / (2.0 * np.pi)

        source = "Reference demo"
        if ui.is_implemented("student1"):
            try:
                module = ui.load_student_module("student1")
                candidate = module.generate_signal(
                    sample_rate=sample_rate,
                    duration=duration,
                    f1=f1,
                    f2=f2,
                    amplitude1=amplitude1,
                    amplitude2=amplitude2,
                    noise_std=noise_std,
                    seed=int(seed),
                )
                candidate_spectrum = module.compute_dtft(np.asarray(candidate), omega)
                signal = np.asarray(candidate, dtype=np.float64)
                spectrum = np.asarray(candidate_spectrum)
                magnitude = np.abs(spectrum)
                phase = np.unwrap(np.angle(spectrum))
                peak_index = int(np.argmax(magnitude[1:])) + 1
                peak_omega = float(omega[peak_index])
                peak_hz = peak_omega * sample_rate / (2.0 * np.pi)
                source = "Student 1 module"
            except Exception as exc:
                st.warning(
                    f"Student 1 module could not be used ({type(exc).__name__}: {exc}); "
                    "showing the reference demo instead."
                )

    st.caption(f"Results source: **{source}**")

    plot_tab, metric_tab = st.tabs(["📈 Plots", "🔢 Metrics"])

    with plot_tab:
        figure, axes = plt.subplots(3, 1, figsize=(10, 9))
        window = min(n_samples, int(0.05 * sample_rate))
        axes[0].plot(n[:window] / sample_rate, signal[:window], color="#1d4ed8", linewidth=1.1)
        axes[0].set_title(f"Time domain (first {window} samples)")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")

        axes[1].plot(omega / np.pi, 20.0 * np.log10(magnitude + 1e-12), color="#0891b2", linewidth=1.1)
        axes[1].set_title("DTFT magnitude (dB)")
        axes[1].set_xlabel("Normalised frequency (ω / π)")
        axes[1].set_ylabel("Magnitude (dB)")

        axes[2].plot(omega / np.pi, phase, color="#7c3aed", linewidth=1.0)
        axes[2].set_title("DTFT phase (unwrapped)")
        axes[2].set_xlabel("Normalised frequency (ω / π)")
        axes[2].set_ylabel("Phase (rad)")

        for axis in axes:
            axis.grid(True, alpha=0.3)
        figure.tight_layout()
        ui.show_figure(figure)

        st.caption(
            f"Expected tones at {f1:.0f} Hz and {f2:.0f} Hz; "
            f"largest DTFT peak at ≈ {peak_hz:.1f} Hz."
        )

    with metric_tab:
        ui.metrics_grid(
            {
                "Sample rate": f"{sample_rate:.0f} Hz",
                "Samples": n_samples,
                "Peak (digital ω)": f"{peak_omega / np.pi:.3f} π",
                "Peak (Hz)": f"{peak_hz:.1f}",
                "Max |X(ω)|": f"{magnitude.max():.2f}",
                "Noise σ": f"{noise_std:.2f}",
                "Seed": int(seed),
                "Grid points": omega.size,
            }
        )
        ui.card(
            "What the DTFT shows",
            "The transform sums every sample against a complex exponential for "
            "each ω. The two input tones produce two peaks; random noise raises "
            "the floor between them. Phase is only meaningful where magnitude is "
            "significant.",
        )

    if ui.is_implemented("student1"):
        with st.expander("Student 1 module output", expanded=False):
            _, log, figures = ui.run_student_main("student1", sample_rate=sample_rate)
            if log:
                st.code(log, language="text")
            for figure in figures:
                ui.show_figure(figure)
