#!/usr/bin/env python3
"""GUI page: Student 3 — Audio Analysis."""

from __future__ import annotations

import contextlib
import io
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

s3 = ui.load_student_module("student3")

ui.setup_page("Audio Analysis", "🎧")
ui.sidebar_context("03", "Audio Analysis", "student3")
ui.header(
    "03",
    "Audio Analysis",
    "Load a clean WAV file (or use the built-in demo signal), add power-line "
    "interference, and identify the culprit frequency from the magnitude spectrum.",
    "student3",
)

with st.expander("Parameters", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        interference_freq = st.slider("Interference frequency (Hz)", 40.0, 70.0, 50.0, 1.0)
    with col2:
        amplitude = st.slider("Interference amplitude", 0.01, 0.40, 0.10, 0.01)
    with col3:
        zoom = st.slider("Spectrum zoom limit (Hz)", 100, 500, 200, 20)

run = st.button("▶ Run audio analysis", type="primary")

input_path = ui.find_input(ui.AUDIO_DIR, ui.helpers.AUDIO_EXTENSIONS)
if input_path is None:
    ui.data_warning("audio")

if not run:
    st.info("Press **Run** to inject interference and analyse the spectrum.")
else:
    with st.spinner("Loading audio and computing spectra…"):
        if s3 is None or not ui.is_implemented("student3"):
            st.error("Student 3 module could not be loaded.")
            st.stop()

        if input_path is not None:
            clean, sample_rate = ui.helpers.load_audio(input_path)
            source_name = input_path.name
        else:
            clean, sample_rate = ui.helpers.make_synthetic_audio()
            source_name = "synthetic demo signal"

        noisy = s3.add_powerline_interference(
            clean, sample_rate, interference_freq=interference_freq, amplitude=amplitude
        )
        clean_freq, clean_mag = s3.compute_spectrum(clean, sample_rate)
        noisy_freq, noisy_mag = s3.compute_spectrum(noisy, sample_rate)

        detection_log = io.StringIO()
        with contextlib.redirect_stdout(detection_log):
            detected = s3.identify_interference_frequency(noisy, sample_rate, candidates=(50.0, 60.0))

        mse, psnr = ui.helpers.calculate_mse_psnr(clean, noisy)
        window = min(clean.size, int(0.05 * sample_rate))

    st.caption(f"Input: **{source_name}** · Student 3 module functions used directly.")

    wave_tab, spectrum_tab, audio_tab, metric_tab = st.tabs(
        ["🌊 Waveforms", "📊 Spectra", "🔊 Audio", "🔢 Metrics"]
    )

    with wave_tab:
        figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        time_axis = np.arange(window) / sample_rate
        axes[0].plot(time_axis, clean[:window], color="#1d4ed8", linewidth=1.1)
        axes[0].set_title(f"Clean audio — first {window} samples")
        axes[1].plot(time_axis, noisy[:window], color="#dc2626", linewidth=1.1)
        axes[1].set_title("Audio with power-line interference")
        axes[1].set_xlabel("Time (s)")
        for axis in axes:
            axis.set_ylabel("Amplitude")
            axis.grid(True, alpha=0.3)
        figure.tight_layout()
        ui.show_figure(figure)

    with spectrum_tab:
        figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(clean_freq, clean_mag, color="#1d4ed8", linewidth=1.0)
        axes[0].set_title("Clean spectrum")
        axes[1].plot(noisy_freq, noisy_mag, color="#dc2626", linewidth=1.0)
        axes[1].set_title("Noisy spectrum — a narrow spike reveals the interference")
        axes[1].set_xlabel("Frequency (Hz)")
        for axis in axes:
            axis.set_xlim(0, zoom)
            axis.set_ylabel("Magnitude")
            axis.grid(True, alpha=0.3)
            for candidate in (50.0, 60.0):
                axis.axvline(candidate, color="grey", linestyle="--", linewidth=0.8)
        figure.tight_layout()
        ui.show_figure(figure)
        with st.expander("Detection log", expanded=False):
            st.code(detection_log.getvalue(), language="text")

    with audio_tab:
        st.caption("Clean signal")
        st.audio(ui.wav_bytes(clean, sample_rate), format="audio/wav")
        st.caption("With power-line interference")
        st.audio(ui.wav_bytes(noisy, sample_rate), format="audio/wav")

    with metric_tab:
        ui.metrics_grid(
            {
                "Injected": f"{interference_freq:.0f} Hz",
                "Detected": f"{detected:.0f} Hz" if detected else "none",
                "Sample rate": f"{sample_rate} Hz",
                "Samples": int(clean.size),
                "Duration": f"{clean.size / sample_rate:.2f} s",
                "MSE": f"{mse:.6f}",
                "PSNR": f"{psnr:.2f} dB" if np.isfinite(psnr) else "∞ dB",
                "Amplitude": f"{amplitude:.2f}",
            }
        )
        ui.card(
            "How detection works",
            "The spectrum is scanned within ±2 Hz of the 50 Hz and 60 Hz mains "
            "candidates. The candidate with the larger local peak is reported as "
            "the interference frequency.",
        )
