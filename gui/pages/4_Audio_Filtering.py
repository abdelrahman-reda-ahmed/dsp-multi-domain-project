#!/usr/bin/env python3
"""GUI page: Student 4 — Audio Filtering."""

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
from scipy import signal

from gui import _shared as ui

s3 = ui.load_student_module("student3")

ui.setup_page("Audio Filtering", "🎚️")
ui.sidebar_context("04", "Audio Filtering", "student4")
ui.header(
    "04",
    "Audio Filtering",
    "Design a second-order notch filter to remove power-line hum from audio, "
    "compare the before/after spectra, and listen to the cleaned result.",
    "student4",
)

with st.expander("Parameters", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        notch_freq = st.slider("Notch frequency (Hz)", 40.0, 70.0, 50.0, 1.0)
        bandwidth = st.slider("Notch bandwidth (Hz)", 1.0, 20.0, 4.0, 0.5)
    with col2:
        interference_amp = st.slider("Injected hum amplitude", 0.01, 0.40, 0.10, 0.01)
        zero_phase = st.checkbox("Zero-phase (filtfilt)", value=True)
    with col3:
        zoom = st.slider("Spectrum zoom limit (Hz)", 100, 500, 200, 20)

run = st.button("▶ Run notch filtering", type="primary")

input_path = ui.find_input(ui.AUDIO_DIR, ui.helpers.AUDIO_EXTENSIONS)
if input_path is None:
    ui.data_warning("audio")

if not run:
    st.info("Press **Run** to add hum, apply the notch filter and compare the results.")
else:
    with st.spinner("Designing the notch filter and filtering audio…"):
        if input_path is not None:
            clean, sample_rate = ui.helpers.load_audio(input_path)
            source_name = input_path.name
        else:
            clean, sample_rate = ui.helpers.make_synthetic_audio()
            source_name = "synthetic demo signal"

        noisy = s3.add_powerline_interference(
            clean, sample_rate, interference_freq=notch_freq, amplitude=interference_amp
        )

        source = "Reference demo"
        try:
            if ui.is_implemented("student4"):
                module = ui.load_student_module("student4")
                b, a = module.design_notch_filter(sample_rate, notch_freq, bandwidth)
                filtered = np.asarray(module.apply_notch(noisy, b, a), dtype=np.float64)
                source = "Student 4 module"
            else:
                quality = notch_freq / bandwidth
                b, a = signal.iirnotch(w0=notch_freq, Q=quality, fs=sample_rate)
                filtered = (
                    signal.filtfilt(b, a, noisy)
                    if zero_phase
                    else signal.lfilter(b, a, noisy)
                )
        except Exception as exc:
            st.warning(
                f"Notch filtering failed ({type(exc).__name__}: {exc}); "
                "using the unfiltered signal for comparison."
            )
            b, a = np.array([1.0]), np.array([1.0])
            filtered = noisy.copy()

        filtered = np.clip(filtered, -1.0, 1.0)
        noisy_freq, noisy_mag = s3.compute_spectrum(noisy, sample_rate)
        _, filtered_mag = s3.compute_spectrum(filtered, sample_rate)

        noisy_mse, noisy_psnr = ui.helpers.calculate_mse_psnr(clean, noisy)
        filtered_mse, filtered_psnr = ui.helpers.calculate_mse_psnr(clean, filtered)

        notch_band = np.abs(noisy_freq - notch_freq) <= bandwidth / 2.0
        attenuation = 0.0
        if notch_band.any():
            before = float(noisy_mag[notch_band].max())
            after = float(filtered_mag[notch_band].max())
            attenuation = 20.0 * np.log10((after + 1e-12) / (before + 1e-12))

    st.caption(f"Input: **{source_name}** · Filter source: **{source}**")

    spectrum_tab, audio_tab, metric_tab = st.tabs(
        ["📊 Spectra", "🔊 Audio", "🔢 Metrics"]
    )

    with spectrum_tab:
        figure, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(noisy_freq, noisy_mag, color="#dc2626", linewidth=1.0)
        axes[0].set_title("Before filtering — hum spike present")
        axes[1].plot(noisy_freq, noisy_mag, color="#94a3b8", linewidth=0.9, label="Noisy")
        axes[1].plot(noisy_freq, filtered_mag, color="#059669", linewidth=1.2, label="Filtered")
        axes[1].set_title("After filtering — spike suppressed")
        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].legend(loc="upper right", fontsize=8)
        for axis in axes:
            axis.set_xlim(0, zoom)
            axis.set_ylabel("Magnitude")
            axis.grid(True, alpha=0.3)
            axis.axvline(notch_freq, color="grey", linestyle="--", linewidth=0.8)
        figure.tight_layout()
        ui.show_figure(figure)

        w, h = signal.freqz(b, a, worN=1024, fs=sample_rate)
        figure, axis = plt.subplots(figsize=(10, 3.5))
        axis.plot(w, 20.0 * np.log10(np.abs(h) + 1e-12), color="#7c3aed", linewidth=1.3)
        axis.set_title(f"Notch filter response (f₀ = {notch_freq:.0f} Hz, BW = {bandwidth:.1f} Hz)")
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("Gain (dB)")
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        ui.show_figure(figure)

    with audio_tab:
        st.caption("With power-line hum")
        st.audio(ui.wav_bytes(noisy, sample_rate), format="audio/wav")
        st.caption("After notch filtering")
        st.audio(ui.wav_bytes(filtered, sample_rate), format="audio/wav")

    with metric_tab:
        ui.metrics_grid(
            {
                "Notch frequency": f"{notch_freq:.0f} Hz",
                "Bandwidth": f"{bandwidth:.1f} Hz",
                "Quality Q": f"{notch_freq / bandwidth:.1f}",
                "MSE before": f"{noisy_mse:.6f}",
                "MSE after": f"{filtered_mse:.6f}",
                "PSNR before": f"{noisy_psnr:.1f} dB" if np.isfinite(noisy_psnr) else "∞ dB",
                "PSNR after": f"{filtered_psnr:.1f} dB" if np.isfinite(filtered_psnr) else "∞ dB",
                "Notch change": f"{attenuation:.1f} dB",
            }
        )
        ui.card(
            "Reading the numbers",
            "A lower MSE and a higher PSNR after filtering mean the output is "
            "closer to the clean reference. The notch change is the gain at the "
            "interference frequency; a negative value means attenuation.",
        )

    if ui.is_implemented("student4"):
        with st.expander("Student 4 module output", expanded=False):
            _, log, figures = ui.run_student_main("student4", audio_path=input_path)
            if log:
                st.code(log, language="text")
            for figure in figures:
                ui.show_figure(figure)
