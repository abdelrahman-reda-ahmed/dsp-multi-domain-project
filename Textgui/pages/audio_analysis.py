#!/usr/bin/env python3
"""Module 03 page: Audio Analysis (power-line interference + spectrum)."""

from __future__ import annotations

import contextlib
import io

import customtkinter as ctk
from matplotlib.figure import Figure
import numpy as np

from components import widgets as ui

# Student 3 is implemented in this repository; load it once at import time.
S3 = ui.load_student_module("student3")


class AudioAnalysisPage(ui.BasePage):
    number = "03"
    title = "Audio Analysis"
    description = (
        "Load a clean WAV file (or use the built-in demo signal), add power-line "
        "interference, and identify the culprit frequency from the magnitude "
        "spectrum. Student 3's functions are used directly."
    )
    module_key = "student3"

    def run_button_text(self) -> str:
        return "▶ Run audio analysis"

    def build_controls(self) -> None:
        self.interference_freq = self.add_slider(
            "Interference frequency (Hz)", 40.0, 70.0, 50.0, 1.0
        )
        self.amplitude = self.add_slider("Interference amplitude", 0.01, 0.40, 0.10, 0.01)
        self.zoom = self.add_slider("Spectrum zoom limit (Hz)", 100.0, 500.0, 200.0, 20.0)
        input_path = ui.find_audio_input()
        self.add_info(
            "Input: "
            + (input_path.name if input_path else "none found — synthetic demo will be used")
        )

    def build_output(self) -> None:
        wave_tab = self.add_tab("🌊 Waveforms")
        self.wave_view = ui.PlotView(wave_tab, toolbar=True)
        self.wave_view.pack(fill="both", expand=True)

        spectrum_tab = self.add_tab("📊 Spectra")
        self.spectrum_view = ui.PlotView(spectrum_tab, toolbar=True)
        self.spectrum_view.pack(fill="both", expand=True)
        self.detection_log = ui.TextView(spectrum_tab, height=90)
        self.detection_log.pack(fill="x", padx=10, pady=(0, 10))
        self.detection_log.set_text("Detection log appears here after a run.")

        audio_tab = self.add_tab("🔊 Audio")
        audio_box = ctk.CTkFrame(audio_tab, fg_color=ui.COLORS["panel"], corner_radius=10)
        audio_box.pack(fill="x", padx=10, pady=10)

        self.audio_status = ctk.CTkLabel(
            audio_box,
            text="Press Run, then use the buttons below to listen.",
            text_color=ui.COLORS["muted"],
            font=ctk.CTkFont(size=11),
            wraplength=560,
            justify="left",
        )
        self.audio_status.pack(fill="x", padx=12, pady=(10, 4))
        self.play_clean = ui.PlayButton(
            audio_box,
            provider=lambda: (self._clean, self._sample_rate),
            status_label=self.audio_status,
            text="▶ Play clean",
        )
        self.play_noisy = ui.PlayButton(
            audio_box,
            provider=lambda: (self._noisy, self._sample_rate),
            status_label=self.audio_status,
            text="▶ Play with interference",
        )
        self.play_clean.pack(fill="x", padx=12, pady=(6, 0))
        self.play_noisy.pack(fill="x", padx=12, pady=(6, 12))
        self._clean = None
        self._noisy = None
        self._sample_rate = 8000

        metric_tab = self.add_tab("🔢 Metrics")
        self.metrics = ui.MetricsView(metric_tab, columns=2)
        self.metrics.pack(fill="both", expand=True, padx=10, pady=10)

    # -- computation -------------------------------------------------------- #

    def on_run(self) -> None:
        self.run_async(self._compute, self._display)

    @staticmethod
    def _add_powerline(clean, sample_rate, freq, amplitude):
        if S3 is not None and callable(getattr(S3, "add_powerline_interference", None)):
            return np.asarray(
                S3.add_powerline_interference(
                    clean, sample_rate, interference_freq=freq, amplitude=amplitude
                ),
                dtype=np.float64,
            )
        return ui.BasePage.add_powerline(clean, sample_rate, freq, amplitude)

    @staticmethod
    def _spectrum(signal, sample_rate):
        if S3 is not None and callable(getattr(S3, "compute_spectrum", None)):
            return S3.compute_spectrum(signal, sample_rate)
        return ui.BasePage.compute_spectrum(signal, sample_rate)

    def _identify(self, noisy, sample_rate):
        if S3 is not None and callable(getattr(S3, "identify_interference_frequency", None)):
            log = io.StringIO()
            with contextlib.redirect_stdout(log):
                detected = S3.identify_interference_frequency(
                    noisy, sample_rate, candidates=(50.0, 60.0)
                )
            return detected, log.getvalue()
        return ui.BasePage.identify_interference_frequency(noisy, sample_rate), ""

    def _compute(self) -> dict:
        freq = float(self.interference_freq.get())
        amplitude = float(self.amplitude.get())
        zoom = float(self.zoom.get())

        input_path = ui.find_audio_input()
        if input_path is not None:
            clean, sample_rate = ui.helpers.load_audio(input_path)
            source_name = input_path.name
        else:
            clean, sample_rate = ui.helpers.make_synthetic_audio()
            source_name = "synthetic demo signal"

        clean = np.asarray(clean, dtype=np.float64)
        noisy = self._add_powerline(clean, sample_rate, freq, amplitude)
        clean_freq, clean_mag = self._spectrum(clean, sample_rate)
        noisy_freq, noisy_mag = self._spectrum(noisy, sample_rate)
        detected, log_text = self._identify(noisy, sample_rate)
        mse, psnr = ui.helpers.calculate_mse_psnr(clean, noisy)

        return dict(
            clean=clean,
            noisy=noisy,
            sample_rate=int(sample_rate),
            source_name=source_name,
            clean_freq=clean_freq,
            clean_mag=clean_mag,
            noisy_freq=noisy_freq,
            noisy_mag=noisy_mag,
            detected=detected,
            log_text=log_text,
            mse=mse,
            psnr=psnr,
            freq=freq,
            amplitude=amplitude,
            zoom=zoom,
        )

    def _display(self, result: dict) -> None:
        clean, noisy = result["clean"], result["noisy"]
        sample_rate = result["sample_rate"]
        window = min(clean.size, int(0.05 * sample_rate))
        time_axis = np.arange(window) / sample_rate

        figure = Figure(figsize=(10, 6))
        axes = figure.subplots(2, 1, sharex=True)
        axes[0].plot(time_axis, clean[:window], color="#1d4ed8", linewidth=1.1)
        axes[0].set_title(f"Clean audio — first {window} samples")
        axes[1].plot(time_axis, noisy[:window], color="#dc2626", linewidth=1.1)
        axes[1].set_title("Audio with power-line interference")
        axes[1].set_xlabel("Time (s)")
        for axis in axes:
            axis.set_ylabel("Amplitude")
            axis.grid(True, alpha=0.3)
        figure.tight_layout()
        self.wave_view.show_figure(figure)

        zoom = result["zoom"]
        figure = Figure(figsize=(10, 6))
        axes = figure.subplots(2, 1, sharex=True)
        axes[0].plot(result["clean_freq"], result["clean_mag"], color="#1d4ed8", linewidth=1.0)
        axes[0].set_title("Clean spectrum")
        axes[1].plot(result["noisy_freq"], result["noisy_mag"], color="#dc2626", linewidth=1.0)
        axes[1].set_title("Noisy spectrum — a narrow spike reveals the interference")
        axes[1].set_xlabel("Frequency (Hz)")
        for axis in axes:
            axis.set_xlim(0, zoom)
            axis.set_ylabel("Magnitude")
            axis.grid(True, alpha=0.3)
            for candidate in (50.0, 60.0):
                axis.axvline(candidate, color="grey", linestyle="--", linewidth=0.8)
        figure.tight_layout()
        self.spectrum_view.show_figure(figure)
        self.detection_log.set_text(result["log_text"] or "(no log produced)")

        self._clean, self._noisy = clean, noisy
        self._sample_rate = sample_rate
        self.audio_status.configure(text=f"Input: {result['source_name']} · ready to play.")

        psnr = result["psnr"]
        self.metrics.set_metrics(
            {
                "Injected": f"{result['freq']:.0f} Hz",
                "Detected": f"{result['detected']:.0f} Hz" if result["detected"] else "none",
                "Sample rate": f"{sample_rate} Hz",
                "Samples": int(clean.size),
                "Duration": f"{clean.size / sample_rate:.2f} s",
                "MSE": f"{result['mse']:.6f}",
                "PSNR": f"{psnr:.2f} dB" if np.isfinite(psnr) else "∞ dB",
                "Amplitude": f"{result['amplitude']:.2f}",
            }
        )
