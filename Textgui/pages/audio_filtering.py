#!/usr/bin/env python3
"""Module 04 page: Audio Filtering (notch filter design + evaluation)."""

from __future__ import annotations

import customtkinter as ctk
from matplotlib.figure import Figure
import numpy as np
from scipy import signal

from components import widgets as ui

S3 = ui.load_student_module("student3")


class AudioFilteringPage(ui.BasePage):
    number = "04"
    title = "Audio Filtering"
    description = (
        "Design a second-order notch filter to remove power-line hum from audio, "
        "compare the before/after spectra and filter response, and listen to the "
        "cleaned result."
    )
    module_key = "student4"

    def run_button_text(self) -> str:
        return "▶ Run notch filtering"

    def build_controls(self) -> None:
        self.notch_freq = self.add_slider("Notch frequency (Hz)", 40.0, 70.0, 50.0, 1.0)
        self.bandwidth = self.add_slider("Notch bandwidth (Hz)", 1.0, 20.0, 4.0, 0.5)
        self.hum_amplitude = self.add_slider("Injected hum amplitude", 0.01, 0.40, 0.10, 0.01)
        self.zoom = self.add_slider("Spectrum zoom limit (Hz)", 100.0, 500.0, 200.0, 20.0)
        self.zero_phase = self.add_checkbox("Zero-phase (filtfilt)", value=True)
        input_path = ui.find_audio_input()
        self.add_info(
            "Input: "
            + (input_path.name if input_path else "none found — synthetic demo will be used")
        )

    def build_output(self) -> None:
        spectrum_tab = self.add_tab("📊 Spectra")
        self.spectrum_view = ui.PlotView(spectrum_tab, toolbar=True)
        self.spectrum_view.pack(fill="both", expand=True)
        self.response_view = ui.PlotView(spectrum_tab)
        self.response_view.pack(fill="both", expand=True)

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
        self.play_noisy = ui.PlayButton(
            audio_box,
            provider=lambda: (self._noisy, self._sample_rate),
            status_label=self.audio_status,
            text="▶ Play with hum",
        )
        self.play_filtered = ui.PlayButton(
            audio_box,
            provider=lambda: (self._filtered, self._sample_rate),
            status_label=self.audio_status,
            text="▶ Play filtered",
        )
        self.play_noisy.pack(fill="x", padx=12, pady=(6, 0))
        self.play_filtered.pack(fill="x", padx=12, pady=(6, 12))
        self._noisy = None
        self._filtered = None
        self._sample_rate = 8000

        metric_tab = self.add_tab("🔢 Metrics")
        self.metrics = ui.MetricsView(metric_tab, columns=2)
        self.metrics.pack(fill="x", padx=10, pady=(10, 4))
        self.notes = ui.TextView(metric_tab, height=120)
        self.notes.pack(fill="both", expand=True, padx=10, pady=(4, 10))
        self.notes.set_text(
            "Reading the numbers\n\n"
            "A lower MSE and a higher PSNR after filtering mean the output is "
            "closer to the clean reference. The notch change is the gain at the "
            "interference frequency; a negative value means attenuation."
        )

    # -- computation -------------------------------------------------------- #

    def on_run(self) -> None:
        self.run_async(self._compute, self._display)

    @staticmethod
    def _spectrum(signal_values, sample_rate):
        if S3 is not None and callable(getattr(S3, "compute_spectrum", None)):
            return S3.compute_spectrum(signal_values, sample_rate)
        return ui.BasePage.compute_spectrum(signal_values, sample_rate)

    def _compute(self) -> dict:
        notch_freq = float(self.notch_freq.get())
        bandwidth = float(self.bandwidth.get())
        hum_amplitude = float(self.hum_amplitude.get())
        zoom = float(self.zoom.get())
        zero_phase = bool(self.zero_phase.get())

        input_path = ui.find_audio_input()
        if input_path is not None:
            clean, sample_rate = ui.helpers.load_audio(input_path)
            source_name = input_path.name
        else:
            clean, sample_rate = ui.helpers.make_synthetic_audio()
            source_name = "synthetic demo signal"
        clean = np.asarray(clean, dtype=np.float64)

        if S3 is not None and callable(getattr(S3, "add_powerline_interference", None)):
            noisy = np.asarray(
                S3.add_powerline_interference(
                    clean, sample_rate, interference_freq=notch_freq, amplitude=hum_amplitude
                ),
                dtype=np.float64,
            )
        else:
            noisy = ui.BasePage.add_powerline(clean, sample_rate, notch_freq, hum_amplitude)

        source = "Reference demo"
        note = ""
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
                    signal.filtfilt(b, a, noisy) if zero_phase else signal.lfilter(b, a, noisy)
                )
        except Exception as exc:
            note = (
                f"Notch filtering failed ({type(exc).__name__}: {exc}); "
                "using the unfiltered signal for comparison."
            )
            b, a = np.array([1.0]), np.array([1.0])
            filtered = noisy.copy()

        filtered = np.clip(filtered, -1.0, 1.0)
        noisy_freq, noisy_mag = self._spectrum(noisy, sample_rate)
        _, filtered_mag = self._spectrum(filtered, sample_rate)

        noisy_mse, noisy_psnr = ui.helpers.calculate_mse_psnr(clean, noisy)
        filtered_mse, filtered_psnr = ui.helpers.calculate_mse_psnr(clean, filtered)

        notch_band = np.abs(noisy_freq - notch_freq) <= bandwidth / 2.0
        attenuation = 0.0
        if notch_band.any():
            before = float(noisy_mag[notch_band].max())
            after = float(filtered_mag[notch_band].max())
            attenuation = 20.0 * np.log10((after + 1e-12) / (before + 1e-12))

        return dict(
            noisy=noisy,
            filtered=filtered,
            sample_rate=int(sample_rate),
            source_name=source_name,
            source=source,
            note=note,
            noisy_freq=noisy_freq,
            noisy_mag=noisy_mag,
            filtered_mag=filtered_mag,
            noisy_mse=noisy_mse,
            noisy_psnr=noisy_psnr,
            filtered_mse=filtered_mse,
            filtered_psnr=filtered_psnr,
            attenuation=attenuation,
            b=b,
            a=a,
            notch_freq=notch_freq,
            bandwidth=bandwidth,
            zoom=zoom,
        )

    def _display(self, result: dict) -> None:
        sample_rate = result["sample_rate"]
        notch_freq = result["notch_freq"]
        zoom = result["zoom"]

        figure = Figure(figsize=(10, 6))
        axes = figure.subplots(2, 1, sharex=True)
        axes[0].plot(result["noisy_freq"], result["noisy_mag"], color="#dc2626", linewidth=1.0)
        axes[0].set_title("Before filtering — hum spike present")
        axes[1].plot(result["noisy_freq"], result["noisy_mag"], color="#94a3b8", linewidth=0.9, label="Noisy")
        axes[1].plot(result["noisy_freq"], result["filtered_mag"], color="#059669", linewidth=1.2, label="Filtered")
        axes[1].set_title("After filtering — spike suppressed")
        axes[1].set_xlabel("Frequency (Hz)")
        axes[1].legend(loc="upper right", fontsize=8)
        for axis in axes:
            axis.set_xlim(0, zoom)
            axis.set_ylabel("Magnitude")
            axis.grid(True, alpha=0.3)
            axis.axvline(notch_freq, color="grey", linestyle="--", linewidth=0.8)
        figure.tight_layout()
        self.spectrum_view.show_figure(figure)

        w, h = signal.freqz(result["b"], result["a"], worN=1024, fs=sample_rate)
        figure = Figure(figsize=(10, 3.5))
        axis = figure.subplots()
        axis.plot(w, 20.0 * np.log10(np.abs(h) + 1e-12), color="#7c3aed", linewidth=1.3)
        axis.set_title(
            f"Notch filter response (f₀ = {notch_freq:.0f} Hz, BW = {result['bandwidth']:.1f} Hz)"
        )
        axis.set_xlabel("Frequency (Hz)")
        axis.set_ylabel("Gain (dB)")
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        self.response_view.show_figure(figure)

        self._noisy, self._filtered = result["noisy"], result["filtered"]
        self._sample_rate = sample_rate
        self.audio_status.configure(
            text=f"Input: {result['source_name']} · filter: {result['source']}"
        )

        psnr_before, psnr_after = result["noisy_psnr"], result["filtered_psnr"]
        self.metrics.set_metrics(
            {
                "Notch frequency": f"{notch_freq:.0f} Hz",
                "Bandwidth": f"{result['bandwidth']:.1f} Hz",
                "Quality Q": f"{notch_freq / result['bandwidth']:.1f}",
                "MSE before": f"{result['noisy_mse']:.6f}",
                "MSE after": f"{result['filtered_mse']:.6f}",
                "PSNR before": f"{psnr_before:.1f} dB" if np.isfinite(psnr_before) else "∞ dB",
                "PSNR after": f"{psnr_after:.1f} dB" if np.isfinite(psnr_after) else "∞ dB",
                "Notch change": f"{result['attenuation']:.1f} dB",
            }
        )
        self.notes.set_text(
            f"Input: {result['source_name']} · Filter source: {result['source']}\n"
            f"{result['note']}\n\n"
            "Reading the numbers\n\n"
            "A lower MSE and a higher PSNR after filtering mean the output is "
            "closer to the clean reference. The notch change is the gain at the "
            "interference frequency; a negative value means attenuation."
        )
