#!/usr/bin/env python3
"""Module 01 page: Mathematical Signal & Discrete-Time Fourier Transform."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from components import widgets as ui


class MathematicalSignalPage(ui.BasePage):
    number = "01"
    title = "Mathematical Signal & DTFT"
    description = (
        "Generate a two-tone discrete-time signal with additive noise, then compute "
        "its Discrete-Time Fourier Transform directly on an omega grid and inspect "
        "the magnitude and phase spectra."
    )
    module_key = "student1"

    def run_button_text(self) -> str:
        return "▶ Run DTFT analysis"

    def build_controls(self) -> None:
        self.sample_rate = self.add_slider("Sample rate (Hz)", 200.0, 4000.0, 1000.0, 100.0)
        self.duration = self.add_slider("Duration (s)", 0.05, 1.0, 0.25, 0.05)
        self.f1 = self.add_slider("Frequency f₁ (Hz)", 10.0, 400.0, 50.0, 5.0)
        self.f2 = self.add_slider("Frequency f₂ (Hz)", 20.0, 600.0, 120.0, 5.0)
        self.amplitude1 = self.add_slider("Amplitude A₁", 0.0, 2.0, 1.0, 0.1)
        self.amplitude2 = self.add_slider("Amplitude A₂", 0.0, 2.0, 0.6, 0.1)
        self.noise_std = self.add_slider("Noise standard deviation", 0.0, 1.0, 0.3, 0.05)
        self.seed = self.add_integer_entry("Random seed", 0, 0, 9999)

    def build_output(self) -> None:
        plot_tab = self.add_tab("📈 Plots")
        self.plot_view = ui.PlotView(plot_tab, toolbar=True)
        self.plot_view.pack(fill="both", expand=True)

        metric_tab = self.add_tab("🔢 Metrics")
        self.metrics = ui.MetricsView(metric_tab, columns=2)
        self.metrics.pack(fill="x", padx=10, pady=(10, 4))
        self.notes = ui.TextView(metric_tab, height=130)
        self.notes.pack(fill="both", expand=True, padx=10, pady=(4, 10))
        self.notes.set_text(
            "What the DTFT shows\n\n"
            "The transform sums every sample against a complex exponential for each omega. "
            "The two input tones produce two peaks; random noise raises the floor between "
            "them. Phase is only meaningful where magnitude is significant."
        )

    # -- computation -------------------------------------------------------- #

    def on_run(self) -> None:
        self.run_async(self._compute, self._display)

    def _compute(self) -> dict:
        sample_rate = float(self.sample_rate.get())
        duration = float(self.duration.get())
        f1 = float(self.f1.get())
        f2 = float(self.f2.get())
        amplitude1 = float(self.amplitude1.get())
        amplitude2 = float(self.amplitude2.get())
        noise_std = float(self.noise_std.get())
        seed = int(float(self.seed.get()))

        n_samples = int(round(sample_rate * duration))
        n = np.arange(n_samples)
        rng = np.random.default_rng(seed)
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
        note = ""

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
                    seed=seed,
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
                note = f"Student 1 module could not be used ({type(exc).__name__}: {exc})."

        return dict(
            signal=signal,
            omega=omega,
            magnitude=magnitude,
            phase=phase,
            n_samples=n_samples,
            sample_rate=sample_rate,
            f1=f1,
            f2=f2,
            peak_omega=peak_omega,
            peak_hz=peak_hz,
            noise_std=noise_std,
            seed=seed,
            source=source,
            note=note,
        )

    def _display(self, result: dict) -> None:
        signal = result["signal"]
        omega = result["omega"]
        magnitude = result["magnitude"]
        phase = result["phase"]
        sample_rate = result["sample_rate"]
        n_samples = result["n_samples"]

        figure = Figure(figsize=(10, 9))
        axes = figure.subplots(3, 1)
        window = min(n_samples, int(0.05 * sample_rate))
        n = np.arange(n_samples)
        axes[0].plot(n[:window] / sample_rate, signal[:window], color="#60a5fa", linewidth=1.1)
        axes[0].set_title(f"Time domain (first {window} samples)")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("Amplitude")

        axes[1].plot(omega / np.pi, 20.0 * np.log10(magnitude + 1e-12), color="#22d3ee", linewidth=1.1)
        axes[1].set_title("DTFT magnitude (dB)")
        axes[1].set_xlabel("Normalised frequency (ω / π)")
        axes[1].set_ylabel("Magnitude (dB)")

        axes[2].plot(omega / np.pi, phase, color="#c084fc", linewidth=1.0)
        axes[2].set_title("DTFT phase (unwrapped)")
        axes[2].set_xlabel("Normalised frequency (ω / π)")
        axes[2].set_ylabel("Phase (rad)")

        for axis in axes:
            axis.grid(True, alpha=0.3)
        figure.tight_layout()
        self.plot_view.show_figure(figure)

        self.metrics.set_metrics(
            {
                "Sample rate": f"{sample_rate:.0f} Hz",
                "Samples": n_samples,
                "Peak (digital ω)": f"{result['peak_omega'] / np.pi:.3f} π",
                "Peak (Hz)": f"{result['peak_hz']:.1f}",
                "Max |X(ω)|": f"{magnitude.max():.2f}",
                "Noise σ": f"{result['noise_std']:.2f}",
                "Seed": result["seed"],
                "Grid points": omega.size,
            }
        )
        expected = (
            f"Expected tones at {result['f1']:.0f} Hz and {result['f2']:.0f} Hz; "
            f"largest DTFT peak at ≈ {result['peak_hz']:.1f} Hz."
        )
        self.notes.set_text(
            f"Results source: {result['source']}\n{expected}\n\n{result['note']}\n\n"
            "What the DTFT shows\n\n"
            "The transform sums every sample against a complex exponential for each omega. "
            "The two input tones produce two peaks; random noise raises the floor between "
            "them. Phase is only meaningful where magnitude is significant."
        )
