#!/usr/bin/env python3
"""Module 02 page: Convolution & Difference Equations."""

from __future__ import annotations

import numpy as np
from matplotlib.figure import Figure

from components import widgets as ui


class ConvolutionPage(ui.BasePage):
    number = "02"
    title = "Convolution & Difference Equations"
    description = (
        "Verify that convolution in the time domain equals multiplication in the "
        "frequency domain, and explore a first-order recursive difference equation "
        "that behaves as a simple low-pass filter."
    )
    module_key = "student2"

    def run_button_text(self) -> str:
        return "▶ Run convolution & difference analysis"

    def build_controls(self) -> None:
        self.signal_length = self.add_slider("Signal length", 64, 512, 200, 16, integer=True)
        self.window_size = self.add_slider("Moving-average window", 2, 25, 5, 1, integer=True)
        self.feedback = self.add_slider("Feedback coefficient a", 0.0, 0.99, 0.90, 0.01)
        self.feedforward = self.add_slider("Feed-forward coefficient b", 0.1, 2.0, 1.0, 0.1)
        self.initial = self.add_slider("Initial condition y₀", -1.0, 1.0, 0.0, 0.1)
        self.noise_std = self.add_slider("Input noise σ", 0.0, 0.5, 0.10, 0.01)

    def build_output(self) -> None:
        conv_tab = self.add_tab("🔁 Convolution")
        self.conv_view = ui.PlotView(conv_tab, toolbar=True)
        self.conv_view.pack(fill="both", expand=True)

        diff_tab = self.add_tab("📉 Difference equation")
        self.diff_view = ui.PlotView(diff_tab, toolbar=True)
        self.diff_view.pack(fill="both", expand=True)
        self.diff_notes = ui.TextView(diff_tab, height=90)
        self.diff_notes.pack(fill="x", padx=10, pady=(0, 10))
        self.diff_notes.set_text(
            "With |a| < 1 the feedback term accumulates past inputs, so the output is "
            "smoothed. Larger a gives a longer memory and a narrower low-pass response; "
            "b controls the overall gain."
        )

        metric_tab = self.add_tab("🔢 Metrics")
        self.metrics = ui.MetricsView(metric_tab, columns=2)
        self.metrics.pack(fill="both", expand=True, padx=10, pady=10)
        self.metric_notes = ui.TextView(metric_tab, height=90)
        self.metric_notes.pack(fill="x", padx=10, pady=(0, 10))
        self.metric_notes.set_text(
            "Why the error is tiny, not zero\n\n"
            "The FFT evaluates the same convolution sum, but floating-point round-off "
            "prevents an exact match. A difference of order 1e-14 confirms the "
            "convolution theorem numerically."
        )

    # -- computation -------------------------------------------------------- #

    def on_run(self) -> None:
        self.run_async(self._compute, self._display)

    def _compute(self) -> dict:
        signal_length = int(self.signal_length.get())
        window_size = int(self.window_size.get())
        feedback = float(self.feedback.get())
        feedforward = float(self.feedforward.get())
        initial = float(self.initial.get())
        noise_std = float(self.noise_std.get())

        n = np.arange(signal_length)
        rng = np.random.default_rng(0)
        x = np.sin(2.0 * np.pi * 0.06 * n) + 0.5 * np.cos(2.0 * np.pi * 0.11 * n)
        x = x + noise_std * rng.standard_normal(signal_length)
        h = np.ones(window_size, dtype=np.float64) / float(window_size)

        y_time = np.convolve(x, h, mode="full")
        fft_length = len(x) + len(h) - 1
        y_freq = np.fft.ifft(np.fft.fft(x, fft_length) * np.fft.fft(h, fft_length)).real

        y_diff = np.empty(signal_length, dtype=np.float64)
        y_diff[0] = feedforward * x[0] + feedback * initial
        for index in range(1, signal_length):
            y_diff[index] = feedforward * x[index] + feedback * y_diff[index - 1]

        max_error = float(np.max(np.abs(y_time - y_freq)))
        mean_error = float(np.mean(np.abs(y_time - y_freq)))
        source = "Reference demo"
        note = ""

        if ui.is_implemented("student2"):
            try:
                module = ui.load_student_module("student2")
                y_time = np.asarray(module.time_domain_convolution(x, h), dtype=np.float64)
                y_freq = np.asarray(module.frequency_domain_convolution(x, h), dtype=np.float64)
                y_diff = np.asarray(
                    module.difference_equation(x, a=feedback, b=feedforward, y0=initial),
                    dtype=np.float64,
                )
                max_error = float(np.max(np.abs(y_time - y_freq)))
                mean_error = float(np.mean(np.abs(y_time - y_freq)))
                source = "Student 2 module"
            except Exception as exc:
                note = f"Student 2 module could not be used ({type(exc).__name__}: {exc})."

        return dict(
            x=x,
            h=h,
            y_time=y_time,
            y_freq=y_freq,
            y_diff=y_diff,
            max_error=max_error,
            mean_error=mean_error,
            signal_length=signal_length,
            window_size=window_size,
            feedback=feedback,
            feedforward=feedforward,
            initial=initial,
            source=source,
            note=note,
        )

    def _display(self, result: dict) -> None:
        x, h = result["x"], result["h"]
        y_time, y_freq, y_diff = result["y_time"], result["y_freq"], result["y_diff"]
        feedback, feedforward = result["feedback"], result["feedforward"]

        figure = Figure(figsize=(10, 7))
        axes = figure.subplots(2, 1, sharex=True)
        axes[0].plot(x, label="x[n]", color="#94a3b8", linewidth=1.0)
        axes[0].plot(h, label="h[n] (moving average)", color="#f59e0b", linewidth=2.0)
        axes[0].set_title("Input signal and impulse response")
        axes[0].legend(loc="upper right", fontsize=8)

        axes[1].plot(y_time, label="Time-domain convolution", color="#60a5fa", linewidth=1.6)
        axes[1].plot(y_freq, "--", label="Frequency-domain result", color="#f87171", linewidth=1.2)
        axes[1].set_title("The two methods overlap after numerical rounding")
        axes[1].set_xlabel("Sample index")
        axes[1].legend(loc="upper right", fontsize=8)

        for axis in axes:
            axis.grid(True, alpha=0.3)
        figure.tight_layout()
        self.conv_view.show_figure(figure)

        figure = Figure(figsize=(10, 4))
        axis = figure.subplots()
        axis.plot(x, label="x[n] (input)", color="#94a3b8", linewidth=1.0)
        axis.plot(y_diff, label="y[n] (recursive output)", color="#22d3ee", linewidth=1.5)
        axis.set_title(
            f"First-order recursion: y[n] = {feedforward:.2f}·x[n] + {feedback:.2f}·y[n-1]"
        )
        axis.set_xlabel("Sample index")
        axis.set_ylabel("Amplitude")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", fontsize=8)
        figure.tight_layout()
        self.diff_view.show_figure(figure)

        self.metrics.set_metrics(
            {
                "Signal length": result["signal_length"],
                "Kernel length": result["window_size"],
                "Max |time − freq|": f"{result['max_error']:.2e}",
                "Mean |time − freq|": f"{result['mean_error']:.2e}",
                "Feedback a": f"{feedback:.2f}",
                "Feed-forward b": f"{feedforward:.2f}",
                "Initial y₀": f"{result['initial']:.2f}",
                "Output energy": f"{float(np.sum(y_diff**2)):.2f}",
            }
        )
        self.metric_notes.set_text(
            f"Results source: {result['source']}\n{result['note']}\n\n"
            "Why the error is tiny, not zero\n\n"
            "The FFT evaluates the same convolution sum, but floating-point round-off "
            "prevents an exact match. A difference of order 1e-14 confirms the "
            "convolution theorem numerically."
        )
