#!/usr/bin/env python3
"""Module 05 page: 2D Fourier & Image Filtering."""

from __future__ import annotations

import customtkinter as ctk
from matplotlib.figure import Figure
import numpy as np
from PIL import Image

from components import widgets as ui

# Student 5 is implemented in this repository; load once and fall back to
# equivalent NumPy operations if an optional dependency is unavailable.
S5 = ui.load_student_module("student5")


def _op(name, fallback):
    """Return the student5 function *name*, or *fallback* when unavailable."""
    if S5 is not None and callable(getattr(S5, name, None)):
        return getattr(S5, name)
    return fallback


def _load_grayscale(path):
    """Load a grayscale image as uint8 (matching student5.load_grayscale_image)."""
    return np.asarray(Image.open(str(path)).convert("L"))


def _demo_image():
    """Fallback demo image as uint8 (matching student5.create_demo_image scale)."""
    synthetic = ui.helpers.make_synthetic_image()
    return ui.to_uint8(synthetic * 255.0)


def _fft_shifted(image):
    return np.fft.fftshift(np.fft.fft2(image))


def _circular_mask(shape, radius, kind="lowpass"):
    height, width = shape[:2]
    yy, xx = np.ogrid[:height, :width]
    distance = np.sqrt((yy - height // 2) ** 2 + (xx - width // 2) ** 2)
    low = (distance <= radius).astype(np.float32)
    return low if kind.lower() == "lowpass" else 1.0 - low


def _apply_mask(shifted, mask):
    return np.real(np.fft.ifft2(np.fft.ifftshift(shifted * mask)))


def _radius_figure(original, shifted, radii=(15, 40, 80)):
    """Fallback radius experiment when student5.plot_radius_experiment is absent."""
    figure = Figure(figsize=(12, 4))
    axes = figure.subplots(1, len(radii))
    for axis, radius in zip(np.atleast_1d(axes), radii):
        mask = _circular_mask(original.shape, radius, "lowpass")
        axis.imshow(_apply_mask(shifted, mask), cmap="gray")
        axis.set_title(f"Low-pass r = {radius}")
        axis.axis("off")
    figure.tight_layout()
    return figure


class ImageFilteringPage(ui.BasePage):
    number = "05"
    title = "2D Fourier & Image Filtering"
    description = (
        "Move an image into the frequency domain with the 2-D FFT and fftshift, "
        "then apply ideal circular low-pass and high-pass masks and reconstruct "
        "the filtered images with the inverse FFT."
    )
    module_key = "student5"

    def run_button_text(self) -> str:
        return "▶ Run 2D Fourier filtering"

    def build_controls(self) -> None:
        self.radius = self.add_slider("Cutoff radius (frequency pixels)", 5.0, 150.0, 40.0, 1.0)
        self.show_masks = self.add_checkbox("Show frequency masks", value=True)
        self.show_experiment = self.add_checkbox("Show radius experiment (15 / 40 / 80)", value=True)
        input_path = ui.find_image_input()
        self.add_info(
            "Input: "
            + (input_path.name if input_path else "none found — synthetic demo will be used")
        )

    def build_output(self) -> None:
        image_tab = self.add_tab("🖼️ Images")
        row = ctk.CTkFrame(image_tab, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=10)
        for column in range(3):
            row.grid_columnconfigure(column, weight=1, uniform="img")
        self.original_view = ui.ImageView(row, max_size=(300, 260))
        self.lowpass_view = ui.ImageView(row, max_size=(300, 260))
        self.highpass_view = ui.ImageView(row, max_size=(300, 260))
        self.original_view.grid(row=0, column=0, sticky="nsew", padx=4)
        self.lowpass_view.grid(row=0, column=1, sticky="nsew", padx=4)
        self.highpass_view.grid(row=0, column=2, sticky="nsew", padx=4)
        ctk.CTkLabel(
            image_tab,
            text="Log-scaled magnitude spectrum (zero frequency at the centre)",
            text_color=ui.COLORS["muted"],
            font=ctk.CTkFont(size=11),
        ).pack(fill="x", padx=12, pady=(4, 0))
        self.spectrum_view = ui.ImageView(image_tab, max_size=(560, 380))
        self.spectrum_view.pack(fill="both", expand=True, padx=10, pady=6)
        self.experiment_view = ui.PlotView(image_tab)
        self.experiment_view.pack(fill="both", expand=True, padx=10, pady=6)

        mask_tab = self.add_tab("🎯 Masks")
        mask_row = ctk.CTkFrame(mask_tab, fg_color="transparent")
        mask_row.pack(fill="x", padx=10, pady=10)
        mask_row.grid_columnconfigure(0, weight=1)
        mask_row.grid_columnconfigure(1, weight=1)
        self.low_mask_view = ui.ImageView(mask_row, max_size=(360, 300))
        self.high_mask_view = ui.ImageView(mask_row, max_size=(360, 300))
        self.low_mask_view.grid(row=0, column=0, sticky="nsew", padx=4)
        self.high_mask_view.grid(row=0, column=1, sticky="nsew", padx=4)
        self.mask_notes = ui.TextView(mask_tab, height=90)
        self.mask_notes.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.mask_notes.set_text(
            "White regions are retained frequencies, black regions are removed. "
            "The low-pass mask keeps the centre (low frequencies); the high-pass "
            "mask keeps the outer region (edges and detail)."
        )

        metric_tab = self.add_tab("🔢 Metrics")
        self.metrics = ui.MetricsView(metric_tab, columns=2)
        self.metrics.pack(fill="x", padx=10, pady=(10, 4))
        self.notes = ui.TextView(metric_tab, height=110)
        self.notes.pack(fill="both", expand=True, padx=10, pady=(4, 10))
        self.notes.set_text(
            "Interpreting the results\n\n"
            "A smaller radius removes more high-frequency content, so the low-pass "
            "image becomes blurrier and its MSE rises. The high-pass image keeps "
            "edges but loses smooth shading."
        )

    # -- computation -------------------------------------------------------- #

    def on_run(self) -> None:
        self.run_async(self._compute, self._display)

    def _compute(self) -> dict:
        radius = int(self.radius.get())
        show_experiment = bool(self.show_experiment.get())

        input_path = ui.find_image_input()
        load_gray = _op("load_grayscale_image", _load_grayscale)
        demo = _op("create_demo_image", _demo_image)
        if input_path is not None:
            original = np.asarray(load_gray(input_path))
            source_name = input_path.name
        else:
            original = np.asarray(demo())
            source_name = "synthetic demo image"

        shifted = _op("compute_fft_shifted", _fft_shifted)(original)
        mask_fn = _op("create_circular_mask", _circular_mask)
        apply_fn = _op("apply_frequency_mask", _apply_mask)
        low_mask = mask_fn(original.shape, radius, kind="lowpass")
        high_mask = mask_fn(original.shape, radius, kind="highpass")
        lowpass = apply_fn(shifted, low_mask)
        highpass = apply_fn(shifted, high_mask)
        spectrum = np.log1p(np.abs(shifted))

        low_retention = float(np.mean(low_mask) * 100.0)
        high_retention = float(np.mean(high_mask) * 100.0)
        mse, psnr = ui.helpers.calculate_mse_psnr(original, lowpass)
        high_display = ui.to_uint8(highpass, normalize=True)
        experiment = None
        if show_experiment:
            try:
                experiment = _op("plot_radius_experiment", _radius_figure)(
                    original, shifted, radii=(15, 40, 80), show=False
                )
            except TypeError:
                experiment = _radius_figure(original, shifted, radii=(15, 40, 80))

        return dict(
            original=original,
            lowpass=lowpass,
            highpass=high_display,
            spectrum=spectrum,
            low_mask=low_mask,
            high_mask=high_mask,
            experiment=experiment,
            source_name=source_name,
            radius=radius,
            low_retention=low_retention,
            high_retention=high_retention,
            mse=mse,
            psnr=psnr,
        )

    def _display(self, result: dict) -> None:
        radius = result["radius"]
        self.original_view.show_image(result["original"], caption="Original grayscale")
        self.lowpass_view.show_image(ui.to_uint8(result["lowpass"]), caption="Low-pass (blurred)")
        self.highpass_view.show_image(result["highpass"], caption="High-pass (normalised)")
        self.spectrum_view.show_image(
            ui.to_uint8(result["spectrum"], normalize=True),
            caption=f"Log spectrum · input: {result['source_name']}",
        )
        if result["experiment"] is not None:
            self.experiment_view.show_figure(result["experiment"])
        else:
            self.experiment_view.show_message("Enable the radius experiment to see it.")

        show_masks = bool(self.show_masks.get())
        if show_masks:
            self.low_mask_view.show_image(
                ui.to_uint8(result["low_mask"] * 255), caption=f"Low-pass mask · r = {radius}"
            )
            self.high_mask_view.show_image(
                ui.to_uint8(result["high_mask"] * 255), caption=f"High-pass mask · r = {radius}"
            )
            self.mask_notes.set_text(
                "White regions are retained frequencies, black regions are removed. "
                "The low-pass mask keeps the centre (low frequencies); the high-pass "
                "mask keeps the outer region (edges and detail)."
            )
        else:
            self.low_mask_view.show_message("Enable 'Show frequency masks' to see them.")
            self.high_mask_view.show_message("Enable 'Show frequency masks' to see them.")
            self.mask_notes.set_text("Enable 'Show frequency masks' in the parameters to see them.")

        psnr = result["psnr"]
        self.metrics.set_metrics(
            {
                "Cutoff radius": radius,
                "Image shape": f"{result['original'].shape[0]}×{result['original'].shape[1]}",
                "Low-pass retained": f"{result['low_retention']:.2f}%",
                "High-pass retained": f"{result['high_retention']:.2f}%",
                "MSE (low-pass)": f"{result['mse']:.2f}",
                "PSNR (low-pass)": f"{psnr:.2f} dB" if np.isfinite(psnr) else "∞ dB",
                "Total retained": f"{result['low_retention'] + result['high_retention']:.2f}%",
                "Spectrum mode": "log(1 + |F|)",
            }
        )
