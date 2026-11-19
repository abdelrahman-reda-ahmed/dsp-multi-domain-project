#!/usr/bin/env python3
"""Module 06 page: Image Noise Removal & Evaluation."""

from __future__ import annotations

import customtkinter as ctk
from matplotlib.figure import Figure
import numpy as np

from components import widgets as ui


class ImageDenoisingPage(ui.BasePage):
    number = "06"
    title = "Image Noise Removal & Evaluation"
    description = (
        "Corrupt an image with periodic sinusoidal noise, design a frequency-domain "
        "notch mask to remove it, and evaluate the restoration with MSE and PSNR."
    )
    module_key = "student6"

    def run_button_text(self) -> str:
        return "▶ Run denoising & evaluation"

    def build_controls(self) -> None:
        self.freq_y = self.add_slider("Noise frequency — vertical (cycles/pixel)", 0.01, 0.30, 0.10, 0.01)
        self.freq_x = self.add_slider("Noise frequency — horizontal (cycles/pixel)", 0.01, 0.30, 0.15, 0.01)
        self.amplitude = self.add_slider("Noise amplitude", 0.02, 0.40, 0.15, 0.01)
        self.notch_radius = self.add_slider("Notch radius (frequency pixels)", 1.0, 20.0, 5.0, 1.0)
        self.show_spectrum = self.add_checkbox("Show noisy spectrum", value=True)
        self.seed = self.add_integer_entry("Demo-image seed", 0, 0, 9999)
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
        self.noisy_view = ui.ImageView(row, max_size=(300, 260))
        self.denoised_view = ui.ImageView(row, max_size=(300, 260))
        self.original_view.grid(row=0, column=0, sticky="nsew", padx=4)
        self.noisy_view.grid(row=0, column=1, sticky="nsew", padx=4)
        self.denoised_view.grid(row=0, column=2, sticky="nsew", padx=4)
        ctk.CTkLabel(
            image_tab,
            text="Notch mask (black = removed frequencies)",
            text_color=ui.COLORS["muted"],
            font=ctk.CTkFont(size=11),
        ).pack(fill="x", padx=12, pady=(4, 0))
        self.mask_view = ui.ImageView(image_tab, max_size=(420, 300))
        self.mask_view.pack(fill="both", expand=True, padx=10, pady=6)

        spectrum_tab = self.add_tab("📊 Spectrum")
        self.spectrum_view = ui.PlotView(spectrum_tab, toolbar=True)
        self.spectrum_view.pack(fill="both", expand=True)
        self.spectrum_notes = ui.TextView(spectrum_tab, height=90)
        self.spectrum_notes.pack(fill="x", padx=10, pady=(0, 10))
        self.spectrum_notes.set_text(
            "Where the noise lives\n\n"
            "A sinusoidal pattern produces sharp impulses away from the centre "
            "of the shifted spectrum. Setting those frequency locations to zero "
            "in the notch mask removes the pattern while keeping most of the image."
        )

        metric_tab = self.add_tab("🔢 Metrics")
        self.metrics = ui.MetricsView(metric_tab, columns=2)
        self.metrics.pack(fill="x", padx=10, pady=(10, 4))
        self.notes = ui.TextView(metric_tab, height=110)
        self.notes.pack(fill="both", expand=True, padx=10, pady=(4, 10))
        self.notes.set_text(
            "Reading MSE and PSNR\n\n"
            "MSE is the mean squared pixel difference from the clean image — lower "
            "is better. PSNR expresses that difference in decibels — higher is "
            "better. A successful notch filter lowers MSE and raises PSNR."
        )

    # -- computation -------------------------------------------------------- #

    def on_run(self) -> None:
        self.run_async(self._compute, self._display)

    def _compute(self) -> dict:
        freq_y = float(self.freq_y.get())
        freq_x = float(self.freq_x.get())
        amplitude = float(self.amplitude.get())
        notch_radius = int(self.notch_radius.get())
        show_spectrum = bool(self.show_spectrum.get())
        seed = int(float(self.seed.get()))

        input_path = ui.find_image_input()
        if input_path is not None:
            original = ui.helpers.load_image(input_path, grayscale=True)
            source_name = input_path.name
        else:
            original = ui.helpers.make_synthetic_image(seed=seed)
            source_name = "synthetic demo image"

        height, width = original.shape
        source = "Reference demo"
        noisy = mask = denoised = None
        student_metrics: dict = {}

        # --- Student 6 path (used automatically once the file is implemented) ---
        if ui.is_implemented("student6"):
            try:
                module = ui.load_student_module("student6")
                noisy = np.asarray(
                    module.add_sinusoidal_noise(
                        original, freq_y=freq_y, freq_x=freq_x, amplitude=amplitude
                    ),
                    dtype=np.float64,
                )
                mask = np.asarray(
                    module.build_notch_mask(
                        original.shape, freq_y=freq_y, freq_x=freq_x, radius=notch_radius
                    )
                )
                denoised = np.asarray(module.denoise(noisy, mask), dtype=np.float64)
                source = "Student 6 module"
                try:
                    evaluated = module.evaluate(original, denoised)
                    if isinstance(evaluated, dict):
                        student_metrics = evaluated
                    elif isinstance(evaluated, (tuple, list)) and len(evaluated) == 2:
                        student_metrics = {"MSE": evaluated[0], "PSNR": evaluated[1]}
                except Exception:
                    student_metrics = {}
            except Exception:
                noisy = mask = denoised = None
                student_metrics = {}

        # --- Reference implementation ---
        if noisy is None:
            y_grid, x_grid = np.mgrid[0:height, 0:width]
            noise = amplitude * np.sin(2.0 * np.pi * (freq_y * y_grid + freq_x * x_grid))
            noisy = np.clip(original + noise, 0.0, 1.0)

            shifted = np.fft.fftshift(np.fft.fft2(noisy))
            yy, xx = np.ogrid[:height, :width]
            mask = np.ones((height, width), dtype=np.float64)
            dy = int(round(freq_y * height))
            dx = int(round(freq_x * width))
            for sign in (1, -1):
                center_y = height // 2 + sign * dy
                center_x = width // 2 + sign * dx
                notch = (yy - center_y) ** 2 + (xx - center_x) ** 2 <= notch_radius**2
                mask[notch] = 0.0

            denoised = np.fft.ifft2(np.fft.ifftshift(shifted * mask)).real
            denoised = np.clip(denoised, 0.0, 1.0)

        noisy_mse, noisy_psnr = ui.helpers.calculate_mse_psnr(original, noisy)
        clean_mse, clean_psnr = ui.helpers.calculate_mse_psnr(original, denoised)
        spectrum = np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(noisy))))

        return dict(
            original=original,
            noisy=noisy,
            denoised=denoised,
            mask=mask,
            spectrum=spectrum,
            height=height,
            width=width,
            source=source,
            source_name=source_name,
            student_metrics=student_metrics,
            noisy_mse=noisy_mse,
            noisy_psnr=noisy_psnr,
            clean_mse=clean_mse,
            clean_psnr=clean_psnr,
            notch_radius=notch_radius,
            show_spectrum=show_spectrum,
        )

    def _display(self, result: dict) -> None:
        self.original_view.show_image(ui.to_uint8(result["original"]), caption="Original")
        self.noisy_view.show_image(
            ui.to_uint8(np.clip(result["noisy"], 0, 1)), caption="With sinusoidal noise"
        )
        self.denoised_view.show_image(
            ui.to_uint8(np.clip(result["denoised"], 0, 1)), caption="Denoised (notch mask)"
        )
        self.mask_view.show_image(
            ui.to_uint8(np.asarray(result["mask"]) * 255),
            caption=f"Notch mask · {result['source_name']}",
        )

        if result["show_spectrum"]:
            figure = Figure(figsize=(8, 6))
            axis = figure.subplots()
            axis.imshow(ui.to_uint8(result["spectrum"], normalize=True), cmap="gray")
            axis.set_title("Noisy image spectrum — bright spots are the periodic noise")
            axis.axis("off")
            figure.tight_layout()
            self.spectrum_view.show_figure(figure)
            self.spectrum_notes.set_text(
                "Where the noise lives\n\n"
                "A sinusoidal pattern produces sharp impulses away from the centre "
                "of the shifted spectrum. Setting those frequency locations to zero "
                "in the notch mask removes the pattern while keeping most of the image."
            )
        else:
            self.spectrum_view.show_message("Enable 'Show noisy spectrum' to see it.")
            self.spectrum_notes.set_text("Enable 'Show noisy spectrum' in the parameters to see it.")

        noisy_psnr, clean_psnr = result["noisy_psnr"], result["clean_psnr"]
        psnr_gain = (
            f"{clean_psnr - noisy_psnr:.2f} dB"
            if np.isfinite(clean_psnr) and np.isfinite(noisy_psnr)
            else "∞ dB"
        )
        self.metrics.set_metrics(
            {
                "MSE before": f"{result['noisy_mse']:.5f}",
                "PSNR before": f"{noisy_psnr:.2f} dB" if np.isfinite(noisy_psnr) else "∞ dB",
                "MSE after": f"{result['clean_mse']:.5f}",
                "PSNR after": f"{clean_psnr:.2f} dB" if np.isfinite(clean_psnr) else "∞ dB",
                "MSE reduction": f"{result['noisy_mse'] - result['clean_mse']:.5f}",
                "PSNR gain": psnr_gain,
                "Notch radius": result["notch_radius"],
                "Image shape": f"{result['height']}×{result['width']}",
            }
        )
        extra = ""
        if result["student_metrics"]:
            extra = f"\nStudent 6 evaluation: {result['student_metrics']}\n"
        self.notes.set_text(
            f"Input: {result['source_name']} · Results source: {result['source']}{extra}\n\n"
            "Reading MSE and PSNR\n\n"
            "MSE is the mean squared pixel difference from the clean image — lower "
            "is better. PSNR expresses that difference in decibels — higher is "
            "better. A successful notch filter lowers MSE and raises PSNR."
        )
