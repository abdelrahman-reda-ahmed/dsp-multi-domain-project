#!/usr/bin/env python3
"""GUI page: Student 6 — Image Noise Removal & Evaluation."""

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

ui.setup_page("Image Noise Removal", "🧹")
ui.sidebar_context("06", "Image Noise Removal & Evaluation", "student6")
ui.header(
    "06",
    "Image Noise Removal & Evaluation",
    "Corrupt an image with periodic sinusoidal noise, design a frequency-domain "
    "notch mask to remove it, and evaluate the restoration with MSE and PSNR.",
    "student6",
)

with st.expander("Parameters", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        freq_y = st.slider("Noise frequency — vertical (cycles/pixel)", 0.01, 0.30, 0.10, 0.01)
        amplitude = st.slider("Noise amplitude", 0.02, 0.40, 0.15, 0.01)
    with col2:
        freq_x = st.slider("Noise frequency — horizontal (cycles/pixel)", 0.01, 0.30, 0.15, 0.01)
        notch_radius = st.slider("Notch radius (frequency pixels)", 1, 20, 5, 1)
    with col3:
        show_spectrum = st.checkbox("Show noisy spectrum", value=True)
        seed = st.number_input("Demo-image seed", min_value=0, max_value=9999, value=0)

run = st.button("▶ Run denoising & evaluation", type="primary")

input_path = ui.find_input(ui.IMAGE_DIR, ui.helpers.IMAGE_EXTENSIONS)
if input_path is None:
    ui.data_warning("image")

if not run:
    st.info("Press **Run** to add periodic noise, apply the notch mask and score the result.")
else:
    with st.spinner("Adding noise, building the notch mask and evaluating…"):
        if input_path is not None:
            original = ui.helpers.load_image(input_path, grayscale=True)
            source_name = input_path.name
        else:
            original = ui.helpers.make_synthetic_image(seed=int(seed))
            source_name = "synthetic demo image"

        height, width = original.shape
        source = "Reference demo"

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
                    else:
                        student_metrics = {}
                except Exception:
                    student_metrics = {}
            except Exception as exc:
                st.warning(
                    f"Student 6 module could not be used ({type(exc).__name__}: {exc}); "
                    "showing the reference demo instead."
                )
                noisy = None
                mask = None
                denoised = None
                student_metrics = {}
        else:
            noisy = None
            mask = None
            denoised = None
            student_metrics = {}

        # --- Reference implementation ---
        if noisy is None:
            y_grid, x_grid = np.mgrid[0:height, 0:width]
            noise = amplitude * np.sin(
                2.0 * np.pi * (freq_y * y_grid + freq_x * x_grid)
            )
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

    st.caption(f"Input: **{source_name}** · Results source: **{source}**")

    image_tab, spectrum_tab, metric_tab = st.tabs(
        ["🖼️ Images", "📊 Spectrum", "🔢 Metrics"]
    )

    with image_tab:
        left, middle, right = st.columns(3)
        with left:
            ui.show_image(ui.to_uint8(original), caption="Original")
        with middle:
            ui.show_image(ui.to_uint8(np.clip(noisy, 0, 1)), caption="With sinusoidal noise")
        with right:
            ui.show_image(ui.to_uint8(np.clip(denoised, 0, 1)), caption="Denoised (notch mask)")

        notch_display = ui.to_uint8(np.asarray(mask) * 255)
        ui.show_image(notch_display, caption="Notch mask (black = removed frequencies)")

    with spectrum_tab:
        if show_spectrum:
            figure, axis = plt.subplots(figsize=(8, 6))
            axis.imshow(ui.to_uint8(spectrum, normalize=True), cmap="gray")
            axis.set_title("Noisy image spectrum — bright spots are the periodic noise")
            axis.axis("off")
            figure.tight_layout()
            ui.show_figure(figure)
            ui.card(
                "Where the noise lives",
                "A sinusoidal pattern produces sharp impulses away from the centre "
                "of the shifted spectrum. Setting those frequency locations to zero "
                "in the notch mask removes the pattern while keeping most of the image.",
            )
        else:
            st.info("Enable **Show noisy spectrum** in the parameters to see it.")

    with metric_tab:
        ui.metrics_grid(
            {
                "MSE before": f"{noisy_mse:.5f}",
                "PSNR before": f"{noisy_psnr:.2f} dB" if np.isfinite(noisy_psnr) else "∞ dB",
                "MSE after": f"{clean_mse:.5f}",
                "PSNR after": f"{clean_psnr:.2f} dB" if np.isfinite(clean_psnr) else "∞ dB",
                "MSE reduction": f"{noisy_mse - clean_mse:.5f}",
                "PSNR gain": f"{clean_psnr - noisy_psnr:.2f} dB"
                if np.isfinite(clean_psnr) and np.isfinite(noisy_psnr)
                else "∞ dB",
                "Notch radius": notch_radius,
                "Image shape": f"{height}×{width}",
            }
        )
        if student_metrics:
            st.caption(f"Student 6 evaluation: `{student_metrics}`")
        ui.card(
            "Reading MSE and PSNR",
            "MSE is the mean squared pixel difference from the clean image — lower "
            "is better. PSNR expresses that difference in decibels — higher is "
            "better. A successful notch filter lowers MSE and raises PSNR.",
        )

    if ui.is_implemented("student6"):
        with st.expander("Student 6 module output", expanded=False):
            _, log, figures = ui.run_student_main("student6", image_path=input_path)
            if log:
                st.code(log, language="text")
            for figure in figures:
                ui.show_figure(figure)

st.caption(
    "The reference demo mirrors the function signatures in "
    "`students/student6_image_noise_removal/student6.py`; once that file is "
    "implemented, this page calls it automatically."
)
