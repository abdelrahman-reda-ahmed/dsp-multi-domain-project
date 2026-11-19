#!/usr/bin/env python3
"""GUI page: Student 5 — 2D Fourier & Image Filtering."""

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

s5 = ui.load_student_module("student5")

ui.setup_page("2D Fourier & Image Filtering", "🖼️")
ui.sidebar_context("05", "2D Fourier & Image Filtering", "student5")
ui.header(
    "05",
    "2D Fourier & Image Filtering",
    "Move an image into the frequency domain with the 2-D FFT and `fftshift`, "
    "then apply ideal circular low-pass and high-pass masks and reconstruct the "
    "filtered images with the inverse FFT.",
    "student5",
)

with st.expander("Parameters", expanded=True):
    col1, col2, col3 = st.columns(3)
    with col1:
        radius = st.slider("Cutoff radius (frequency pixels)", 5, 150, 40, 1)
    with col2:
        show_masks = st.checkbox("Show frequency masks", value=True)
    with col3:
        show_experiment = st.checkbox("Show radius experiment (15 / 40 / 80)", value=True)

run = st.button("▶ Run 2D Fourier filtering", type="primary")

input_path = ui.find_input(ui.IMAGE_DIR, ui.helpers.IMAGE_EXTENSIONS)
if input_path is None:
    ui.data_warning("image")

if not run:
    st.info("Press **Run** to filter the image in the frequency domain.")
else:
    with st.spinner("Computing the 2-D FFT and applying the masks…"):
        if s5 is None or not ui.is_implemented("student5"):
            st.error("Student 5 module could not be loaded.")
            st.stop()

        if input_path is not None:
            original = s5.load_grayscale_image(input_path)
            source_name = input_path.name
        else:
            original = s5.create_demo_image()
            source_name = "synthetic demo image"

        shifted = s5.compute_fft_shifted(original)
        low_mask = s5.create_circular_mask(original.shape, radius, kind="lowpass")
        high_mask = s5.create_circular_mask(original.shape, radius, kind="highpass")
        lowpass = s5.apply_frequency_mask(shifted, low_mask)
        highpass = s5.apply_frequency_mask(shifted, high_mask)
        spectrum = np.log1p(np.abs(shifted))

        low_retention = float(np.mean(low_mask) * 100.0)
        high_retention = float(np.mean(high_mask) * 100.0)
        mse, psnr = ui.helpers.calculate_mse_psnr(original, lowpass)

        high_display = ui.to_uint8(highpass, normalize=True)
        figure_experiment = (
            s5.plot_radius_experiment(original, shifted, radii=(15, 40, 80), show=False)
            if show_experiment
            else None
        )

    st.caption(f"Input: **{source_name}** · Student 5 module functions used directly.")

    image_tab, mask_tab, metric_tab = st.tabs(["🖼️ Images", "🎯 Masks", "🔢 Metrics"])

    with image_tab:
        left, middle, right = st.columns(3)
        with left:
            ui.show_image(original, caption="Original grayscale")
        with middle:
            ui.show_image(ui.to_uint8(lowpass), caption="Low-pass (blurred)")
        with right:
            ui.show_image(high_display, caption="High-pass (normalised)")

        st.caption("Log-scaled magnitude spectrum (zero frequency at the centre)")
        ui.show_image(ui.to_uint8(spectrum, normalize=True))

        if figure_experiment is not None:
            st.caption("Effect of the cutoff radius")
            ui.show_figure(figure_experiment)

    with mask_tab:
        if show_masks:
            left, right = st.columns(2)
            with left:
                ui.show_image(ui.to_uint8(low_mask * 255), caption=f"Low-pass mask · r = {radius}")
            with right:
                ui.show_image(ui.to_uint8(high_mask * 255), caption=f"High-pass mask · r = {radius}")
            ui.card(
                "Reading the masks",
                "White regions are retained frequencies, black regions are removed. "
                "The low-pass mask keeps the centre (low frequencies); the high-pass "
                "mask keeps the outer region (edges and detail).",
            )
        else:
            st.info("Enable **Show frequency masks** in the parameters to see them.")

    with metric_tab:
        ui.metrics_grid(
            {
                "Cutoff radius": radius,
                "Image shape": f"{original.shape[0]}×{original.shape[1]}",
                "Low-pass retained": f"{low_retention:.2f}%",
                "High-pass retained": f"{high_retention:.2f}%",
                "MSE (low-pass)": f"{mse:.2f}",
                "PSNR (low-pass)": f"{psnr:.2f} dB" if np.isfinite(psnr) else "∞ dB",
                "Total retained": f"{low_retention + high_retention:.2f}%",
                "Spectrum mode": "log(1 + |F|)",
            }
        )
        ui.card(
            "Interpreting the results",
            "A smaller radius removes more high-frequency content, so the low-pass "
            "image becomes blurrier and its MSE rises. The high-pass image keeps "
            "edges but loses smooth shading.",
        )
        st.caption("Full-size artifacts are also written to `outputs/student5/` by the module.")
