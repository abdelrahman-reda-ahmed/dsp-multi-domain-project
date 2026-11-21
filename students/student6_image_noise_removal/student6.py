#!/usr/bin/env python3
"""
Student 6 - Image Noise Removal & Evaluation
============================================

Required tasks:
1. Add periodic sinusoidal noise to a grayscale image.
2. Display the noisy-image spectrum and build a notch mask from the KNOWN noise
   frequency.
3. Apply the mask in the frequency domain and reconstruct with the inverse 2D FFT.
4. Compare original / noisy / restored images.
5. Evaluate the restoration with MSE and PSNR.

Conventions used throughout
---------------------------
* Images are float64 in [0, 1], shape (rows, cols) = (H, W).
* Noise frequencies are in cycles/pixel:  freq_y along rows, freq_x along columns.
* Noise model:  n[y, x] = A * sin(2*pi*(freq_y*y + freq_x*x))
* In the fftshift-ed spectrum, DC sits at (H//2, W//2).  A real sinusoid gives
  TWO impulses (conjugate pair) at
        (H//2 + freq_y*H,  W//2 + freq_x*W)   and
        (H//2 - freq_y*H,  W//2 - freq_x*W)
  so the notch mask must remove both.
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Allow running this file directly (python student6.py) as well as through main.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import helpers  # noqa: E402


# --------------------------------------------------------------------------- #
# 1. Periodic noise
# --------------------------------------------------------------------------- #
def add_sinusoidal_noise(image, freq_y=0.10, freq_x=0.15, amplitude=0.15):
    """Add a 2D sinusoidal pattern to a grayscale image.

    The result is intentionally NOT clipped: clipping to [0, 1] would flatten
    the sinusoid's peaks, create harmonics, and make the noise no longer a
    single-frequency pattern (so a notch at the fundamental could not remove it
    fully).  Clipping is only applied for display.
    """
    image = np.asarray(image, dtype=np.float64)
    h, w = image.shape
    y, x = np.mgrid[0:h, 0:w]
    noise = amplitude * np.sin(2.0 * np.pi * (freq_y * y + freq_x * x))
    return image + noise


# --------------------------------------------------------------------------- #
# 2. Notch mask
# --------------------------------------------------------------------------- #
def build_notch_mask(shape, freq_y=0.10, freq_x=0.15, radius=3):
    """Return an (H, W) mask for the fftshift-ed spectrum: 1 = keep, 0 = remove.

    Two circular notches are placed at the symmetric impulse locations.
    Note: freq*H (or freq*W) is generally not an integer, so the noise energy
    "leaks" over neighbouring bins; the radius must be large enough to cover that
    leakage but small enough not to delete image content.
    """
    h, w = shape
    cy, cx = h // 2, w // 2
    dy = int(round(freq_y * h))
    dx = int(round(freq_x * w))

    yy, xx = np.ogrid[:h, :w]
    mask = np.ones((h, w), dtype=np.float64)
    for sign in (+1, -1):                       # conjugate-symmetric pair
        notch = (yy - (cy + sign * dy)) ** 2 + (xx - (cx + sign * dx)) ** 2 <= radius ** 2
        mask[notch] = 0.0
    return mask


# --------------------------------------------------------------------------- #
# 3. Denoising (FFT -> multiply -> inverse FFT)
# --------------------------------------------------------------------------- #
def denoise(noisy, notch_mask):
    """Apply the notch mask in the frequency domain and return the restored image."""
    spectrum = np.fft.fftshift(np.fft.fft2(noisy))            # centre DC
    filtered = spectrum * notch_mask                           # kill noise impulses
    restored = np.fft.ifft2(np.fft.ifftshift(filtered)).real   # back to space
    return np.clip(restored, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# 4. Evaluation
# --------------------------------------------------------------------------- #
def evaluate(original, denoised):
    """Return (MSE, PSNR in dB) between original and restored image.

    MSE  = mean((original - restored)^2)
    PSNR = 10*log10(MAX^2 / MSE), with MAX = 1.0 because pixels are in [0, 1].
    """
    return helpers.calculate_mse_psnr(original, denoised, peak=1.0)


# --------------------------------------------------------------------------- #
# 5. Plots
# --------------------------------------------------------------------------- #
def _log_spectrum(image):
    return np.log1p(np.abs(np.fft.fftshift(np.fft.fft2(image))))


def plot_comparison(original, noisy, denoised, notch_mask, show=True):
    """2x3 figure: original/noisy/restored images, noisy spectrum, mask, restored spectrum."""
    mse_n, psnr_n = helpers.calculate_mse_psnr(original, np.clip(noisy, 0, 1), peak=1.0)
    mse_d, psnr_d = evaluate(original, denoised)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    panels = [
        (original, "Original", True),
        (np.clip(noisy, 0, 1), f"Noisy\nMSE={mse_n:.5f}, PSNR={psnr_n:.2f} dB", True),
        (denoised, f"Restored (notch filter)\nMSE={mse_d:.5f}, PSNR={psnr_d:.2f} dB", True),
        (_log_spectrum(noisy), "Noisy spectrum (log) - bright dots = noise", False),
        (notch_mask, "Notch mask (black = removed)", True),
        (_log_spectrum(denoised), "Restored spectrum (log)", False),
    ]
    for ax, (img, title, fixed_range) in zip(axes.ravel(), panels):
        ax.imshow(img, cmap="gray",
                  vmin=0 if fixed_range else None, vmax=1 if fixed_range else None)
        ax.set_title(title, fontsize=10)
        ax.axis("off")
    fig.suptitle("Student 6 - Periodic noise removal with a 2D notch filter", fontsize=13)
    fig.tight_layout()

    out_dir = PROJECT_ROOT / "outputs" / "student6"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "comparison.png", dpi=130)

    if show:
        plt.show()
    return fig


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(image_path=None, show=True):
    freq_y, freq_x, amplitude, radius = 0.10, 0.15, 0.15, 3

    if image_path is None:
        image_path = helpers.find_data_file(helpers.IMAGE_DIR, helpers.IMAGE_EXTENSIONS)
    if image_path is not None:
        original = helpers.load_image(image_path, grayscale=True)
        print(f"  Image: {Path(image_path).name}  shape={original.shape}")
    else:
        original = helpers.make_synthetic_image()
        print("  No image found - using synthetic demo image.")

    noisy = add_sinusoidal_noise(original, freq_y, freq_x, amplitude)
    mask = build_notch_mask(original.shape, freq_y, freq_x, radius)
    restored = denoise(noisy, mask)

    mse_n, psnr_n = helpers.calculate_mse_psnr(original, np.clip(noisy, 0, 1), peak=1.0)
    mse_d, psnr_d = evaluate(original, restored)
    print(f"  Noisy    : MSE = {mse_n:.6f}, PSNR = {psnr_n:.2f} dB")
    print(f"  Restored : MSE = {mse_d:.6f}, PSNR = {psnr_d:.2f} dB")
    print(f"  Gain     : {psnr_d - psnr_n:+.2f} dB")

    plot_comparison(original, noisy, restored, mask, show=show)
    return mse_d, psnr_d


if __name__ == "__main__":
    main()
