#!/usr/bin/env python3
"""Student 6 - Image Noise Removal (SKELETON).

Your tasks:
    1. Add periodic sinusoidal noise to the grayscale image.
    2. Build a notch mask in the frequency domain to remove that noise.
    3. Reconstruct with the inverse FFT and compute MSE and PSNR.
    4. Compare the original, noisy and denoised images.

Run standalone with:
    python students/student6_image_noise_removal/student6.py

Implement every TODO below. The placeholder bodies let the script run today.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import helpers  # optional shared image loading + MSE/PSNR helpers


def add_sinusoidal_noise(image: np.ndarray, freq_y: float = 0.10,
                         freq_x: float = 0.15, amplitude: float = 0.15) -> np.ndarray:
    """Add a 2-D periodic sinusoidal pattern to an image.

    Parameters
    ----------
    image : numpy.ndarray
        Original grayscale image.
    freq_y, freq_x : float
        Normalised spatial frequencies (cycles per pixel) along rows/columns.
    amplitude : float
        Noise amplitude.

    Returns
    -------
    numpy.ndarray
        Noisy image of the same shape.

    TODO (student 6):
        - Build row/column index grids and compute
          ``amplitude * sin(2*pi*(freq_y*y + freq_x*x))``.
        - Add the pattern to the image and clip to a valid pixel range.
    """
    # TODO: replace this placeholder with your implementation.
    return np.array(image, dtype=float, copy=True)


def build_notch_mask(shape, freq_y: float = 0.10, freq_x: float = 0.15,
                     radius: int = 3) -> np.ndarray:
    """Build a frequency-domain notch mask that rejects the sinusoidal peaks.

    Parameters
    ----------
    shape : tuple[int, int]
        Image shape ``(rows, cols)``.
    freq_y, freq_x : float
        Normalised frequencies of the added noise.
    radius : int
        Radius (in frequency bins) of each notch region.

    Returns
    -------
    numpy.ndarray
        Mask with zeros at the noise locations and ones elsewhere.

    TODO (student 6):
        - Use ``np.fft.fftshift(np.fft.fftfreq(...))`` to place notches at both
          (+fy, +fx) and (-fy, -fx) in the centred spectrum.
        - Use a distance grid and set ``mask[distance <= radius] = 0``.
        - Explain how notch radius affects image detail.
    """
    # TODO: replace this placeholder with your implementation.
    return np.ones(shape, dtype=float)


def denoise(noisy: np.ndarray, notch_mask: np.ndarray) -> np.ndarray:
    """Remove the periodic noise using the notch mask and an inverse FFT.

    Steps: FFT, ``fftshift``, multiply by the mask, ``ifftshift``, ``ifft2``,
    take the real part and clip to ``[0, 1]``.

    TODO (student 6): implement the steps above in order.
    """
    # TODO: replace this placeholder with your implementation.
    return np.array(noisy, dtype=float, copy=True)


def evaluate(original: np.ndarray, denoised: np.ndarray):
    """Compute MSE and PSNR between the original and the denoised image.

    Returns
    -------
    tuple[float, float]
        ``(mse, psnr_db)``.

    TODO (student 6):
        - Compute ``mse = mean((original - denoised)**2)``.
        - Compute ``psnr = 10*log10(peak**2 / mse)`` with peak = 1.0 for [0,1] data.
        - (Optional) compare against ``helpers.calculate_mse_psnr``.
    """
    # TODO: replace this placeholder with your implementation.
    print("  TODO: compute MSE and PSNR; handle the identical-images case.")
    return float("nan"), float("nan")


def plot_comparison(original: np.ndarray, noisy: np.ndarray,
                    denoised: np.ndarray, notch_mask: np.ndarray,
                    show: bool = True) -> None:
    """Show original, noisy, notch mask and denoised images side by side."""
    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].set_title("Original - TODO: imshow")
    axes[0, 1].set_title("With sinusoidal noise - TODO: imshow")
    axes[1, 0].set_title("Notch mask - TODO: imshow")
    axes[1, 1].set_title("Denoised - TODO: imshow")
    for ax in axes.ravel():
        ax.set_axis_off()
        ax.text(0.5, 0.5, "TODO: add image", transform=ax.transAxes,
                ha="center", va="center", color="grey")
    figure.tight_layout()
    if show:
        plt.show()


def main(image_path=None, show: bool = True) -> None:
    """Run the Student 6 experiment; fall back to a demo image if none exists."""
    print("SKELETON MODE - implement the TODOs in this file to complete Student 6.")

    if image_path is None or not Path(image_path).is_file():
        print("  [!] No image supplied - using a synthetic demo image.")
        print("      Place a grayscale image in data/images/ to use a real one.")
        image_path = None
    original = helpers.make_synthetic_image() if image_path is None else helpers.load_image(
        image_path, grayscale=True)
    print(f"  Image shape: {original.shape}")

    noisy = add_sinusoidal_noise(original)
    mask = build_notch_mask(original.shape)
    denoised = denoise(noisy, mask)
    mse, psnr = evaluate(original, denoised)
    print(f"  MSE = {mse}, PSNR = {psnr} dB (TODO: fill in)")

    plot_comparison(original, noisy, denoised, mask, show=show)
    print("Student 6 skeleton finished.")


if __name__ == "__main__":
    main()
