#!/usr/bin/env python3
"""Student 5 - 2D Fourier & Image Filtering (SKELETON).

Your tasks:
    1. Load the grayscale image from ``data/images/``.
    2. Compute its 2-D FFT and centre it with ``fftshift``.
    3. Build ideal circular low-pass and high-pass masks and apply them.
    4. Show the original image, the spectrum, and both filtered results.

Run standalone with:
    python students/student5_2d_fourier_image_filtering/student5.py

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

from utils import helpers  # optional shared image loading helpers


def load_grayscale_image(image_path=None) -> np.ndarray:
    """Load a grayscale image as a float array in ``[0, 1]``.

    Falls back to a synthetic test image when no file is supplied.
    """
    if image_path is None or not Path(image_path).is_file():
        print("  [!] No image supplied - using a synthetic demo image.")
        print("      Place a grayscale image in data/images/ to use a real one.")
        return helpers.make_synthetic_image()
    return helpers.load_image(image_path, grayscale=True)


def compute_fft_shifted(image: np.ndarray):
    """Compute the centred 2-D Fourier transform of an image.

    Parameters
    ----------
    image : numpy.ndarray
        2-D grayscale image.

    Returns
    -------
    numpy.ndarray
        Complex centred spectrum: ``np.fft.fftshift(np.fft.fft2(image))``.

    TODO (student 5):
        - Apply ``np.fft.fft2`` then ``np.fft.fftshift``.
        - Display the log magnitude ``np.log1p(np.abs(shifted))``.
    """
    # TODO: replace this placeholder with your implementation.
    return np.zeros(image.shape, dtype=complex)


def create_circular_mask(shape, radius: float, kind: str = "lowpass") -> np.ndarray:
    """Build an ideal circular frequency mask centred on the DC component.

    Parameters
    ----------
    shape : tuple[int, int]
        Image/spectrum shape ``(rows, cols)``.
    radius : float
        Cut-off radius in samples from the centre of the spectrum.
    kind : {"lowpass", "highpass"}
        ``"lowpass"`` keeps frequencies inside the circle; ``"highpass"`` keeps
        frequencies outside it.

    Returns
    -------
    numpy.ndarray
        Boolean mask with the same shape.

    TODO (student 5):
        - Build a distance grid from the centre using ``np.ogrid``/``np.mgrid``.
        - lowpass: distance <= radius; highpass: distance > radius.
        - Try several radii and describe the trade-off in comments.
    """
    # TODO: replace this placeholder with your implementation.
    return np.zeros(shape, dtype=bool)


def apply_frequency_mask(shifted: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a mask in the frequency domain and return the filtered image.

    Steps: multiply the shifted spectrum by the mask, undo the shift with
    ``np.fft.ifftshift``, invert with ``np.fft.ifft2``, take the real part.

    TODO (student 5): implement these four steps in order.
    """
    # TODO: replace this placeholder with your implementation.
    return np.zeros(shifted.shape, dtype=float)


def plot_results(original: np.ndarray, shifted: np.ndarray,
                 lowpass: np.ndarray, highpass: np.ndarray,
                 show: bool = True) -> None:
    """Show the original image, log spectrum, low-pass and high-pass results."""
    figure, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes[0, 0].set_title("Original - TODO: imshow")
    axes[0, 1].set_title("Log magnitude spectrum - TODO: imshow")
    axes[1, 0].set_title("Ideal low-pass result - TODO: imshow")
    axes[1, 1].set_title("Ideal high-pass result - TODO: imshow")
    for ax in axes.ravel():
        ax.set_axis_off()
        ax.text(0.5, 0.5, "TODO: add image", transform=ax.transAxes,
                ha="center", va="center", color="grey")
    figure.tight_layout()
    if show:
        plt.show()


def main(image_path=None, show: bool = True) -> None:
    """Run the Student 5 experiment; fall back to a demo image if none exists."""
    print("SKELETON MODE - implement the TODOs in this file to complete Student 5.")

    image = load_grayscale_image(image_path)
    print(f"  Image shape: {image.shape}")

    shifted = compute_fft_shifted(image)
    lowpass_mask = create_circular_mask(image.shape, radius=30, kind="lowpass")
    highpass_mask = create_circular_mask(image.shape, radius=30, kind="highpass")
    lowpass = apply_frequency_mask(shifted, lowpass_mask)
    highpass = apply_frequency_mask(shifted, highpass_mask)

    print("  TODO: report what each mask removes and why ringing may appear.")
    plot_results(image, shifted, lowpass, highpass, show=show)
    print("Student 5 skeleton finished.")


if __name__ == "__main__":
    main()
