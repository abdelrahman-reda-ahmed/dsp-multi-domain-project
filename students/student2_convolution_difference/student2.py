#!/usr/bin/env python3
"""Student 2 - Convolution & Difference Equation (SKELETON).

Your tasks:
    1. Moving average using ``numpy.convolve``.
    2. Compare time-domain convolution with frequency-domain multiplication
       after zero-padding both sequences.
    3. Implement a simple first-order difference equation.

Run standalone with:
    python students/student2_convolution_difference/student2.py

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

# from utils import helpers  # optional shared utilities


def moving_average(signal: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Smooth a 1-D signal with a length-``window_size`` moving-average FIR filter.

    Parameters
    ----------
    signal : numpy.ndarray
        Input samples.
    window_size : int
        Number of taps in the averaging window.

    Returns
    -------
    numpy.ndarray
        Smoothed samples with the same length as the input.

    TODO (student 2):
        - Build the kernel ``np.ones(window_size) / window_size``.
        - Apply it with ``np.convolve(signal, kernel, mode="same")``.
        - Compare odd/even window sizes and comment on delay and edge effects.
    """
    # TODO: replace this placeholder with your implementation.
    return np.zeros_like(np.asarray(signal, dtype=float))


def time_domain_convolution(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Compute the full linear convolution y[n] = x[n] * h[n] directly.

    Returns an array of length ``len(x) + len(h) - 1``.
    Use ``np.convolve(x, h, mode="full")``.

    TODO (student 2): implement this function.
    """
    # TODO: replace this placeholder with your implementation.
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    return np.zeros(x.size + h.size - 1)


def frequency_domain_convolution(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Compute the same linear convolution via FFT multiplication.

    Zero-pad both inputs to ``len(x) + len(h) - 1`` first, multiply their FFTs,
    then take the inverse FFT and return the real part.

    Returns
    -------
    numpy.ndarray
        Convolution result of length ``len(x) + len(h) - 1``.

    TODO (student 2):
        - Verify ``max(abs(direct - via_fft))`` is near machine precision.
        - Explain in comments why zero-padding is required (circular aliasing).
    """
    # TODO: replace this placeholder with your implementation.
    x = np.asarray(x, dtype=float)
    h = np.asarray(h, dtype=float)
    return np.zeros(x.size + h.size - 1)


def difference_equation(x: np.ndarray, a: float = 0.9, b: float = 1.0,
                        y0: float = 0.0) -> np.ndarray:
    """Solve the first-order recursion y[n] = a*y[n-1] + b*x[n].

    Parameters
    ----------
    x : numpy.ndarray
        Input sequence x[n].
    a : float
        Feedback coefficient (|a| < 1 for a stable filter).
    b : float
        Input scaling coefficient.
    y0 : float
        Initial condition for y[-1].

    Returns
    -------
    numpy.ndarray
        Output sequence y[n], same length as ``x``.

    TODO (student 2):
        - Loop over the samples and compute ``a * previous + b * sample``.
        - Update the stored previous output each iteration.
        - Derive H(z) = b / (1 - a*z^-1) on paper; try a < 0 and a > 1.
    """
    # TODO: replace this placeholder with your implementation.
    return np.zeros_like(np.asarray(x, dtype=float))


def compare_convolutions(x: np.ndarray, h: np.ndarray, show: bool = True) -> float:
    """Compare the two convolution methods and plot both results.

    Returns the maximum absolute difference between them.
    """
    direct = time_domain_convolution(x, h)
    via_fft = frequency_domain_convolution(x, h)
    max_difference = float(np.max(np.abs(direct - via_fft)))

    print(f"  Direct result length : {direct.size}")
    print(f"  FFT result length    : {via_fft.size}")
    print(f"  Max |direct - FFT|   : {max_difference:.3e} (placeholder)")

    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].set_title("Convolution - TODO: plot direct vs FFT results")
    axes[0].set_ylabel("Amplitude")
    axes[1].set_title("TODO: plot the absolute difference")
    axes[1].set_xlabel("Sample n")
    axes[1].set_ylabel("|difference|")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.text(0.5, 0.5, "TODO: add your curve", transform=ax.transAxes,
                ha="center", va="center", color="grey")
    figure.tight_layout()
    if show:
        plt.show()
    return max_difference


def main(show: bool = True) -> None:
    """Run the Student 2 experiment end to end."""
    print("SKELETON MODE - implement the TODOs in this file to complete Student 2.")
    print("Demonstrating convolution and a first-order difference equation...")

    t = np.arange(200)
    x = np.sin(2 * np.pi * 0.05 * t)
    h = np.ones(5) / 5.0

    smoothed = moving_average(x, window_size=5)
    compare_convolutions(x, h, show=show)
    y = difference_equation(x, a=0.9, b=0.1, y0=0.0)

    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].set_title("TODO: plot input vs moving-average output")
    axes[0].set_ylabel("Amplitude")
    axes[1].set_title("TODO: plot the IIR output y[n]")
    axes[1].set_xlabel("Sample n")
    axes[1].set_ylabel("Amplitude")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.text(0.5, 0.5, "TODO: add your curve", transform=ax.transAxes,
                ha="center", va="center", color="grey")
    figure.tight_layout()
    if show:
        plt.show()
    print("Student 2 skeleton finished.")


if __name__ == "__main__":
    main()
