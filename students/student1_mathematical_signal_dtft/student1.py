#!/usr/bin/env python3
"""Student 1 - Mathematical Signal & DTFT (SKELETON).

Your tasks:
    1. Generate a discrete-time signal: two sinusoids + Gaussian noise.
    2. Compute the DTFT manually with loops (no FFT allowed for this exercise).
    3. Plot the magnitude and phase spectra.

Run standalone with:
    python students/student1_mathematical_signal_dtft/student1.py

Implement every TODO below. The placeholder bodies let the script run today.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Allow direct execution of this file (adds the project root to sys.path).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# from utils import helpers  # optional shared utilities


def generate_signal(sample_rate: float = 1000.0, duration: float = 0.25,
                    f1: float = 50.0, f2: float = 120.0,
                    amplitude1: float = 1.0, amplitude2: float = 0.6,
                    noise_std: float = 0.3, seed: int = 0):
    """Build x[n] = A1*sin(2*pi*f1*n/fs) + A2*cos(2*pi*f2*n/fs) + noise.

    Parameters
    ----------
    sample_rate : float
        Sampling frequency fs in Hz.
    duration : float
        Length of the signal in seconds.
    f1, f2 : float
        Frequencies of the two sinusoids, in Hz.
    amplitude1, amplitude2 : float
        Amplitudes of the two sinusoids.
    noise_std : float
        Standard deviation of the additive Gaussian noise.
    seed : int
        Seed for reproducible random noise.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(n, signal)``: integer sample indices and the generated samples.

    TODO (student 1):
        - Build ``n = np.arange(int(sample_rate * duration))``.
        - Create both sinusoids and add ``noise_std * rng.standard_normal(n.size)``.
        - Use ``np.random.default_rng(seed)`` so results are reproducible.
    """
    # TODO: replace this placeholder with your implementation.
    n = np.arange(int(sample_rate * duration))
    signal = np.zeros_like(n, dtype=float)
    return n, signal


def compute_dtft(signal: np.ndarray, omega_values: np.ndarray) -> np.ndarray:
    """Evaluate X(e^jw) = sum_n x[n] * exp(-j*w*n) on a grid of frequencies.

    Parameters
    ----------
    signal : numpy.ndarray
        The discrete-time samples x[n].
    omega_values : numpy.ndarray
        Normalised angular frequencies in radians/sample.

    Returns
    -------
    numpy.ndarray
        Complex DTFT values, one per element of ``omega_values``.

    TODO (student 1):
        - Loop over every frequency w, then over every sample n.
        - Accumulate ``sample * np.exp(-1j * omega * n)`` into a complex sum.
        - Do NOT use ``numpy.fft`` anywhere in this function.
        - Check that peaks appear near +/- 2*pi*f/fs for both sinusoids.
    """
    # TODO: replace this placeholder with an explicit double loop (no FFT).
    return np.zeros(omega_values.size, dtype=complex)


def plot_magnitude_phase(omega: np.ndarray, spectrum: np.ndarray,
                         sample_rate: float, show: bool = True) -> None:
    """Plot the DTFT magnitude and phase against frequency in Hz.

    Parameters
    ----------
    omega : numpy.ndarray
        Normalised angular frequencies (radians/sample).
    spectrum : numpy.ndarray
        Complex DTFT values from :func:`compute_dtft`.
    sample_rate : float
        Sampling frequency used to convert w to Hz.
    show : bool
        Call ``plt.show()`` when True.

    TODO (student 1):
        - Convert with ``f = omega * sample_rate / (2 * np.pi)``.
        - Top axes: |X(e^jw)|; bottom axes: np.angle(X, deg=True).
        - Label axes and explain erratic phase away from the peaks.
    """
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].set_title("DTFT magnitude - TODO: plot |X(e^jw)|")
    axes[0].set_ylabel("Magnitude")
    axes[1].set_title("DTFT phase - TODO: plot phase in degrees")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Phase (deg)")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.text(0.5, 0.5, "TODO: add your curve", transform=ax.transAxes,
                ha="center", va="center", color="grey")
    figure.tight_layout()
    if show:
        plt.show()


def main(sample_rate: float = 1000.0, show: bool = True) -> None:
    """Run the Student 1 experiment end to end."""
    print("SKELETON MODE - implement the TODOs in this file to complete Student 1.")
    print("Generating two sinusoids plus Gaussian noise for a manual DTFT...")

    n, signal = generate_signal(sample_rate=sample_rate)
    omega = np.linspace(-np.pi, np.pi, 512, endpoint=False)
    spectrum = compute_dtft(signal, omega)

    print(f"  Samples generated     : {signal.size}")
    print(f"  Frequency grid points : {omega.size}")
    print("  TODO: report the two dominant frequencies and compare with f1/f2.")

    plot_magnitude_phase(omega, spectrum, sample_rate, show=show)
    print("Student 1 skeleton finished.")


if __name__ == "__main__":
    main()
