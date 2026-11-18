#!/usr/bin/env python3
"""Student 1 - Mathematical Signal & DTFT.

Tasks covered here:
    1. Generate a discrete-time signal: two sinusoids + Gaussian noise.
    2. Compute the DTFT manually with loops (no FFT allowed for this exercise).
    3. Plot the magnitude and phase spectra.

Run standalone with:
    python students/student1_mathematical_signal_dtft/student1.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Keep plotting non-blocking when there is no display server (e.g. CI).
if not os.environ.get("DISPLAY") and sys.platform != "win32":
    os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib.pyplot as plt

# Allow direct execution of this file (adds the project root to sys.path).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def generate_signal(sample_rate=1000.0, duration=0.25, f1=50.0, f2=120.0,
                    amplitude1=1.0, amplitude2=0.6, noise_std=0.3, seed=0):
    """Build x[n] = A1*sin(2*pi*f1*n/fs) + A2*cos(2*pi*f2*n/fs) + noise.

    Parameters
    ----------
    sample_rate : float
        Sampling frequency fs in Hz.
    duration : float
        Length of the signal in seconds (kept short so the manual DTFT is fast).
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
    """
    n = np.arange(int(sample_rate * duration))
    rng = np.random.default_rng(seed)

    sinusoid_1 = amplitude1 * np.sin(2 * np.pi * f1 * n / sample_rate)
    sinusoid_2 = amplitude2 * np.cos(2 * np.pi * f2 * n / sample_rate)
    noise = noise_std * rng.standard_normal(n.size)
    signal = sinusoid_1 + sinusoid_2 + noise

    return n, signal


def compute_dtft(signal, omega_values):
    """Evaluate X(e^jw) = sum_n x[n] * exp(-j*w*n) on a grid of frequencies.

    This uses explicit Python loops and deliberately does NOT call numpy.fft.
    Each output is a complex inner product between the signal and a complex
    sinusoid at frequency w.

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
    """
    signal = np.asarray(signal, dtype=np.float64)
    result = np.zeros(omega_values.size, dtype=complex)

    for k, omega in enumerate(omega_values):       # loop over frequencies
        accumulator = 0.0 + 0.0j
        for n, sample in enumerate(signal):        # loop over time samples
            accumulator += sample * np.exp(-1j * omega * n)
        result[k] = accumulator

    return result


def plot_magnitude_phase(omega, spectrum, sample_rate, show=True):
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
    """
    frequency_hz = omega * sample_rate / (2 * np.pi)

    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(frequency_hz, np.abs(spectrum), color="tab:blue")
    axes[0].set_title("DTFT magnitude")
    axes[0].set_ylabel("|X(e^jw)|")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(frequency_hz, np.angle(spectrum, deg=True), color="tab:orange")
    axes[1].set_title("DTFT phase")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Phase (degrees)")
    axes[1].grid(True, alpha=0.3)

    figure.tight_layout()
    if show and plt.get_backend().lower() != "agg":  # skip on non-interactive backends
        plt.show()


def main(sample_rate=1000.0, show=True):
    """Run the Student 1 experiment end to end."""
    print("Generating two sinusoids plus Gaussian noise (manual DTFT, no FFT)...")

    n, signal = generate_signal(sample_rate=sample_rate)
    omega = np.linspace(-np.pi, np.pi, 512, endpoint=False)
    spectrum = compute_dtft(signal, omega)

    print(f"  Samples generated      : {signal.size}")
    print(f"  Frequency grid points  : {omega.size}")
    print(f"  Max DTFT magnitude     : {np.abs(spectrum).max():.3f}")

    # Locate the two strongest positive-frequency peaks, keeping them at least
    # 20 Hz apart so sidelobe leakage around f1 does not mask the f2 peak.
    positive = omega >= 0
    magnitude = np.abs(spectrum[positive])
    positive_hz = omega[positive] * sample_rate / (2 * np.pi)
    selected = []
    for index in np.argsort(magnitude)[::-1]:
        if all(abs(positive_hz[index] - positive_hz[other]) > 20 for other in selected):
            selected.append(index)
        if len(selected) == 2:
            break
    peak_hz = np.sort(positive_hz[selected])
    print(f"  Dominant frequencies   : {peak_hz} Hz (expected near 50 and 120 Hz)")

    plot_magnitude_phase(omega, spectrum, sample_rate, show=show)
    print("Student 1 finished.")


if __name__ == "__main__":
    main()
