#!/usr/bin/env python3
"""Student 3 - Audio Analysis (SKELETON).

Your tasks:
    1. Load the clean WAV file from ``data/audio/``.
    2. Add 50 Hz or 60 Hz sinusoidal power-line interference.
    3. Plot waveforms and spectra, and identify the interference frequency.

Run standalone with:
    python students/student3_audio_analysis/student3.py

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

from utils import helpers  # optional shared load/plot helpers


def add_powerline_interference(clean: np.ndarray, sample_rate: float,
                               interference_freq: float = 50.0,
                               amplitude: float = 0.1) -> np.ndarray:
    """Add a sinusoidal power-line interference to a clean audio signal.

    Parameters
    ----------
    clean : numpy.ndarray
        Clean mono audio samples.
    sample_rate : float
        Sampling frequency in Hz.
    interference_freq : float
        Frequency of the added interference (50 or 60 Hz).
    amplitude : float
        Amplitude of the added sine wave.

    Returns
    -------
    numpy.ndarray
        The corrupted signal, same length as ``clean``.

    TODO (student 3):
        - Build a time vector ``t = np.arange(clean.size) / sample_rate``.
        - Return ``clean + amplitude * np.sin(2*pi*interference_freq*t)``.
        - Try both 50 Hz and 60 Hz and note which matches your region.
    """
    # TODO: replace this placeholder with your implementation.
    return np.array(clean, dtype=float, copy=True)


def compute_spectrum(signal: np.ndarray, sample_rate: float):
    """Compute a single-sided magnitude spectrum of a 1-D signal.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(frequencies_hz, magnitude)``.

    TODO (student 3):
        - Use ``np.fft.rfft`` and ``np.fft.rfftfreq``.
        - Normalise the magnitude by the number of samples.
        - (Optional) start from ``helpers.plot_spectrum`` and adapt it.
    """
    # TODO: replace this placeholder with your implementation.
    signal = np.asarray(signal, dtype=float)
    frequencies = np.fft.rfftfreq(signal.size, d=1.0 / sample_rate)
    return frequencies, np.zeros_like(frequencies)


def plot_waveforms(clean: np.ndarray, noisy: np.ndarray, sample_rate: float,
                   show: bool = True) -> None:
    """Plot clean and corrupted waveforms on the same time axis.

    TODO (student 3): show a short time window so the interference is visible.
    """
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].set_title("Clean audio waveform - TODO: plot")
    axes[1].set_title("Audio with interference - TODO: plot")
    axes[1].set_xlabel("Time (s)")
    for ax in axes:
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
        ax.text(0.5, 0.5, "TODO: add your curve", transform=ax.transAxes,
                ha="center", va="center", color="grey")
    figure.tight_layout()
    if show:
        plt.show()


def plot_spectra(clean: np.ndarray, noisy: np.ndarray, sample_rate: float,
                 show: bool = True) -> None:
    """Plot clean and corrupted magnitude spectra and mark the interference.

    TODO (student 3): use :func:`compute_spectrum` and zoom near 0-200 Hz.
    """
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].set_title("Clean spectrum - TODO: plot")
    axes[1].set_title("Noisy spectrum - TODO: locate the interference peak")
    axes[1].set_xlabel("Frequency (Hz)")
    for ax in axes:
        ax.set_ylabel("Magnitude")
        ax.grid(True, alpha=0.3)
        ax.text(0.5, 0.5, "TODO: add your curve", transform=ax.transAxes,
                ha="center", va="center", color="grey")
    figure.tight_layout()
    if show:
        plt.show()


def identify_interference_frequency(noisy: np.ndarray, sample_rate: float,
                                    candidates=(50.0, 60.0)):
    """Estimate which candidate frequency dominates the corrupted spectrum.

    Parameters
    ----------
    noisy : numpy.ndarray
        Corrupted audio samples.
    sample_rate : float
        Sampling frequency in Hz.
    candidates : tuple[float, ...]
        Frequencies to test, typically 50 and 60 Hz.

    Returns
    -------
    float or None
        The estimated interference frequency in Hz.

    TODO (student 3):
        - Compute the spectrum, then measure energy near each candidate (+-2 Hz).
        - Return the candidate with the larger spectral magnitude.
        - Print the result and justify it from your spectrum plot.
    """
    # TODO: replace this placeholder with your implementation.
    print("  TODO: determine whether the interference is 50 Hz or 60 Hz.")
    return None


def main(audio_path=None, show: bool = True) -> None:
    """Run the Student 3 experiment; fall back to a demo signal if no WAV exists."""
    print("SKELETON MODE - implement the TODOs in this file to complete Student 3.")

    if audio_path is None or not Path(audio_path).is_file():
        print("  [!] No WAV file supplied - using a synthetic demo signal.")
        print("      Place a clean .wav file in data/audio/ to use real audio.")
        clean, sample_rate = helpers.make_synthetic_audio()
    else:
        clean, sample_rate = helpers.load_audio(audio_path)
        print(f"  Loaded {Path(audio_path).name} at {sample_rate} Hz.")

    noisy = add_powerline_interference(clean, sample_rate, interference_freq=50.0)
    freq, magnitude = compute_spectrum(noisy, sample_rate)
    print(f"  Spectrum bins: {freq.size} (TODO: fill in the magnitude values)")

    plot_waveforms(clean, noisy, sample_rate, show=show)
    plot_spectra(clean, noisy, sample_rate, show=show)
    detected = identify_interference_frequency(noisy, sample_rate)
    print(f"  Identified interference: {detected}")
    print("Student 3 skeleton finished.")


if __name__ == "__main__":
    main()
