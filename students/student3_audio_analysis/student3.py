#!/usr/bin/env python3
"""Student 3 - Audio Analysis.

Tasks covered here:
    1. Load a clean WAV file from ``data/audio/`` (or use a demo signal).
    2. Add 50 Hz / 60 Hz power-line interference.
    3. Plot waveforms and spectra, then identify the interference frequency.

Run standalone with:
    python students/student3_audio_analysis/student3.py
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

from utils import helpers


def add_powerline_interference(clean, sample_rate, interference_freq=50.0, amplitude=0.1):
    """Add sinusoidal power-line interference to a clean audio signal.

    Parameters
    ----------
    clean : array-like
        Clean mono audio samples.
    sample_rate : float
        Sampling frequency in Hz.
    interference_freq : float
        Interference frequency in Hz (50 or 60 for mains power).
    amplitude : float
        Amplitude of the added sine wave.

    Returns
    -------
    numpy.ndarray
        Corrupted signal with the same length as ``clean``, clipped to [-1, 1].
    """
    clean = np.asarray(clean, dtype=np.float64)
    t = np.arange(clean.size) / float(sample_rate)
    interference = amplitude * np.sin(2.0 * np.pi * interference_freq * t)
    return np.clip(clean + interference, -1.0, 1.0)


def compute_spectrum(signal, sample_rate):
    """Compute the single-sided magnitude spectrum of a 1-D signal.

    Uses the real FFT and scales the magnitudes so a sinusoid of amplitude A
    produces a peak of about A. DC and the Nyquist bin are not doubled.

    Parameters
    ----------
    signal : array-like
        Time-domain samples.
    sample_rate : float
        Sampling frequency in Hz.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(frequencies_hz, magnitude)``.
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = signal.size
    spectrum = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
    magnitude = (2.0 / n) * np.abs(spectrum)
    magnitude[0] /= 2.0                     # DC is not doubled
    if n % 2 == 0:                          # Nyquist bin is not doubled either
        magnitude[-1] /= 2.0
    return frequencies, magnitude


def plot_waveforms(clean, noisy, sample_rate, show=True):
    """Plot a short window of the clean and corrupted waveforms.

    The window is limited to 50 ms (or the full signal if shorter) so the
    high-frequency interference is visible on the time axis.
    """
    clean = np.asarray(clean, dtype=np.float64)
    noisy = np.asarray(noisy, dtype=np.float64)
    window = min(clean.size, int(0.05 * sample_rate))
    t = np.arange(window) / float(sample_rate)

    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(t, clean[:window], linewidth=1.0, color="tab:blue")
    axes[0].set_title(f"Clean audio (first {window} samples)")
    axes[1].plot(t, noisy[:window], linewidth=1.0, color="tab:red")
    axes[1].set_title("Audio with power-line interference")
    axes[1].set_xlabel("Time (s)")

    for ax in axes:
        ax.set_ylabel("Amplitude")
        ax.grid(True, alpha=0.3)
    figure.tight_layout()
    if show and plt.get_backend().lower() != "agg":
        plt.show()


def plot_spectra(clean, noisy, sample_rate, show=True):
    """Plot zoomed spectra (0-200 Hz) and mark the 50 Hz / 60 Hz candidates."""
    clean_freq, clean_mag = compute_spectrum(clean, sample_rate)
    noisy_freq, noisy_mag = compute_spectrum(noisy, sample_rate)

    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(clean_freq, clean_mag, linewidth=1.0, color="tab:blue")
    axes[0].set_title("Clean audio spectrum")
    axes[1].plot(noisy_freq, noisy_mag, linewidth=1.0, color="tab:red")
    axes[1].set_title("Noisy spectrum - interference peak near 50 or 60 Hz")
    axes[1].set_xlabel("Frequency (Hz)")

    for ax in axes:
        ax.set_xlim(0, 200)
        ax.set_ylabel("Magnitude")
        ax.grid(True, alpha=0.3)
        for candidate in (50.0, 60.0):
            ax.axvline(candidate, color="grey", linestyle="--", linewidth=0.8,
                       label=f"{candidate:.0f} Hz")
        ax.legend(loc="upper right", fontsize=8)
    figure.tight_layout()
    if show and plt.get_backend().lower() != "agg":
        plt.show()


def identify_interference_frequency(noisy, sample_rate, candidates=(50.0, 60.0)):
    """Estimate which candidate frequency dominates the corrupted spectrum.

    Measures the maximum spectral magnitude within +/-2 Hz of each candidate
    and returns the frequency with the larger value.

    Returns
    -------
    float or None
        Estimated interference frequency in Hz, or ``None`` when the spectrum
        cannot be computed (empty signal).
    """
    frequencies, magnitude = compute_spectrum(noisy, sample_rate)
    if frequencies.size == 0:
        print("  Could not identify interference: empty signal.")
        return None

    scores = {}
    for candidate in candidates:
        nearby = (frequencies >= candidate - 2.0) & (frequencies <= candidate + 2.0)
        scores[candidate] = float(magnitude[nearby].max()) if nearby.any() else 0.0
        print(f"  Energy near {candidate:.0f} Hz: {scores[candidate]:.4f}")

    detected = max(scores, key=scores.get)
    if scores[detected] <= 0.0:
        print("  No interference peak found near the candidate frequencies.")
        return None
    print(f"  -> Interference identified as {detected:.0f} Hz")
    return detected


def main(audio_path=None, show=True):
    """Run the Student 3 experiment; fall back to a demo signal if no WAV exists."""
    print("Loading audio and adding power-line interference...")

    if audio_path is None or not Path(audio_path).is_file():
        print("  [!] No WAV file supplied - using a synthetic demo signal.")
        print("      Place a clean .wav file in data/audio/ to use real audio.")
        clean, sample_rate = helpers.make_synthetic_audio()
    else:
        clean, sample_rate = helpers.load_audio(audio_path)
        print(f"  Loaded {Path(audio_path).name} at {sample_rate} Hz.")

    interference_freq = 50.0
    noisy = add_powerline_interference(clean, sample_rate,
                                       interference_freq=interference_freq,
                                       amplitude=0.1)

    frequencies, magnitude = compute_spectrum(noisy, sample_rate)
    print(f"  Samples: {noisy.size}, spectrum bins: {frequencies.size}")
    print(f"  Injected interference: {interference_freq:.0f} Hz")

    plot_waveforms(clean, noisy, sample_rate, show=show)
    plot_spectra(clean, noisy, sample_rate, show=show)
    detected = identify_interference_frequency(noisy, sample_rate)
    print(f"  Detected frequency: {detected} Hz (expected {interference_freq:.0f} Hz)")
    print("Student 3 finished.")


if __name__ == "__main__":
    main()
