#!/usr/bin/env python3
"""Student 4 - Audio Filtering (SKELETON).

Your tasks:
    1. Design a second-order notch filter (calculate the coefficients b, a).
    2. Apply it to the interfered audio and save the filtered WAV.
    3. Compare the spectra before and after filtering.

Run standalone with:
    python students/student4_audio_filtering/student4.py

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

from utils import helpers  # optional shared load/save/plot helpers


def design_notch_filter(sample_rate: float, notch_freq: float,
                        bandwidth: float):
    """Design a second-order IIR notch filter.

    Parameters
    ----------
    sample_rate : float
        Sampling frequency in Hz.
    notch_freq : float
        Frequency to reject, in Hz (e.g. 50 or 60).
    bandwidth : float
        Width of the rejected band around ``notch_freq``, in Hz.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Numerator coefficients ``b`` and denominator coefficients ``a``.

    TODO (student 4):
        - Compute omega_0 = 2*pi*notch_freq/sample_rate and Q = notch_freq/bandwidth.
        - Calculate the 2nd-order coefficients yourself, or use
          ``scipy.signal.iirnotch(notch_freq, Q, fs=sample_rate)`` as a cross-check.
        - Plot the frequency response with ``scipy.signal.freqz(b, a)``.
        - Explain the pole radius / bandwidth trade-off in comments.
    """
    # TODO: replace this placeholder with your implementation.
    b = np.array([1.0])
    a = np.array([1.0])
    return b, a


def apply_notch(audio: np.ndarray, b: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Filter audio with the designed notch filter.

    Parameters
    ----------
    audio : numpy.ndarray
        Input samples (possibly containing interference).
    b, a : numpy.ndarray
        Filter coefficients from :func:`design_notch_filter`.

    Returns
    -------
    numpy.ndarray
        Filtered samples.

    TODO (student 4):
        - Use ``scipy.signal.filtfilt(b, a, audio)`` for zero-phase filtering,
          or ``scipy.signal.lfilter`` and comment on the phase delay.
    """
    # TODO: replace this placeholder with your implementation.
    return np.array(audio, dtype=float, copy=True)


def save_filtered_audio(path: Path | str, filtered: np.ndarray,
                        sample_rate: float) -> Path:
    """Save the filtered signal as a WAV file in ``data/audio/``.

    TODO (student 4): call ``helpers.save_audio(path, filtered, sample_rate)``
    once your filter works. Suggested filename: ``data/audio/filtered.wav``.
    """
    print(f"  TODO: save filtered audio to {path} (not written in skeleton mode).")
    return Path(path)


def plot_spectra_comparison(noisy: np.ndarray, filtered: np.ndarray,
                            sample_rate: float, show: bool = True) -> None:
    """Plot spectra before and after notch filtering.

    TODO (student 4): use ``helpers.plot_spectrum`` or :mod:`numpy.fft`,
    zoom near the notch frequency, and mark the rejected band.
    """
    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].set_title("Before filtering - TODO: plot spectrum")
    axes[1].set_title("After notch filtering - TODO: plot spectrum")
    axes[1].set_xlabel("Frequency (Hz)")
    for ax in axes:
        ax.set_ylabel("Magnitude")
        ax.grid(True, alpha=0.3)
        ax.text(0.5, 0.5, "TODO: add your curve", transform=ax.transAxes,
                ha="center", va="center", color="grey")
    figure.tight_layout()
    if show:
        plt.show()


def main(audio_path=None, show: bool = True) -> None:
    """Run the Student 4 experiment; fall back to a demo signal if no WAV exists."""
    print("SKELETON MODE - implement the TODOs in this file to complete Student 4.")

    if audio_path is None or not Path(audio_path).is_file():
        print("  [!] No WAV file supplied - using a synthetic demo signal.")
        print("      Place a clean .wav file in data/audio/ to use real audio.")
        noisy, sample_rate = helpers.make_synthetic_audio()
    else:
        noisy, sample_rate = helpers.load_audio(audio_path)
        print(f"  Loaded {Path(audio_path).name} at {sample_rate} Hz.")

    notch_freq = 50.0
    b, a = design_notch_filter(sample_rate, notch_freq, bandwidth=5.0)
    print(f"  Coefficients (placeholders): b={b}, a={a}")
    filtered = apply_notch(noisy, b, a)

    output_path = helpers.AUDIO_DIR / "filtered_demo.wav"
    save_filtered_audio(output_path, filtered, sample_rate)
    plot_spectra_comparison(noisy, filtered, sample_rate, show=show)
    print("Student 4 skeleton finished.")


if __name__ == "__main__":
    main()
