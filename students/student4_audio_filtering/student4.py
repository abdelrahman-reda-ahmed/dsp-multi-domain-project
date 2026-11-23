#!/usr/bin/env python3
"""Student 4 - Audio Filtering.

Tasks covered here:
    1. Design a second-order IIR notch filter to remove power-line hum (50/60 Hz).
    2. Apply the filter to a corrupted audio signal with zero phase.
    3. Save the filtered WAV and compare the before/after spectra.

Run standalone with:
    python students/student4_audio_filtering/student4.py
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
from scipy import signal

# Allow direct execution of this file (adds the project root to sys.path).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import helpers


def design_notch_filter(sample_rate, notch_freq, bandwidth):
    """Design a second-order IIR notch filter.

    Parameters
    ----------
    sample_rate : float
        Sampling frequency in Hz.
    notch_freq : float
        Centre frequency of the notch in Hz (e.g. 50 or 60 for mains hum).
    bandwidth : float
        Width of the notch in Hz at the -3 dB points.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(b, a)`` numerator/denominator coefficients of the notch filter.
    """
    sample_rate = float(sample_rate)
    notch_freq = float(notch_freq)
    bandwidth = float(bandwidth)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not 0.0 < notch_freq < sample_rate / 2.0:
        raise ValueError("notch_freq must lie between 0 and the Nyquist frequency")
    if bandwidth <= 0:
        raise ValueError("bandwidth must be positive")

    # A second-order notch's quality factor is centre frequency / bandwidth.
    quality = notch_freq / bandwidth
    b, a = signal.iirnotch(w0=notch_freq, Q=quality, fs=sample_rate)
    return b, a


def apply_notch(audio, b, a):
    """Apply a notch filter to a 1-D signal with zero phase.

    ``filtfilt`` is used so the filtered audio has no phase distortion; short
    signals that cannot accommodate its padding fall back to ``lfilter``.

    Parameters
    ----------
    audio : array-like
        Mono input samples.
    b, a : array-like
        Numerator and denominator filter coefficients.

    Returns
    -------
    numpy.ndarray
        Filtered samples with the same length as ``audio``.
    """
    audio = np.asarray(audio, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a = np.asarray(a, dtype=np.float64)

    padlen = 3 * max(a.size, b.size)
    if audio.size > padlen:
        return signal.filtfilt(b, a, audio)
    return signal.lfilter(b, a, audio)


def save_filtered_audio(path, filtered, sample_rate):
    """Save filtered samples as a 16-bit PCM WAV and return the output path."""
    return helpers.save_audio(path, filtered, int(sample_rate))


def plot_spectra_comparison(noisy, filtered, sample_rate, show=True):
    """Plot zoomed spectra of the noisy and filtered signals side by side."""
    noisy_freq, noisy_mag = _spectrum(noisy, sample_rate)
    _, filtered_mag = _spectrum(filtered, sample_rate)

    figure, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(noisy_freq, noisy_mag, linewidth=1.0, color="tab:red", label="With hum")
    axes[0].set_title("Before filtering - hum spike present")
    axes[1].plot(noisy_freq, filtered_mag,
                 linewidth=1.0, color="tab:green", label="Filtered")
    axes[1].set_title("After filtering - spike suppressed")
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


def _spectrum(values, sample_rate):
    """Return the single-sided magnitude spectrum of a 1-D signal."""
    values = np.asarray(values, dtype=np.float64)
    n = values.size
    spectrum = np.fft.rfft(values)
    frequencies = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
    magnitude = (2.0 / n) * np.abs(spectrum)
    magnitude[0] /= 2.0
    if n % 2 == 0:
        magnitude[-1] /= 2.0
    return frequencies, magnitude


def main(audio_path=None, show=True):
    """Run the Student 4 experiment; fall back to a demo signal if no WAV exists."""
    print("Designing a notch filter and removing power-line hum...")

    notch_freq = 50.0
    bandwidth = 4.0
    hum_amplitude = 0.1

    if audio_path is None:
        audio_path = helpers.find_data_file(helpers.AUDIO_DIR, helpers.AUDIO_EXTENSIONS)

    if audio_path is None or not Path(audio_path).is_file():
        print("  [!] No WAV file supplied - using a synthetic demo signal.")
        clean, sample_rate = helpers.make_synthetic_audio()
        output_name = "filtered_demo.wav"
    else:
        clean, sample_rate = helpers.load_audio(audio_path)
        print(f"  Loaded {Path(audio_path).name} at {sample_rate} Hz.")
        output_name = "filtered.wav"

    clean = np.asarray(clean, dtype=np.float64)
    t = np.arange(clean.size) / float(sample_rate)
    noisy = np.clip(clean + hum_amplitude * np.sin(2.0 * np.pi * notch_freq * t),
                    -1.0, 1.0)

    b, a = design_notch_filter(sample_rate, notch_freq, bandwidth)
    filtered = np.clip(apply_notch(noisy, b, a), -1.0, 1.0)

    noisy_mse, noisy_psnr = helpers.calculate_mse_psnr(clean, noisy)
    filtered_mse, filtered_psnr = helpers.calculate_mse_psnr(clean, filtered)
    print(f"  Notch: f0={notch_freq:.0f} Hz, BW={bandwidth:.1f} Hz, "
          f"Q={notch_freq / bandwidth:.1f}")
    print(f"  MSE  before/after: {noisy_mse:.6f} / {filtered_mse:.6f}")
    print(f"  PSNR before/after: {noisy_psnr:.2f} dB / {filtered_psnr:.2f} dB")

    output_path = save_filtered_audio(
        helpers.AUDIO_DIR / output_name, filtered, sample_rate
    )
    print(f"  Filtered audio saved to {output_path.relative_to(PROJECT_ROOT)}")

    plot_spectra_comparison(noisy, filtered, sample_rate, show=show)
    print("Student 4 finished.")


if __name__ == "__main__":
    main()
