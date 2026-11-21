#!/usr/bin/env python3
"""
Student 1 - Mathematical Signal & DTFT
=====================================================

Programming tasks:
1. Generate a signal containing two sinusoidal components and Gaussian noise.
2. Compute the DTFT from its defining equation using loops for a limited
   number of samples and frequency points.
3. Plot the magnitude and phase spectra.

Theory explained in this module:
- Discrete-time signals: x[n] obtained by sampling continuous time at t = n / fs.
- Angular frequency (omega): Digital angular frequency omega = 2 * pi * f / fs (rad/sample),
  where the unique frequency range for real signals is [0, pi] (up to Nyquist fs / 2).
- Discrete-Time Fourier Transform (DTFT):
      X(e^{j*omega}) = sum_{n=0}^{N-1} x[n] * e^{-j * omega * n}
  which maps a discrete-time aperiodic sequence into a continuous, 2*pi-periodic
  complex spectrum in omega.
- Magnitude and Phase:
      |X(e^{j*omega})| = sqrt(Re(X)^2 + Im(X)^2)  (energy/strength at frequency omega)
      angle(X(e^{j*omega})) = atan2(Im(X), Re(X)) (initial phase shift of frequency omega)

Run standalone with:
    python students/student1_mathematical_signal_dtft/student1.py
"""

from __future__ import annotations

import cmath
import os
import sys
from pathlib import Path

# Keep plotting non-blocking when there is no display server (e.g. headless/CI).
if not os.environ.get("DISPLAY") and sys.platform != "win32":
    os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import matplotlib.pyplot as plt

# Allow direct execution of this file (adds the project root to sys.path).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# 1. GENERATE DISCRETE-TIME MATHEMATICAL SIGNAL
# ============================================================

def generate_signal(sample_rate=1000.0, duration=0.25, f1=50.0, f2=120.0,
                    amplitude1=1.0, amplitude2=0.6, noise_std=0.3, seed=0):
    """
    Generate a discrete-time signal containing two sinusoidal components
    and additive white Gaussian noise (AWGN).

    Mathematical model:
        x[n] = A1 * sin(omega1 * n) + A2 * cos(omega2 * n) + w[n]
    where:
        n      = 0, 1, ..., N - 1   (discrete sample index, N = round(fs * duration))
        omega1 = 2 * pi * f1 / fs   (normalized angular frequency 1 in rad/sample)
        omega2 = 2 * pi * f2 / fs   (normalized angular frequency 2 in rad/sample)
        w[n]   ~ N(0, noise_std^2)  (zero-mean Gaussian noise)

    Parameters
    ----------
    sample_rate : float
        Sampling frequency fs in Hz.
    duration : float
        Signal duration in seconds.
    f1 : float
        Frequency of the first sinusoidal component in Hz.
    f2 : float
        Frequency of the second sinusoidal component in Hz.
    amplitude1 : float
        Amplitude A1 of the first sinusoid.
    amplitude2 : float
        Amplitude A2 of the second sinusoid.
    noise_std : float
        Standard deviation (sigma) of the zero-mean Gaussian noise.
    seed : int
        Random number generator seed for reproducibility.

    Returns
    -------
    signal : numpy.ndarray
        1-D float64 array of length N containing the discrete-time signal x[n].
    """
    n_samples = int(round(float(sample_rate) * float(duration)))
    n = np.arange(n_samples, dtype=np.float64)

    # Normalized angular frequencies (rad/sample): omega = 2 * pi * f / fs
    omega1 = 2.0 * np.pi * float(f1) / float(sample_rate)
    omega2 = 2.0 * np.pi * float(f2) / float(sample_rate)

    # Deterministic two-tone component + Gaussian noise
    rng = np.random.default_rng(int(seed))
    tone1 = float(amplitude1) * np.sin(omega1 * n)
    tone2 = float(amplitude2) * np.cos(omega2 * n)
    noise = float(noise_std) * rng.standard_normal(n_samples)

    signal = tone1 + tone2 + noise
    return signal.astype(np.float64)


# ============================================================
# 2. MANUAL DTFT FROM DEFINING EQUATION USING LOOPS
# ============================================================

def compute_dtft(signal, omega_values):
    """
    Compute the Discrete-Time Fourier Transform (DTFT) directly from its
    defining equation using explicit nested loops over a limited number of
    samples (N) and frequency points (M):

        X(e^{j*omega_k}) = sum_{n=0}^{N-1} x[n] * exp(-j * omega_k * n)

    Using Euler's formula:
        exp(-j * omega * n) = cos(omega * n) - j * sin(omega * n)

    Parameters
    ----------
    signal : array-like
        1-D discrete-time signal x[n] of length N.
    omega_values : array-like
        1-D array of angular frequencies omega (in rad/sample) of length M.

    Returns
    -------
    spectrum : numpy.ndarray
        1-D complex128 array of length M containing X(e^{j*omega}) for each omega.
    """
    x = np.asarray(signal, dtype=np.float64)
    omega_grid = np.asarray(omega_values, dtype=np.float64)

    n_samples = x.size
    n_freqs = omega_grid.size
    spectrum = np.zeros(n_freqs, dtype=np.complex128)

    # Convert to plain Python lists so the explicit loops run without NumPy scalar overhead
    x_list = x.tolist()
    omega_list = omega_grid.tolist()
    exp = cmath.exp

    # Outer loop: iterate over each angular frequency point omega_k
    for k in range(n_freqs):
        omega_k = omega_list[k]
        acc = 0.0 + 0.0j
        # Inner loop: sum over all discrete-time sample indices n = 0 .. N-1
        for n in range(n_samples):
            acc += x_list[n] * exp(-1j * omega_k * n)
        spectrum[k] = acc

    return spectrum


# ============================================================
# 3. PLOT MAGNITUDE AND PHASE SPECTRA
# ============================================================

def plot_magnitude_phase(omega, spectrum, sample_rate, show=True):
    """
    Plot the DTFT magnitude and phase spectra against normalized angular
    frequency (omega / pi) and physical frequency (Hz).

    Parameters
    ----------
    omega : array-like
        1-D array of angular frequencies in rad/sample (typically in [0, pi]).
    spectrum : array-like
        1-D complex DTFT values X(e^{j*omega}).
    sample_rate : float
        Sampling rate fs in Hz (used to map omega to physical frequency f = omega * fs / (2*pi)).
    show : bool
        If True and an interactive backend is active, display the figure.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated matplotlib figure containing magnitude and phase subplots.
    """
    omega = np.asarray(omega, dtype=np.float64)
    spectrum = np.asarray(spectrum, dtype=np.complex128)

    magnitude = np.abs(spectrum)
    magnitude_db = 20.0 * np.log10(magnitude + 1e-12)
    phase_wrapped = np.angle(spectrum)
    phase_unwrapped = np.unwrap(phase_wrapped)

    norm_omega = omega / np.pi
    freqs_hz = omega * float(sample_rate) / (2.0 * np.pi)

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    # 1. Linear Magnitude Spectrum |X(e^{j*omega})|
    axes[0].plot(norm_omega, magnitude, color="#3b82f6", linewidth=1.3)
    axes[0].set_title(f"DTFT Linear Magnitude Spectrum |X(e^(jω))|  (fs = {sample_rate:.0f} Hz)")
    axes[0].set_ylabel("Magnitude |X(ω)|")
    axes[0].grid(True, alpha=0.3)

    # Annotate top 2 spectral peaks (excluding DC bin 0)
    if magnitude.size > 4:
        peak1_idx = int(np.argmax(magnitude[1:])) + 1
        masked = magnitude.copy()
        w = max(3, magnitude.size // 50)
        masked[max(0, peak1_idx - w):min(magnitude.size, peak1_idx + w + 1)] = 0.0
        masked[0] = 0.0
        peak2_idx = int(np.argmax(masked))
        axes[0].axvline(norm_omega[peak1_idx], color="#ef4444", linestyle="--", alpha=0.75,
                        label=f"Peak 1: ω={norm_omega[peak1_idx]:.2f}π ({freqs_hz[peak1_idx]:.1f} Hz)")
        axes[0].axvline(norm_omega[peak2_idx], color="#f59e0b", linestyle="--", alpha=0.75,
                        label=f"Peak 2: ω={norm_omega[peak2_idx]:.2f}π ({freqs_hz[peak2_idx]:.1f} Hz)")
        axes[0].legend(loc="upper right", fontsize=9)

    # 2. Logarithmic Magnitude Spectrum (dB)
    axes[1].plot(norm_omega, magnitude_db, color="#06b6d4", linewidth=1.2)
    axes[1].set_title("DTFT Magnitude Spectrum in Decibels (20 log₁₀ |X(e^(jω))|)")
    axes[1].set_ylabel("Magnitude (dB)")
    axes[1].grid(True, alpha=0.3)

    # 3. Phase Spectrum angle(X(e^{j*omega}))
    axes[2].plot(norm_omega, phase_unwrapped, color="#a855f7", linewidth=1.1, label="Unwrapped phase")
    axes[2].set_title("DTFT Phase Spectrum ∠X(e^(jω))")
    axes[2].set_xlabel("Normalized Angular Frequency (ω / π)  [1.0 = Nyquist fs/2]")
    axes[2].set_ylabel("Phase (radians)")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right", fontsize=9)

    fig.suptitle("Student 1: Discrete-Time Fourier Transform (DTFT) Magnitude & Phase", fontsize=14)
    fig.tight_layout()

    if show and plt.get_backend().lower() != "agg":
        plt.show()

    return fig


def plot_time_and_clean_vs_noisy(signal, clean_signal, sample_rate, f1, f2, show=True):
    """
    Plot the generated discrete-time signal in the time domain (clean vs. noisy
    and discrete stem plot) to illustrate discrete-time sampling.
    """
    n_samples = signal.size
    n = np.arange(n_samples)
    t_ms = (n / float(sample_rate)) * 1000.0
    window = min(n_samples, int(0.08 * sample_rate))  # First 80 ms

    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    axes[0].plot(t_ms[:window], clean_signal[:window], color="#22c55e", linewidth=1.4,
                 label=f"Clean 2-tone ({f1:.0f} Hz + {f2:.0f} Hz)")
    axes[0].plot(t_ms[:window], signal[:window], color="#3b82f6", linewidth=1.0, alpha=0.8,
                 label="Noisy signal x[n]")
    axes[0].set_title(f"Time-Domain Signal Comparison (First {window} samples)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend(loc="upper right", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    markerline, stemlines, baseline = axes[1].stem(
        t_ms[:window], signal[:window], linefmt="C0-", markerfmt="C0o", basefmt="k-"
    )
    plt.setp(markerline, markersize=3.5)
    plt.setp(stemlines, linewidth=0.8)
    axes[1].set_title("Discrete-Time Sequence x[n] (Stem Representation)")
    axes[1].set_xlabel("Time (ms)")
    axes[1].set_ylabel("x[n]")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    if show and plt.get_backend().lower() != "agg":
        plt.show()

    return fig


# ============================================================
# 4. MAIN PIPELINE
# ============================================================

def main(sample_rate=1000.0, show=True):
    """
    Execute the complete Student 1 pipeline:
    1. Generate a two-sinusoid discrete-time signal with Gaussian noise.
    2. Compute the DTFT via explicit loops over limited samples and frequency points.
    3. Verify loop-based DTFT accuracy against vectorized NumPy reference.
    4. Plot time-domain signal and DTFT magnitude & phase spectra.
    5. Save figures to outputs/student1/.
    """
    duration = 0.25          # N = 250 samples at fs = 1000 Hz
    f1 = 50.0                # First sinusoid frequency (Hz)
    f2 = 120.0               # Second sinusoid frequency (Hz)
    amplitude1 = 1.0         # First sinusoid amplitude
    amplitude2 = 0.6         # Second sinusoid amplitude
    noise_std = 0.3          # Gaussian noise standard deviation
    seed = 0
    num_freq_points = 512    # Limited number of frequency points in [0, pi]

    print(f"Generating discrete-time signal (fs = {sample_rate:.0f} Hz, duration = {duration} s)...")
    signal = generate_signal(
        sample_rate=sample_rate,
        duration=duration,
        f1=f1,
        f2=f2,
        amplitude1=amplitude1,
        amplitude2=amplitude2,
        noise_std=noise_std,
        seed=seed,
    )
    clean_signal = generate_signal(
        sample_rate=sample_rate,
        duration=duration,
        f1=f1,
        f2=f2,
        amplitude1=amplitude1,
        amplitude2=amplitude2,
        noise_std=0.0,
        seed=seed,
    )

    n_samples = signal.size
    omega1 = 2.0 * np.pi * f1 / sample_rate
    omega2 = 2.0 * np.pi * f2 / sample_rate

    print(f"  Samples (N)          : {n_samples}")
    print(f"  Component 1          : f1 = {f1:.1f} Hz  ->  omega1 = {omega1 / np.pi:.3f}*pi rad/sample (A1 = {amplitude1})")
    print(f"  Component 2          : f2 = {f2:.1f} Hz  ->  omega2 = {omega2 / np.pi:.3f}*pi rad/sample (A2 = {amplitude2})")
    print(f"  Gaussian noise std   : sigma = {noise_std}")

    # Frequency grid in [0, pi]
    omega = np.linspace(0.0, np.pi, num_freq_points)
    print(f"Computing manual DTFT using nested loops ({n_samples} samples x {num_freq_points} frequency points)...")
    spectrum = compute_dtft(signal, omega)

    # Verify accuracy against vectorized inner product
    n_idx = np.arange(n_samples, dtype=np.float64)
    reference_spectrum = np.exp(-1j * np.outer(omega, n_idx)) @ signal
    max_diff = float(np.max(np.abs(spectrum - reference_spectrum)))

    magnitude = np.abs(spectrum)
    freqs_hz = omega * float(sample_rate) / (2.0 * np.pi)

    # Locate the two dominant peaks
    peak1_idx = int(np.argmax(magnitude[1:])) + 1
    peak1_hz = float(freqs_hz[peak1_idx])
    peak1_omega = float(omega[peak1_idx])

    # Mask around peak 1 to find the second peak
    masked_mag = magnitude.copy()
    mask_width = max(3, num_freq_points // 50)
    low_b = max(0, peak1_idx - mask_width)
    high_b = min(num_freq_points, peak1_idx + mask_width + 1)
    masked_mag[low_b:high_b] = 0.0
    masked_mag[0] = 0.0
    peak2_idx = int(np.argmax(masked_mag))
    peak2_hz = float(freqs_hz[peak2_idx])
    peak2_omega = float(omega[peak2_idx])

    print(f"  Detected Peak 1      : {peak1_hz:.2f} Hz (omega = {peak1_omega / np.pi:.3f}*pi, |X| = {magnitude[peak1_idx]:.2f})")
    print(f"  Detected Peak 2      : {peak2_hz:.2f} Hz (omega = {peak2_omega / np.pi:.3f}*pi, |X| = {magnitude[peak2_idx]:.2f})")
    print(f"  Max error vs. NumPy  : {max_diff:.2e}")

    # Plot and save artifacts
    time_fig = plot_time_and_clean_vs_noisy(signal, clean_signal, sample_rate, f1, f2, show=show)
    dtft_fig = plot_magnitude_phase(omega, spectrum, sample_rate, show=show)

    output_dir = PROJECT_ROOT / "outputs" / "student1"
    output_dir.mkdir(parents=True, exist_ok=True)
    time_fig.savefig(output_dir / "time_domain_signal.png", dpi=200, bbox_inches="tight")
    dtft_fig.savefig(output_dir / "dtft_magnitude_phase.png", dpi=200, bbox_inches="tight")
    print(f"  Saved plots to       : {output_dir.relative_to(PROJECT_ROOT)}")
    print("Student 1 finished.")

    return {
        "signal": signal,
        "clean_signal": clean_signal,
        "omega": omega,
        "spectrum": spectrum,
        "magnitude": magnitude,
        "phase": np.unwrap(np.angle(spectrum)),
        "sample_rate": sample_rate,
        "output_dir": output_dir,
    }


if __name__ == "__main__":
    main()
