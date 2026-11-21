import numpy as np
import matplotlib.pyplot as plt


def generate_signal(sample_rate=1000.0, duration=0.25, f1=50.0, f2=120.0,
                    amplitude1=1.0, amplitude2=0.6, noise_std=0.3, seed=0):
    n_samples = int(round(sample_rate * duration))
    n = np.arange(n_samples)
    rng = np.random.default_rng(seed)

    # x[n] = A1*sin(w1*n) + A2*cos(w2*n) + noise
    w1 = 2 * np.pi * f1 / sample_rate
    w2 = 2 * np.pi * f2 / sample_rate
    noise = noise_std * rng.standard_normal(n_samples)

    return amplitude1 * np.sin(w1 * n) + amplitude2 * np.cos(w2 * n) + noise


def compute_dtft(signal, omega_values):
    # X(e^jw) = sum(x[n] * e^(-j*w*n))
    N = len(signal)
    M = len(omega_values)
    spectrum = np.zeros(M, dtype=complex)

    for k in range(M):
        w = omega_values[k]
        total = 0j
        for n in range(N):
            total += signal[n] * np.exp(-1j * w * n)
        spectrum[k] = total

    return spectrum


def plot_magnitude_phase(omega, spectrum, sample_rate, show=True):
    magnitude = np.abs(spectrum)
    phase = np.unwrap(np.angle(spectrum))
    freqs_hz = omega * sample_rate / (2 * np.pi)

    fig, axs = plt.subplots(2, 1, figsize=(8, 6))

    axs[0].plot(freqs_hz, magnitude)
    axs[0].set_title("DTFT Magnitude Spectrum")
    axs[0].set_xlabel("Frequency (Hz)")
    axs[0].set_ylabel("|X(ω)|")
    axs[0].grid(True)

    axs[1].plot(freqs_hz, phase, color="tab:orange")
    axs[1].set_title("DTFT Phase Spectrum")
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_ylabel("Phase (rad)")
    axs[1].grid(True)

    plt.tight_layout()
    if show:
        plt.show()

    return magnitude, phase


def main(sample_rate=1000.0, show=True):
    signal = generate_signal(sample_rate=sample_rate)
    omega = np.linspace(0, np.pi, 500)
    spectrum = compute_dtft(signal, omega)

    print("Signal length:", len(signal))
    print("Frequency points:", len(omega))
    print("Max magnitude:", np.max(np.abs(spectrum)))

    if show:
        plt.figure(figsize=(8, 3))
        plt.plot(signal)
        plt.title("Generated Signal (2 Sinusoids + Gaussian Noise)")
        plt.xlabel("Sample index (n)")
        plt.ylabel("Amplitude")
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    plot_magnitude_phase(omega, spectrum, sample_rate, show=show)


if __name__ == "__main__":
    main()
