import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import lfilter


def moving_average(signal, window_size=5):
    # moving average = convolution مع نافذة كلها 1/window_size
    window = np.ones(window_size) / window_size
    return np.convolve(signal, window, mode="same")


def time_domain_convolution(x, h):
    return np.convolve(x, h)


def frequency_domain_convolution(x, h):
    # طول الناتج = len(x) + len(h) - 1 (zero-padding عشان نتفادى circular convolution)
    n = len(x) + len(h) - 1
    X = np.fft.fft(x, n)
    H = np.fft.fft(h, n)
    return np.real(np.fft.ifft(X * H))


def difference_equation(x, a=0.9, b=1.0, y0=0.0):
    # y[n] = a*y[n-1] + b*x[n]
    return lfilter([b], [1, -a], x, zi=[a * y0])[0]


def compare_convolutions(x, h, show=True):
    y_time = time_domain_convolution(x, h)
    y_freq = frequency_domain_convolution(x, h)
    max_error = np.max(np.abs(y_time - y_freq))

    if show:
        fig, axs = plt.subplots(3, 1, figsize=(8, 8))
        axs[0].stem(y_time)
        axs[0].set_title("Time-domain convolution")
        axs[1].stem(y_freq)
        axs[1].set_title("Frequency-domain convolution")
        axs[2].stem(np.abs(y_time - y_freq))
        axs[2].set_title(f"Absolute difference (max = {max_error:.2e})")
        plt.tight_layout()
        plt.show()

    return y_time, y_freq, max_error


def main(show=True):
    x = np.array([1, 2, 3, 4, 5, 4, 3, 2, 1], dtype=float)
    h = np.array([1, 0.5, 0.25], dtype=float)

    ma = moving_average(x, window_size=3)
    y_diff = difference_equation(x, a=0.9, b=1.0, y0=0.0)
    y_time, y_freq, err = compare_convolutions(x, h, show=False)

    print("Moving average:", ma)
    print("Difference equation:", y_diff)
    print("Max error (time vs freq):", err)

    if show:
        fig, axs = plt.subplots(3, 1, figsize=(8, 8))
        axs[0].stem(x)
        axs[0].stem(ma, linefmt="r-", markerfmt="ro")
        axs[0].set_title("Input (blue) and moving average (red)")
        axs[1].stem(y_diff)
        axs[1].set_title("Difference equation: y[n] = 0.9 y[n-1] + x[n]")
        axs[2].stem(y_time)
        axs[2].set_title("Convolution of x and h")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()