"""Shared utility functions for the DSP multi-domain project.

These helpers are optional: students may use them or write their own versions.
They keep I/O, plotting, and evaluation code consistent across the six modules.

All image data is handled as ``float64`` in the range ``[0, 1]`` and all audio is
returned as mono ``float64`` samples, which makes Fourier operations predictable.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = PROJECT_ROOT / "data" / "audio"
IMAGE_DIR = PROJECT_ROOT / "data" / "images"

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
AUDIO_EXTENSIONS = (".wav", ".wave")


# --------------------------------------------------------------------------- #
# File discovery
# --------------------------------------------------------------------------- #

def find_data_file(directory: Path | str, extensions: tuple[str, ...]):
    """Return the first file in *directory* matching *extensions*, or ``None``.

    Parameters
    ----------
    directory:
        Folder to search (non-recursive).
    extensions:
        Case-insensitive filename extensions to accept, e.g. ``(".wav",)``.

    Returns
    -------
    pathlib.Path or None
        Path of the first match, sorted by filename for deterministic behaviour.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return None
    matches = sorted(
        p for p in directory.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )
    return matches[0] if matches else None


def ensure_directories() -> None:
    """Create the ``data/audio`` and ``data/images`` folders if they are absent."""
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# Audio
# --------------------------------------------------------------------------- #

def _as_float_mono(data, mono: bool = True) -> np.ndarray:
    """Convert PCM or float audio samples to a normalised float array."""
    data = np.asarray(data)
    if mono and data.ndim > 1:             # average the channels down to mono
        data = data.mean(axis=1)
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        data = data.astype(np.float64) / max(abs(info.min), info.max)
    else:
        data = data.astype(np.float64)
    return data


def load_audio(path: Path | str, mono: bool = True):
    """Load an audio file and return ``(samples, sample_rate)``.

    Tries :mod:`soundfile` first (supports many formats) and falls back to
    :func:`scipy.io.wavfile.read` for plain WAV files when soundfile is absent.
    Integer PCM samples are scaled to ``[-1, 1]``.

    Parameters
    ----------
    path:
        Path to the audio file.
    mono:
        If ``True`` (default), multi-channel audio is averaged to mono.

    Returns
    -------
    tuple[numpy.ndarray, int]
        ``(samples, sample_rate)`` where ``samples`` is a 1-D float64 array.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    try:
        import soundfile as sf
        data, sample_rate = sf.read(str(path), always_2d=False)
        return _as_float_mono(data, mono), int(sample_rate)
    except ImportError:
        # Fallback: SciPy ships with the required dependencies.
        from scipy.io import wavfile
        sample_rate, data = wavfile.read(str(path))
        return _as_float_mono(data, mono), int(sample_rate)


def save_audio(path: Path | str, samples, sample_rate: int) -> Path:
    """Save float samples as a 16-bit PCM WAV file and return the output path.

    Uses :mod:`soundfile` when available, otherwise :func:`scipy.io.wavfile.write`.
    Samples are clipped to ``[-1, 1]`` before writing.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    samples = np.clip(np.asarray(samples, dtype=np.float64), -1.0, 1.0)

    try:
        import soundfile as sf
        sf.write(str(path), samples, int(sample_rate), subtype="PCM_16")
    except ImportError:
        from scipy.io import wavfile
        wavfile.write(str(path), int(sample_rate), (samples * 32767).astype(np.int16))
    return path


def make_synthetic_audio(sample_rate: int = 8000, duration: float = 1.0,
                         seed: int = 0):
    """Return ``(audio, sample_rate)`` with a short synthetic 'clean' signal.

    Used as a demo fallback when no WAV file has been provided. The signal is a
    sum of three sinusoids plus a little noise, which gives the analysis and
    filtering students something meaningful to work with.
    """
    t = np.arange(int(sample_rate * duration)) / float(sample_rate)
    rng = np.random.default_rng(seed)
    audio = (0.45 * np.sin(2 * np.pi * 220 * t)
             + 0.30 * np.sin(2 * np.pi * 440 * t)
             + 0.15 * np.cos(2 * np.pi * 660 * t)
             + 0.02 * rng.standard_normal(t.size))
    return audio, int(sample_rate)


# --------------------------------------------------------------------------- #
# Images
# --------------------------------------------------------------------------- #

def load_image(path: Path | str, grayscale: bool = True) -> np.ndarray:
    """Load an image as a float array in ``[0, 1]``.

    Uses OpenCV when installed; falls back to ``matplotlib.image.imread`` so the
    project still runs in minimal environments. Grayscale conversion uses the
    standard luminance weights when required.

    Parameters
    ----------
    path:
        Path to the image file.
    grayscale:
        Return a 2-D luminance image when ``True`` (default).

    Returns
    -------
    numpy.ndarray
        2-D (grayscale) or 3-D RGB image with values in ``[0, 1]``.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file not found: {path}")

    try:
        import cv2
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        image = cv2.imread(str(path), flag)
        if image is None:
            raise ValueError(f"OpenCV could not decode image: {path}")
        if not grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except ImportError:
        import matplotlib.image as mpimg
        image = mpimg.imread(str(path))
        if grayscale and image.ndim == 3:
            image = (0.299 * image[..., 0]
                     + 0.587 * image[..., 1]
                     + 0.114 * image[..., 2])
        elif not grayscale and image.ndim == 2:
            image = np.stack([image] * 3, axis=-1)

    image = np.asarray(image, dtype=np.float64)
    if image.max() > 1.0:                  # uint8-style data -> [0, 1]
        image = image / 255.0
    return image


def make_synthetic_image(size: int = 256, seed: int = 0) -> np.ndarray:
    """Return a synthetic grayscale test image (gradient + shapes) in ``[0, 1]``.

    Used as a demo fallback when no image file has been provided.
    """
    y, x = np.mgrid[0:size, 0:size].astype(np.float64)
    image = 0.25 + 0.45 * (x / size) + 0.20 * (y / size)
    image += 0.30 * ((x - size * 0.35) ** 2 + (y - size * 0.40) ** 2 < (size * 0.12) ** 2)
    image -= 0.25 * ((x - size * 0.70) ** 2 + (y - size * 0.65) ** 2 < (size * 0.10) ** 2)
    image += 0.05 * np.random.default_rng(seed).standard_normal((size, size))
    return np.clip(image, 0.0, 1.0)


# --------------------------------------------------------------------------- #
# Plotting and evaluation
# --------------------------------------------------------------------------- #

def plot_spectrum(signal, sampling_rate: float, title: str = "Magnitude spectrum",
                  ax=None):
    """Plot the single-sided magnitude spectrum of a 1-D signal.

    Parameters
    ----------
    signal:
        1-D time-domain samples.
    sampling_rate:
        Sampling frequency in Hz.
    title:
        Axes title.
    ax:
        Optional matplotlib axes to draw on. A new figure is created if omitted.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(frequencies_hz, magnitude)`` for the plotted spectrum.
    """
    signal = np.asarray(signal, dtype=np.float64)
    n = signal.size
    spectrum = np.fft.rfft(signal)
    frequencies = np.fft.rfftfreq(n, d=1.0 / float(sampling_rate))
    magnitude = (2.0 / n) * np.abs(spectrum)
    magnitude[0] /= 2.0                    # DC is not doubled
    if n % 2 == 0:                         # Nyquist bin is not doubled either
        magnitude[-1] /= 2.0

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(frequencies, magnitude, linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude")
    ax.grid(True, alpha=0.3)
    return frequencies, magnitude


def calculate_mse_psnr(original, processed, peak: float | None = None):
    """Compute mean squared error and peak signal-to-noise ratio.

    Parameters
    ----------
    original, processed:
        Arrays of the same shape (typically two images in ``[0, 1]``).
    peak:
        Maximum possible pixel value. Defaults to ``1.0`` for images scaled to
        ``[0, 1]``, or ``255.0`` for 8-bit data.

    Returns
    -------
    tuple[float, float]
        ``(mse, psnr_db)``. ``psnr_db`` is ``inf`` when the images are identical.

    TODO (student 6): verify how the choice of ``peak`` changes PSNR and explain
    the units in your report.
    """
    original = np.asarray(original, dtype=np.float64)
    processed = np.asarray(processed, dtype=np.float64)
    if original.shape != processed.shape:
        raise ValueError("original and processed must have the same shape")

    mse = float(np.mean((original - processed) ** 2))
    if peak is None:
        peak = 1.0 if np.nanmax(original) <= 1.0 else 255.0
    psnr = float("inf") if mse == 0.0 else float(10.0 * np.log10(peak ** 2 / mse))
    return mse, psnr
