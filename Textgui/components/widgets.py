#!/usr/bin/env python3
"""Shared widgets and helpers for the CustomTkinter DSP desktop application.

This module is the desktop equivalent of the old ``gui/_shared.py``: it keeps
input discovery, student-module loading, image/audio conversion, and the
reusable UI building blocks in one place.  Every page imports from here so the
six modules share the same look and the same graceful-degradation behaviour.

Only projects dependencies are used (numpy, matplotlib, customtkinter, Pillow
via matplotlib, and the standard library).  Audio playback delegates to the
operating system's default player, so no extra audio package is required.
"""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import platform
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import wave
from pathlib import Path
from types import ModuleType

import customtkinter as ctk
import matplotlib

# Student modules call ``plt.show()`` with their default ``show=True``.  Under a
# real display that would open blocking Tk windows on top of this app.  The Agg
# backend makes those calls no-ops while embedded FigureCanvasTkAgg canvases
# (used by every page) still render normally.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image

# --------------------------------------------------------------------------- #
# Project paths -- Textgui/components/widgets.py -> project root is parents[2]
# --------------------------------------------------------------------------- #

PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUDIO_DIR = PROJECT_ROOT / "data" / "audio"
IMAGE_DIR = PROJECT_ROOT / "data" / "images"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import helpers  # noqa: E402  (path setup must happen first)

# Repository-relative student files (identical map to the old Streamlit GUI).
MODULE_FILES = {
    "student1": "students/student1_mathematical_signal_dtft/student1.py",
    "student2": "students/student2_convolution_difference/student2.py",
    "student3": "students/student3_audio_analysis/student3.py",
    "student4": "students/student4_audio_filtering/student4.py",
    "student5": "students/student5_2d_fourier_image_filtering/student5.py",
    "student6": "students/student6_image_noise_removal/student6.py",
}

MODULE_TITLES = {
    "student1": "Mathematical Signal & DTFT",
    "student2": "Convolution & Difference Equations",
    "student3": "Audio Analysis",
    "student4": "Audio Filtering",
    "student5": "2D Fourier & Image Filtering",
    "student6": "Image Noise Removal",
}

# Academic dark palette shared by the window and the embedded matplotlib plots.
COLORS = {
    "bg": "#161a20",
    "panel": "#20252d",
    "panel_alt": "#272d37",
    "accent": "#3b82f6",
    "accent_hover": "#2f6fd0",
    "ok": "#22c55e",
    "warn": "#f59e0b",
    "error": "#ef4444",
    "text": "#e6e9ee",
    "muted": "#9aa4b2",
}

# Make embedded figures match the dark theme instead of matplotlib's white default.
plt.rcParams.update(
    {
        "figure.facecolor": COLORS["panel"],
        "axes.facecolor": COLORS["panel"],
        "axes.edgecolor": "#4b5563",
        "axes.labelcolor": COLORS["text"],
        "axes.titlecolor": COLORS["text"],
        "text.color": COLORS["text"],
        "xtick.color": "#b6bec9",
        "ytick.color": "#b6bec9",
        "grid.color": "#3a424e",
        "legend.facecolor": COLORS["panel"],
        "legend.edgecolor": "#4b5563",
        "figure.titlesize": 12,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
    }
)


# --------------------------------------------------------------------------- #
# Discovery and student-module helpers
# --------------------------------------------------------------------------- #

def find_input(directory: Path, extensions: tuple[str, ...]) -> Path | None:
    """Return the first matching input file, or ``None`` when data is missing."""
    return helpers.find_data_file(directory, extensions)


def find_audio_input() -> Path | None:
    return find_input(AUDIO_DIR, helpers.AUDIO_EXTENSIONS)


def find_image_input() -> Path | None:
    return find_input(IMAGE_DIR, helpers.IMAGE_EXTENSIONS)


def is_implemented(key: str) -> bool:
    """Return True when a student file has real function bodies (not ``pass``)."""
    relative = MODULE_FILES.get(key)
    if relative is None:
        return False
    path = PROJECT_ROOT / relative
    if not path.is_file():
        return False
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except (OSError, SyntaxError):
        return False
    functions = [
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if not functions:
        return False
    for function in functions:
        body = function.body
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]  # ignore docstrings
        if not all(isinstance(node, ast.Pass) for node in body):
            return True
    return False


def load_student_module(key: str) -> ModuleType | None:
    """Import a ``studentN.py`` file by path, mirroring ``main.load_module``."""
    relative = MODULE_FILES.get(key)
    if relative is None:
        return None
    path = PROJECT_ROOT / relative
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(f"textgui_{key}", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        # A student module may need an optional dependency (e.g. OpenCV) that is
        # not installed.  Keep the desktop app alive; callers fall back.
        return None
    return module


def run_student_main(key: str, **kwargs) -> tuple[object, str, list]:
    """Run a student ``main(show=False)`` and capture its log and new figures."""
    module = load_student_module(key)
    if module is None or not callable(getattr(module, "main", None)):
        return None, "", []
    before = set(plt.get_fignums())
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        try:
            result = module.main(show=False, **kwargs)
        except TypeError:
            result = module.main(**kwargs)
        except Exception as exc:  # keep the desktop app alive
            result = None
            buffer.write(f"Student module error: {type(exc).__name__}: {exc}\n")
    figures = [
        plt.figure(number)
        for number in sorted(set(plt.get_fignums()) - before)
    ]
    return result, buffer.getvalue(), figures


def status_badge(key: str) -> str:
    """Short human-readable status string for the sidebar / page headers."""
    return "Implemented" if is_implemented(key) else "Reference demo"


# --------------------------------------------------------------------------- #
# Image and audio conversion
# --------------------------------------------------------------------------- #

def to_uint8(array, normalize: bool = False) -> np.ndarray:
    """Convert an image array to uint8 for reliable photo rendering.

    Fourier results can span a much wider range than ``[0, 1]``; this mirrors
    the old Streamlit helper exactly, including optional min/max normalisation.
    """
    values = np.asarray(array, dtype=np.float64)
    if normalize:
        low, high = float(values.min()), float(values.max())
        values = (values - low) / max(high - low, 1e-12) * 255.0
    elif (
        np.issubdtype(np.asarray(array).dtype, np.floating)
        and values.size
        and np.isfinite(values).any()
        and float(np.nanmax(values)) <= 1.0
    ):
        # Float images use the project's [0, 1] convention; without this step
        # every pixel below 1.0 truncates to 0 and the image renders black.
        values = values * 255.0
    return np.clip(np.nan_to_num(values, nan=0.0), 0.0, 255.0).astype(np.uint8)


def wav_bytes(audio, sample_rate: int) -> bytes:
    """Encode mono samples as 16-bit PCM WAV bytes."""
    samples = np.clip(np.asarray(audio, dtype=np.float64), -1.0, 1.0)
    pcm = (samples * 32767.0).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm)
    return buffer.getvalue()


_AUDIO_TMP_DIR: Path | None = None


def _audio_tmp_dir() -> Path:
    global _AUDIO_TMP_DIR
    if _AUDIO_TMP_DIR is None:
        _AUDIO_TMP_DIR = Path(tempfile.mkdtemp(prefix="dsp_desktop_audio_"))
    return _AUDIO_TMP_DIR


def save_temp_wav(audio, sample_rate: int, name: str = "clip.wav") -> Path:
    """Write audio to a temporary WAV file and return its path."""
    path = _audio_tmp_dir() / name
    path.write_bytes(wav_bytes(audio, sample_rate))
    return path


def play_wav_file(path: Path | str) -> bool:
    """Play a WAV file with the OS default player. Returns True if launched."""
    path = str(path)
    system = platform.system()
    try:
        if system == "Windows":  # winsound is in the Windows standard library
            import winsound

            winsound.PlaySound(path, winsound.SND_ASYNC | winsound.SND_FILENAME)
        elif system == "Darwin":
            subprocess.Popen(
                ["afplay", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            for player in ("paplay", "aplay", "play"):
                if shutil.which(player):
                    subprocess.Popen(
                        [player, path],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    return True
            subprocess.Popen(  # last resort: desktop association
                ["xdg-open", path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        return True
    except Exception:
        return False


def play_audio(audio, sample_rate: int, name: str = "clip.wav") -> bool:
    """Encode and play a mono audio array through the system player."""
    return play_wav_file(save_temp_wav(audio, sample_rate, name))


# --------------------------------------------------------------------------- #
# Reusable display widgets
# --------------------------------------------------------------------------- #

class PlotView(ctk.CTkFrame):
    """Frame that embeds a matplotlib figure with an optional navigation bar."""

    def __init__(self, master, toolbar: bool = False, **kwargs):
        kwargs.setdefault("fg_color", COLORS["panel"])
        kwargs.setdefault("corner_radius", 10)
        super().__init__(master, **kwargs)
        self._toolbar_enabled = toolbar
        # NOTE: do not name this ``_canvas`` -- CTkFrame uses that attribute for
        # its own Tk drawing canvas and would fail to redraw.
        self._figure_canvas: FigureCanvasTkAgg | None = None
        self._toolbar = None
        self._placeholder = ctk.CTkLabel(
            self,
            text="Run the module to generate plots.",
            text_color=COLORS["muted"],
            font=ctk.CTkFont(size=12),
        )
        self._placeholder.pack(expand=True)

    def clear(self) -> None:
        if self._toolbar is not None:
            self._toolbar.destroy()
            self._toolbar = None
        if self._figure_canvas is not None:
            self._figure_canvas.get_tk_widget().destroy()
            self._figure_canvas = None

    def show_figure(self, figure, keep: bool = False) -> None:
        """Replace the current content with *figure*.

        When ``keep`` is False the figure is closed after it is drawn, which
        prevents figures from piling up in memory across repeated runs.
        """
        self.clear()
        self._placeholder.pack_forget()
        canvas = FigureCanvasTkAgg(figure, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True, padx=6, pady=6)
        if self._toolbar_enabled:
            toolbar = NavigationToolbar2Tk(canvas, self, pack_toolbar=False)
            toolbar.config(background=COLORS["panel"])
            for widget in toolbar.winfo_children():
                try:
                    widget.config(background=COLORS["panel"], foreground=COLORS["text"])
                except Exception:
                    pass
            toolbar.update()
            toolbar.pack(side="bottom", fill="x")
            self._toolbar = toolbar
        self._figure_canvas = canvas

    def show_message(self, text: str) -> None:
        self.clear()
        self._placeholder.configure(text=text)
        self._placeholder.pack(expand=True)


class ImageView(ctk.CTkFrame):
    """Frame that displays a numpy image (or a file path) using Pillow."""

    def __init__(self, master, max_size: tuple[int, int] = (720, 520), **kwargs):
        kwargs.setdefault("fg_color", COLORS["panel"])
        kwargs.setdefault("corner_radius", 10)
        super().__init__(master, **kwargs)
        self._max_size = max_size
        self._photo: ctk.CTkImage | None = None
        self._label = ctk.CTkLabel(
            self,
            text="No image yet.",
            text_color=COLORS["muted"],
            font=ctk.CTkFont(size=12),
        )
        self._label.pack(expand=True, fill="both", padx=8, pady=8)

    def clear(self) -> None:
        self._photo = None
        self._label.configure(image=None, text="No image yet.")

    def show_image(self, data, caption: str | None = None, normalize: bool = False) -> None:
        """Display an array or image path with optional min/max normalisation."""
        try:
            if isinstance(data, (str, Path)):
                image = Image.open(str(data))
            else:
                array = data
                if not (isinstance(array, np.ndarray) and array.dtype == np.uint8):
                    array = to_uint8(array, normalize=normalize)
                if array.ndim == 2:
                    image = Image.fromarray(array, mode="L")
                elif array.ndim == 3 and array.shape[2] == 4:
                    image = Image.fromarray(array, mode="RGBA")
                else:
                    image = Image.fromarray(array, mode="RGB")

            width, height = image.size
            max_w, max_h = self._max_size
            scale = min(max_w / width, max_h / height, 3.0)
            if scale < 1.0 or (scale > 1.0 and width < max_w and height < max_h):
                size = (max(1, int(width * scale)), max(1, int(height * scale)))
                resample = getattr(Image, "Resampling", Image).LANCZOS
                image = image.resize(size, resample)

            # CTkLabel expects a CTkImage (not a raw PhotoImage) so it can scale
            # correctly on HighDPI displays.  The image is already resized to
            # fit ``_max_size`` above, so pass those exact dimensions through.
            self._photo = ctk.CTkImage(light_image=image, dark_image=image, size=image.size)
            self._label.configure(
                image=self._photo,
                text=caption or "",
                compound="top" if caption else "none",
                text_color=COLORS["text"] if caption else COLORS["muted"],
            )
        except Exception as exc:
            self._label.configure(image=None, text=f"Could not display image: {exc}")

    def show_message(self, text: str) -> None:
        self.clear()
        self._label.configure(text=text)


class MetricsView(ctk.CTkFrame):
    """Read-only grid of ``label: value`` metric pairs."""

    def __init__(self, master, columns: int = 2, **kwargs):
        kwargs.setdefault("fg_color", COLORS["panel"])
        kwargs.setdefault("corner_radius", 10)
        super().__init__(master, **kwargs)
        self._columns = columns
        self._placeholder = ctk.CTkLabel(
            self,
            text="Metrics appear here after a run.",
            text_color=COLORS["muted"],
            font=ctk.CTkFont(size=12),
        )
        self._placeholder.pack(expand=True)

    def clear(self) -> None:
        for child in list(self.winfo_children()):
            child.destroy()

    def set_metrics(self, items: dict[str, object]) -> None:
        self.clear()
        if not items:
            self._placeholder = ctk.CTkLabel(self, text="No metrics.", text_color=COLORS["muted"])
            self._placeholder.pack(expand=True)
            return
        for index, (label, value) in enumerate(items.items()):
            row, column = divmod(index, self._columns)
            self.grid_columnconfigure(column, weight=1, uniform="metric")
            cell = ctk.CTkFrame(self, fg_color=COLORS["panel_alt"], corner_radius=8)
            cell.grid(row=row, column=column, sticky="ew", padx=5, pady=5)
            ctk.CTkLabel(
                cell,
                text=str(label),
                text_color=COLORS["muted"],
                font=ctk.CTkFont(size=10),
                anchor="w",
            ).pack(fill="x", padx=10, pady=(7, 0))
            ctk.CTkLabel(
                cell,
                text=str(value),
                text_color=COLORS["text"],
                font=ctk.CTkFont(size=14, weight="bold"),
                anchor="w",
            ).pack(fill="x", padx=10, pady=(2, 8))


class PlayButton(ctk.CTkButton):
    """Button that plays audio produced by a ``provider() -> (audio, sr)``."""

    def __init__(self, master, provider, status_label=None, **kwargs):
        self._provider = provider
        self._status_label = status_label
        kwargs.setdefault("text", "▶ Play")
        kwargs.setdefault("fg_color", COLORS["accent"])
        kwargs.setdefault("hover_color", COLORS["accent_hover"])
        super().__init__(master, command=self._play, **kwargs)

    def _play(self) -> None:
        try:
            audio, sample_rate = self._provider()
            ok = play_audio(audio, sample_rate)
            message = "Playing through the system audio player." if ok else "Could not start an audio player."
        except Exception as exc:
            message = f"Audio error: {type(exc).__name__}: {exc}"
        if self._status_label is not None:
            self._status_label.configure(text=message)


class TextView(ctk.CTkTextbox):
    """Read-only scrollable text area for logs and captions."""

    def __init__(self, master, **kwargs):
        kwargs.setdefault("fg_color", COLORS["panel"])
        kwargs.setdefault("text_color", COLORS["text"])
        kwargs.setdefault("font", ctk.CTkFont(family="Consolas", size=11))
        super().__init__(master, **kwargs)
        self.configure(state="disabled")

    def set_text(self, text: str) -> None:
        self.configure(state="normal")
        self.delete("1.0", "end")
        self.insert("1.0", text or "(empty)")
        self.configure(state="disabled")


# --------------------------------------------------------------------------- #
# Base page: shared layout for all six modules
# --------------------------------------------------------------------------- #

class BasePage(ctk.CTkFrame):
    """Template for a module page: header, controls, tabbed output, run button."""

    number = "00"
    title = "Module"
    description = ""
    module_key = ""

    def __init__(self, master, app):
        super().__init__(master, fg_color="transparent")
        self.app = app
        self._busy = False
        self._event_queue: queue.Queue = queue.Queue()
        self._polling = False
        self._build_shell()

    # -- construction ------------------------------------------------------- #

    def _build_shell(self) -> None:
        header = ctk.CTkFrame(self, fg_color=COLORS["panel"], corner_radius=12)
        header.pack(fill="x", padx=14, pady=(14, 8))

        badge_color = COLORS["ok"] if is_implemented(self.module_key) else COLORS["warn"]
        badge_text = status_badge(self.module_key)

        top = ctk.CTkFrame(header, fg_color="transparent")
        top.pack(fill="x", padx=16, pady=(13, 0))
        ctk.CTkLabel(
            top,
            text=f"MODULE {self.number} / 06",
            text_color=COLORS["accent"],
            font=ctk.CTkFont(size=11, weight="bold"),
        ).pack(side="left")
        ctk.CTkLabel(
            top,
            text=badge_text,
            text_color=badge_color,
            font=ctk.CTkFont(size=11, weight="bold"),
        ).pack(side="right")

        ctk.CTkLabel(
            header,
            text=self.title,
            text_color=COLORS["text"],
            font=ctk.CTkFont(size=21, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=16, pady=(4, 0))
        ctk.CTkLabel(
            header,
            text=self.description,
            text_color=COLORS["muted"],
            font=ctk.CTkFont(size=12),
            anchor="w",
            wraplength=980,
            justify="left",
        ).pack(fill="x", padx=16, pady=(4, 13))

        body = ctk.CTkFrame(self, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        body.grid_columnconfigure(0, weight=0, minsize=300)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        self.controls = ctk.CTkScrollableFrame(
            body,
            width=305,
            fg_color=COLORS["panel"],
            corner_radius=12,
            label_text="Parameters",
            label_font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.controls.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        self.output = ctk.CTkFrame(body, fg_color="transparent")
        self.output.grid(row=0, column=1, sticky="nsew")
        self.output.grid_rowconfigure(0, weight=1)
        self.output.grid_columnconfigure(0, weight=1)

        self.tabs = ctk.CTkTabview(
            self.output,
            fg_color=COLORS["panel"],
            segmented_button_fg_color=COLORS["panel_alt"],
            segmented_button_unselected_color=COLORS["panel_alt"],
            segmented_button_selected_color=COLORS["accent"],
        )
        self.tabs.grid(row=0, column=0, sticky="nsew")

        self.status_label = ctk.CTkLabel(
            self.controls,
            text="Adjust the parameters and press Run.",
            text_color=COLORS["muted"],
            font=ctk.CTkFont(size=11),
            wraplength=265,
            justify="left",
            anchor="w",
        )

        self.build_controls()
        self.status_label.pack(fill="x", padx=12, pady=(6, 2))
        self._run_button = ctk.CTkButton(
            self.controls,
            text=self.run_button_text(),
            command=self.on_run,
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            font=ctk.CTkFont(size=13, weight="bold"),
            height=38,
        )
        self._run_button.pack(fill="x", padx=12, pady=(2, 14))
        self.build_output()

    # -- hooks implemented by each page -------------------------------------- #

    def run_button_text(self) -> str:
        return "▶ Run"

    def build_controls(self) -> None:
        raise NotImplementedError

    def build_output(self) -> None:
        raise NotImplementedError

    def on_run(self) -> None:
        raise NotImplementedError

    # -- control helpers ----------------------------------------------------- #

    def add_slider(self, label, from_, to, value, step=1.0, integer=False):
        variable = ctk.IntVar(value=int(value)) if integer else ctk.DoubleVar(value=float(value))
        row = ctk.CTkFrame(self.controls, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(8, 0))

        value_text = ctk.StringVar(value=self._format_value(value, integer))
        header = ctk.CTkFrame(row, fg_color="transparent")
        header.pack(fill="x")
        ctk.CTkLabel(
            header,
            text=label,
            text_color=COLORS["text"],
            font=ctk.CTkFont(size=11),
            anchor="w",
        ).pack(side="left")
        ctk.CTkLabel(
            header,
            textvariable=value_text,
            text_color=COLORS["accent"],
            font=ctk.CTkFont(size=11, weight="bold"),
        ).pack(side="right")

        steps = max(1, int(round((to - from_) / step)))
        slider = ctk.CTkSlider(
            row,
            from_=from_,
            to=to,
            number_of_steps=steps,
            variable=variable,
            height=16,
            progress_color=COLORS["accent"],
            button_color=COLORS["accent_hover"],
        )
        slider.pack(fill="x", pady=(2, 0))

        def _update(_=None):
            value_text.set(self._format_value(variable.get(), integer))

        slider.configure(command=_update)
        _update()
        return variable

    @staticmethod
    def _format_value(value, integer: bool) -> str:
        return f"{int(value)}" if integer else f"{float(value):.2f}"

    def add_checkbox(self, label: str, value: bool = True):
        variable = ctk.BooleanVar(value=value)
        ctk.CTkCheckBox(
            self.controls,
            text=label,
            variable=variable,
            fg_color=COLORS["accent"],
            hover_color=COLORS["accent_hover"],
            font=ctk.CTkFont(size=11),
        ).pack(fill="x", padx=12, pady=(10, 0))
        return variable

    def add_integer_entry(self, label: str, value: int, minimum: int = 0, maximum: int = 9999):
        row = ctk.CTkFrame(self.controls, fg_color="transparent")
        row.pack(fill="x", padx=12, pady=(10, 0))
        ctk.CTkLabel(
            row,
            text=label,
            text_color=COLORS["text"],
            font=ctk.CTkFont(size=11),
            anchor="w",
        ).pack(fill="x")
        variable = ctk.StringVar(value=str(value))

        def _clamp(*_):
            try:
                number = int(float(variable.get()))
            except (TypeError, ValueError):
                number = minimum
            number = max(minimum, min(maximum, number))
            variable.set(str(number))

        entry = ctk.CTkEntry(
            row,
            textvariable=variable,
            fg_color=COLORS["panel_alt"],
            border_color=COLORS["panel_alt"],
        )
        entry.pack(fill="x", pady=(3, 0))
        entry.bind("<FocusOut>", _clamp)
        return variable

    def add_info(self, text: str) -> None:
        ctk.CTkLabel(
            self.controls,
            text=text,
            text_color=COLORS["muted"],
            font=ctk.CTkFont(size=10),
            wraplength=265,
            justify="left",
            anchor="w",
        ).pack(fill="x", padx=12, pady=(10, 0))

    def add_tab(self, name: str):
        self.tabs.add(name)
        return self.tabs.tab(name)

    # -- status and background execution ------------------------------------ #

    def set_status(self, text: str, kind: str = "info") -> None:
        color = {
            "info": COLORS["muted"],
            "ok": COLORS["ok"],
            "busy": COLORS["accent"],
            "warn": COLORS["warn"],
            "error": COLORS["error"],
        }.get(kind, COLORS["muted"])
        self.status_label.configure(text=text, text_color=color)

    def run_async(self, worker, on_done=None) -> None:
        """Run *worker* in a daemon thread, then call *on_done* on the UI thread."""
        if self._busy:
            return
        self._busy = True
        self._run_button.configure(state="disabled")
        self.set_status("Working…", "busy")

        def target():
            try:
                result, error = worker(), None
            except Exception as exc:  # surface the error in the status label
                result, error = None, exc
            self._event_queue.put((on_done, result, error))

        threading.Thread(target=target, daemon=True).start()
        if not self._polling:
            self._polling = True
            self._poll_events()

    def _poll_events(self) -> None:
        try:
            on_done, result, error = self._event_queue.get_nowait()
        except queue.Empty:
            self.after(70, self._poll_events)
            return

        self._busy = False
        self._run_button.configure(state="normal")
        if error is not None:
            self.set_status(f"Error: {type(error).__name__}: {error}", "error")
        else:
            self.set_status("Done.", "ok")
            if on_done is not None:
                try:
                    on_done(result)
                except Exception as exc:
                    self.set_status(f"Display error: {type(exc).__name__}: {exc}", "error")
        # Keep polling in case another event arrived while handling this one.
        if not self._event_queue.empty():
            self.after(70, self._poll_events)
        else:
            self._polling = False

    # -- fallbacks for implemented-student helpers --------------------------- #

    @staticmethod
    def add_powerline(clean, sample_rate, freq=50.0, amplitude=0.1):
        """Fallback for student3.add_powerline_interference."""
        clean = np.asarray(clean, dtype=np.float64)
        t = np.arange(clean.size) / float(sample_rate)
        return np.clip(clean + amplitude * np.sin(2.0 * np.pi * freq * t), -1.0, 1.0)

    @staticmethod
    def compute_spectrum(signal, sample_rate):
        """Fallback for student3.compute_spectrum."""
        signal = np.asarray(signal, dtype=np.float64)
        n = signal.size
        spectrum = np.fft.rfft(signal)
        frequencies = np.fft.rfftfreq(n, d=1.0 / float(sample_rate))
        magnitude = (2.0 / n) * np.abs(spectrum)
        magnitude[0] /= 2.0
        if n % 2 == 0:
            magnitude[-1] /= 2.0
        return frequencies, magnitude

    @classmethod
    def identify_interference_frequency(cls, noisy, sample_rate, candidates=(50.0, 60.0)):
        """Fallback for student3.identify_interference_frequency."""
        frequencies, magnitude = cls.compute_spectrum(noisy, sample_rate)
        if frequencies.size == 0:
            return None
        scores = {}
        for candidate in candidates:
            nearby = (frequencies >= candidate - 2.0) & (frequencies <= candidate + 2.0)
            scores[candidate] = float(magnitude[nearby].max()) if nearby.any() else 0.0
        detected = max(scores, key=scores.get)
        return detected if scores[detected] > 0.0 else None
