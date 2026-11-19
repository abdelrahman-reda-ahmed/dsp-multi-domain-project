"""Shared presentation and utility helpers for the Streamlit GUI.

The GUI follows the same conventions as ``main.py``: it discovers inputs in
``data/audio`` and ``data/images`` and never asks the user for a filesystem path.
Every string rendered as HTML is escaped, and all file access is limited to the
project's own folders.
"""

from __future__ import annotations

import ast
import contextlib
import html
import importlib.util
import inspect
import io
import sys
import wave
from pathlib import Path
from types import ModuleType

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIO_DIR = PROJECT_ROOT / "data" / "audio"
IMAGE_DIR = PROJECT_ROOT / "data" / "images"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import helpers  # noqa: E402  (path setup must happen first)

# Module keys map to the repository-relative student file paths.
MODULE_FILES = {
    "student1": "students/student1_mathematical_signal_dtft/student1.py",
    "student2": "students/student2_convolution_difference/student2.py",
    "student3": "students/student3_audio_analysis/student3.py",
    "student4": "students/student4_audio_filtering/student4.py",
    "student5": "students/student5_2d_fourier_image_filtering/student5.py",
    "student6": "students/student6_image_noise_removal/student6.py",
}

_CSS = """
<style>
.stApp { background: #f6f8fc; }
.block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1200px; }
[data-testid="stSidebar"] { background: #eef2f9; border-right: 1px solid #dbe3ef; }
.dsp-hero {
    background: linear-gradient(120deg, #0f172a 0%, #1d4ed8 100%);
    color: #ffffff;
    padding: 1.5rem 1.8rem;
    border-radius: 16px;
    box-shadow: 0 10px 30px rgba(15, 23, 42, 0.18);
    margin-bottom: 1rem;
}
.dsp-hero h1 { margin: 0 0 0.4rem 0; font-size: 1.85rem; font-weight: 700; }
.dsp-hero p { margin: 0; color: #dbeafe; font-size: 0.97rem; line-height: 1.5; }
.dsp-kicker {
    display: inline-block;
    font-size: 0.7rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    background: rgba(255, 255, 255, 0.16);
    padding: 0.25rem 0.6rem;
    border-radius: 999px;
    margin-bottom: 0.7rem;
}
.dsp-badge {
    display: inline-block;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.28rem 0.7rem;
    border-radius: 999px;
    margin: 0.5rem 0 0.2rem 0;
}
.dsp-badge.ok { background: #dcfce7; color: #166534; }
.dsp-badge.pending { background: #fef3c7; color: #92400e; }
.dsp-card {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 14px;
    padding: 1rem 1.2rem;
    margin: 0.6rem 0;
    box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
}
.dsp-card h3 { margin: 0 0 0.45rem 0; font-size: 1rem; }
.dsp-card p { margin: 0; color: #475569; font-size: 0.9rem; line-height: 1.5; }
div[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid #e2e8f0;
    border-radius: 12px;
    padding: 0.7rem 0.9rem;
}
.stButton > button { border-radius: 10px; font-weight: 600; }
.stTabs [data-baseweb="tab"] { border-radius: 10px; }
h2, h3 { color: #0f172a; }
</style>
"""


def setup_page(title: str, icon: str = "📡") -> None:
    """Configure every page consistently and inject the shared stylesheet."""
    st.set_page_config(
        page_title=f"{title} — DSP Project",
        page_icon=icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CSS, unsafe_allow_html=True)


def sidebar_context(number: str, title: str, key: str) -> None:
    """Show compact module status and input-data status in the sidebar."""
    with st.sidebar:
        st.markdown(f"### {number} · {html.escape(title)}")
        st.markdown(
            '<span class="dsp-badge ok">Implemented</span>'
            if is_implemented(key)
            else '<span class="dsp-badge pending">Reference demo</span>',
            unsafe_allow_html=True,
        )
        st.divider()
        st.caption("Project inputs")
        audio = find_input(AUDIO_DIR, helpers.AUDIO_EXTENSIONS)
        image = find_input(IMAGE_DIR, helpers.IMAGE_EXTENSIONS)
        st.markdown(f"- Audio: `{html.escape(audio.name)}`" if audio else "- Audio: _missing (demo signal)_")
        st.markdown(f"- Image: `{html.escape(image.name)}`" if image else "- Image: _missing (demo image)_")


def header(number: str, title: str, description: str, key: str) -> None:
    """Render the academic-style page header with an implementation badge."""
    badge = (
        '<span class="dsp-badge ok">Uses student module</span>'
        if is_implemented(key)
        else '<span class="dsp-badge pending">Reference demo · student file is a placeholder</span>'
    )
    st.markdown(
        f'<div class="dsp-hero">'
        f'<span class="dsp-kicker">Module {html.escape(number)} / 06</span>'
        f'<h1>{html.escape(title)}</h1>'
        f'<p>{html.escape(description)}</p>{badge}</div>',
        unsafe_allow_html=True,
    )


def card(title: str, body: str) -> None:
    """Render a small explanatory card."""
    st.markdown(
        f'<div class="dsp-card"><h3>{html.escape(title)}</h3>'
        f'<p>{html.escape(body)}</p></div>',
        unsafe_allow_html=True,
    )


def metrics_grid(items: dict[str, object]) -> None:
    """Display numeric results in responsive metric columns."""
    columns = st.columns(min(len(items), 4)) if items else []
    for column, (label, value) in zip(columns, items.items()):
        column.metric(label, value)


def find_input(directory: Path, extensions: tuple[str, ...]) -> Path | None:
    """Return the first matching input file, or ``None`` when data is missing."""
    return helpers.find_data_file(directory, extensions)


def load_student_module(key: str) -> ModuleType | None:
    """Import a ``studentN.py`` file by path, mirroring ``main.load_module``."""
    relative = MODULE_FILES.get(key)
    if relative is None:
        return None
    path = PROJECT_ROOT / relative
    if not path.is_file():
        return None
    spec = importlib.util.spec_from_file_location(f"gui_{key}", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception:
        # A student module may require an optional dependency that is not
        # installed (e.g. OpenCV). Keep the GUI alive; callers fall back.
        return None
    return module


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
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
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


def run_student_main(key: str, **kwargs) -> tuple[object, str, list]:
    """Run a student ``main(show=False)`` and capture its log and new figures.

    This lets the GUI surface work as soon as a student implements their file,
    without changing the GUI code. Placeholder modules are skipped by callers.
    """
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
        except Exception as exc:  # keep the GUI alive if a student module fails
            result = None
            buffer.write(f"Student module error: {type(exc).__name__}: {exc}\n")
    figures = [plt.figure(number) for number in sorted(set(plt.get_fignums()) - before)]
    return result, buffer.getvalue(), figures


def wav_bytes(audio, sample_rate: int) -> bytes:
    """Encode mono samples as 16-bit PCM WAV bytes for ``st.audio``."""
    samples = np.clip(np.asarray(audio, dtype=np.float64), -1.0, 1.0)
    pcm = (samples * 32767.0).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(int(sample_rate))
        handle.writeframes(pcm)
    return buffer.getvalue()


def _full_width_kwargs(widget) -> dict:
    """Return the width keyword understood by the installed Streamlit version."""
    try:
        if "width" in inspect.signature(widget).parameters:
            return {"width": "stretch"}
    except (TypeError, ValueError):
        pass
    return {"use_container_width": True}


def show_figure(figure) -> None:
    """Display a matplotlib figure at full container width."""
    st.pyplot(figure, **_full_width_kwargs(st.pyplot))
    plt.close(figure)


def show_image(data, caption: str | None = None, clamp: bool | None = None) -> None:
    """Display an image at full container width with optional caption/clamp."""
    options = _full_width_kwargs(st.image)
    if caption is not None:
        options["caption"] = caption
    if clamp is not None:
        options["clamp"] = clamp
    st.image(data, **options)


def full_width_button(label: str, **kwargs) -> bool:
    """Button that fills the container on both old and new Streamlit releases."""
    kwargs.update(_full_width_kwargs(st.button))
    return st.button(label, **kwargs)


def to_uint8(array, normalize: bool = False) -> np.ndarray:
    """Convert an image array to uint8 for reliable ``st.image`` rendering.

    Streamlit interprets float images as values in ``[0, 1]``, so Fourier
    results (which may span a much wider range) must be converted explicitly.
    """
    values = np.asarray(array, dtype=np.float64)
    if normalize:
        low, high = float(values.min()), float(values.max())
        values = (values - low) / max(high - low, 1e-12) * 255.0
    return np.clip(np.nan_to_num(values, nan=0.0), 0.0, 255.0).astype(np.uint8)


def data_warning(kind: str) -> None:
    """Explain that a missing input file triggers the synthetic fallback."""
    if kind == "audio":
        st.info(
            "No WAV file found in `data/audio/`. The GUI is using a synthetic "
            "demo signal — add a `.wav` file there to analyse real audio."
        )
    else:
        st.info(
            "No image found in `data/images/`. The GUI is using a synthetic "
            "demo image — add a `.png`/`.jpg` file there to process a real image."
        )
