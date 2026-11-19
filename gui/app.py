#!/usr/bin/env python3
"""Streamlit home page for the DSP multi-domain project.

Launch with::

    streamlit run gui/app.py
"""

from __future__ import annotations

import contextlib
import html
import importlib.util
import io
import sys
from pathlib import Path

PROJECT_ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "main.py").is_file()
)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from gui import _shared as ui

ui.setup_page("Digital Signal Processing — Multi-Domain Project", "📡")

# --------------------------------------------------------------------------- #
# Sidebar: project identity and quick status
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## DSP Project")
    st.caption("Multi-domain signal processing · collaborative coursework")
    st.divider()
    st.markdown("### Navigation")
    st.caption("Use the page menu above to open a module.")
    st.divider()
    st.markdown("### Module status")
    for number, title, key in [
        ("01", "Mathematical Signal & DTFT", "student1"),
        ("02", "Convolution & Difference", "student2"),
        ("03", "Audio Analysis", "student3"),
        ("04", "Audio Filtering", "student4"),
        ("05", "2D Fourier & Image Filtering", "student5"),
        ("06", "Image Noise Removal", "student6"),
    ]:
        dot = "🟢" if ui.is_implemented(key) else "🟡"
        st.markdown(f"{dot} {number} · {html.escape(title)}")
    st.divider()
    st.caption("Inputs are read from `data/audio/` and `data/images/`.")

# --------------------------------------------------------------------------- #
# Hero
# --------------------------------------------------------------------------- #
st.markdown(
    '<div class="dsp-hero">'
    '<span class="dsp-kicker">Collaborative DSP coursework · 6 domains</span>'
    '<h1>Digital Signal Processing – Multi-Domain Signal Processing Project</h1>'
    '<p>Explore signals in the discrete-time, frequency, audio and image domains. '
    'Each of the six modules is owned by one student; this interface brings them '
    'together into a single interactive workspace.</p>'
    '</div>',
    unsafe_allow_html=True,
)

overview, pipeline = st.columns([1.05, 0.95], gap="large")

with overview:
    st.subheader("Project overview")
    st.write(
        "The project combines mathematical signal analysis, convolution and "
        "difference equations, power-line interference in audio, notch filtering, "
        "2-D Fourier image filtering, and periodic-noise removal with MSE/PSNR "
        "evaluation. Every page exposes the relevant parameters, a **Run** button "
        "and dedicated output areas for plots, images, audio and metrics."
    )

    st.markdown("#### The six modules")
    modules = [
        ("01 · Mathematical Signal & DTFT", "Build a two-tone signal and inspect its DTFT magnitude and phase.", "student1"),
        ("02 · Convolution & Difference", "Compare time- and frequency-domain convolution and a recursive filter.", "student2"),
        ("03 · Audio Analysis", "Inject 50/60 Hz interference and locate it in the spectrum.", "student3"),
        ("04 · Audio Filtering", "Design a notch filter and listen to the cleaned signal.", "student4"),
        ("05 · 2D Fourier & Image Filtering", "Apply ideal circular low-pass and high-pass masks.", "student5"),
        ("06 · Image Noise Removal", "Remove sinusoidal noise and score the result with MSE/PSNR.", "student6"),
    ]
    for index in range(0, len(modules), 2):
        left, right = st.columns(2)
        for column, (title, body, key) in zip((left, right), modules[index : index + 2]):
            badge = "🟢 module" if ui.is_implemented(key) else "🟡 reference"
            with column:
                ui.card(f"{badge} · {title}", body)

with pipeline:
    st.subheader("Run the complete pipeline")
    st.write(
        "Execute all six modules in order with the same master runner used by "
        "`python main.py`. The console log and any generated image artifacts are "
        "displayed below."
    )

    if ui.full_width_button("▶ Run full project pipeline", type="primary"):
        buffer = io.StringIO()
        with st.spinner("Running all six DSP modules…"):
            spec = importlib.util.spec_from_file_location(
                "dsp_master_runner", PROJECT_ROOT / "main.py"
            )
            runner = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(runner)
            with contextlib.redirect_stdout(buffer):
                runner.main()
        log = buffer.getvalue()

        st.success("Pipeline finished — see the log below.")
        sections = [line.strip("= ") for line in log.splitlines() if line.startswith("=== ")]
        if sections:
            st.markdown("**Modules executed**")
            for line in sections:
                st.markdown(f"- {html.escape(line)}")

        with st.expander("Console output", expanded=False):
            st.code(log or "(no console output)", language="text")

        artifacts = sorted(ui.OUTPUT_DIR.rglob("*.png"))
        if artifacts:
            st.markdown("**Generated artifacts**")
            for artifact in artifacts[:4]:
                ui.show_image(str(artifact), caption=html.escape(artifact.name))
            if len(artifacts) > 4:
                st.caption(f"+ {len(artifacts) - 4} more files in `outputs/`.")
        else:
            st.caption("No image artifacts were generated by this run.")

st.divider()
st.subheader("Input data")
audio = ui.find_input(ui.AUDIO_DIR, ui.helpers.AUDIO_EXTENSIONS)
image = ui.find_input(ui.IMAGE_DIR, ui.helpers.IMAGE_EXTENSIONS)
col_audio, col_image = st.columns(2)
with col_audio:
    if audio:
        st.success(f"Audio: `{html.escape(audio.name)}`")
        st.audio(ui.wav_bytes(*ui.helpers.load_audio(audio)), format="audio/wav")
    else:
        ui.data_warning("audio")
with col_image:
    if image:
        st.success(f"Image: `{html.escape(image.name)}`")
        ui.show_image(str(image), caption="Sample image")
    else:
        ui.data_warning("image")

st.caption(
    "Tip: open a module page from the sidebar to change its parameters and run it "
    "independently. Start the server with `streamlit run gui/app.py`."
)
