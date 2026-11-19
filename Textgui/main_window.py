#!/usr/bin/env python3
"""CustomTkinter desktop application for the DSP multi-domain project.

Launch from the project root with::

    python Textgui/main_window.py

or as a package::

    python -m Textgui.main_window

The sidebar switches between an overview page (including the full-pipeline
runner) and the six module pages.  Each page owns its parameters, plots, images,
audio playback and metrics; shared behaviour lives in ``components.widgets``.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import queue
import sys
import threading
from pathlib import Path

import customtkinter as ctk

# Allow both ``python Textgui/main_window.py`` and ``python -m Textgui.main_window``
# to resolve ``components`` and ``pages`` as top-level packages.
TEXTGUI_DIR = Path(__file__).resolve().parent
if str(TEXTGUI_DIR) not in sys.path:
    sys.path.insert(0, str(TEXTGUI_DIR))

from components import widgets as ui  # noqa: E402
from pages.audio_analysis import AudioAnalysisPage  # noqa: E402
from pages.audio_filtering import AudioFilteringPage  # noqa: E402
from pages.convolution import ConvolutionPage  # noqa: E402
from pages.image_denoising import ImageDenoisingPage  # noqa: E402
from pages.image_filtering import ImageFilteringPage  # noqa: E402
from pages.mathematical_signal import MathematicalSignalPage  # noqa: E402

# --------------------------------------------------------------------------- #
# Theme
# --------------------------------------------------------------------------- #

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

APP_TITLE = "Digital Signal Processing – Multi-Domain Project"

MODULES = [
    ("01", "Mathematical Signal & DTFT", "Build a two-tone signal and inspect its DTFT magnitude and phase.", "student1", "mathematical_signal", "📐"),
    ("02", "Convolution & Difference", "Compare time- and frequency-domain convolution and a recursive filter.", "student2", "convolution", "🧮"),
    ("03", "Audio Analysis", "Inject 50/60 Hz interference and locate it in the spectrum.", "student3", "audio_analysis", "🎵"),
    ("04", "Audio Filtering", "Design a notch filter and listen to the cleaned signal.", "student4", "audio_filtering", "🎚️"),
    ("05", "2D Fourier & Image Filtering", "Apply ideal circular low-pass and high-pass masks.", "student5", "image_filtering", "🖼️"),
    ("06", "Image Noise Removal", "Remove sinusoidal noise and score the result with MSE/PSNR.", "student6", "image_denoising", "✨"),
]


# --------------------------------------------------------------------------- #
# Overview / home page
# --------------------------------------------------------------------------- #

class HomePage(ctk.CTkFrame):
    """Landing page with project summary, module cards and the pipeline runner."""

    def __init__(self, master, app):
        super().__init__(master, fg_color="transparent")
        self.app = app
        self._busy = False
        self._events: queue.Queue = queue.Queue()
        self._artifact_views: list[ui.ImageView] = []

        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.pack(fill="both", expand=True)
        scroll.grid_columnconfigure(0, weight=1)

        self._build_hero(scroll)
        self._build_module_cards(scroll)
        self._build_pipeline(scroll)

    # -- layout ------------------------------------------------------------- #

    def _build_hero(self, parent) -> None:
        hero = ctk.CTkFrame(parent, fg_color=ui.COLORS["panel"], corner_radius=14)
        hero.pack(fill="x", padx=18, pady=(18, 10))

        accent = ctk.CTkFrame(hero, fg_color=ui.COLORS["accent"], height=4, corner_radius=2)
        accent.pack(fill="x", padx=18, pady=(18, 0))

        inner = ctk.CTkFrame(hero, fg_color="transparent")
        inner.pack(fill="x", padx=18, pady=(12, 18))

        ctk.CTkLabel(
            inner,
            text="COLLABORATIVE DSP COURSEWORK  ·  6 DOMAINS",
            text_color=ui.COLORS["accent"],
            font=ctk.CTkFont(size=11, weight="bold"),
            anchor="w",
        ).pack(fill="x")
        ctk.CTkLabel(
            inner,
            text=APP_TITLE,
            text_color=ui.COLORS["text"],
            font=ctk.CTkFont(size=25, weight="bold"),
            anchor="w",
            wraplength=900,
            justify="left",
        ).pack(fill="x", pady=(6, 0))
        ctk.CTkLabel(
            inner,
            text=(
                "Explore signals in the discrete-time, frequency, audio and image domains. "
                "Each of the six modules is owned by one student; this desktop workspace "
                "brings them together with consistent controls, plots, audio playback and "
                "MSE/PSNR evaluation."
            ),
            text_color=ui.COLORS["muted"],
            font=ctk.CTkFont(size=12),
            anchor="w",
            wraplength=880,
            justify="left",
        ).pack(fill="x", pady=(8, 0))

        implemented = sum(ui.is_implemented(key) for _, _, _, key, _, _ in MODULES)
        ctk.CTkLabel(
            inner,
            text=f"{implemented} student modules implemented  ·  {len(MODULES) - implemented} reference demos available",
            text_color=ui.COLORS["ok"] if implemented else ui.COLORS["warn"],
            font=ctk.CTkFont(size=11, weight="bold"),
            anchor="w",
        ).pack(fill="x", pady=(10, 0))

    def _build_module_cards(self, parent) -> None:
        section = ctk.CTkFrame(parent, fg_color="transparent")
        section.pack(fill="x", padx=18, pady=(6, 4))
        ctk.CTkLabel(
            section,
            text="The six modules",
            text_color=ui.COLORS["text"],
            font=ctk.CTkFont(size=15, weight="bold"),
            anchor="w",
        ).pack(fill="x")

        grid = ctk.CTkFrame(parent, fg_color="transparent")
        grid.pack(fill="x", padx=18, pady=(4, 8))
        grid.grid_columnconfigure(0, weight=1, uniform="cards")
        grid.grid_columnconfigure(1, weight=1, uniform="cards")

        for index, (number, title, body, key, page_key, icon) in enumerate(MODULES):
            row, column = divmod(index, 2)
            card = ctk.CTkFrame(grid, fg_color=ui.COLORS["panel"], corner_radius=12)
            card.grid(row=row, column=column, sticky="nsew", padx=5, pady=5)

            header = ctk.CTkFrame(card, fg_color="transparent")
            header.pack(fill="x", padx=14, pady=(13, 0))
            ctk.CTkLabel(
                header,
                text=f"{icon}  {number} · {title}",
                text_color=ui.COLORS["text"],
                font=ctk.CTkFont(size=13, weight="bold"),
                anchor="w",
                wraplength=400,
                justify="left",
            ).pack(side="left", fill="x", expand=True)
            ready = ui.is_implemented(key)
            ctk.CTkLabel(
                header,
                text="● module" if ready else "● reference",
                text_color=ui.COLORS["ok"] if ready else ui.COLORS["warn"],
                font=ctk.CTkFont(size=10, weight="bold"),
            ).pack(side="right")

            ctk.CTkLabel(
                card,
                text=body,
                text_color=ui.COLORS["muted"],
                font=ctk.CTkFont(size=11),
                anchor="w",
                wraplength=430,
                justify="left",
            ).pack(fill="x", padx=14, pady=(8, 4))

            ctk.CTkButton(
                card,
                text="Open module  →",
                command=lambda page=page_key: self.app.show_page(page),
                fg_color=ui.COLORS["panel_alt"],
                hover_color=ui.COLORS["accent"],
                text_color=ui.COLORS["text"],
                font=ctk.CTkFont(size=11, weight="bold"),
                height=30,
                corner_radius=8,
            ).pack(fill="x", padx=14, pady=(4, 13))

    def _build_pipeline(self, parent) -> None:
        card = ctk.CTkFrame(parent, fg_color=ui.COLORS["panel"], corner_radius=14)
        card.pack(fill="x", padx=18, pady=(8, 24))

        ctk.CTkLabel(
            card,
            text="Run the complete pipeline",
            text_color=ui.COLORS["text"],
            font=ctk.CTkFont(size=15, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=16, pady=(14, 0))
        ctk.CTkLabel(
            card,
            text=(
                "Execute all six modules in order with the same master runner used by "
                "`python main.py`. The captured console log and generated image artifacts "
                "are shown below."
            ),
            text_color=ui.COLORS["muted"],
            font=ctk.CTkFont(size=11),
            anchor="w",
            wraplength=880,
            justify="left",
        ).pack(fill="x", padx=16, pady=(6, 0))

        controls = ctk.CTkFrame(card, fg_color="transparent")
        controls.pack(fill="x", padx=16, pady=(10, 4))
        self._pipeline_button = ctk.CTkButton(
            controls,
            text="▶ Run full project pipeline",
            command=self._run_pipeline,
            fg_color=ui.COLORS["accent"],
            hover_color=ui.COLORS["accent_hover"],
            font=ctk.CTkFont(size=12, weight="bold"),
            height=36,
        )
        self._pipeline_button.pack(side="left")
        self._pipeline_status = ctk.CTkLabel(
            controls,
            text="Ready. Individual module pages can be run independently.",
            text_color=ui.COLORS["muted"],
            font=ctk.CTkFont(size=11),
            anchor="w",
        )
        self._pipeline_status.pack(side="left", padx=12)

        self._log = ui.TextView(card, height=150)
        self._log.pack(fill="x", padx=16, pady=(6, 4))
        self._log.set_text(
            "Console output appears here after the pipeline finishes.\n"
            "Tip: open a module page from the sidebar to change its parameters."
        )

        ctk.CTkLabel(
            card,
            text="Generated artifacts",
            text_color=ui.COLORS["text"],
            font=ctk.CTkFont(size=12, weight="bold"),
            anchor="w",
        ).pack(fill="x", padx=16, pady=(8, 0))
        self._artifacts = ctk.CTkFrame(card, fg_color="transparent")
        self._artifacts.pack(fill="x", padx=16, pady=(4, 16))
        self._refresh_artifacts()

    # -- pipeline execution ------------------------------------------------- #

    def _run_pipeline(self) -> None:
        if self._busy:
            return
        self._busy = True
        self._pipeline_button.configure(state="disabled", text="⏳ Running…")
        self._pipeline_status.configure(
            text="Running all six DSP modules…", text_color=ui.COLORS["accent"]
        )
        self._log.set_text("Running all six DSP modules… please wait.")

        def worker() -> None:
            buffer = io.StringIO()
            try:
                spec = importlib.util.spec_from_file_location(
                    "dsp_master_runner", ui.PROJECT_ROOT / "main.py"
                )
                if spec is None or spec.loader is None:
                    raise ImportError("Cannot load main.py")
                runner = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(runner)
                with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
                    runner.main()
                self._events.put((buffer.getvalue(), None))
            except Exception as exc:
                self._events.put((buffer.getvalue(), exc))

        threading.Thread(target=worker, daemon=True).start()
        self.after(80, self._poll_pipeline)

    def _poll_pipeline(self) -> None:
        try:
            log, error = self._events.get_nowait()
        except queue.Empty:
            self.after(80, self._poll_pipeline)
            return

        self._busy = False
        self._pipeline_button.configure(state="normal", text="▶ Run full project pipeline")
        if error is not None:
            self._pipeline_status.configure(
                text=f"Pipeline error: {type(error).__name__}: {error}",
                text_color=ui.COLORS["error"],
            )
            self._log.set_text(log or f"Pipeline failed: {error}")
            return

        sections = [line.strip("= ") for line in log.splitlines() if line.startswith("=== ")]
        self._pipeline_status.configure(
            text=f"Pipeline finished · {len(sections)} section(s) executed.",
            text_color=ui.COLORS["ok"],
        )
        header = "Modules executed:\n" + "\n".join(f"  • {name}" for name in sections) + "\n\n"
        self._log.set_text(header + "── Console output ─────────────────────────────\n" + (log or "(no console output)"))
        self._refresh_artifacts()

    def _refresh_artifacts(self) -> None:
        for child in list(self._artifacts.winfo_children()):
            child.destroy()
        self._artifact_views.clear()

        artifacts = sorted(ui.OUTPUT_DIR.rglob("*.png")) if ui.OUTPUT_DIR.is_dir() else []
        if not artifacts:
            ctk.CTkLabel(
                self._artifacts,
                text="No image artifacts yet — run the pipeline (or an image module) to generate some.",
                text_color=ui.COLORS["muted"],
                font=ctk.CTkFont(size=11),
                anchor="w",
            ).pack(fill="x", padx=4, pady=4)
            return

        for artifact in artifacts[:4]:
            cell = ctk.CTkFrame(self._artifacts, fg_color=ui.COLORS["panel_alt"], corner_radius=8)
            cell.pack(side="left", padx=5, pady=5)
            view = ui.ImageView(cell, max_size=(230, 150))
            view.pack(padx=6, pady=6)
            view.show_image(str(artifact), caption=artifact.name)
            self._artifact_views.append(view)

        if len(artifacts) > 4:
            ctk.CTkLabel(
                self._artifacts,
                text=f"+ {len(artifacts) - 4} more file(s)\nin outputs/",
                text_color=ui.COLORS["muted"],
                font=ctk.CTkFont(size=11),
                justify="left",
            ).pack(side="left", padx=10)


# --------------------------------------------------------------------------- #
# Main application window
# --------------------------------------------------------------------------- #

class DSPApp(ctk.CTk):
    """Resizable main window with a sidebar and stackable module pages."""

    def __init__(self) -> None:
        super().__init__()
        self.title(APP_TITLE)
        self.geometry("1340x880")
        self.minsize(1060, 700)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.nav_buttons: dict[str, ctk.CTkButton] = {}
        self.pages: dict[str, ctk.CTkFrame] = {}
        self.current_page = "overview"

        self._build_sidebar()
        self._build_content()

    # -- construction ------------------------------------------------------- #

    def _build_sidebar(self) -> None:
        sidebar = ctk.CTkFrame(self, width=266, corner_radius=0, fg_color=ui.COLORS["panel"])
        sidebar.grid(row=0, column=0, sticky="nsew")
        sidebar.grid_propagate(False)
        sidebar.grid_rowconfigure(5, weight=1)

        # Brand
        brand = ctk.CTkFrame(sidebar, fg_color="transparent")
        brand.grid(row=0, column=0, sticky="ew", padx=18, pady=(22, 16))
        badge = ctk.CTkLabel(
            brand,
            text="DSP",
            text_color="#ffffff",
            font=ctk.CTkFont(size=16, weight="bold"),
            fg_color=ui.COLORS["accent"],
            corner_radius=8,
            width=44,
            height=36,
        )
        badge.grid(row=0, column=0, sticky="w")
        ctk.CTkLabel(
            brand,
            text="Digital Signal\nProcessing",
            text_color=ui.COLORS["text"],
            font=ctk.CTkFont(size=14, weight="bold"),
            justify="left",
        ).grid(row=0, column=1, sticky="w", padx=(10, 0))
        ctk.CTkLabel(
            brand,
            text="MULTI-DOMAIN PROJECT",
            text_color=ui.COLORS["muted"],
            font=ctk.CTkFont(size=9, weight="bold"),
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

        ctk.CTkFrame(sidebar, fg_color=ui.COLORS["panel_alt"], height=1).grid(
            row=1, column=0, sticky="ew", padx=18
        )

        # Workspace section
        ctk.CTkLabel(
            sidebar,
            text="WORKSPACE",
            text_color=ui.COLORS["muted"],
            font=ctk.CTkFont(size=10, weight="bold"),
            anchor="w",
        ).grid(row=2, column=0, sticky="ew", padx=20, pady=(16, 6))
        self._add_nav_button(sidebar, "🏠", "Overview", "overview", 3)

        # Module section
        ctk.CTkLabel(
            sidebar,
            text="MODULES",
            text_color=ui.COLORS["muted"],
            font=ctk.CTkFont(size=10, weight="bold"),
            anchor="w",
        ).grid(row=4, column=0, sticky="ew", padx=20, pady=(16, 6))

        nav_frame = ctk.CTkFrame(sidebar, fg_color="transparent")
        nav_frame.grid(row=5, column=0, sticky="nsew", padx=10)
        for index, (number, title, _body, _key, page_key, icon) in enumerate(MODULES):
            self._add_nav_button(
                nav_frame,
                icon,
                f"{number}  {title}",
                page_key,
                index,
                parent_is_grid=True,
            )

        # Footer
        footer = ctk.CTkFrame(sidebar, fg_color="transparent")
        footer.grid(row=6, column=0, sticky="ew", padx=18, pady=(0, 18))
        implemented = sum(ui.is_implemented(key) for _, _, _, key, _, _ in MODULES)
        ctk.CTkLabel(
            footer,
            text=f"{implemented} implemented  ·  {len(MODULES) - implemented} reference",
            text_color=ui.COLORS["muted"],
            font=ctk.CTkFont(size=10),
            anchor="w",
        ).pack(fill="x")
        ctk.CTkLabel(
            footer,
            text="CustomTkinter desktop edition",
            text_color=ui.COLORS["muted"],
            font=ctk.CTkFont(size=9),
            anchor="w",
        ).pack(fill="x", pady=(2, 0))

    def _add_nav_button(self, parent, icon, label, key, row, parent_is_grid=False) -> None:
        button = ctk.CTkButton(
            parent,
            text=f"{icon}   {label}",
            command=lambda: self.show_page(key),
            anchor="w",
            height=38 if parent_is_grid else 40,
            corner_radius=8,
            fg_color="transparent",
            hover_color=ui.COLORS["panel_alt"],
            text_color=ui.COLORS["muted"],
            font=ctk.CTkFont(size=12),
        )
        if parent_is_grid:
            button.grid(row=row, column=0, sticky="ew", pady=2)
        else:
            button.grid(row=row, column=0, sticky="ew", padx=10)
        self.nav_buttons[key] = button

    def _build_content(self) -> None:
        content = ctk.CTkFrame(self, fg_color=ui.COLORS["bg"], corner_radius=0)
        content.grid(row=0, column=1, sticky="nsew")
        content.grid_rowconfigure(0, weight=1)
        content.grid_columnconfigure(0, weight=1)

        factories: list[tuple[str, type]] = [
            ("overview", HomePage),
            ("mathematical_signal", MathematicalSignalPage),
            ("convolution", ConvolutionPage),
            ("audio_analysis", AudioAnalysisPage),
            ("audio_filtering", AudioFilteringPage),
            ("image_filtering", ImageFilteringPage),
            ("image_denoising", ImageDenoisingPage),
        ]
        for key, factory in factories:
            try:
                page = factory(content, self)
            except Exception as exc:
                # A broken page must not take down the whole application.
                page = ctk.CTkFrame(content, fg_color="transparent")
                ctk.CTkLabel(
                    page,
                    text=f"Could not build page '{key}': {type(exc).__name__}: {exc}",
                    text_color=ui.COLORS["error"],
                    font=ctk.CTkFont(size=13),
                    wraplength=700,
                ).pack(expand=True)
            page.grid(row=0, column=0, sticky="nsew")
            self.pages[key] = page

    # -- navigation --------------------------------------------------------- #

    def show_page(self, key: str) -> None:
        page = self.pages.get(key)
        if page is None:
            return
        page.tkraise()
        self.current_page = key
        self.title(f"{APP_TITLE}  ·  {self._page_name(key)}")
        for nav_key, button in self.nav_buttons.items():
            active = nav_key == key
            button.configure(
                fg_color=ui.COLORS["accent"] if active else "transparent",
                hover_color=ui.COLORS["accent_hover"] if active else ui.COLORS["panel_alt"],
                text_color="#ffffff" if active else ui.COLORS["muted"],
            )

    @staticmethod
    def _page_name(key: str) -> str:
        if key == "overview":
            return "Overview"
        for number, title, _body, _student, page_key, _icon in MODULES:
            if page_key == key:
                return f"{number} · {title}"
        return key


def main() -> None:
    """Create the window and start the Tk event loop."""
    app = DSPApp()
    app.mainloop()


if __name__ == "__main__":
    main()
