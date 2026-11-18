#!/usr/bin/env python3
"""Master runner for the DSP multi-domain team project.

Runs all six student modules in order, prints a labelled section header for each,
locates the shared input data, and reports when a data file is missing. Each
student module falls back to a small synthetic demo so the full project still
executes before the real WAV/image files are supplied.

Usage
-----
    python main.py
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep plotting non-blocking in headless/CI environments (no display server).
if not os.environ.get("DISPLAY") and sys.platform != "win32":
    os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt

from utils import helpers

# (Section title, path of the module relative to the project root, data argument)
STUDENTS = [
    ("Student 1: Mathematical Signal & DTFT",
     "students/student1_mathematical_signal_dtft/student1.py", None),
    ("Student 2: Convolution & Difference Equation",
     "students/student2_convolution_difference/student2.py", None),
    ("Student 3: Audio Analysis",
     "students/student3_audio_analysis/student3.py", "audio_path"),
    ("Student 4: Audio Filtering",
     "students/student4_audio_filtering/student4.py", "audio_path"),
    ("Student 5: 2D Fourier & Image Filtering",
     "students/student5_2d_fourier_image_filtering/student5.py", "image_path"),
    ("Student 6: Image Noise Removal",
     "students/student6_image_noise_removal/student6.py", "image_path"),
]


def load_module(module_name: str, file_path: Path):
    """Import a Python module directly from a file path.

    The student folders are not packages, so ``importlib`` is used to load each
    ``studentN.py`` file under a unique module name.
    """
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def locate_inputs():
    """Find the shared WAV/image files and print friendly status messages."""
    helpers.ensure_directories()

    audio_path = helpers.find_data_file(helpers.AUDIO_DIR, helpers.AUDIO_EXTENSIONS)
    image_path = helpers.find_data_file(helpers.IMAGE_DIR, helpers.IMAGE_EXTENSIONS)

    print("-" * 72)
    print("Input data check")
    if audio_path is None:
        print("  [!] No WAV file found in data/audio/.")
        print("      Students 3 and 4 will use a synthetic demo signal instead.")
        print("      -> Drop a clean .wav file into data/audio/ to use real audio.")
    else:
        print(f"  [OK] Audio : {audio_path.relative_to(PROJECT_ROOT)}")

    if image_path is None:
        print("  [!] No image found in data/images/.")
        print("      Students 5 and 6 will use a synthetic demo image instead.")
        print("      -> Drop a grayscale image into data/images/ to use a real image.")
    else:
        print(f"  [OK] Image : {image_path.relative_to(PROJECT_ROOT)}")
    print("-" * 72)

    return audio_path, image_path


def main() -> None:
    """Run every student module in sequence and print a final summary."""
    print("=" * 72)
    print("DSP MULTI-DOMAIN PROJECT - MASTER RUNNER")
    print("=" * 72)

    audio_path, image_path = locate_inputs()
    completed = []
    failed = []

    for index, (title, relative_path, data_argument) in enumerate(STUDENTS, start=1):
        print()
        print(f"=== {title} ===")

        module_path = PROJECT_ROOT / relative_path
        if not module_path.is_file():
            print(f"  [!] Missing module: {relative_path} - skipping.")
            failed.append(title)
            continue

        try:
            module = load_module(f"student{index}", module_path)
            entry_point = getattr(module, "main", None)
            if not callable(entry_point):
                print("  [!] Module has no callable main() - skipping.")
                failed.append(title)
                continue

            kwargs = {}
            if data_argument == "audio_path":
                kwargs["audio_path"] = audio_path
            elif data_argument == "image_path":
                kwargs["image_path"] = image_path

            entry_point(**kwargs)
            completed.append(title)
        except Exception as exc:  # keep the pipeline running for the other students
            print(f"  [ERROR] {title} raised {type(exc).__name__}: {exc}")
            failed.append(title)
        finally:
            plt.close("all")

    print()
    print("=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"  Modules completed : {len(completed)}/{len(STUDENTS)}")
    if failed:
        print("  Modules needing attention:")
        for title in failed:
            print(f"    - {title}")
    if len(completed) == len(STUDENTS):
        print("  The full multi-domain DSP project has finished successfully.")
    else:
        print("  The full multi-domain DSP project has finished with warnings.")
    print("=" * 72)


if __name__ == "__main__":
    main()
