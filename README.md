# DSP Multi-Domain Project

A collaborative **Digital Signal Processing (DSP)** team project that explores
signals in several domains: discrete-time signals and the DTFT, convolution and
difference equations, 1-D audio analysis and filtering, and 2-D image processing
with the Fourier transform.

Each of the six students owns one self-contained Python module. A single master
script (`main.py`) runs every module in order, so the individual contributions
come together as one complete multi-domain project.

---

## Team structure

| # | Student role | Module | Domain |
|---|--------------|--------|--------|
| 1 | **Mathematical Signal & DTFT** | `students/student1_mathematical_signal_dtft/student1.py` | Discrete-time signals, manual DTFT, magnitude/phase |
| 2 | **Convolution & Difference Equation** | `students/student2_convolution_difference/student2.py` | Moving average, time vs. frequency domain convolution, first-order recursion |
| 3 | **Audio Analysis** | `students/student3_audio_analysis/student3.py` | WAV loading, 50/60 Hz power-line interference, waveform and spectrum analysis |
| 4 | **Audio Filtering** | `students/student4_audio_filtering/student4.py` | Second-order notch filter design, filtered WAV output, spectral comparison |
| 5 | **2D Fourier & Image Filtering** | `students/student5_2d_fourier_image_filtering/student5.py` | Grayscale images, 2-D FFT, `fftshift`, ideal circular low-pass/high-pass masks |
| 6 | **Image Noise Removal** | `students/student6_image_noise_removal/student6.py` | Periodic sinusoidal noise, frequency-domain notch mask, MSE/PSNR evaluation |

Shared code lives in `utils/helpers.py` (audio/image loading, spectrum plotting,
MSE/PSNR calculation, and small demo signal generators).

---

## Repository structure

```text
dsp-multi-domain-project/
├── README.md
├── requirements.txt
├── main.py                          # Master script that runs the entire project in order
├── Textgui/                         # CustomTkinter desktop application (one page per module)
│   ├── main_window.py               # Entry point: window, sidebar navigation, pipeline runner
│   ├── components/
│   │   ├── widgets.py               # Shared widgets, helpers and the common page template
│   │   └── __init__.py
│   └── pages/
│       ├── mathematical_signal.py
│       ├── convolution.py
│       ├── audio_analysis.py
│       ├── audio_filtering.py
│       ├── image_filtering.py
│       ├── image_denoising.py
│       └── __init__.py
├── data/
│   ├── audio/
│   │   ├── clean.wav                # Bundled sample signal (used by the demo)
│   │   └── .gitkeep
│   └── images/
│       ├── sample.jpg               # Bundled sample image (used by the demo)
│       └── .gitkeep
├── students/
│   ├── student1_mathematical_signal_dtft/
│   │   └── student1.py
│   ├── student2_convolution_difference/
│   │   └── student2.py
│   ├── student3_audio_analysis/
│   │   └── student3.py
│   ├── student4_audio_filtering/
│   │   └── student4.py
│   ├── student5_2d_fourier_image_filtering/
│   │   └── student5.py
│   └── student6_image_noise_removal/
│       └── student6.py
└── utils/
    └── helpers.py                   # Optional shared utility functions
```

---

## Installation (Linux)

Use Python 3.9 or newer. All commands below are for a **Linux terminal** and are
run from the project root (the folder containing `main.py`, `README.md` and
`Textgui/`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`python3` is the standard interpreter name on Linux. On Debian/Ubuntu the Tk
library used by the GUI is a separate system package — install it with:

```bash
sudo apt install python3-tk
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `opencv-python`, `soundfile`, and
`customtkinter` (for the desktop interface; pulls in Pillow automatically).

---

## Quick start (Linux)

Follow these steps from a terminal. Every command assumes Linux and is run from
the project root.

**1. Enter the project folder**

```bash
cd ~/path/to/dsp-multi-domain-project
```

**2. Create and activate the virtual environment (once)**

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Your prompt should now show `(.venv)`. Activate it again with
`source .venv/bin/activate` in every new terminal; leave it with `deactivate`.

**3A. Run in the terminal (no GUI)**

```bash
python3 main.py
```

Runs all six modules in order and prints a labelled section for each one.

**3B. Run the desktop GUI**

```bash
python3 Textgui/main_window.py
```

Opens the dark-themed CustomTkinter window with the sidebar and all six module
pages. (Equivalent: `python3 -m Textgui.main_window`.) If `tkinter` fails to
import, install `python3-tk` as shown in the Installation section.

---

## Input data

The project uses two input files:

1. A **clean WAV file** — place it in `data/audio/` (any `*.wav` file).
2. A **grayscale image** — place it in `data/images/` (e.g. `*.png`, `*.jpg`, `*.bmp`).

Two ready-to-use samples are included in the repository:
`data/audio/clean.wav` and `data/images/sample.jpg`. You may replace either one
with your own file.

The master script automatically picks the first matching file in each folder.
If a file is missing, it prints a friendly message and the affected students fall
back to a small synthetic demo signal/image so the whole project still runs.

---

## Running the full project

From the project root:

```bash
python3 main.py
```

This prints a clearly labelled section for each student:

```text
=== Student 1: Mathematical Signal & DTFT ===
=== Student 2: Convolution & Difference Equation ===
=== Student 3: Audio Analysis ===
=== Student 4: Audio Filtering ===
=== Student 5: 2D Fourier & Image Filtering ===
=== Student 6: Image Noise Removal ===
```

and finishes with a summary confirming that the full multi-domain project has
completed. Once Student 4 implements the TODOs, the filtered output will be
written to `data/audio/filtered.wav` (or `filtered_demo.wav` with the demo signal).

---

## Desktop application (CustomTkinter)

A modern desktop interface exposes all six modules with per-module parameters,
run buttons, plots, audio playback, images and evaluation metrics. From the
project root:

```bash
python3 Textgui/main_window.py
```

(Equivalently: `python3 -m Textgui.main_window`.) No browser or local server is
needed — the app opens in its own window with a dark theme and a sidebar. The
**Overview** page shows the project summary, live module status and a
**Run full project pipeline** button that executes `main.py` and displays the
captured console log and generated image artifacts. Use the sidebar to open any
individual module.

Each module page has the same structure: adjustable parameters, a **Run** button
(computation happens in a background thread so the window stays responsive), and
output tabs for plots, images, audio playback and numeric results (MSE / PSNR
where applicable). A page calls the corresponding student module automatically
once its functions are implemented; until then it shows a clearly labelled
**reference demo** with the same controls and outputs, so every page is usable
while the student files are still in progress. Missing files in `data/` are
handled with synthetic demo signals/images — no manual path entry is needed.

---

## How each student should work

- Edit **only your own file** under `students/studentN_*/`. The master script and
  the helper module are shared by everyone.
- Students 3 and 5 are fully implemented and can be used as references. Students
  1, 2, 4 and 6 contain **only the required function names (with their parameters)
  and empty `pass` placeholders** — you supply the imports, docstrings,
  implementation, and plots for your own file.
- Running a placeholder-only module before implementing it does nothing (each
  placeholder returns immediately). Verify that every module still loads with:

  ```bash
  python3 main.py
  ```

- Keep function names and parameters intact — `main.py` calls each module's
  `main()` by name, and teammates may rely on the shared helpers.
- If you need a common utility (loading, plotting, MSE/PSNR), add or use a
  function in `utils/helpers.py` instead of duplicating code.
- Import shared helpers from the project root, e.g. `from utils import helpers`
  when running `python3 main.py`. When running a student file directly, add the
  project root to `sys.path` as shown in the provided skeletons.

---

## Assessment note

The focus is on **correct implementation and demonstrated understanding** of the
DSP concepts — not on achieving perfect denoising or flawless audio quality.
Document your reasoning in comments, explain the choices you made (window sizes,
filter order, notch radius, etc.), and make sure your plots clearly show what is
happening in each domain.
