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
├── data/
│   ├── audio/
│   │   └── .gitkeep                 # Placeholder for the provided clean WAV file
│   └── images/
│       └── .gitkeep                 # Placeholder for the provided grayscale image
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

## Installation

Use Python 3.9 or newer. From the project root:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Dependencies: `numpy`, `scipy`, `matplotlib`, `opencv-python`, and `soundfile`.

---

## Input data

The project expects two provided input files:

1. A **clean WAV file** — place it in `data/audio/` (any `*.wav` file).
2. A **grayscale image** — place it in `data/images/` (e.g. `*.png`, `*.jpg`, `*.bmp`).

The master script automatically picks the first matching file in each folder.
If a file is missing, it prints a friendly message and the affected students fall
back to a small synthetic demo signal/image so the whole project still runs.

> Note: the raw data files are intentionally not committed to the repository.

---

## Running the full project

From the project root:

```bash
python main.py
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

## How each student should work

- Edit **only your own file** under `students/studentN_*/`. The master script and
  the helper module are shared by everyone.
- Every file is a **runnable skeleton**: it contains a `main()` function, clearly
  named helper functions with docstrings, `TODO` comments marking the work you
  must implement, and placeholder plots. No DSP logic is filled in for you.
- You can run your module before writing any code — it prints a "SKELETON MODE"
  notice and shows empty placeholder figures:

  ```bash
  python students/student3_audio_analysis/student3.py
  ```

- Keep function names and signatures intact — `main.py` calls each module's
  `main()` by name, and teammates may rely on the shared helpers.
- If you need a common utility (loading, plotting, MSE/PSNR), add or use a
  function in `utils/helpers.py` instead of duplicating code.
- Import shared helpers from the project root, e.g. `from utils import helpers`
  when running `python main.py`. When running a student file directly, add the
  project root to `sys.path` as shown in the provided skeletons.

---

## Assessment note

The focus is on **correct implementation and demonstrated understanding** of the
DSP concepts — not on achieving perfect denoising or flawless audio quality.
Document your reasoning in comments, explain the choices you made (window sizes,
filter order, notch radius, etc.), and make sure your plots clearly show what is
happening in each domain.
