Lunar Lumina: Non-AI Lunar Image Enhancement (NYCU Fall 2025)

NYCU_IP_2025Fall — Final Project
Team 20 — Lunar Lumina | Swarnajit Bhattacharya

Overview

This project implements a complete non-AI enhancement pipeline for high-resolution lunar astrophotography. The workflow integrates multi-frame stacking, wavelet-based denoising, adaptive contrast enhancement (CLAHE), and multi-pass sharpening to improve detail visibility, SNR, PSNR, and SSIM—while preserving scientific authenticity. The project demonstrates that classical image-processing methods can achieve high-quality lunar enhancement on standard hardware without machine learning or heavy computation.

Repo Layout
LunarLumina/
├── src/
│   ├── stacking.py
│   ├── wavelet_denoise.py
│   ├── clahe_enhance.py
│   ├── sharpen.py
│   ├── metrics.py
│   └── pipeline.py
├── data/
│   ├── input/
│   └── output/
├── results/
│   ├── histograms/
│   ├── snr_analysis/
│   ├── difference_maps/
│   └── processed/
└── report/
    └── Lunar_Lumina_Project_Report.pdf

Setup

Install dependencies:

pip install -r requirements.txt


Requirements include:
opencv-python, pywavelets, numpy, scipy, matplotlib, scikit-image

Run the Enhancement Pipeline (Quick Start)

Open pipeline.py and configure:

use_stacking = True
use_wavelet_denoising = True
use_clahe = True
use_unsharp_mask = True

path_input = "data/input/lunar_raw.jpg"
path_output = "data/output/lunar_enhanced.jpg"


Run:

python src/pipeline.py


Tip: You can toggle:

use_wavelet_denoising (multi-scale denoise)

use_clahe (local contrast)

use_unsharp_mask (edge enhancement)

wavelet_type, clip_limit, sigma, etc.

Metrics & Evaluation (Optional)

To compute SNR, PSNR, SSIM, histograms, Gaussian-blur robustness, and difference maps:

python src/metrics.py


Outputs saved under results/.

Dataset (Optional)

For multi-frame processing, place lunar frames under:

data/input/frames/


Enable stacking in pipeline.py:

use_stacking = True
stack_folder = "data/input/frames"

Acknowledgement

This project is developed for the NYCU Fall 2025 Image Processing course.
It is based entirely on classical image-processing techniques, inspired by established astronomy workflows (stacking, wavelets, contrast enhancement), and is intended strictly for educational and research purposes.
