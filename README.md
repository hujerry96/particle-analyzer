# SEM Particle Size Analyzer

SEM image particle / pore analysis tool, supporting two core algorithms: Watershed and Hough Circle.

[English](README.md) | [繁體中文](README.zh.md)

## Features

- **Particle analysis**: Watershed segmentation for overlapping particles, Hough Circle detection for round particles
- **Pore analysis**: Threshold, Watershed, and Hough Circle modes
- **Scale bar calibration**: interactive click-to-calibrate / auto-detect SEM scale bar
- **Dual-axis log distribution plot**: Channel (%) + Passing (%) cumulative curve
- **Export results**: CSV diameter data, statistics summary TXT, overlay image, distribution plot
- **GUI (Tkinter)** and **CLI** operation
- **Bilingual UI**: switch 中文 / English instantly from the `語言 Language` dropdown at the top of the control panel

## Usage

### Graphical interface (recommended)

```bash
python particle_analyzer_gui.py
```

### Command line

```bash
# Watershed analysis (interactive scale calibration)
python particle_analyzer.py sample.tif -m watershed

# Hough Circle analysis (specify scale bar)
python particle_analyzer.py sample.tif -m hough -r 1.0

# Non-interactive batch processing
python particle_analyzer.py *.tif -m watershed --no-interactive -l 1.0

# Specify μm/pixel ratio directly (skip calibration)
python particle_analyzer.py sample.tif -m watershed -p 0.01
```

### Parameters

| Parameter | Description |
|-----------|-------------|
| `-m` | Algorithm: `watershed` (default) or `hough` |
| `-r` | Actual length the scale bar represents (μm) |
| `-p` | Directly specify μm/pixel ratio, skip calibration |
| `--no-interactive` | Non-interactive mode, auto-detect scale bar |
| `--no-plot` | Do not show plots, only save files |
| `-o` | Output directory |
| `--min-dist` | Watershed min center distance (px, default 12) |
| `--min-area` | Watershed min area (px², default 30) |
| `--hough-min-r` | Hough Circle min radius (px, default 5) |
| `--hough-max-r` | Hough Circle max radius (px) |
| `--param1` | Hough Canny high threshold (default 50) |
| `--param2` | Hough accumulator threshold (default 30) |

## Install dependencies

```bash
pip install opencv-python numpy pandas matplotlib scipy scikit-image pillow
```

## Build a standalone executable

```bash
pip install pyinstaller
pyinstaller ParticleAnalyzer.spec
```

Output: `dist/ParticleAnalyzer.exe` (includes all dependencies, ~130 MB).

## Algorithm notes

### Watershed
Best for dense, overlapping particles. CLAHE enhancement → Gaussian blur → Otsu threshold → distance transform → local maxima as seeds → watershed segmentation.

### Hough Circle
Best for particles with clear circular edges. CLAHE enhancement → Gaussian blur → HoughCircles detection.

## Output files

After analysis, the following files are produced:

| File | Description |
|------|-------------|
| `*_diameters.csv` | Particle diameter data (μm) |
| `*_statistics.txt` | Statistics summary (Count, Mean, D50, D10, D90...) |
| `*_overlay.png` | Segmentation overlay + scale bar |
| `*_distribution.png` | Dual-axis log distribution plot |
