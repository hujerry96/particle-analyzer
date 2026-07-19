<input type="radio" id="tab-zh" name="lang" checked>
<input type="radio" id="tab-en" name="lang">

<div style="border-bottom:1px solid #ddd;margin-bottom:16px">
  <label for="tab-zh" style="display:inline-block;padding:8px 16px;cursor:pointer;border:1px solid #ddd;border-bottom:none;border-radius:6px 6px 0 0;margin-bottom:-1px;background:#f6f8fa;font-weight:bold">中文</label>
  <label for="tab-en" style="display:inline-block;padding:8px 16px;cursor:pointer;border:1px solid #ddd;border-bottom:none;border-radius:6px 6px 0 0;margin-bottom:-1px;background:#f6f8fa;font-weight:bold">English</label>
</div>

<!-- ===================== 中文 ===================== -->
<div class="lang-zh">

# SEM 粒徑分析工具 Particle Size Analyzer

SEM 影像顆粒/孔隙分析工具，支援分水嶺（Watershed）與霍夫圓（Hough Circle）兩種演算法。

## 功能

- **顆粒分析**：分水嶺分割重疊顆粒、霍夫圓偵測圓形顆粒
- **孔隙分析**：二值化閾值、分水嶺、霍夫圓三種模式
- **比例尺校正**：互動式點擊校正 / 自動偵測 SEM 比例尺
- **雙軸對數分布圖**：Channel (%) + Passing (%) 累積曲線
- **匯出結果**：CSV 粒徑數據、統計摘要 TXT、疊加圖、分布圖
- **GUI 圖形介面**（Tkinter）與 CLI 命令列兩種操作方式
- **雙語介面**：控制面板頂部 `語言 Language` 下拉可即時切換中文 / English

## 執行方式

### 圖形介面（建議）

```bash
python particle_analyzer_gui.py
```

### 命令列

```bash
# 分水嶺分析（互動比例尺校正）
python particle_analyzer.py sample.tif -m watershed

# 霍夫圓分析（指定比例尺）
python particle_analyzer.py sample.tif -m hough -r 1.0

# 非互動批次處理
python particle_analyzer.py *.tif -m watershed --no-interactive -l 1.0

# 指定 μm/pixel 比例（跳過校正）
python particle_analyzer.py sample.tif -m watershed -p 0.01
```

### 參數說明

| 參數 | 說明 |
|------|------|
| `-m` | 演算法：`watershed`（預設）或 `hough` |
| `-r` | 比例尺對應的實際長度（μm） |
| `-p` | 直接指定 μm/pixel 比例，跳過校正 |
| `--no-interactive` | 非互動模式，自動偵測比例尺 |
| `--no-plot` | 不顯示圖表，僅儲存檔案 |
| `-o` | 輸出目錄 |
| `--min-dist` | 分水嶺最小中心距（px，預設 12） |
| `--min-area` | 分水嶺最小面積（px²，預設 30） |
| `--hough-min-r` | 霍夫圓最小半徑（px，預設 5） |
| `--hough-max-r` | 霍夫圓最大半徑（px） |
| `--param1` | Hough Canny 高閾值（預設 50） |
| `--param2` | Hough 累計器閾值（預設 30） |

## 安裝相依套件

```bash
pip install opencv-python numpy pandas matplotlib scipy scikit-image pillow
```

## 打包成單一執行檔

```bash
pip install pyinstaller
pyinstaller ParticleAnalyzer.spec
```

產出在 `dist/ParticleAnalyzer.exe`（含所有相依，約 130 MB）。

## 演算法說明

### 分水嶺 Watershed
適合密集、重疊的顆粒。對灰階影像做 CLAHE 增強 → Gaussian 模糊 → Otsu 二值化 → 距離變換 → 找區域極大作為種子 → 分水嶺分割。

### 霍夫圓 Hough Circle
適合邊緣明顯的圓形顆粒。CLAHE 增強 → Gaussian 模糊 → HoughCircles 偵測。

## 產出檔案

分析完成後會產生以下檔案：

| 檔案 | 說明 |
|------|------|
| `*_diameters.csv` | 粒徑數據（μm） |
| `*_statistics.txt` | 統計摘要（Count, Mean, D50, D10, D90...） |
| `*_overlay.png` | 分割疊加圖 + 比例尺 |
| `*_distribution.png` | 雙軸對數分布圖 |

</div>

<!-- ===================== English ===================== -->
<div class="lang-en">

# SEM Particle Size Analyzer

SEM image particle / pore analysis tool, supporting two core algorithms: Watershed and Hough Circle.

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

</div>

<style>
#tab-zh:checked ~ .lang-zh { display: block; }
#tab-zh:checked ~ .lang-en { display: none; }
#tab-en:checked ~ .lang-zh { display: none; }
#tab-en:checked ~ .lang-en { display: block; }
</style>
