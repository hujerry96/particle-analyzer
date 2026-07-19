# SEM 粒徑分析工具 Particle Size Analyzer

SEM 影像顆粒/孔隙分析工具，支援分水嶺（Watershed）與霍夫圓（Hough Circle）兩種演算法。

[English](README.md) | [繁體中文](README.zh.md)

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
