"""
粒徑分析工具 Particle Size Analyzer
=====================================
支援兩種核心演算法：
  1) 分水嶺 Watershed  — 適合密集/重疊顆粒
  2) 霍夫圓 Hough Circle — 適合圓形輪廓清晰之顆粒

使用方式：
  python particle_analyzer.py <image_path> [options]

互動式比例尺校正：
  - 圖片彈出後，在比例尺線段兩端各點一下
  - 輸入該線段代表的實際長度（例如 1 um）
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
from matplotlib.backend_bases import MouseButton
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.segmentation import watershed as sk_watershed
from i18n import t, LANG

warnings.filterwarnings('ignore', category=UserWarning)


def imread_unicode(path):
    """支援 Unicode 路徑的 OpenCV 讀圖（解決中文路徑亂碼問題）。"""
    path = str(path)
    try:
        with open(path, 'rb') as f:
            file_bytes = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    return cv2.imread(path)


def imwrite_unicode(path, img):
    """支援 Unicode 路徑的 OpenCV 寫圖。"""
    path = str(path)
    ext = '.' + path.rsplit('.', 1)[-1] if '.' in path else '.png'
    success, encoded = cv2.imencode(ext, img)
    if success:
        with open(path, 'wb') as f:
            f.write(encoded.tobytes())
        return True
    return False


# ============================================================================
#  比例尺校正
# ============================================================================

class ScaleCalibrator:
    """透過在圖片上點擊兩點來計算 pixel-to-micron 比例。"""

    def __init__(self, image_rgb, known_length_um=None):
        self.image = image_rgb
        self.pts = []
        self.known_length = known_length_um
        self.pixel_to_micron = None
        self.pixel_dist = None
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.imshow(image_rgb)
        self.ax.set_title(t('calib_window_title'))
        self.cid = self.fig.canvas.mpl_connect('button_press_event', self._on_click)

    def _on_click(self, event):
        if event.inaxes != self.ax:
            return
        self.pts.append((event.xdata, event.ydata))
        self.ax.plot(event.xdata, event.ydata, 'ro', markersize=6)
        if len(self.pts) == 2:
            p1, p2 = self.pts
            pixel_dist = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            self.pixel_dist = pixel_dist
            if self.known_length is not None:
                self.pixel_to_micron = self.known_length / pixel_dist
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', lw=2)
            self.ax.set_title(
                t('calib_window_done', dist=pixel_dist, known=self.known_length,
                  ratio=self.pixel_to_micron)
            )
            plt.pause(0.5)
            plt.close(self.fig)

    def get_ratio(self):
        plt.show()
        return self.pixel_to_micron, self.pixel_dist


def auto_detect_scale_bar(gray_img, expected_px_range=(30, 300)):
    """
    嘗試自動偵測 SEM 比例尺（白色水平線段）。
    回傳 (pixel_length, x_center, y_center) 或 None。
    """
    h, w = gray_img.shape
    search_regions = [
        ('bottom_info', gray_img[int(h * 0.88):h-5, :]),
        ('main_area', gray_img[int(h * 0.85):int(h * 0.95), :]),
    ]
    for region_name, region in search_regions:
        _, binary = cv2.threshold(region, 200, 255, cv2.THRESH_BINARY)
        kernel = np.ones((2, 6), np.uint8)
        dilated = cv2.dilate(binary, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, bw, bh = cv2.boundingRect(cnt)
            if bh < 1:
                continue
            aspect = bw / bh
            if 5 < aspect < 500 and expected_px_range[0] < bw < expected_px_range[1]:
                offset_y = int(h * 0.88) if region_name == 'bottom_info' else int(h * 0.85)
                y_global = offset_y + y + bh // 2
                return bw, x + bw // 2, y_global
    return None


# ============================================================================
#  SEM 圖預處理：自動裁切底部資訊欄
# ============================================================================

def auto_crop_sem(image):
    """
    偵測底部深色資訊欄並裁切。回傳 (cropped_image, crop_offset_y)。
    保留上半部影像內容，移除底部黑色資訊列（比例尺在校正時已取得）。
    """
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    row_means = np.mean(gray, axis=1)
    threshold = np.mean(row_means[:int(h*0.5)]) * 0.6
    dark_rows = np.where(row_means < threshold)[0]
    cut_y = h
    if len(dark_rows) > 0:
        first_dark = dark_rows[0]
        if first_dark > h * 0.5:
            cut_y = first_dark
    return image[:cut_y, :].copy()


# ============================================================================
#  分水嶺算法
# ============================================================================

def segment_watershed(gray, min_distance=12, min_area=30, max_area_ratio=0.5):
    """
    分水嶺分割：先距離變換找出局部峰值作為標記，再執行分水嶺。
    - min_distance: 兩個顆粒中心點的最小距離（像素）
    - min_area: 有效顆粒的最小面積（像素）
    - max_area_ratio: 顆粒佔全圖面積的最大比例（過濾大面積異常）
    """
    img_area = gray.shape[0] * gray.shape[1]
    max_area = img_area * max_area_ratio

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    _, otsu_thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(otsu_thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    dist = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    coords = peak_local_max(
        dist, min_distance=min_distance,
        threshold_abs=dist.max() * 0.3,
        labels=opening
    )

    mask = np.zeros(dist.shape, dtype=bool)
    mask[tuple(coords.T)] = True
    markers, _ = ndi.label(mask)

    labels = sk_watershed(-dist, markers, mask=opening)

    props = []
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        label_mask = np.uint8(labels == label_id) * 255
        contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        area_px = cv2.contourArea(cnt)
        if area_px < min_area or area_px > max_area:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        props.append({
            'label': label_id,
            'area_px': area_px,
            'diameter_eq_px': 2 * np.sqrt(area_px / np.pi),
            'contour': cnt,
            'center': (cx, cy),
        })

    return props, dist_norm, labels


# ============================================================================
#  霍夫圓變換
# ============================================================================

def segment_hough_circles(gray, min_radius=5, max_radius=None, param1=50, param2=30):
    """
    霍夫圓變換：適合邊緣明顯的圓形顆粒。
    - min_radius / max_radius: 顆粒半徑範圍（像素）
    - param1: Canny 邊緣檢測的高閾值
    - param2: 累計器閾值，越小越容易偵測到圓（但也容易誤判）
    """
    if max_radius is None:
        max_radius = min(gray.shape) // 4

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_radius * 2,
        param1=param1, param2=param2,
        minRadius=min_radius, maxRadius=max_radius
    )

    props = []
    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        for i, (x, y, r) in enumerate(circles):
            area_px = np.pi * r * r
            contour = cv2.ellipse2Poly((x, y), (r, r), 0, 0, 360, 5)
            props.append({
                'label': i + 1,
                'area_px': area_px,
                'diameter_eq_px': 2 * r,
                'contour': contour,
                'center': (x, y),
            })

    return props, blurred


# ============================================================================
#  孔隙分析
# ============================================================================

def segment_pores_threshold(gray, min_area=30, max_area_ratio=0.5, morph_kernel=3):
    """
    閾值二值化偵測孔隙。
    Otsu 反轉二值化 → 型態學開運算 → 連通元件 → 依面積過濾。
    - min_area: 最小孔隙面積（像素）
    - max_area_ratio: 最大孔隙面積佔影像比例
    - morph_kernel: 型態學開運算核心大小（奇數），用來切斷孔隙間的細微相連
    """
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if morph_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    total_area = gray.shape[0] * gray.shape[1]
    max_area = total_area * max_area_ratio

    props = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < min_area or area > max_area:
            continue
        label_mask = np.uint8(labels == i) * 255
        contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        props.append({
            'label': i,
            'area_px': area,
            'diameter_eq_px': 2 * np.sqrt(area / np.pi),
            'contour': cnt,
            'center': (cx, cy),
        })

    return props, binary


def segment_pores_watershed(gray, min_distance=12, min_area=30, max_area_ratio=0.5, morph_kernel=3):
    """
    反轉分水嶺分割相連孔隙。
    反轉影像 → Otsu 二值化 → 型態學開運算 → 距離變換 → 找局部極大 → 分水嶺。
    - min_distance: 種子點最小間距（像素）
    - min_area: 最小孔隙面積（像素²）
    - max_area_ratio: 最大孔隙面積佔影像比例
    - morph_kernel: 型態學開運算核心大小（奇數），用來切斷孔隙間的細微相連
    """
    inverted = cv2.bitwise_not(gray)
    # Otsu 取得孔隙二值遮罩（暗色孔隙 → 白色前景）
    _, binary = cv2.threshold(inverted, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if morph_kernel > 1:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_kernel, morph_kernel))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    dist = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    dist_norm = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    coords = peak_local_max(dist, min_distance=min_distance, exclude_border=5)
    markers = np.zeros(gray.shape, dtype=np.int32)
    for i, coord in enumerate(coords, 1):
        markers[coord[0], coord[1]] = i
    if len(coords) == 0:
        return [], dist_norm, np.zeros_like(gray)

    labels = sk_watershed(-dist, markers, mask=binary)
    total_area = gray.shape[0] * gray.shape[1]
    max_area = total_area * max_area_ratio

    props = []
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        label_mask = np.uint8(labels == label_id) * 255
        contours, _ = cv2.findContours(label_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        cnt = max(contours, key=cv2.contourArea)
        area_px = cv2.contourArea(cnt)
        if area_px < min_area or area_px > max_area:
            continue
        (cx, cy), r = cv2.minEnclosingCircle(cnt)
        props.append({
            'label': label_id,
            'area_px': area_px,
            'diameter_eq_px': 2 * np.sqrt(area_px / np.pi),
            'contour': cnt,
            'center': (cx, cy),
        })

    return props, dist_norm, labels


def segment_pores_hough(gray, min_radius=5, max_radius=None, param1=50, param2=30):
    """
    霍夫圓偵測圓形孔隙（自動反轉影像）。
    反轉影像 → CLAHE → GaussianBlur → HoughCircles。
    """
    if max_radius is None:
        max_radius = min(gray.shape) // 4

    inverted = cv2.bitwise_not(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(inverted)
    blurred = cv2.GaussianBlur(enhanced, (5, 5), 0)

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_radius * 2,
        param1=param1, param2=param2,
        minRadius=min_radius, maxRadius=max_radius
    )

    props = []
    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        for i, (x, y, r) in enumerate(circles):
            area_px = np.pi * r * r
            contour = cv2.ellipse2Poly((x, y), (r, r), 0, 0, 360, 5)
            props.append({
                'label': i + 1,
                'area_px': area_px,
                'diameter_eq_px': 2 * r,
                'contour': contour,
                'center': (x, y),
            })

    return props, blurred


def compute_porosity(pore_areas_px, total_area_px):
    """計算孔隙率 = (孔隙總面積 / 影像總面積) × 100%。"""
    return (sum(pore_areas_px) / total_area_px) * 100.0


# ============================================================================
#  統計與輸出
# ============================================================================

def compute_statistics(diameters_um):
    """計算粒徑統計數據。"""
    if not diameters_um:
        return {}
    arr = np.array(diameters_um)
    return {
        'count': len(arr),
        'mean_nm': np.mean(arr) * 1000,
        'median_nm': np.median(arr) * 1000,
        'std_nm': np.std(arr) * 1000,
        'min_nm': np.min(arr) * 1000,
        'max_nm': np.max(arr) * 1000,
        'd10_nm': np.percentile(arr, 10) * 1000,
        'd90_nm': np.percentile(arr, 90) * 1000,
        'mean_um': np.mean(arr),
        'median_um': np.median(arr),
        'std_um': np.std(arr),
        'd10_um': np.percentile(arr, 10),
        'd90_um': np.percentile(arr, 90),
    }


def export_csv(diameters_um, output_path):
    """輸出粒徑數據到 CSV。"""
    df = pd.DataFrame({'Diameter_um': diameters_um})
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    return df


def export_statistics_txt(stats, output_path):
    """輸出統計摘要到 TXT。"""
    lines = [
        "=" * 45,
        "  粒徑分析統計結果 Particle Size Statistics",
        "=" * 45,
        f"  顆粒總數 (Count):            {stats['count']}",
        f"  平均粒徑 (Mean):              {stats['mean_nm']:.1f} nm  ({stats['mean_um']:.3f} μm)",
        f"  中位數 D50 (Median):          {stats['median_nm']:.1f} nm  ({stats['median_um']:.3f} μm)",
        f"  標準差 (Std):                 {stats['std_nm']:.1f} nm  ({stats['std_um']:.3f} μm)",
        f"  最小值 (Min):                 {stats['min_nm']:.1f} nm  ({stats['min_um']:.3f} μm)",
        f"  最大值 (Max):                 {stats['max_nm']:.1f} nm  ({stats['max_um']:.3f} μm)",
        f"  D10 (10% 小於此值):           {stats['d10_nm']:.1f} nm  ({stats['d10_um']:.3f} μm)",
        f"  D90 (90% 小於此值):           {stats['d90_nm']:.1f} nm  ({stats['d90_um']:.3f} μm)",
        "=" * 45,
    ]
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    return lines


# ============================================================================
#  視覺化
# ============================================================================

def plot_results(image_rgb, props, diameters_um, stats, method_name, pixel_to_micron,
                 save_path=None, show_plot=True, scale_bar_um=None):
    """繪製分割結果與粒徑分布圖。"""
    result_img = image_rgb.copy()
    for p in props:
        cv2.drawContours(result_img, [p['contour']], -1, (0, 255, 0), 2)
        d_px = p['diameter_eq_px']
        d_um = d_px * pixel_to_micron
        d_nm = d_um * 1000
        label_text = f"{d_nm:.0f}"
        cx, cy = int(p['center'][0]), int(p['center'][1])
        cv2.putText(result_img, label_text, (cx - 15, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    if scale_bar_um is not None:
        result_bgr = cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        draw_scale_bar(result_bgr, scale_bar_um, pixel_to_micron)
        result_img = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    axes[0].imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"{method_name} — 偵測顆粒數: {stats['count']}", fontsize=13)
    axes[0].axis('off')

    if diameters_um:
        arr = np.array(diameters_um)
        vmin = max(arr.min() * 0.5, arr.min() * 0.8)
        vmax = arr.max() * 1.5
        bins = np.logspace(np.log10(vmin), np.log10(vmax), 35)
        bin_centers = np.sqrt(bins[:-1] * bins[1:])
        counts, _ = np.histogram(arr, bins=bins)
        total = counts.sum()
        channel_pct = counts / total * 100
        passing_pct = np.cumsum(channel_pct)

        ax1 = axes[1]
        bar_width = np.diff(np.log10(bins))
        ax1.bar(bin_centers, channel_pct, width=bins[:-1] * (10**bar_width - 1),
                color='#5B9BD5', edgecolor='white', alpha=0.85, align='center',
                label='Channel (%)', zorder=2)
        ax1.set_xscale('log')
        ax1.set_xlabel('Particle size (μm)', fontsize=12)
        ax1.set_ylabel('Channel (%)', fontsize=12, color='#5B9BD5')
        ax1.tick_params(axis='y', labelcolor='#5B9BD5')
        ax1.set_ylim(0, max(channel_pct) * 1.25 if max(channel_pct) > 0 else 100)

        ax2 = ax1.twinx()
        ax2.plot(bin_centers, passing_pct, color='#003366', lw=2.5,
                 label='Passing (%)', zorder=3)
        ax2.set_ylabel('Passing (%)', fontsize=12, color='#003366')
        ax2.tick_params(axis='y', labelcolor='#003366')
        ax2.set_ylim(-3, 103)

        ax1.grid(True, which='both', alpha=0.25, linestyle='--', zorder=0)
        ax1.set_axisbelow(False)

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
                   fontsize=10, framealpha=0.85)
    else:
        axes[1].text(0.5, 0.5, "No particles detected", ha='center', va='center', fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  圖表已儲存: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)


def draw_scale_bar(image, pixel_length_um, pixel_to_micron, position='br'):
    """
    在影像右下角繪製比例尺。
    image: BGR numpy array (modified in-place)
    pixel_length_um: 比例尺代表的實際長度 (μm)
    pixel_to_micron: 換算比例
    position: 'br' 右下角
    """
    from PIL import Image, ImageDraw, ImageFont
    h, w = image.shape[:2]
    bar_um = _nice_scale_value(pixel_length_um)
    bar_px = int(bar_um / pixel_to_micron)
    margin = int(max(20, h * 0.03))
    thickness = max(3, h // 250)

    x1 = w - margin - bar_px
    x2 = w - margin
    y = h - margin

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(pil_img)

    mean_val = np.mean(image[max(0,y-10):min(h,y+10), max(0,x1-10):min(w,x1+10)])
    color = (255, 255, 255) if mean_val < 128 else (0, 0, 0)

    draw.line([(x1, y), (x2, y)], fill=color, width=thickness)
    draw.line([(x1, y - 6), (x1, y + 6)], fill=color, width=thickness)
    draw.line([(x2, y - 6), (x2, y + 6)], fill=color, width=thickness)

    label = f"{bar_um:.0f} \u03bcm" if bar_um >= 1 else f"{bar_um*1000:.0f} nm"
    font_size = max(18, h // 35)
    try:
        font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", font_size)
    except Exception:
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    tx = x1 + (bar_px - tw) // 2
    ty = y - font_size - 6
    draw.text((tx, ty), label, fill=color, font=font)

    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    image[:] = result


def _nice_scale_value(target_um):
    """取一個適合顯示的整數比例尺值（1, 2, 5, 10, 20, 50...）。"""
    if target_um <= 0:
        return 1.0
    magnitude = 10 ** np.floor(np.log10(target_um))
    residual = target_um / magnitude
    if residual < 1.5:
        return magnitude
    elif residual < 3.5:
        return 2 * magnitude
    elif residual < 7.5:
        return 5 * magnitude
    else:
        return 10 * magnitude


def plot_particle_size_distribution(diameters_um, stats, save_path=None, show_plot=True, figsize=(7, 7), unit='um', title=None):
    """
    繪製藍色雙軸對數粒徑分布圖（1:1 比例）。
    - X 軸：對數刻度 (μm 或 nm)
    - 左 Y 軸：Channel (%) — 藍色長條圖
    - 右 Y 軸：Passing (%) — 深藍色累積曲線（無數據點）
    """
    scale = 1000.0 if unit == 'nm' else 1.0
    ulb = unit
    arr = np.array(diameters_um) * scale
    if len(arr) == 0:
        return None

    vmin = max(arr.min() * 0.5, arr.min() * 0.8)
    vmax = arr.max() * 1.5
    bins = np.logspace(np.log10(vmin), np.log10(vmax), 35)
    bin_centers = np.sqrt(bins[:-1] * bins[1:])

    counts, _ = np.histogram(arr, bins=bins)
    total = counts.sum()
    channel_pct = counts / total * 100
    passing_pct = np.cumsum(channel_pct)

    fig, ax1 = plt.subplots(figsize=figsize)

    bar_width = np.diff(np.log10(bins))
    ax1.bar(bin_centers, channel_pct, width=bins[:-1] * (10**bar_width - 1),
            color='#5B9BD5', edgecolor='white', alpha=0.85, align='center',
            label='Channel (%)', zorder=2)
    ax1.set_xscale('log')
    ax1.set_xlabel(f'Particle size ({ulb})', fontsize=12)
    ax1.set_ylabel('Channel (%)', fontsize=12, color='#5B9BD5')
    ax1.tick_params(axis='both', which='both', direction='in', pad=6)
    ax1.tick_params(axis='y', labelcolor='#5B9BD5')
    ax1.set_ylim(0, max(channel_pct) * 1.25 if max(channel_pct) > 0 else 100)

    # Ensure x-axis limits don't cut off tick labels
    ax1.set_xlim(bins[0] * 0.8, bins[-1] * 1.25)

    ax2 = ax1.twinx()
    ax2.plot(bin_centers, passing_pct, color='#003366', lw=2.5,
             label='Passing (%)', zorder=3)
    ax2.set_ylabel('Passing (%)', fontsize=12, color='#003366')
    ax2.tick_params(axis='y', which='both', direction='in', labelcolor='#003366')
    ax2.set_ylim(-3, 103)

    ax1.grid(True, which='both', alpha=0.25, linestyle='--', zorder=0)
    ax1.set_axisbelow(False)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
               fontsize=10, framealpha=0.85)

    # Add more bottom margin to prevent X-axis labels from being covered by bars
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.97)
        plt.tight_layout(rect=[0, 0.12, 1, 0.93])
    else:
        plt.tight_layout(rect=[0, 0.12, 1, 1])

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"  分布圖已儲存: {save_path}")

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return fig


def print_stats_table(stats):
    """在終端機印出統計表格。"""
    print()
    print("=" * 55)
    print("  粒徑分析統計結果 Particle Size Statistics")
    print("=" * 55)
    print(f"  顆粒總數 (Count)              : {stats['count']}")
    print(f"  平均粒徑 (Mean)               : {stats['mean_nm']:>8.1f} nm   ({stats['mean_um']:.3f} μm)")
    print(f"  中位數 D50 (Median)           : {stats['median_nm']:>8.1f} nm   ({stats['median_um']:.3f} μm)")
    print(f"  標準差 (Std)                  : {stats['std_nm']:>8.1f} nm   ({stats['std_um']:.3f} μm)")
    print(f"  最小值 (Min)                  : {stats['min_nm']:>8.1f} nm   ({stats['min_um']:.3f} μm)")
    print(f"  最大值 (Max)                  : {stats['max_nm']:>8.1f} nm   ({stats['max_um']:.3f} μm)")
    print(f"  D10 (10% 小於此值)            : {stats['d10_nm']:>8.1f} nm   ({stats['d10_um']:.3f} μm)")
    print(f"  D90 (90% 小於此值)            : {stats['d90_nm']:>8.1f} nm   ({stats['d90_um']:.3f} μm)")
    print("=" * 55)


# ============================================================================
#  主程式
# ============================================================================

def analyze_one_image(
    image_path,
    method='watershed',
    known_length_um=None,
    interactive_scale=True,
    min_distance=12,
    min_area=30,
    hough_min_r=5,
    hough_max_r=None,
    hough_param1=50,
    hough_param2=30,
    output_dir=None,
    show_plot=True,
):
    """
    對單張 SEM 影像執行粒徑分析。

    參數
    ----
    image_path : str
    method : 'watershed' | 'hough'
    known_length_um : float, optional — 比例尺代表的實際長度 (μm)
    interactive_scale : bool — 是否互動式校正比例尺
    min_distance : int — 分水嶺最小顆粒中心距 (px)
    min_area : int — 分水嶺最小顆粒面積 (px)
    hough_min_r / hough_max_r : int — 霍夫圓半徑範圍 (px)
    hough_param1 / hough_param2 : int — HoughCircles 參數
    output_dir : str, optional — 輸出目錄
    """
    img_path = Path(image_path)
    if not img_path.exists():
        print(f"錯誤: 找不到圖片 {image_path}")
        return None

    img = imread_unicode(str(img_path))
    if img is None:
        print(f"錯誤: 無法讀取圖片 {image_path}")
        return None

    print(f"\n讀取圖片: {image_path}  ({img.shape[1]}x{img.shape[0]})")

    img_full_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ----- 比例尺校正（在完整影像上進行） -----
    if known_length_um is None:
        known_length_um = 1.0

    scale_bar_px = None
    pixel_to_micron = None
    if interactive_scale:
        print("=> 請在彈出的視窗中點擊比例尺的兩個端點（在圖片下方的資訊欄內）...")
        cal = ScaleCalibrator(img_full_rgb, known_length_um=known_length_um)
        pixel_to_micron, px_dist = cal.get_ratio()
        scale_bar_px = px_dist
        if pixel_to_micron is None:
            gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            auto_result = auto_detect_scale_bar(gray_full)
            if auto_result is not None:
                px_len, cx, cy = auto_result
                scale_bar_px = px_len
                pixel_to_micron = known_length_um / px_len
                print(f"  (自動偵測) 比例尺長度 = {px_len} px, 比例 = {pixel_to_micron:.6f} μm/px")
            else:
                print("  未進行比例尺校正，使用預設比例 (1 px = 0.01 μm)")
                pixel_to_micron = 0.01
    else:
        gray_full = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        auto_result = auto_detect_scale_bar(gray_full)
        if auto_result is not None:
            px_len, cx, cy = auto_result
            scale_bar_px = px_len
            pixel_to_micron = known_length_um / px_len
            print(f"  (自動偵測) 比例尺長度 = {px_len} px, 比例 = {pixel_to_micron:.6f} μm/px")
        else:
            print("  警告: 無法自動偵測比例尺，使用預設比例 (1 px = 0.01 μm/px)")
            pixel_to_micron = 0.01

    # ----- 裁切底部資訊欄（校正後再切） -----
    img_cropped = auto_crop_sem(img)
    gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
    print(f"  裁切後尺寸: {img_cropped.shape[1]}x{img_cropped.shape[0]}")

    if scale_bar_px is None and pixel_to_micron > 0:
        test_length = _nice_scale_value(1.0)
        scale_bar_px = int(test_length / pixel_to_micron)
        if scale_bar_px > gray.shape[1] * 0.4:
            test_length = _nice_scale_value(0.5)
            scale_bar_px = int(test_length / pixel_to_micron)

    print(f"  比例係數: {pixel_to_micron:.6f} μm/pixel  "
          f"(= {pixel_to_micron*1000:.4f} nm/pixel)")

    # ----- 分割 -----
    if method == 'watershed':
        print("=> 執行分水嶺演算法...")
        props, dist_img, labels = segment_watershed(
            gray, min_distance=min_distance, min_area=min_area
        )
    elif method == 'hough':
        print("=> 執行霍夫圓變換...")
        props, _ = segment_hough_circles(
            gray, min_radius=hough_min_r, max_radius=hough_max_r,
            param1=hough_param1, param2=hough_param2
        )
    else:
        print(f"錯誤: 不支援的演算法 '{method}'")
        return None

    if not props:
        print("  未偵測到任何顆粒。請調整參數。")
        return None

    diameters_um = [p['diameter_eq_px'] * pixel_to_micron for p in props]
    stats = compute_statistics(diameters_um)
    stats['min_um'] = stats['min_nm'] / 1000
    stats['max_um'] = stats['max_nm'] / 1000
    stats['d10_um'] = stats['d10_nm'] / 1000
    stats['d90_um'] = stats['d90_nm'] / 1000

    print_stats_table(stats)

    # ----- 輸出 -----
    if output_dir:
        out_dir = Path(output_dir)
    else:
        out_dir = img_path.parent / f"{img_path.stem}_results"
    out_dir.mkdir(parents=True, exist_ok=True)

    base = img_path.stem
    method_tag = 'watershed' if method == 'watershed' else 'hough'

    csv_path = out_dir / f"{base}_{method_tag}_diameters.csv"
    txt_path = out_dir / f"{base}_{method_tag}_statistics.txt"
    plot_path = out_dir / f"{base}_{method_tag}_result.png"
    overlay_path = out_dir / f"{base}_{method_tag}_overlay.png"
    dist_path = out_dir / f"{base}_{method_tag}_distribution.png"

    export_csv(diameters_um, csv_path)
    export_statistics_txt(stats, txt_path)

    overlay_bgr = img_cropped.copy()
    for p in props:
        cv2.drawContours(overlay_bgr, [p['contour']], -1, (0, 255, 0), 2)
    bar_um = known_length_um if scale_bar_px is None else None
    if scale_bar_px is not None:
        draw_scale_bar(overlay_bgr, scale_bar_px * pixel_to_micron, pixel_to_micron)
    imwrite_unicode(str(overlay_path), overlay_bgr)

    print(f"\n  輸出資料夾: {out_dir}")
    print(f"  CSV 粒徑數據: {csv_path}")
    print(f"  TXT 統計摘要: {txt_path}")
    print(f"  分割疊加圖: {overlay_path.name}")

    print("=> 繪製分布圖表...")
    scale_bar_um = scale_bar_px * pixel_to_micron if scale_bar_px else None
    plot_results(
        img_rgb, props, diameters_um, stats,
        f"Watershed" if method == 'watershed' else "Hough Circles",
        pixel_to_micron, save_path=plot_path, show_plot=show_plot,
        scale_bar_um=scale_bar_um
    )

    plot_particle_size_distribution(
        diameters_um, stats, save_path=str(dist_path), show_plot=False
    )

    return stats


def batch_analyze(
    image_paths,
    method='watershed',
    known_length_um=None,
    interactive_scale=False,
    **kwargs,
):
    """批次分析多張圖片（非互動模式）。"""
    all_stats = []
    for i, img_path in enumerate(image_paths):
        print(f"\n{'='*55}")
        print(f"[{i+1}/{len(image_paths)}] {img_path}")
        print(f"{'='*55}")
        stats = analyze_one_image(
            img_path, method=method,
            known_length_um=known_length_um,
            interactive_scale=interactive_scale,
            **kwargs,
        )
        if stats:
            all_stats.append({'image': str(img_path), **stats})
    return all_stats


# ============================================================================
#  命令列入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='粒徑分析工具 — SEM 影像 Watershed / Hough Circle',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 分水嶺分析（互動比例尺校正）
  python particle_analyzer.py np-3x20000.tif -m watershed

  # 霍夫圓分析（指定比例尺 1 μm = 160 px）
  python particle_analyzer.py np-3x20000.tif -m hough -r 160

  # 非互動批次處理
  python particle_analyzer.py *.tif -m watershed --no-interactive -l 1.0
        """
    )
    parser.add_argument('images', nargs='+', help='SEM 圖片路徑（支援多張）')
    parser.add_argument('-m', '--method', choices=['watershed', 'hough'],
                        default='watershed', help='分割演算法 (預設: watershed)')
    parser.add_argument('-r', '--scale-ratio', type=float, default=None,
                        help='比例尺對應的實際長度 (μm), 如 1.0')
    parser.add_argument('-p', '--pixel-ratio', type=float, default=None,
                        help='直接指定 μm/pixel 比例 (跳過校正)')
    parser.add_argument('--no-interactive', action='store_true',
                        help='非互動模式：自動偵測比例尺')
    parser.add_argument('--no-plot', action='store_true',
                        help='不顯示圖表視窗（僅儲存檔案）')
    parser.add_argument('-o', '--output-dir', default=None,
                        help='輸出目錄 (預設: 圖片所在資料夾/_results)')

    # Watershed 參數
    parser.add_argument('--min-dist', type=int, default=12,
                        help='分水嶺最小中心距 (px, 預設: 12)')
    parser.add_argument('--min-area', type=int, default=30,
                        help='分水嶺最小面積 (px, 預設: 30)')

    # Hough 參數
    parser.add_argument('--hough-min-r', type=int, default=5,
                        help='霍夫圓最小半徑 (px, 預設: 5)')
    parser.add_argument('--hough-max-r', type=int, default=None,
                        help='霍夫圓最大半徑 (px, 預設: 圖長邊/4)')
    parser.add_argument('--param1', type=int, default=50,
                        help='Hough param1 (Canny 高閾值, 預設: 50)')
    parser.add_argument('--param2', type=int, default=30,
                        help='Hough param2 (累計器閾值, 預設: 30)')

    args = parser.parse_args()

    if args.pixel_ratio is not None:
        # 直接使用給定的比例，不進行互動校正
        print(f"使用指定比例: {args.pixel_ratio} μm/px")
        _analyze_with_fixed_ratio(args, args.pixel_ratio)
    else:
        interactive = not args.no_interactive
        kwargs = dict(
            method=args.method,
            known_length_um=args.scale_ratio,
            interactive_scale=interactive,
            min_distance=args.min_dist,
            min_area=args.min_area,
            hough_min_r=args.hough_min_r,
            hough_max_r=args.hough_max_r,
            hough_param1=args.param1,
            hough_param2=args.param2,
            output_dir=args.output_dir,
            show_plot=not args.no_plot,
        )
        if len(args.images) == 1:
            analyze_one_image(args.images[0], **kwargs)
        else:
            kwargs['interactive_scale'] = False
            batch_analyze(args.images, **kwargs)


def _analyze_with_fixed_ratio(args, pixel_to_micron):
    """使用固定比例分析（跳過校正）。"""
    for img_path in args.images:
        img = cv2.imread(img_path)
        if img is None:
            continue
        img_cropped = auto_crop_sem(img)
        gray = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)

        if args.method == 'watershed':
            props, _, _ = segment_watershed(gray, min_distance=args.min_dist, min_area=args.min_area)
        else:
            props, _ = segment_hough_circles(
                gray, min_radius=args.hough_min_r, max_radius=args.hough_max_r,
                param1=args.param1, param2=args.param2
            )

        diameters_um = [p['diameter_eq_px'] * pixel_to_micron for p in props]
        stats = compute_statistics(diameters_um)
        stats['min_um'] = stats['min_nm'] / 1000
        stats['max_um'] = stats['max_nm'] / 1000
        stats['d10_um'] = stats['d10_nm'] / 1000
        stats['d90_um'] = stats['d90_nm'] / 1000

        print_stats_table(stats)
        plot_results(img_rgb, props, diameters_um, stats,
                     "Watershed" if args.method == 'watershed' else "Hough Circles",
                     pixel_to_micron, show_plot=not args.no_plot,
                     scale_bar_um=None)
        plot_particle_size_distribution(
            diameters_um, stats, show_plot=not args.no_plot
        )


if __name__ == '__main__':
    main()
