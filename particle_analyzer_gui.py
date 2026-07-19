"""
粒徑分析工具 — GUI 版
======================
基於分水嶺 (Watershed) 與霍夫圓 (Hough Circle) 的 SEM 粒徑分析圖形介面。

執行方式：
  python particle_analyzer_gui.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure

# 匯入核心分析函數
from particle_analyzer import (
    auto_crop_sem, auto_detect_scale_bar,
    segment_watershed, segment_hough_circles,
    segment_pores_threshold, segment_pores_watershed, segment_pores_hough,
    compute_statistics, compute_porosity,
    draw_scale_bar, _nice_scale_value,
    ScaleCalibrator, imread_unicode, imwrite_unicode, plot_particle_size_distribution
)
from i18n import t, set_lang
import i18n


class ToolTip:
    """懸浮提示視窗。"""
    def __init__(self, widget, text, delay=400):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tipwindow = None
        self.after_id = None
        widget.bind('<Enter>', self._enter)
        widget.bind('<Leave>', self._leave)

    def _enter(self, event):
        self._schedule()

    def _leave(self, event):
        self._unschedule()
        self._hide()

    def _schedule(self):
        self._unschedule()
        self.after_id = self.widget.after(self.delay, self._show)

    def _unschedule(self):
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None

    def _show(self):
        if self.tipwindow:
            return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.tipwindow = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f'+{x}+{y}')
        lbl = tk.Label(tw, text=self.text, justify=tk.LEFT,
                       background='#FFFFE0', relief=tk.SOLID, borderwidth=1,
                       font=('Consolas', 9), padx=6, pady=3)
        lbl.pack()

    def _hide(self):
        if self.tipwindow:
            self.tipwindow.destroy()
            self.tipwindow = None


class ParticleAnalyzerGUI:
    def __init__(self, master):
        self.master = master
        master.title(t('app_title'))
        master.geometry("1400x800")

        # 狀態變數
        self.image_path = None
        self.img_original = None          # BGR 原始
        self.img_rgb_full = None          # RGB 完整（含資訊欄）
        self.img_cropped_bgr = None       # BGR 裁切後
        self.img_cropped_rgb = None       # RGB 裁切後
        self.gray = None                  # 裁切後灰階
        self.scale_bar_px = None
        self.pixel_to_micron = None
        self.props = None
        self.diameters_um = None
        self.stats = None
        self.known_length_um = 1.0
        self._param_keys = {}

        self._build_ui()
        self._apply_language()

    # ========================================================================
    #  UI 建構
    # ========================================================================

    def _build_ui(self):
        main_paned = ttk.PanedWindow(self.master, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

        # 左側 — 圖片顯示區
        left_frame = ttk.Frame(main_paned, padding=(6, 4, 6, 4))
        main_paned.add(left_frame, weight=3)

        left_paned = ttk.PanedWindow(left_frame, orient=tk.VERTICAL)
        left_paned.pack(fill=tk.BOTH, expand=True)

        # 上半部：分割疊加圖
        top_frame = ttk.Frame(left_paned, padding=(4, 4, 4, 2))
        left_paned.add(top_frame, weight=2)

        self.fig = Figure(figsize=(8, 5), dpi=100)
        self.fig.subplots_adjust(left=0, right=1, bottom=0.01, top=0.93)
        self.ax_img = self.fig.add_subplot(111)
        self.ax_img.axis('off')
        self.canvas = FigureCanvasTkAgg(self.fig, master=top_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar_frame = ttk.Frame(top_frame)
        toolbar_frame.pack(fill=tk.X)
        NavigationToolbar2Tk(self.canvas, toolbar_frame)

        # 下半部：分布圖
        bottom_frame = ttk.Frame(left_paned, padding=(4, 2, 4, 4))
        left_paned.add(bottom_frame, weight=1)

        self.fig_dist = Figure(figsize=(8, 4), dpi=100)
        self.ax_dist = self.fig_dist.add_subplot(111)
        self.ax_dist.set_xlabel(t('dist_xlabel', u='μm'))
        self.ax_dist.set_ylabel(t('dist_ylabel'))
        self.ax_dist.set_xscale('log')
        self.ax_dist.set_title(t('dist_title'), fontsize=11)
        self.fig_dist.subplots_adjust(left=0.15, right=0.83, top=0.92, bottom=0.18)
        self.canvas_dist = FigureCanvasTkAgg(self.fig_dist, master=bottom_frame)
        self.canvas_dist.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 右側 — 控制面板
        right_frame = ttk.Frame(main_paned, width=380)
        main_paned.add(right_frame, weight=1)
        right_frame.pack_propagate(False)

        # --- 語言切換 ---
        lang_frame = ttk.LabelFrame(right_frame, text=t('lang'), padding=6)
        lang_frame.pack(fill=tk.X, padx=8, pady=(5, 2))
        self.lang_var = tk.StringVar(value='zh' if i18n.LANG == 'zh' else 'en')
        self.lang_cb = ttk.Combobox(lang_frame, textvariable=self.lang_var,
                                    values=[t('lang_zh'), t('lang_en')],
                                    state='readonly', width=20)
        self.lang_cb.pack(fill=tk.X)
        self.lang_cb.bind('<<ComboboxSelected>>', self._on_lang_change)
        self._lang_frame = lang_frame

        # --- 按鈕區 ---
        btn_frame = ttk.LabelFrame(right_frame, text=t('op'), padding=10)
        btn_frame.pack(fill=tk.X, padx=8, pady=5)
        self._btn_load = ttk.Button(btn_frame, text=t('load_image'), command=self._load_image)
        self._btn_load.pack(fill=tk.X, pady=2)
        self.btn_calibrate = ttk.Button(btn_frame, text=t('calibrate'), command=self._calibrate_scale, state=tk.DISABLED)
        self.btn_calibrate.pack(fill=tk.X, pady=2)
        self._btn_run = ttk.Button(btn_frame, text=t('run'), command=self._run_analysis)
        self._btn_run.pack(fill=tk.X, pady=5)
        self._btn_export = ttk.Button(btn_frame, text=t('export'), command=self._export_results)
        self._btn_export.pack(fill=tk.X, pady=2)
        self._btn_frame = btn_frame

        # --- 比例尺資訊 ---
        self.scale_frame = ttk.LabelFrame(right_frame, text=t('scale'), padding=8)
        self.scale_frame.pack(fill=tk.X, padx=8, pady=5)
        row = ttk.Frame(self.scale_frame)
        row.pack(fill=tk.X)
        self._lbl_scale_len = ttk.Label(row, text=t('scale_len_um'))
        self._lbl_scale_len.pack(side=tk.LEFT)
        self.var_scale_len = tk.DoubleVar(value=1.0)
        ttk.Entry(row, textvariable=self.var_scale_len, width=8).pack(side=tk.RIGHT)
        self.lbl_scale = ttk.Label(self.scale_frame, text=t('scale_not_calibrated'))
        self.lbl_scale.pack(anchor=tk.W, pady=(4, 0))

        # --- 模式選擇 ---
        mode_frame = ttk.LabelFrame(right_frame, text=t('mode'), padding=8)
        mode_frame.pack(fill=tk.X, padx=8, pady=5)
        self.mode_var = tk.StringVar(value=t('mode_particle'))
        mode_cb = ttk.Combobox(mode_frame, textvariable=self.mode_var,
                               state='readonly', width=20)
        mode_cb.pack(fill=tk.X)
        mode_cb.bind('<<ComboboxSelected>>', self._on_mode_change)
        self._mode_cb = mode_cb
        self._mode_frame = mode_frame

        # --- 演算法選擇 ---
        alg_frame = ttk.LabelFrame(right_frame, text=t('alg'), padding=8)
        alg_frame.pack(fill=tk.X, padx=8, pady=5)
        self.alg_var = tk.StringVar(value='watershed')
        self.alg_cb = ttk.Combobox(alg_frame, textvariable=self.alg_var,
                                   state='readonly', width=22)
        self.alg_cb.pack(fill=tk.X)
        self.alg_cb.bind('<<ComboboxSelected>>', self._on_alg_change)
        self._alg_tooltip = ToolTip(self.alg_cb, '')
        self._alg_frame = alg_frame

        # --- 通用參數（所有演算法共用閾值參數框架）---
        self.ws_frame = ttk.LabelFrame(right_frame, text=t('ws_params'), padding=8)
        self._add_param(self.ws_frame, 'min_dist', t('min_dist'), 12, 3, 50)
        self._add_param(self.ws_frame, 'min_area', t('min_area'), 30, 10, 500)
        self._add_param(self.ws_frame, 'morph_kernel', t('morph_kernel'), 3, 1, 15)
        self.ws_frame.pack_forget()

        self.hg_frame = ttk.LabelFrame(right_frame, text=t('hg_params'), padding=8)
        self._add_param(self.hg_frame, 'hough_min_r', t('hough_min_r'), 5, 2, 50)
        self._add_param(self.hg_frame, 'hough_max_r', t('hough_max_r'), 80, 10, 300)
        self._add_param(self.hg_frame, 'param1', t('param1'), 50, 10, 200)
        self._add_param(self.hg_frame, 'param2', t('param2'), 30, 5, 150)
        self.hg_frame.pack_forget()

        self.th_frame = ttk.LabelFrame(right_frame, text=t('th_params'), padding=8)
        self._add_param(self.th_frame, 'th_min_area', t('th_min_area'), 30, 10, 500)
        self._add_param(self.th_frame, 'th_max_ratio', t('th_max_ratio'), 50, 5, 99)
        self._add_param(self.th_frame, 'th_morph', t('th_morph'), 3, 1, 15)
        self.th_frame.pack_forget()

        # --- 單位切換 ---
        unit_frame = ttk.LabelFrame(right_frame, text=t('display_unit'), padding=8)
        unit_frame.pack(fill=tk.X, padx=8, pady=5)
        self.unit_var = tk.StringVar(value='nm')
        ttk.Radiobutton(unit_frame, text='nm', variable=self.unit_var,
                        value='nm').pack(side=tk.LEFT, padx=(0, 15))
        ttk.Radiobutton(unit_frame, text='μm', variable=self.unit_var,
                        value='um', command=self._on_unit_change).pack(side=tk.LEFT)
        # 'um' 按鈕觸發更新（nm 切換相同）
        # 綁定兩個按鈕，用 trace 監聽變數變化
        self.unit_var.trace_add('write', self._on_unit_change)
        self._unit_frame = unit_frame

        # --- 統計結果 ---
        self.result_frame = ttk.LabelFrame(right_frame, text=t('result'), padding=8)
        self.result_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=5)
        self.result_text = tk.Text(self.result_frame, height=12, width=35, font=('Consolas', 9))
        self.result_text.pack(fill=tk.BOTH, expand=True)
        self._result_frame = self.result_frame

        self._update_alg_list()
        self._update_alg_tip()
        self._update_param_frames()

    # ---- 演算法設定 ----

    def _mode_key(self):
        label = self.mode_var.get()
        for k in ('particle', 'pore'):
            if t(f'mode_{k}') == label:
                return k
        return 'particle'

    ALG_OPTIONS = {
        'particle': [
            ('watershed', 'alg_watershed', 'tip_watershed_particle'),
            ('hough', 'alg_hough', 'tip_hough_particle'),
        ],
        'pore': [
            ('threshold', 'alg_threshold', 'tip_threshold_pore'),
            ('watershed', 'alg_watershed', 'tip_watershed_pore'),
            ('hough', 'alg_hough', 'tip_hough_pore'),
        ],
    }

    def _alg_list(self):
        mode = self._mode_key()
        return [t(label_key) for _, label_key, _ in self.ALG_OPTIONS[mode]]

    def _alg_key(self, label):
        mode = self._mode_key()
        for key, lbl_key, _ in self.ALG_OPTIONS[mode]:
            if t(lbl_key) == label:
                return key
        return 'watershed'

    def _alg_label(self, key):
        mode = self._mode_key()
        for k, lbl_key, _ in self.ALG_OPTIONS[mode]:
            if k == key:
                return t(lbl_key)
        return key

    def _alg_tip(self, key):
        mode = self._mode_key()
        for k, _, tip_key in self.ALG_OPTIONS[mode]:
            if k == key:
                return t(tip_key)
        return ''

    def _update_alg_list(self):
        labels = self._alg_list()
        self.alg_cb['values'] = labels
        current = self.alg_var.get()
        mode = self._mode_key()
        # remap key → display label, or fallback to first option
        found = False
        for key, lbl_key, _ in self.ALG_OPTIONS[mode]:
            if current == key or current == t(lbl_key):
                self.alg_var.set(t(lbl_key))
                found = True
                break
        if not found:
            self.alg_var.set(labels[0] if labels else '')

    def _update_alg_tip(self):
        lbl = self.alg_var.get()
        key = self._alg_key(lbl)
        tip = self._alg_tip(key)
        if hasattr(self, '_alg_tooltip'):
            self._alg_tooltip.text = tip

    def _update_param_frames(self):
        key = self._alg_key(self.alg_var.get())
        frames = [self.ws_frame, self.hg_frame, self.th_frame]
        for f in frames:
            f.pack_forget()
        if key == 'watershed':
            self.ws_frame.pack(fill=tk.X, padx=8, pady=3)
        elif key == 'hough':
            self.hg_frame.pack(fill=tk.X, padx=8, pady=3)
        elif key == 'threshold':
            self.th_frame.pack(fill=tk.X, padx=8, pady=3)

    def _on_mode_change(self, event=None):
        self._update_alg_list()
        self._update_alg_tip()
        self._update_param_frames()
        self._clear_results()

    def _on_alg_change(self, event=None):
        self._update_alg_tip()
        self._update_param_frames()
        self._clear_results()

    def _on_lang_change(self, event=None):
        set_lang('zh' if self.lang_var.get() == t('lang_zh') else 'en')
        self._apply_language()

    def _update_mode_list(self):
        mode_labels = [t('mode_particle'), t('mode_pore')]
        self._mode_cb['values'] = mode_labels
        key = self._mode_key()
        self._mode_cb.set(t(f'mode_{key}'))

    def _apply_language(self):
        """重設所有介面文字（靜態 widget + 動態顯示）。"""
        self.master.title(t('app_title'))
        # 語言下拉本身
        self._lang_frame.configure(text=t('lang'))
        self.lang_cb['values'] = [t('lang_zh'), t('lang_en')]
        self.lang_var.set('zh' if i18n.LANG == 'zh' else 'en')
        # 按鈕
        self._btn_frame.configure(text=t('op'))
        self._btn_load.configure(text=t('load_image'))
        self.btn_calibrate.configure(text=t('calibrate'))
        self._btn_run.configure(text=t('run'))
        self._btn_export.configure(text=t('export'))
        # 比例尺
        self.scale_frame.configure(text=t('scale'))
        self._lbl_scale_len.configure(text=t('scale_len_um'))
        if self.pixel_to_micron is None:
            self.lbl_scale.configure(text=t('scale_not_calibrated'))
        # 模式
        self._mode_frame.configure(text=t('mode'))
        self._update_mode_list()
        # 演算法
        self._alg_frame.configure(text=t('alg'))
        self._update_alg_list()
        self._update_alg_tip()
        # 參數框架標題
        self.ws_frame.configure(text=t('ws_params'))
        self.hg_frame.configure(text=t('hg_params'))
        self.th_frame.configure(text=t('th_params'))
        for name, key in self._param_keys.items():
            getattr(self, f'_{name}_lbl').configure(text=t(key))
        # 單位
        self._unit_frame.configure(text=t('display_unit'))
        # 結果
        self._result_frame.configure(text=t('result'))
        # 圖表標題
        self.ax_dist.set_title(t('dist_title'), fontsize=11)
        # 若已分析，重畫動態內容
        if self.stats:
            self._show_results()
            self._show_overlay()
            self._show_distribution()
        else:
            self.canvas_dist.draw()

    # ---- 單位輔助 ----
    def _unit_scale(self):
        """回傳 1.0 (μm) 或 1000 (nm)。"""
        return 1000.0 if self.unit_var.get() == 'nm' else 1.0

    def _unit_label(self):
        return self.unit_var.get()  # 'nm' or 'um'

    def _on_unit_change(self, *args):
        """單位切換時重繪顯示（不重新分析）。"""
        if self.stats:
            self._show_results()
            self._show_distribution()
            self._show_overlay()

    def _add_param(self, parent, name, label_key, default, min_v, max_v):
        row = ttk.Frame(parent)
        row.pack(fill=tk.X, pady=1)
        lbl = ttk.Label(row, text=t(label_key), width=22, anchor=tk.W)
        lbl.pack(side=tk.LEFT)
        setattr(self, f'var_{name}', tk.IntVar(value=default))
        setattr(self, f'_{name}_lbl', lbl)
        self._param_keys[name] = label_key
        ttk.Spinbox(row, from_=min_v, to=max_v, textvariable=getattr(self, f'var_{name}'),
                        width=6).pack(side=tk.RIGHT)

    # ========================================================================
    #  載入圖片
    # ========================================================================

    def _load_image(self):
        path = filedialog.askopenfilename(
            title=t('select_image_title'),
            filetypes=[("Image files", "*.tif *.tiff *.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
        )
        if not path:
            return
        self.image_path = path
        self.img_original = imread_unicode(path)
        if self.img_original is None:
            messagebox.showerror(t('err_title'), t('err_read', path=path))
            return
        self.img_rgb_full = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2RGB)
        self._show_image(self.img_rgb_full, f"{os.path.basename(path)}")
        self.btn_calibrate.configure(state=tk.NORMAL)
        self._clear_results()
        self.scale_bar_px = None
        self.pixel_to_micron = None
        self.lbl_scale.configure(text=t('scale_not_calibrated'))
        self._log(t('log_loaded', name=os.path.basename(path),
                    w=self.img_original.shape[1], h=self.img_original.shape[0]))

    # ========================================================================
    #  比例尺校正
    # ========================================================================

    def _calibrate_scale(self):
        if self.img_rgb_full is None:
            return
        length_um = self.var_scale_len.get()
        if length_um <= 0:
            messagebox.showerror(t('err_title'), t('calib_len_err'))
            return

        cal = ScaleCalibrator(self.img_rgb_full, known_length_um=length_um)
        ratio, px_dist = cal.get_ratio()

        if ratio is not None and px_dist is not None:
            self.pixel_to_micron = ratio
            self.scale_bar_px = px_dist
            self.lbl_scale.configure(
                text=t('scale_manual', px=px_dist, um=length_um,
                       ratio=ratio, nmp=ratio*1000)
            )
            self._log(t('log_calibrated', px=px_dist, um=length_um, ratio=ratio))
        else:
            messagebox.showwarning(t('calib_fail'), t('calib_fail_msg'))
            gray_full = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
            result = auto_detect_scale_bar(gray_full)
            if result is not None:
                px_len, cx, cy = result
                self.pixel_to_micron = length_um / px_len
                self.scale_bar_px = px_len
                self.lbl_scale.configure(
                    text=t('scale_auto', px=px_len, um=length_um,
                           ratio=self.pixel_to_micron)
                )
                self._log(t('log_auto', px=px_len, ratio=self.pixel_to_micron))
            else:
                self.pixel_to_micron = 0.01
                self.lbl_scale.configure(text=t('scale_default'))

    # ========================================================================
    #  執行分析
    # ========================================================================

    def _run_analysis(self):
        try:
            self._run_analysis_inner()
        except Exception as e:
            import traceback
            messagebox.showerror(t('err_title'),
                                 f"{t('err_analysis')}\n\n{traceback.format_exc()[-1500:]}")

    def _run_analysis_inner(self):
        if self.img_original is None:
            messagebox.showwarning(t('warn_no_image'), t('warn_load_first'))
            return
        if self.pixel_to_micron is None:
            # 嘗試自動偵測
            gray_full = cv2.cvtColor(self.img_original, cv2.COLOR_BGR2GRAY)
            result = auto_detect_scale_bar(gray_full)
            if result is not None:
                px_len, cx, cy = result
                self.pixel_to_micron = self.var_scale_len.get() / px_len
                self.scale_bar_px = px_len
                self.lbl_scale.configure(
                    text=t('scale_auto', px=px_len, um=self.var_scale_len.get(),
                           ratio=self.pixel_to_micron)
                )
            else:
                messagebox.showwarning(t('warn_no_image'), t('warn_no_scale'))
                return

        # 裁切底部資訊欄
        self.img_cropped_bgr = auto_crop_sem(self.img_original)
        self.gray = cv2.cvtColor(self.img_cropped_bgr, cv2.COLOR_BGR2GRAY)
        self.img_cropped_rgb = cv2.cvtColor(self.img_cropped_bgr, cv2.COLOR_BGR2RGB)

        # 確保 scale_bar_px 有值
        if self.scale_bar_px is None:
            test_length = _nice_scale_value(1.0)
            self.scale_bar_px = int(test_length / self.pixel_to_micron)
            if self.scale_bar_px > self.gray.shape[1] * 0.4:
                test_length = _nice_scale_value(0.5)
                self.scale_bar_px = int(test_length / self.pixel_to_micron)

        mode = self._mode_key()
        key = self._alg_key(self.alg_var.get())
        self._log(t('running', mode=mode, key=key))
        self.master.update()

        if mode == 'particle':
            if key == 'watershed':
                self.props, _, _ = segment_watershed(
                    self.gray,
                    min_distance=self.var_min_dist.get(),
                    min_area=self.var_min_area.get())
            else:
                self.props, _ = segment_hough_circles(
                    self.gray,
                    min_radius=self.var_hough_min_r.get(),
                    max_radius=self.var_hough_max_r.get(),
                    param1=self.var_param1.get(),
                    param2=self.var_param2.get())
        else:
            if key == 'threshold':
                self.props, _ = segment_pores_threshold(
                    self.gray,
                    min_area=self.var_th_min_area.get(),
                    max_area_ratio=self.var_th_max_ratio.get() / 100.0,
                    morph_kernel=self.var_th_morph.get())
            elif key == 'watershed':
                self.props, _, _ = segment_pores_watershed(
                    self.gray,
                    min_distance=self.var_min_dist.get(),
                    min_area=self.var_min_area.get(),
                    morph_kernel=self.var_morph_kernel.get())
            else:
                self.props, _ = segment_pores_hough(
                    self.gray,
                    min_radius=self.var_hough_min_r.get(),
                    max_radius=self.var_hough_max_r.get(),
                    param1=self.var_param1.get(),
                    param2=self.var_param2.get())

        if not self.props:
            msg = t('info_no_particles') if mode == 'particle' else t('info_no_pores')
            self._log(t('log_no_detect', msg=msg))
            messagebox.showinfo(t('info_result'), msg)
            return

        self.diameters_um = [p['diameter_eq_px'] * self.pixel_to_micron for p in self.props]
        self.stats = compute_statistics(self.diameters_um)
        self.stats['min_um'] = self.stats['min_nm'] / 1000
        self.stats['max_um'] = self.stats['max_nm'] / 1000
        self.stats['d10_um'] = self.stats['d10_nm'] / 1000
        self.stats['d90_um'] = self.stats['d90_nm'] / 1000

        if mode == 'pore':
            total_area = self.gray.shape[0] * self.gray.shape[1]
            pore_areas = [p['area_px'] for p in self.props]
            self.stats['porosity'] = compute_porosity(pore_areas, total_area)

        self._show_results()
        self._show_overlay()
        self._show_distribution()

    # ========================================================================
    #  顯示結果
    # ========================================================================

    def _show_overlay(self):
        """在左上側顯示分割疊加圖 + 比例尺。"""
        mode = self._mode_key()
        color = (0, 0, 255) if mode == 'pore' else (0, 255, 0)
        overlay = self.img_cropped_rgb.copy()
        for p in self.props:
            cv2.drawContours(overlay, [p['contour']], -1, color, 2)

        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
        bar_um = self.scale_bar_px * self.pixel_to_micron
        draw_scale_bar(overlay_bgr, bar_um, self.pixel_to_micron)
        overlay = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)

        alg_label = self.alg_var.get()
        unit_label = t('label_pore') if mode == 'pore' else t('label_particle')
        u = self.unit_var.get()
        sfx = '_nm' if u == 'nm' else '_um'
        ulb = u
        title = (f"{alg_label} — {self.stats['count']} {unit_label} | "
                 f"Mean {self.stats[f'mean{sfx}']:.0f} {ulb} | "
                 f"D50 {self.stats[f'median{sfx}']:.0f} {ulb}")
        self._show_image(overlay, title)

    def _show_distribution(self):
        """在左下側顯示藍色雙軸對數分布圖（支援 nm/μm 切換）。"""
        self.fig_dist.clf()
        self.ax_dist = self.fig_dist.add_subplot(111)

        scale = self._unit_scale()      # 1000 (nm) or 1 (um)
        ulb = self._unit_label()        # 'nm' or 'um'
        arr = np.array(self.diameters_um) * scale
        if len(arr) == 0:
            self.ax_dist.text(0.5, 0.5, t('no_data'), ha='center', va='center')
            self.canvas_dist.draw()
            return

        vmin = max(arr.min() * 0.5, arr.min() * 0.8)
        vmax = arr.max() * 1.5
        bins = np.logspace(np.log10(vmin), np.log10(vmax), 35)
        bin_centers = np.sqrt(bins[:-1] * bins[1:])
        counts, _ = np.histogram(arr, bins=bins)
        total = counts.sum()
        channel_pct = counts / total * 100
        passing_pct = np.cumsum(channel_pct)

        bar_width = np.diff(np.log10(bins))
        self.ax_dist.bar(bin_centers, channel_pct, width=bins[:-1] * (10**bar_width - 1),
                         color='#5B9BD5', edgecolor='white', alpha=0.85, align='center',
                         label=t('dist_ylabel'), zorder=2)
        self.ax_dist.set_xscale('log')
        self.ax_dist.set_xlabel(t('dist_xlabel', u=ulb), fontsize=11)
        self.ax_dist.set_ylabel(t('dist_ylabel'), fontsize=11, color='#5B9BD5')
        self.ax_dist.tick_params(axis='both', which='both', direction='in', pad=6)
        self.ax_dist.tick_params(axis='y', labelcolor='#5B9BD5')
        self.ax_dist.set_ylim(0, max(channel_pct) * 1.25 if max(channel_pct) > 0 else 100)

        

        ax2 = self.ax_dist.twinx()
        ax2.plot(bin_centers, passing_pct, color='#003366', lw=2,
                 label=t('dist_ylabel2'), zorder=3)
        ax2.set_ylabel(t('dist_ylabel2'), fontsize=11, color='#003366')
        ax2.tick_params(axis='y', which='both', direction='in', labelcolor='#003366')
        ax2.set_ylim(-3, 103)

        self.ax_dist.grid(True, which='both', alpha=0.25, linestyle='--', zorder=0)
        self.ax_dist.set_axisbelow(False)

        lines1, labels1 = self.ax_dist.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        self.ax_dist.legend(lines1 + lines2, labels1 + labels2, loc='upper left',
                            fontsize=9, framealpha=0.85)

        self.ax_dist.set_title(t('dist_title'), fontsize=11)
        # Add more bottom margin to prevent X-axis labels from being covered by bars
        self.fig_dist.subplots_adjust(left=0.15, right=0.83, top=0.92, bottom=0.18)
        self.canvas_dist.draw()

    def _show_results(self):
        """在右側結果區顯示統計數據。"""
        mode = self._mode_key()
        u = self.unit_var.get()
        sfx = '_nm' if u == 'nm' else '_um'
        ulb = 'nm' if u == 'nm' else 'μm'
        s = self.stats
        label = t('label_pore') if mode == 'pore' else t('label_particle')
        text = (
            f"{t('r_count', label=label)} : {s['count']}\n"
            f"{t('r_mean')}      : {s[f'mean{sfx}']:>7.1f} {ulb}\n"
            f"{t('r_median')}: {s[f'median{sfx}']:>7.1f} {ulb}\n"
            f"{t('r_std')}     : {s[f'std{sfx}']:>7.1f} {ulb}\n"
            f"{t('r_min')}     : {s[f'min{sfx}']:>7.1f} {ulb}\n"
            f"{t('r_max')}     : {s[f'max{sfx}']:>7.1f} {ulb}\n"
            f"{t('r_d10')}        : {s[f'd10{sfx}']:>7.1f} {ulb}\n"
            f"{t('r_d90')}        : {s[f'd90{sfx}']:>7.1f} {ulb}\n"
        )
        if mode == 'pore' and 'porosity' in s:
            text += f"{t('r_porosity')}: {s['porosity']:>6.2f} %\n"
        text += (
            f"{'─'*30}\n"
            f"{t('r_ratio')}       : {self.pixel_to_micron:.6f} μm/px\n"
            f"{t('r_area')}       : {self.gray.shape[1]}x{self.gray.shape[0]} px"
        )
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(1.0, text)
        self._log(t('log_done', count=s['count'], label=label,
                    mean=s[f'mean{sfx}'], u=ulb))

    def _show_image(self, img_rgb, title=""):
        self.ax_img.clear()
        self.ax_img.imshow(img_rgb, aspect='equal')
        self.ax_img.set_title(title, fontsize=11, pad=6)
        self.ax_img.axis('off')
        self.canvas.draw()

    def _clear_results(self):
        self.result_text.delete(1.0, tk.END)
        self.props = None
        self.diameters_um = None
        self.stats = None

    def _log(self, msg):
        print(f"[GUI] {msg}")

    # ========================================================================
    #  匯出

    # ========================================================================
    #  匯出
    # ========================================================================

    def _export_results(self):
        if not self.stats:
            messagebox.showwarning(t('export_warn'), t('export_no_result'))
            return

        out_dir = filedialog.askdirectory(title=t('export_title'))
        if not out_dir:
            return

        out_path = Path(out_dir)
        base = Path(self.image_path).stem
        mode = self._mode_key()
        key = self._alg_key(self.alg_var.get())
        tag = f"{mode}_{key}"
        label_s = t('label_pore') if mode == 'pore' else t('label_particle')
        u = self.unit_var.get()
        sfx = '_nm' if u == 'nm' else '_um'
        ulb = 'nm' if u == 'nm' else 'μm'
        s = self.stats

        # CSV
        col = f'Diameter_{ulb}'
        scale = self._unit_scale()
        csv_path = out_path / f"{base}_{tag}_diameters.csv"
        pd.DataFrame({col: [d * scale for d in self.diameters_um]}).to_csv(csv_path, index=False, encoding='utf-8-sig')

        # 統計 TXT
        txt_path = out_path / f"{base}_{tag}_statistics.txt"
        lines = [
            f"{t('r_count', label=label_s)}      : {s['count']}",
            f"{t('r_mean')}           : {s[f'mean{sfx}']:.1f} {ulb}",
            f"{t('r_median')}   : {s[f'median{sfx}']:.1f} {ulb}",
            f"{t('r_std')}          : {s[f'std{sfx}']:.1f} {ulb}",
            f"{t('r_min')}          : {s[f'min{sfx}']:.1f} {ulb}",
            f"{t('r_max')}          : {s[f'max{sfx}']:.1f} {ulb}",
            f"D10                  : {s[f'd10{sfx}']:.1f} {ulb}",
            f"D90                  : {s[f'd90{sfx}']:.1f} {ulb}",
        ]
        if mode == 'pore' and 'porosity' in s:
            lines.append(f"{t('r_porosity')}     : {s['porosity']:.2f} %")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        # 疊加圖
        from PIL import Image, ImageDraw, ImageFont
        color = (0, 0, 255) if mode == 'pore' else (0, 255, 0)
        overlay_bgr = self.img_cropped_bgr.copy()
        for p in self.props:
            cv2.drawContours(overlay_bgr, [p['contour']], -1, color, 2)
        bar_um = self.scale_bar_px * self.pixel_to_micron
        draw_scale_bar(overlay_bgr, bar_um, self.pixel_to_micron)

        h_img = overlay_bgr.shape[0]
        font_size = max(22, h_img // 28)
        try:
            pil_font = ImageFont.truetype(r"C:\Windows\Fonts\msyh.ttc", font_size)
        except Exception:
            pil_font = ImageFont.load_default()

        u = self.unit_var.get()
        sfx = '_nm' if u == 'nm' else '_um'
        ulb = u
        info_lines = []
        if mode == 'pore' and 'porosity' in s:
            info_lines.append(t('overlay_info_porosity', p=s['porosity']))
        info_lines.append(f"{label_s}: {s['count']}")
        info_lines.append(t('overlay_info_mean', m=s[f'mean{sfx}'], d=s[f'median{sfx}'], u=ulb))
        info_text = "  |  ".join(info_lines)

        rgb = cv2.cvtColor(overlay_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        draw = ImageDraw.Draw(pil_img)
        bbox = draw.textbbox((0, 0), info_text, font=pil_font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        margin = max(10, h_img // 80)
        draw.rectangle([margin - 4, margin - 2, margin + tw + 8, margin + th + 8],
                       fill=(0, 0, 0, 180))
        draw.text((margin, margin), info_text, fill=(255, 255, 255), font=pil_font)
        overlay_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        overlay_path = out_path / f"{base}_{tag}_overlay.png"
        imwrite_unicode(str(overlay_path), overlay_bgr)

        # 分布圖
        dist_path = out_path / f"{base}_{tag}_distribution.png"
        plot_particle_size_distribution(
            self.diameters_um, self.stats,
            save_path=str(dist_path), show_plot=False,
            figsize=(7.5, 7), unit=u
        )

        messagebox.showinfo(t('export_done'),
            t('export_files', dir=out_dir, csv=csv_path.name, txt=txt_path.name,
              overlay=overlay_path.name, dist=dist_path.name))


if __name__ == '__main__':
    root = tk.Tk()
    app = ParticleAnalyzerGUI(root)
    root.mainloop()
