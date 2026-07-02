# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['particle_analyzer_gui.py'],
    pathex=[],
    binaries=[],
    datas=[('particle_analyzer.py', '.')],
    hiddenimports=['tkinter', 'matplotlib.backends.backend_tkagg', 'cv2', 'numpy', 'pandas', 'scipy', 'skimage', 'PIL', 'skimage.feature', 'skimage.segmentation', 'skimage.morphology', 'skimage.filters', 'skimage.exposure', 'skimage.draw', 'skimage.color', 'skimage.transform', 'skimage.measure', 'skimage.restoration', 'skimage.metrics'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    a.zipfiles,
    a.zipped_data,
    name='ParticleAnalyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
