"""Native file picker for local training UI.

Windows 优先走 comdlg32.GetOpenFileNameW（不依赖 tkinter/Tcl/Tk），
其它平台或 Win32 失败时再回退到 tkinter。
"""
from __future__ import annotations

import sys
from typing import Sequence


def _pick_with_win32(
    title: str,
    filetypes: Sequence[tuple[str, str]],
) -> str | None:
    """Windows 系统原生打开对话框（ctypes + comdlg32）。"""
    import ctypes
    from ctypes import wintypes

    class OPENFILENAMEW(ctypes.Structure):
        _fields_ = [
            ("lStructSize", wintypes.DWORD),
            ("hwndOwner", wintypes.HWND),
            ("hInstance", wintypes.HINSTANCE),
            ("lpstrFilter", wintypes.LPCWSTR),
            ("lpstrCustomFilter", wintypes.LPWSTR),
            ("nMaxCustFilter", wintypes.DWORD),
            ("nFilterIndex", wintypes.DWORD),
            ("lpstrFile", wintypes.LPWSTR),
            ("nMaxFile", wintypes.DWORD),
            ("lpstrFileTitle", wintypes.LPWSTR),
            ("nMaxFileTitle", wintypes.DWORD),
            ("lpstrInitialDir", wintypes.LPCWSTR),
            ("lpstrTitle", wintypes.LPCWSTR),
            ("Flags", wintypes.DWORD),
            ("nFileOffset", wintypes.WORD),
            ("nFileExtension", wintypes.WORD),
            ("lpstrDefExt", wintypes.LPCWSTR),
            ("lCustData", wintypes.LPARAM),
            ("lpfnHook", ctypes.c_void_p),
            ("lpTemplateName", wintypes.LPCWSTR),
            ("pvReserved", ctypes.c_void_p),
            ("dwReserved", wintypes.DWORD),
            ("FlagsEx", wintypes.DWORD),
        ]

    OFN_FILEMUSTEXIST = 0x00001000
    OFN_PATHMUSTEXIST = 0x00000800
    OFN_EXPLORER = 0x00080000
    OFN_NOCHANGEDIR = 0x00000008
    OFN_HIDEREADONLY = 0x00000004

    # Filter: "Label\0pattern\0Label2\0pattern2\0\0"
    parts: list[str] = []
    for label, pattern in filetypes:
        parts.append(label)
        parts.append(pattern)
    filter_buf = ctypes.create_unicode_buffer("\0".join(parts) + "\0")

    max_path = 32768
    file_buf = ctypes.create_unicode_buffer(max_path)

    ofn = OPENFILENAMEW()
    ofn.lStructSize = ctypes.sizeof(OPENFILENAMEW)
    ofn.lpstrFilter = ctypes.cast(filter_buf, wintypes.LPCWSTR)
    ofn.nFilterIndex = 1
    # create_unicode_buffer 直接赋给 LPWSTR 在部分 Python 会报类型不兼容
    ofn.lpstrFile = ctypes.cast(file_buf, wintypes.LPWSTR)
    ofn.nMaxFile = max_path
    ofn.lpstrTitle = title
    ofn.Flags = (
        OFN_EXPLORER
        | OFN_FILEMUSTEXIST
        | OFN_PATHMUSTEXIST
        | OFN_NOCHANGEDIR
        | OFN_HIDEREADONLY
    )

    comdlg32 = ctypes.WinDLL("comdlg32", use_last_error=True)
    GetOpenFileNameW = comdlg32.GetOpenFileNameW
    GetOpenFileNameW.argtypes = [ctypes.POINTER(OPENFILENAMEW)]
    GetOpenFileNameW.restype = wintypes.BOOL

    ok = GetOpenFileNameW(ctypes.byref(ofn))
    if not ok:
        return None
    path = file_buf.value.strip()
    return path or None


def _pick_with_tkinter(
    title: str,
    filetypes: Sequence[tuple[str, str]],
) -> str | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        raise RuntimeError("当前环境不支持图形文件选择（缺少 tkinter）")

    root = tk.Tk()
    root.withdraw()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    path = filedialog.askopenfilename(title=title, filetypes=list(filetypes))
    root.destroy()
    return path or None


def _pick_open_file(
    title: str,
    filetypes: Sequence[tuple[str, str]],
) -> str | None:
    if sys.platform == "win32":
        try:
            return _pick_with_win32(title, filetypes)
        except Exception:
            # Win32 失败时再试 tkinter（完整 Python 安装可用）
            pass
    return _pick_with_tkinter(title, filetypes)


def pick_parquet_file() -> str | None:
    return _pick_open_file(
        title="选择 K 线 Parquet 文件",
        filetypes=[
            ("Parquet K线", "*.parquet"),
            ("所有文件", "*.*"),
        ],
    )


def pick_strategy_file() -> str | None:
    return _pick_open_file(
        title="选择策略 JSON 文件",
        filetypes=[
            ("策略 JSON", "*.json"),
            ("所有文件", "*.*"),
        ],
    )
