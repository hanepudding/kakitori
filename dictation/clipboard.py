"""The system clipboard as a whole: snapshot() every format it holds, restore() them, write_text() replaces them."""
import sys

if sys.platform == "darwin":
    from AppKit import NSData, NSPasteboard, NSPasteboardItem, NSPasteboardTypeString

    _pasteboard = NSPasteboard.generalPasteboard()

    def snapshot() -> list[list[tuple]]:
        """One list of (type, data) per pasteboard item; a Finder copy of several files is several items."""
        # A promised type (a file not yet written, for one) has no data until a paste asks for it, and is lost
        return [[(t, d) for t in item.types() if (d := item.dataForType_(t)) is not None]
                for item in _pasteboard.pasteboardItems()]

    def restore(items: list[list[tuple]]) -> None:
        _pasteboard.clearContents()
        objects = []
        for types in items:
            item = NSPasteboardItem.alloc().init()
            for t, d in types:
                item.setData_forType_(d, t)
            objects.append(item)
        if objects:
            _pasteboard.writeObjects_(objects)

    def write_text(text: str) -> None:
        data = text.encode("utf-8")
        restore([[(NSPasteboardTypeString, NSData.dataWithBytes_length_(data, len(data)))]])

else:
    import contextlib
    import ctypes
    import time
    from ctypes import wintypes

    _user32 = ctypes.WinDLL("user32", use_last_error=True)
    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _gdi32 = ctypes.WinDLL("gdi32", use_last_error=True)

    def _api(dll, name, restype, *argtypes):
        f = getattr(dll, name)
        f.restype, f.argtypes = restype, argtypes
        return f

    _H = wintypes.HANDLE
    _CreateWindowExW = _api(_user32, "CreateWindowExW", wintypes.HWND, wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR,
                            wintypes.DWORD, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.HWND,
                            wintypes.HMENU, wintypes.HINSTANCE, wintypes.LPVOID)
    _DestroyWindow = _api(_user32, "DestroyWindow", wintypes.BOOL, wintypes.HWND)
    _OpenClipboard = _api(_user32, "OpenClipboard", wintypes.BOOL, wintypes.HWND)
    _CloseClipboard = _api(_user32, "CloseClipboard", wintypes.BOOL)
    _EmptyClipboard = _api(_user32, "EmptyClipboard", wintypes.BOOL)
    _EnumClipboardFormats = _api(_user32, "EnumClipboardFormats", wintypes.UINT, wintypes.UINT)
    _GetClipboardData = _api(_user32, "GetClipboardData", _H, wintypes.UINT)
    _SetClipboardData = _api(_user32, "SetClipboardData", _H, wintypes.UINT, _H)
    _GlobalAlloc = _api(_kernel32, "GlobalAlloc", _H, wintypes.UINT, ctypes.c_size_t)
    _GlobalLock = _api(_kernel32, "GlobalLock", wintypes.LPVOID, _H)
    _GlobalUnlock = _api(_kernel32, "GlobalUnlock", wintypes.BOOL, _H)
    _GlobalSize = _api(_kernel32, "GlobalSize", ctypes.c_size_t, _H)
    _GlobalFree = _api(_kernel32, "GlobalFree", _H, _H)
    _GetEnhMetaFileBits = _api(_gdi32, "GetEnhMetaFileBits", wintypes.UINT, _H, wintypes.UINT, ctypes.c_char_p)
    _SetEnhMetaFileBits = _api(_gdi32, "SetEnhMetaFileBits", _H, wintypes.UINT, ctypes.c_char_p)
    _DeleteEnhMetaFile = _api(_gdi32, "DeleteEnhMetaFile", wintypes.BOOL, _H)

    _CF_UNICODETEXT = 13
    _CF_ENHMETAFILE = 14
    _GHND = 0x0042  # GMEM_MOVEABLE | GMEM_ZEROINIT
    # Formats whose handle is a GDI object or owner-defined, not global memory. Windows synthesizes CF_BITMAP from
    # CF_DIB and CF_METAFILEPICT from CF_ENHMETAFILE, so a picture copied in either form still comes back
    _UNCOPYABLE = {2, 3, 9, 0x80, 0x82, 0x83, 0x8E} | set(range(0x200, 0x400))
    _OPEN_TIMEOUT_SEC = 0.5

    def _check(ok):
        if not ok:
            raise ctypes.WinError(ctypes.get_last_error())
        return ok

    @contextlib.contextmanager
    def _opened():
        # EmptyClipboard after OpenClipboard(NULL) leaves the clipboard without an owner, and then SetClipboardData fails
        hwnd = _check(_CreateWindowExW(0, "STATIC", None, 0, 0, 0, 0, 0, None, None, None, None))
        try:
            # Another process holding the clipboard open, a clipboard-history listener right after an update among
            # them, makes OpenClipboard fail until it closes
            deadline = time.monotonic() + _OPEN_TIMEOUT_SEC
            while not _OpenClipboard(hwnd):
                if time.monotonic() > deadline:
                    raise ctypes.WinError(ctypes.get_last_error())
                time.sleep(0.01)
            try:
                yield
            finally:
                _CloseClipboard()
        finally:
            _DestroyWindow(hwnd)

    def _read(fmt: int) -> bytes | None:
        handle = _GetClipboardData(fmt)
        # NULL when the owner fails to render the format, or offers it only through OLE (FileContents, for one)
        if not handle:
            return None
        if fmt == _CF_ENHMETAFILE:
            buffer = ctypes.create_string_buffer(_GetEnhMetaFileBits(handle, 0, None))
            _GetEnhMetaFileBits(handle, len(buffer), buffer)
            return buffer.raw
        pointer = _GlobalLock(handle)
        if not pointer:
            return None
        try:
            return ctypes.string_at(pointer, _GlobalSize(handle))
        finally:
            _GlobalUnlock(handle)

    def _write(fmt: int, data: bytes) -> None:
        if fmt == _CF_ENHMETAFILE:
            handle = _check(_SetEnhMetaFileBits(len(data), data))
            if not _SetClipboardData(fmt, handle):
                _DeleteEnhMetaFile(handle)
                raise ctypes.WinError(ctypes.get_last_error())
            return
        # A zero-byte GlobalAlloc returns a discarded block that GlobalLock refuses
        handle = _check(_GlobalAlloc(_GHND, max(len(data), 1)))
        ctypes.memmove(_GlobalLock(handle), data, len(data))
        _GlobalUnlock(handle)
        if not _SetClipboardData(fmt, handle):
            _GlobalFree(handle)
            raise ctypes.WinError(ctypes.get_last_error())

    def snapshot() -> list[tuple[int, bytes]]:
        """(format, data) in the clipboard's own order, which is the preference order apps pick a format by.

        Every delay-rendered format is rendered now: copying a large Excel range makes this take a moment.
        """
        formats = []
        with _opened():
            fmt = _EnumClipboardFormats(0)
            while fmt:
                if fmt not in _UNCOPYABLE and (data := _read(fmt)) is not None:
                    formats.append((fmt, data))
                fmt = _EnumClipboardFormats(fmt)
        return formats

    def restore(formats: list[tuple[int, bytes]]) -> None:
        with _opened():
            _check(_EmptyClipboard())
            for fmt, data in formats:
                _write(fmt, data)

    def write_text(text: str) -> None:
        restore([(_CF_UNICODETEXT, (text + "\0").encode("utf-16-le"))])
