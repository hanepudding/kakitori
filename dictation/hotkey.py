"""Global hotkey chord; its trigger key is swallowed so the focused app never sees it.

listen() puts "down" on the queue when the chord completes and "up" when any of its keys is released.
"""
import ctypes
import queue
import sys

from pynput import keyboard
from pynput.keyboard import Key

# The Windows low-level keyboard hook reports side-specific key codes, so <ctrl> in a chord must match both sides
_MODIFIER_FAMILIES = {
    Key.ctrl: (Key.ctrl, Key.ctrl_l, Key.ctrl_r),
    Key.shift: (Key.shift, Key.shift_l, Key.shift_r),
    Key.alt: (Key.alt, Key.alt_l, Key.alt_r),
    Key.cmd: (Key.cmd, Key.cmd_l, Key.cmd_r),
}
_WM_KEYDOWN = 0x0100
_WM_SYSKEYDOWN = 0x0104
_LLKHF_INJECTED = 0x10


class Chord:
    """State of one hotkey. Every part but the last is a modifier; the last is the trigger.

    press/release return True when the event must be swallowed: the trigger key, from the
    press that completes the chord until that key's own release.
    """

    def __init__(self, parts: list[set], events: queue.Queue) -> None:
        self.parts = parts
        self.trigger = len(parts) - 1
        self.events = events
        self.held: set[int] = set()
        self.active = False
        self.swallowing = False

    def _index(self, token) -> int | None:
        return next((i for i, part in enumerate(self.parts) if token in part), None)

    def press(self, token) -> bool:
        i = self._index(token)
        if i is None:
            return False
        self.held.add(i)
        if i == self.trigger and not self.swallowing and len(self.held) == len(self.parts):
            self.active = self.swallowing = True
            self.events.put("down")
        return i == self.trigger and self.swallowing

    def release(self, token) -> bool:
        i = self._index(token)
        if i is None:
            return False
        self.held.discard(i)
        if self.active:
            self.active = False
            self.events.put("up")
        if i == self.trigger and self.swallowing:
            self.swallowing = False
            return True
        return False


def _win32_vks(key) -> set[int]:
    if key in _MODIFIER_FAMILIES:
        return {k.value.vk for k in _MODIFIER_FAMILIES[key]}
    if key.char is not None:
        vk_key_scan = ctypes.WinDLL("user32").VkKeyScanW
        vk_key_scan.argtypes = (ctypes.c_wchar,)
        vk_key_scan.restype = ctypes.c_short
        return {vk_key_scan(key.char) & 0xFF}
    return {key.vk}


def listen(hotkey: str) -> queue.Queue:
    """Start listening for `hotkey`, written in pynput's format ("<ctrl>+h", "<f13>")."""
    events: queue.Queue = queue.Queue()
    keys = keyboard.HotKey.parse(hotkey)

    if sys.platform == "win32":
        chord = Chord([_win32_vks(k) for k in keys], events)

        # The filter runs synchronously inside the hook callback, and a suppressed event never reaches
        # on_press/on_release, so the chord can only be tracked here
        def event_filter(msg, data):
            # Ctrl injected by paste.py passes through the hook too; counting it would clear a Ctrl the user
            # is still holding and emit an extra "up"
            if data.flags & _LLKHF_INJECTED:
                return
            track = chord.press if msg in (_WM_KEYDOWN, _WM_SYSKEYDOWN) else chord.release
            if track(data.vkCode):
                listener.suppress_event()

        listener = keyboard.Listener(win32_event_filter=event_filter)
    else:
        chord = Chord([{k} for k in keys], events)
        swallow = False

        def on_press(key):
            nonlocal swallow
            swallow = chord.press(listener.canonical(key))

        def on_release(key):
            nonlocal swallow
            swallow = chord.release(listener.canonical(key))

        # For each event pynput calls on_press/on_release synchronously and then intercept,
        # so swallow describes only the current event and is cleared once read
        def intercept(event_type, event):
            nonlocal swallow
            swallowed, swallow = swallow, False
            return None if swallowed else event

        listener = keyboard.Listener(on_press=on_press, on_release=on_release, darwin_intercept=intercept)

    listener.start()
    listener.wait()
    return events
