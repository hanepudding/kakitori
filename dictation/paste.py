"""Insert text into the focused window through the clipboard, then put the old clipboard text back."""
import sys
import time

import pyperclip
from pynput.keyboard import Controller, Key

PASTE_MODIFIER = Key.cmd if sys.platform == "darwin" else Key.ctrl

# Built on the main thread before the listener starts: on macOS the constructor calls input-source APIs,
# and concurrent calls from the listener thread abort the process
_keyboard = Controller()


def paste(text: str, paste_delay: float, restore_delay: float) -> None:
    """paste_delay: clipboard write -> paste keystroke; restore_delay: paste keystroke -> restore (seconds)."""
    previous = pyperclip.paste()
    pyperclip.copy(text)
    time.sleep(paste_delay)
    with _keyboard.pressed(PASTE_MODIFIER):
        _keyboard.tap("v")
    time.sleep(restore_delay)
    # With no text on the clipboard (empty, image, files) paste() returns "" (None on macOS). Images and
    # files are already gone after copy(text); this only avoids wiping the pasted text or writing "None"
    if previous:
        pyperclip.copy(previous)
