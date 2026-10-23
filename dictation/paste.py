"""Insert text into the focused window through the clipboard, then put back everything the clipboard held."""
import sys
import time

from pynput.keyboard import Controller, Key

from dictation import clipboard

PASTE_MODIFIER = Key.cmd if sys.platform == "darwin" else Key.ctrl

# Built on the main thread before the listener starts: on macOS the constructor calls input-source APIs,
# and concurrent calls from the listener thread abort the process
_keyboard = Controller()


def paste(text: str, paste_delay: float, restore_delay: float) -> None:
    """paste_delay: clipboard write -> paste keystroke; restore_delay: paste keystroke -> restore (seconds)."""
    previous = clipboard.snapshot()
    clipboard.write_text(text)
    time.sleep(paste_delay)
    with _keyboard.pressed(PASTE_MODIFIER):
        _keyboard.tap("v")
    time.sleep(restore_delay)
    clipboard.restore(previous)
