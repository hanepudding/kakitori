"""Insert text into the focused window through the clipboard, then put back everything the clipboard held."""
import sys
import time

from dictation import clipboard

# dwExtraInfo on the paste keystroke, so the hotkey hook can tell it from injected keys of other origin, such as a
# mouse driver that sends Ctrl+H for a button
INJECTED_MARK = 0x6B616B69

if sys.platform == "darwin":
    from pynput.keyboard import Controller, Key

    # Built on the main thread before the listener starts: the constructor calls input-source APIs, and concurrent
    # calls from the listener thread abort the process
    _keyboard = Controller()

    def _send_paste() -> None:
        with _keyboard.pressed(Key.cmd):
            _keyboard.tap("v")

else:
    import ctypes

    from pynput._util.win32 import INPUT, INPUT_union, KEYBDINPUT, MapVirtualKey, SendInput

    _VK_CONTROL, _VK_V = 0x11, 0x56

    def _send_paste() -> None:
        # pynput's Controller leaves dwExtraInfo at 0
        events = [(_VK_CONTROL, 0), (_VK_V, 0), (_VK_V, KEYBDINPUT.KEYUP), (_VK_CONTROL, KEYBDINPUT.KEYUP)]
        inputs = (INPUT * len(events))(*(
            INPUT(type=INPUT.KEYBOARD, value=INPUT_union(ki=KEYBDINPUT(
                wVk=vk, wScan=MapVirtualKey(vk, MapVirtualKey.MAPVK_VK_TO_VSC), dwFlags=flags, dwExtraInfo=INJECTED_MARK)))
            for vk, flags in events))
        SendInput(len(inputs), inputs, ctypes.sizeof(INPUT))


def paste(text: str, paste_delay: float, restore_delay: float) -> None:
    """paste_delay: clipboard write -> paste keystroke; restore_delay: paste keystroke -> restore (seconds)."""
    previous = clipboard.snapshot()
    clipboard.write_text(text)
    time.sleep(paste_delay)
    _send_paste()
    time.sleep(restore_delay)
    clipboard.restore(previous)
