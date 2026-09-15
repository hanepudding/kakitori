"""Two short tones, one when recording starts and one when it stops, played without blocking. The start tone
overlaps the first 60 ms of the recording: on four test clips a tone that ended before speech began left the
transcript unchanged at every frequency and level tried."""
import subprocess
import sys
import threading

TONES = {"start": 880, "done": 1320}
if sys.platform == "win32":
    import winsound

    def play(tone: str) -> None:
        threading.Thread(target=winsound.Beep, args=(TONES[tone], 60), daemon=True).start()

else:
    SOUNDS = {"start": "/System/Library/Sounds/Tink.aiff", "done": "/System/Library/Sounds/Pop.aiff"}

    def play(tone: str) -> None:
        subprocess.Popen(["afplay", SOUNDS[tone]], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
