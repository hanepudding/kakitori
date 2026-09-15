"""Dictate into the focused window. Run: python dictate.py — settings come from .env (see .env.example), Ctrl+C quits."""
import sys

from dictation import config

# Under pythonw (the scheduled task) there is no console and sys.stdout is None, so prints and tracebacks
# would be lost. Redirect before importing anything else, so import failures get logged too.
if sys.stdout is None:
    log_file = config.log_file()
    log_file.parent.mkdir(parents=True, exist_ok=True)
    sys.stdout = sys.stderr = open(log_file, "a", encoding="utf-8", buffering=1)

import queue
import time

from dictation import hotkey, paste, recorder, vocab
from dictation.asr import SAMPLE_RATE, Qwen3ASR, ServerError

STOP_EDGE = {"hold": "up", "toggle": "down"}
OFFLINE_NOTICE = "[dictation] ASR server offline, not recording"
FAILED_NOTICE = "[dictation] transcription failed"


def say(message: str) -> None:
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {message}")


def wait_for(events: queue.Queue, edge: str) -> None:
    while events.get() != edge:
        pass


if __name__ == "__main__":
    settings = config.load()
    stop = STOP_EDGE[settings.mode]

    t = time.perf_counter()
    asr = Qwen3ASR(settings.server, settings.max_new_tokens, settings.timeout_sec)
    say(f"Waiting for {settings.server} ...")
    asr.wait_until_ready()
    asr.warm_up(vocab.load_prompt(settings.vocab), settings.language)
    # Listen only once transcription works; until then the hotkey reaches apps as usual
    events = hotkey.listen(settings.hotkey)
    say(f"Ready in {time.perf_counter() - t:.1f} s. {settings.hotkey} records ({settings.mode}), Ctrl+C quits.")

    while True:
        wait_for(events, "down")
        if not asr.ready():
            say(f"{settings.server} is offline, not recording")
            paste.paste(OFFLINE_NOTICE, settings.paste_delay_sec, settings.restore_delay_sec)
            continue
        # Read on every press, so an edit to the vocabulary file takes effect without a restart
        prompt = vocab.load_prompt(settings.vocab)
        say("Recording...")
        samples = recorder.record(SAMPLE_RATE, until=lambda: wait_for(events, stop))
        if samples.size == 0:
            say("No audio recorded")
            continue
        t = time.perf_counter()
        try:
            text = asr.transcribe(samples, prompt, settings.language)
        except ServerError as e:
            say(f"Transcription failed: {e}")
            paste.paste(FAILED_NOTICE, settings.paste_delay_sec, settings.restore_delay_sec)
            continue
        say(f"[{len(samples) / SAMPLE_RATE:.1f} s audio, transcribed in {time.perf_counter() - t:.2f} s] {text}")
        if not text:
            continue
        paste.paste(text, settings.paste_delay_sec, settings.restore_delay_sec)
