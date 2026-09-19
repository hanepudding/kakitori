"""Settings from the project's .env. A blank or unset key takes the default, which .env.example documents."""
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).parent.parent
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    hotkey: str
    mode: str
    language: str | None
    vocab: Path
    server: str
    normalizer: str | None
    normalize_steps: tuple[str, ...]
    max_new_tokens: int
    timeout_sec: float
    normalizer_timeout_sec: float
    normalizer_sec_per_char: float
    chime: bool
    paste_delay_sec: float
    restore_delay_sec: float


def _get(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, "").strip() or default


def _path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def log_file() -> Path:
    return _path(_get("DICTATION_LOG_FILE", "~/.local/state/dictation/dictation.log"))


def load() -> Settings:
    return Settings(
        hotkey=_get("DICTATION_HOTKEY", "<ctrl>+h"),
        mode=_get("DICTATION_MODE", "hold"),
        language=_get("DICTATION_LANGUAGE"),
        vocab=_path(_get("DICTATION_VOCAB", "vocab.txt")),
        server=_get("DICTATION_SERVER", "http://127.0.0.1:8080"),
        normalizer=_get("DICTATION_NORMALIZER"),
        normalize_steps=tuple(s.strip() for s in _get("DICTATION_NORMALIZE_STEPS", "digits").split(",")),
        max_new_tokens=int(_get("DICTATION_MAX_NEW_TOKENS", "512")),
        timeout_sec=float(_get("DICTATION_TIMEOUT_SEC", "60")),
        normalizer_timeout_sec=float(_get("DICTATION_NORMALIZER_TIMEOUT_SEC", "5")),
        normalizer_sec_per_char=float(_get("DICTATION_NORMALIZER_SEC_PER_CHAR", "0.02")),
        chime=_get("DICTATION_CHIME", "1") != "0",
        paste_delay_sec=float(_get("DICTATION_PASTE_DELAY_SEC", "0.05")),
        restore_delay_sec=float(_get("DICTATION_RESTORE_DELAY_SEC", "0.5")),
    )
