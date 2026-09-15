"""Qwen3-ASR served by llama-server's OpenAI-compatible chat API; one request transcribes one utterance."""
import base64
import io
import time
import wave

import httpx
import numpy as np

SAMPLE_RATE = 16000
# Windows takes about 2 s to refuse a connection to a closed local port;
# a hotkey press while the server is down should not wait that long
HEALTH_TIMEOUT_SEC = 0.5


class ServerError(Exception):
    """llama-server was unreachable or rejected the request, e.g. audio too long for its context."""


class Qwen3ASR:
    def __init__(self, server: str, max_new_tokens: int, timeout_sec: float) -> None:
        self.server = server.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.http = httpx.Client(timeout=timeout_sec)

    def ready(self) -> bool:
        """Whether GET /health answers 200; llama-server answers 503 while it is still loading the model."""
        try:
            return self.http.get(f"{self.server}/health", timeout=HEALTH_TIMEOUT_SEC).status_code == 200
        except httpx.TransportError:
            return False

    def wait_until_ready(self) -> None:
        while not self.ready():
            time.sleep(1)

    def transcribe(self, samples: np.ndarray, prompt: str | None, language: str | None) -> str:
        """`samples` is mono float32 at SAMPLE_RATE; `language` is a full name such as "Chinese", or None to detect."""
        wav = io.BytesIO()
        with wave.open(wav, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes((np.clip(samples, -1.0, 1.0) * 32767).astype("<i2").tobytes())
        audio = {"type": "input_audio", "input_audio": {"data": base64.b64encode(wav.getvalue()).decode(), "format": "wav"}}
        messages = [{"role": "system", "content": prompt or ""}, {"role": "user", "content": [audio]}]
        if language is not None:
            # Qwen3-ASR is held to a language by prefilling the start of its answer
            messages.append({"role": "assistant", "content": f"language {language}<asr_text>"})

        body = {"messages": messages, "max_tokens": self.max_new_tokens, "temperature": 0}
        try:
            r = self.http.post(f"{self.server}/v1/chat/completions", json=body)
        except httpx.TransportError as e:
            raise ServerError(f"{self.server}: {e!r}") from e
        if r.status_code != 200:
            raise ServerError(f"{r.status_code} {r.text}")

        # The answer reads "language <Name><asr_text><transcript>", with the prefill included when there is one
        content = r.json()["choices"][0]["message"]["content"]
        _, tag, text = content.partition("<asr_text>")
        return (text if tag else content).strip()

    def warm_up(self, prompt: str | None, language: str | None) -> None:
        self.transcribe(np.zeros(SAMPLE_RATE, dtype=np.float32), prompt, language)
