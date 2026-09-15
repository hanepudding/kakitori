# Kakitori

Local push-to-talk dictation for Windows and macOS. Hold a hotkey, speak, release: the sentence is transcribed by a
local speech model and pasted into whatever window has focus. Chinese, English and Japanese can be mixed in one sentence.

Everything runs on your own machine. This program itself holds no model: it records, talks to an HTTP server, and
pastes. The model behind that server is yours to choose.

## How it is built

The whole thing is about 650 lines of Python. This section is the wiring; with it, a coding assistant can rebuild
the project or port any piece of it.

**Pipeline, one utterance**

1. `pynput` listens for a global key chord (default `Ctrl+H`). The trigger key is swallowed at the OS hook level so
   the focused app never sees it (Ctrl+H is "replace" in editors and "backspace" in terminals). Press and release
   edges go into a queue; `hold` mode records between them, `toggle` mode between two presses.
2. `sounddevice` (PortAudio) records mono float32 at 16 kHz from the default input device for as long as the key is
   held.
3. The audio is written to an in-memory 16-bit WAV and sent with `httpx` to a llama-server as one OpenAI-style chat
   completion: system message = the vocabulary prompt (`Vocabulary: term、term。`, built from `vocab.txt`), user
   message = `input_audio` (base64 WAV), `temperature 0`. To pin the language, the assistant turn is prefilled with
   `language Chinese<asr_text>`; by default it is left empty and the model detects the language per utterance. The
   answer reads `language <Name><asr_text><transcript>`; everything after `<asr_text>` is the transcript.
4. `pyperclip` puts the text on the clipboard, `pynput` sends Ctrl+V (Cmd+V on macOS), and the previous clipboard
   text is restored after a short delay. Typing the characters instead was rejected because Chinese IMEs capture key
   presses into their composition window.

**Stack**: Python 3.12+, `pynput` (pinned: the hotkey code depends on the order it calls the event filter, the
callbacks and the macOS intercept), `sounddevice`, `numpy`, `httpx`, `pyperclip`, `python-dotenv`. Settings come from
`.env` (see `.env.example`), the vocabulary from `vocab.txt` (re-read on every press, so edits need no restart). The
process never exits on a failure: an offline server or a failed request costs one utterance and pastes a one-line
notice instead, so it can run unattended as a scheduled task.

**The model we use, and what can replace it**

- Transcription: [Qwen3-ASR-1.7B](https://huggingface.co/ggml-org/Qwen3-ASR-1.7B-GGUF) (bf16 GGUF plus its mmproj)
  on [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server`. About 4.5 GB of VRAM, 0.1 to 0.5 s per
  utterance on a current desktop GPU. Any server that accepts the OpenAI `input_audio` content part and an assistant-turn
  prefill will do; Ollama does not accept either. Other quantizations of the same model only need the `-hf` argument
  changed. A different ASR model needs its own output parsing in `dictation/asr.py` (the `<asr_text>` split and the
  language prefill are Qwen3-ASR conventions).
## Setup

A llama-server, reachable over HTTP from this machine (it does not have to run on it), then the client.

```
llama-server -hf ggml-org/Qwen3-ASR-1.7B-GGUF --host 127.0.0.1 --port 8080 -ngl 99 -np 1 -c 2048 --no-webui
```

It downloads the model on first use (`brew install llama.cpp` on macOS; on Windows take a CUDA build from the
llama.cpp releases). Then, from the project directory, into the machine's own interpreter:

```
uv pip install -r requirements.txt
cp .env.example .env
cp vocab.example.txt vocab.txt
python dictate.py
```

Hold Ctrl+H to talk, release to paste. Every setting is explained in `.env.example`.

Audio costs about 13 tokens per second, and one request must fit a server slot together with the vocabulary prompt
and the transcript: with 2048 tokens per slot an utterance can run to about 100 s. The client waits at startup until
the ASR server's `/health` answers, and while the server is offline (stopped to free the GPU, say) a press pastes
`[dictation] ASR server offline, not recording`; a failed request pastes `[dictation] transcription failed`.

On Windows, the hotkey and the paste do not reach an elevated (administrator) window. On macOS, the interpreter that
runs `dictate.py` needs Input Monitoring, Accessibility and Microphone. Bluetooth earphones switch to their call
profile when the input stream opens, which silences the first second: press, wait a beat, then speak.

### Background task (Windows)

Runs as the scheduled task `dictation` under `pythonw`, with no console; output goes to `DICTATION_LOG_FILE`. It
starts at logon, on unlock and every hour, and a trigger is ignored while it is already running. Register it once
from PowerShell, with your interpreter and this directory:

```powershell
$action = New-ScheduledTaskAction -Execute "C:\path\to\pythonw.exe" -Argument "`"C:\path\to\dictation\dictate.py`""
# Logon and unlock triggers are bound to the current user: any-user triggers are refused without an elevated shell
$logon = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
$hourly = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Hours 1)
$hourly.Repetition.StopAtDurationEnd = $false
# StateChange 8 is session unlock
$unlock = New-CimInstance -CimClass (Get-CimClass -Namespace ROOT\Microsoft\Windows\TaskScheduler -ClassName MSFT_TaskSessionStateChangeTrigger) -Property @{ StateChange = 8; UserId = $env:USERNAME } -ClientOnly
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
Register-ScheduledTask -TaskName dictation -Action $action -Trigger $logon, $hourly, $unlock -Settings $settings -Principal (New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive)
```

`Start-ScheduledTask dictation` and `Stop-ScheduledTask dictation` control it; restart it after editing `.env`, and
stop it before running `python dictate.py` by hand, otherwise both react to the hotkey. The llama-server can be
registered the same way. To free the GPU for a while, disable the server task rather than ending it, since an
hourly trigger restarts an ended task; the client keeps running and only pastes the offline notice.

## Layout

```
dictate.py             entry: records on the hotkey, transcribes, pastes, logs results and timings
.env.example           every setting with its default and what it does
vocab.example.txt      vocabulary template; copy to vocab.txt (not in git), one term per line
dictation/config.py    .env loading and parsing
dictation/hotkey.py    global chord: press/release edges into a queue, trigger key swallowed
dictation/recorder.py  mono recording from the default microphone
dictation/asr.py       llama-server client: health wait, WAV request, language prefill, transcript parsing
dictation/vocab.py     vocab.txt -> prompt
dictation/paste.py     paste via the clipboard, then restore the previous clipboard text (images and files are lost)
```

## Design decisions

- **Transcription on a shared llama-server, no model in this process.** Over 28 clips of 1-15 s with the same
  vocabulary prompt, the server took 0.19 s (median) against 0.76 s for transformers in-process on the same GPU; 15
  of 28 transcripts were identical and the rest differed by 4.6 % of characters, mostly punctuation, fillers and clip
  edges. It also frees about 4.4 GB of VRAM and starts instantly, and other programs can use the same server.
- **Automatic language detection by default.** Speech mixes Chinese, English and Japanese, and an occasional utterance
  taken as the wrong language is preferred over switching; the trade-off is spelled out in `.env.example`.
- **Clipboard paste instead of typing.** pynput's `type()` sends lowercase letters and digits as real key presses, and
  Microsoft Pinyin in Chinese mode swallows them into its composition window.
- **pynput for the hotkey.** It is the only maintained cross-platform listener with key-up events and left/right
  modifiers: `keyboard` is unmaintained and needs root on macOS, and pyautogui cannot listen and sends Cmd+V without the
  Command flag on macOS.
- **Failures never end the process.** Startup waits for `/health` as long as it takes; afterwards an offline server or
  a failed request only costs that utterance, with a pasted notice, because a scheduled task restarts a dead process no
  sooner than its next trigger. The notice is pasted rather than shown elsewhere because the process has no console.

## Known limitations

- Every transcript lands in the clipboard history and in `DICTATION_LOG_FILE`.
- Whether the Windows keyboard hook keeps working after sleep or lock is untested; the task triggers only restart a
  process that has exited.
- A stray "v" instead of a paste has been reported for pynput's Ctrl+V under Chinese IMEs (CapsWriter-Offline #426);
  not seen here so far.
- Whether a microphone connected after startup is picked up without a restart is unverified.
