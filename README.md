# Kakitori

Local push-to-talk dictation for Windows and macOS. Hold a hotkey, speak, release: the sentence is transcribed by a
local speech model, optionally rewritten into written form (digits, unit symbols, fillers removed) by a local chat
model, and pasted into whatever window has focus. Chinese, English and Japanese can be mixed in one sentence.

Everything runs on your own machine. This program holds no model: it records, talks to two HTTP servers, and pastes.

This was vibe coded and is kept deliberately small. The rest of this file is written for the coding assistant you
will hand it to. If you are reading it yourself, all of it is:

1. Run two `llama-server` instances, sizes from the table below.
2. Copy `.env.example` to `.env`, `vocab.example.txt` to `vocab.txt`, and run `python dictate.py`.
3. Hold Ctrl+H, speak, release.

## Choosing models

Take the last row that fits in the memory you can leave to the servers — on a dedicated card, its size less whatever
the desktop already holds:

| Memory to spare | Transcription | Written form | |
|---|---|---|---|
| 2 GB | `Qwen3-ASR-0.6B:Q8_0` | none |
| 4 GB | `Qwen3-ASR-0.6B:Q8_0` | `gemma-4-E2B-it-qat` |
| 6 GB | `Qwen3-ASR-1.7B:Q8_0` | `gemma-4-E2B-it-qat` |
| 7 GB | `Qwen3-ASR-1.7B:Q8_0` | `gemma-4-E4B-it-qat` | (what we run) |
| 11 GB | `Qwen3-ASR-1.7B:Q8_0` | `gemma-4-12B-it-qat` |

Start at the third row if you dictate much Japanese. Transcription needs a server that accepts the OpenAI
`input_audio` content part and an assistant-turn prefill; Ollama accepts neither, and anything other than
[Qwen3-ASR](https://huggingface.co/ggml-org/Qwen3-ASR-1.7B-GGUF) needs its own parsing in `dictation/asr.py`. Written
form is a plain chat completions endpoint: point `DICTATION_NORMALIZER` at anything you already run. It sees every
sentence you dictate, so pick one you trust; Qwen3-ASR transcribes verbatim and ignores formatting instructions,
which is the whole reason for the second model.

## Setup

Two llama-servers, reachable over HTTP from this machine (neither has to run on it), then the client.

```
# transcription, required
llama-server -hf ggml-org/Qwen3-ASR-1.7B-GGUF:Q8_0 --host 127.0.0.1 --port 8080 -ngl 99 -np 1 -c 2048 --no-webui

# written form, optional
llama-server -hf unsloth/gemma-4-E4B-it-qat-GGUF:UD-Q4_K_XL --no-mmproj --host 127.0.0.1 --port 8081 -ngl 99 -np 1 -c 4096 --no-webui
```

Both download their model on first use (`brew install llama.cpp` on macOS; on Windows take a CUDA build from the
llama.cpp releases). Then, from the project directory, into the machine's own interpreter:

```
uv pip install -r requirements.txt
cp .env.example .env
cp vocab.example.txt vocab.txt
python dictate.py
```

Set `DICTATION_NORMALIZER=http://127.0.0.1:8081` in `.env` if you started the second server; leave it blank to paste
transcripts as spoken. Hold Ctrl+H to talk, release to paste. At `-c 2048` an utterance can run to about 100 s; a
press with the server or the microphone unavailable pastes a `[dictation] ...` notice instead of a transcript.

On Windows, the hotkey and the paste do not reach an elevated window. On macOS, the interpreter that runs
`dictate.py` needs Input Monitoring, Accessibility and Microphone. Bluetooth earphones switch to their call profile
when the input stream opens, which silences the first second: press, wait a beat, then speak.

### In the background

On Windows, as the scheduled task `dictation` under `pythonw`, started at logon, on unlock and every hour, with
output to `DICTATION_LOG_FILE`. Register it once from PowerShell, with your interpreter and this directory:

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

`Start-ScheduledTask` and `Stop-ScheduledTask` control it; stop it before running `python dictate.py` by hand,
otherwise both react to the hotkey. The llama-servers can be registered the same way; disable those tasks rather
than ending them, since an hourly trigger restarts an ended task.

On macOS, launchd runs it at login and whenever it exits. Put your interpreter, this directory and the log path into
`launchd/local.dictation.plist`, copy it to `~/Library/LaunchAgents/`, then
`launchctl bootstrap gui/$UID ~/Library/LaunchAgents/local.dictation.plist`. The permissions go to the interpreter
launchd starts, not to a terminal: add that `python3` binary to Accessibility in System Settings. Reload with
`launchctl kickstart -k gui/$UID/local.dictation` after editing `.env` or the plist.

## Measurements

Measured with `-ngl 99 -np 1`, `-c 2048` for transcription and `-c 4096` for written form, on a current desktop GPU
and a current Apple Silicon machine. CUDA is VRAM above the idle desktop, Metal the server process's resident set.

| Model | Quant | CUDA | Metal |
|---|---|---|---|
| Qwen3-ASR-1.7B | bf16 | 5222 MiB | 4893 MiB |
| Qwen3-ASR-1.7B | Q8_0 | 3270 MiB | 2805 MiB |
| Qwen3-ASR-0.6B | bf16 | 2783 MiB | 2199 MiB |
| Qwen3-ASR-0.6B | Q8_0 | 1975 MiB | 1368 MiB |
| gemma-4-12B-it-qat | UD-Q4_K_XL | 7339 MiB | |
| gemma-4-E4B-it-qat | UD-Q4_K_XL | 2977 MiB | 4276 MiB |
| gemma-4-E2B-it-qat | UD-Q4_K_XL | 1651 MiB | 2739 MiB |

Transcription takes 0.06 to 0.18 s for a 5.9 s utterance and 0.26 to 0.98 s for a 36 s one, slowest at 1.7B bf16 and
fastest at 0.6B Q8_0. Context and slots move memory further than the model does: the 1.7B Q8_0 server holds 3270 MiB
at `-np 1 -c 2048` and 4863 MiB at `-np 8 -c 16384`. 1.7B is the largest Qwen3-ASR published as GGUF.

Accuracy over 150 FLEURS test clips per language, language pinned, CER for Chinese and Japanese, WER for English:

| Model | Quant | zh | ja | en |
|---|---|---|---|---|
| Qwen3-ASR-1.7B | bf16 | 7.36 % | 5.76 % | 4.13 % |
| Qwen3-ASR-1.7B | Q8_0 | 7.34 % | 5.68 % | 4.06 % |
| Qwen3-ASR-0.6B | bf16 | 8.07 % | 9.64 % | 5.61 % |
| Qwen3-ASR-0.6B | Q8_0 | 8.06 % | 9.35 % | 5.84 % |

A tenth of a point is noise at this sample size. Read speech in one language is easier than dictation, and
`vocab.txt` has no effect on these numbers.

## How it works

About 650 lines of Python. `pynput` watches for the chord and swallows the trigger key at the OS hook level, so the
focused app never sees Ctrl+H; `sounddevice` records mono 16 kHz while it is held. `asr.py` sends the WAV to
llama-server as one chat completion whose system message is the `vocab.txt` prompt and whose answer reads
`language <Name><asr_text><transcript>`. `normalize.py` optionally sends the transcript on to the second server,
then verifies the rewrite edit by edit with `difflib`: an edit survives only if the same speech could have produced
both forms — a numeral written as the same value in digits, a unit as its symbol, a filler from a fixed list
removed — and anything else is undone on its own, so the rest of the sentence still benefits. The rewrite is a
sequence of steps named in `DICTATION_NORMALIZE_STEPS`, each a prompt sent as its own request on the previous step's
output; `digits` passes the check. `paste.py` puts the result on the clipboard, sends Ctrl+V, and restores what was
there before.

`pynput` is pinned: the hotkey code depends on the order it calls the event filter, the callbacks and the macOS
intercept. `vocab.txt` is re-read on every press. Every setting lives in `.env.example`.

## Notes

- Pasting rather than typing is deliberate: Chinese IMEs swallow synthetic key presses into their composition
  window. `pynput` is the only maintained cross-platform listener with key-up events and left/right modifiers.
- The rewrite is checked in code because the prompt alone did not hold it: the model translated requests, dropped
  half clauses and wrote an approximate range as exact digits. Having it mark numerals for the code to convert, converting
  every numeral in code and having it revert the false ones, and WeTextProcessing all lost to the current design.
- Unit symbols are applied only after Chinese numbers; Japanese and English keep their unit words, and the `UNITS`
  table bounds which symbol a word may become, so a second can never come out as a minute.
- The check lets through a deleted filler that did carry meaning, and keeps half-width punctuation after a digit.
- Every transcript lands in the clipboard history and in `DICTATION_LOG_FILE`. Whether the Windows keyboard hook
  survives sleep is untested. With `DICTATION_NORMALIZER` set and its server down, a press waits 0.5 s first.
