# 🎙️ VoiceAssistant — Real-Time AI Voice Q&A System

A fully local, GPU-accelerated voice assistant pipeline built for low-latency, real-time interaction. Ask questions out loud — get spoken answers back, in a cloned voice.

**Pipeline:** `Microphone → VAD → ASR → LLM → TTS → Speaker`

---

## ✨ Features

- 🎤 **Automatic speech detection** via Silero VAD (no push-to-talk)
- 🧠 **State-of-the-art ASR** with faster-whisper (Whisper Medium, GPU-accelerated)
- 💬 **Local LLM inference** via llama.cpp (Llama 3.2 3B, Q4_K_M quantization)
- 🔊 **Voice cloning TTS** with Chatterbox Turbo (zero-shot voice cloning from a 5-sec clip)
- ⚡ **Streaming pipeline** — TTS starts speaking sentence-by-sentence while LLM is still generating
- 🖥️ **Fully local** — no cloud APIs, no internet required after setup

---

## 🏗️ System Architecture

```
┌─────────────┐    ┌──────────────────┐    ┌─────────────────────┐    ┌────────────────────┐
│  Microphone │───▶│  Silero VAD      │───▶│  Whisper Medium     │───▶│  Llama 3.2 3B      │
│             │    │  (speech detect) │    │  (faster-whisper)   │    │  (llama.cpp CUDA)  │
└─────────────┘    └──────────────────┘    └─────────────────────┘    └────────────────────┘
                                                                                │
                                                                                ▼ (streamed, sentence-by-sentence)
                                                                       ┌────────────────────┐    ┌─────────┐
                                                                       │  Chatterbox Turbo  │───▶│ Speaker │
                                                                       │  (voice cloning)   │    │         │
                                                                       └────────────────────┘    └─────────┘
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.10+
- NVIDIA GPU with 6+ GB VRAM (tested on 8 GB)
- CUDA 12.x + cuDNN
- [Git](https://git-scm.com/) and [Conda](https://docs.conda.io/) (recommended) or venv

### 1. Clone the repository

```bash
git clone https://github.com/Berayksl/AI-Voice-Assistant.git
cd voice-assistant
```

### 2. Create a virtual environment

```bash
conda create -n voice-assistant python=3.10 -y
conda activate voice-assistant
```

### 3. Install PyTorch with CUDA

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 4. Install llama-cpp-python with CUDA support

This must be done **before** the rest of the requirements, with the CUDA flag:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
```

> **Windows users:** Use PowerShell and set the env variable differently:
> ```powershell
> $env:CMAKE_ARGS="-DGGML_CUDA=on"
> pip install llama-cpp-python --upgrade --force-reinstall --no-cache-dir
> ```

### 5. Install remaining dependencies

```bash
pip install -r requirements.txt
```

### 6. Install ffmpeg (required by pydub for audio export)

```bash
conda install ffmpeg -c conda-forge
```

---

## 📥 Downloading Models

### LLM — Llama 3.2 3B (Q4_K_M)

Download from Hugging Face and place in `models/`:

```bash
pip install huggingface_hub
huggingface-cli download bartowski/Llama-3.2-3B-Instruct-GGUF \
  --include "Llama-3.2-3B-Instruct-Q4_K_M.gguf" \
  --local-dir ./models
```

### ASR — Whisper Medium

Downloaded automatically by faster-whisper on first run. No action needed.

### TTS — Chatterbox Turbo

Downloaded automatically from Hugging Face on first run (`ResembleAI/chatterbox-turbo`). No action needed.

---

## 🚀 Running the Voice Assistant

```bash
python VoiceChatBot.py
```

- Speak after the `🎙️ Listening...` prompt appears
- The assistant will transcribe, think, and respond in a cloned voice
- Press `Ctrl+C` to exit

### Optional: set your voice cloning reference

Place a clean 5–30 second WAV recording of the target voice in `clone_audio/` and update the path in `pipeline.py`:

```python
TTS_REF_AUDIO = "./clone_audio/your_voice_sample.wav"
```

---

## 🧪 Testing Individual Modules

```bash
# Test ASR alone (benchmarks small vs. medium on test audio files)
python asr_test.py

# Test LLM alone (measures tokens/sec)
python llm_test.py

# Test TTS + voice cloning
python tts_test.py
```

## 🔧 Configuration

All key parameters are at the top of `pipeline.py`:

| Parameter | Default | Description |
|---|---|---|
| `ASR_MODEL_SIZE` | `"medium"` | Whisper model size (`small`, `medium`, `large`) |
| `VAD_THRESHOLD` | `0.5` | Speech detection sensitivity (0–1) |
| `SILENCE_LIMIT` | `20` | Silence frames before recording stops (~0.6s) |
| `LLM_MAX_TOKENS` | `128` | Max tokens per LLM response |
| `TTS_EXAGGERATION` | `0.5` | Voice expressiveness (0 = neutral, 1 = dramatic) |
| `TTS_REF_AUDIO` | `./clone_audio/...` | Path to voice cloning reference audio |

---

## 🧩 Tech Stack

| Component | Model / Library |
|---|---|
| ASR | [Whisper Medium](https://github.com/openai/whisper) via [faster-whisper](https://github.com/SYSTRAN/faster-whisper) |
| VAD | [Silero VAD](https://github.com/snakers4/silero-vad) |
| LLM | [Llama 3.2 3B Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) via [llama.cpp](https://github.com/ggerganov/llama.cpp) |
| TTS | [Chatterbox Turbo](https://github.com/resemble-ai/chatterbox) by Resemble AI |
| Audio I/O | sounddevice, soundfile, pydub |

---

## 🙏 Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper)
- [SYSTRAN faster-whisper](https://github.com/SYSTRAN/faster-whisper)
- [ggerganov llama.cpp](https://github.com/ggerganov/llama.cpp)
- [Resemble AI Chatterbox](https://github.com/resemble-ai/chatterbox)
- [Silero VAD](https://github.com/snakers4/silero-vad)
