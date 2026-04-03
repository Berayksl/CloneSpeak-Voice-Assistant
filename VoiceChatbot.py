import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import time
import queue
import threading
import tempfile
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from collections import deque
from faster_whisper import WhisperModel
from silero_vad import load_silero_vad
from llama_cpp import Llama
from chatterbox.tts_turbo import ChatterboxTurboTTS
from voice_selector_ui import select_voice

# ── Config ────────────────────────────────────────────────────────────────────
# ASR
ASR_MODEL_SIZE  = "medium"
ASR_DEVICE      = "cuda"
ASR_COMPUTE     = "float16"

# Recording / VAD
SAMPLE_RATE     = 16000
CHANNELS        = 1
FRAME_SIZE      = 512
VAD_THRESHOLD   = 0.5
SILENCE_LIMIT   = 20
PRE_BUFFER_SIZE = FRAME_SIZE * 20

# LLM
LLM_MODEL_PATH  = "./models/Llama-3.2-3B-Instruct-Q4_K_M.gguf"
LLM_GPU_LAYERS  = -1
LLM_CONTEXT     = 4096
LLM_MAX_TOKENS  = 256

# TTS
TTS_REF_AUDIO    = r"./clone audio/stevejobs sample.wav"
TTS_EXAGGERATION = 0.5
TTS_CFG_WEIGHT   = 0.5

# Sentence boundary characters — flush TTS at these
SENTENCE_END = {'.', '!', '?'}


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 1 — Voice Input + ASR
# ══════════════════════════════════════════════════════════════════════════════

def load_asr_model():
    print("[ASR] Loading Whisper medium...")
    t0 = time.time()
    model = WhisperModel(ASR_MODEL_SIZE, device=ASR_DEVICE, compute_type=ASR_COMPUTE)
    print(f"[ASR] Model loaded in {time.time()-t0:.2f}s\n")
    return model


def load_vad_model():
    print("[VAD] Loading Silero VAD...")
    model = load_silero_vad()
    print("[VAD] Ready.\n")
    return model


def record_audio(vad_model) -> np.ndarray:
    print("[MIC] 🎙️  Listening... (speak to start)\n")

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=CHANNELS,
        dtype="float32", blocksize=FRAME_SIZE,
    )
    stream.start()

    pre_buffer     = deque(maxlen=PRE_BUFFER_SIZE)
    recorded       = []
    speech_started = False
    silence_frames = 0

    try:
        while True:
            chunk, _ = stream.read(FRAME_SIZE)
            chunk = chunk[:, 0]
            pre_buffer.extend(chunk)

            speech_prob = vad_model(
                torch.from_numpy(chunk).unsqueeze(0), SAMPLE_RATE
            ).item()

            if speech_prob > VAD_THRESHOLD:
                if not speech_started:
                    print("[MIC] 🔴 Recording...")
                    speech_started = True
                    recorded.extend(pre_buffer)
                recorded.extend(chunk)
                silence_frames = 0
            elif speech_started:
                recorded.extend(chunk)
                silence_frames += 1
                if silence_frames > SILENCE_LIMIT:
                    print("[MIC] ✅ Done.\n")
                    break
    except KeyboardInterrupt:
        print("\n[MIC] Interrupted.\n")
    finally:
        stream.stop()
        stream.close()

    audio = np.array(recorded, dtype=np.float32)
    if len(audio) == 0:
        print("[MIC] ⚠️  No speech detected.\n")
    else:
        print(f"[MIC] Recorded {len(audio)/SAMPLE_RATE:.2f}s\n")
    return audio


def transcribe(asr_model: WhisperModel, audio: np.ndarray) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    sf.write(tmp_path, audio, SAMPLE_RATE)

    t0 = time.time()
    segments, _ = asr_model.transcribe(
        tmp_path, language="en", beam_size=5,
        vad_filter=True, vad_parameters=dict(min_silence_duration_ms=500),
    )
    transcript = " ".join(seg.text.strip() for seg in segments)
    elapsed = time.time() - t0
    os.unlink(tmp_path)

    print(f"[ASR] Transcript : {transcript}")
    print(f"[ASR] Time       : {elapsed:.2f}s\n")
    return transcript


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 2 — LLM (streaming)
# ══════════════════════════════════════════════════════════════════════════════

def load_llm():
    print("[LLM] Loading Llama 3.2 3B Q4_K_M...")
    t0 = time.time()
    llm = Llama(
        model_path=LLM_MODEL_PATH,
        n_gpu_layers=LLM_GPU_LAYERS,
        n_ctx=LLM_CONTEXT,
        n_threads=8,
        verbose=False,
    )
    print(f"[LLM] Model loaded in {time.time()-t0:.2f}s\n")
    return llm


def stream_sentences(llm: Llama, question: str, sentence_queue: queue.Queue):
    """
    Stream tokens from LLM and push complete sentences to sentence_queue.
    Runs in a background thread so TTS can start immediately.
    Puts None when done to signal the TTS thread to stop.
    """
    system_prompt = (
        "You are a helpful voice assistant. "
        "Answer the user's question in 2-3 short sentences. "
        "Do not use markdown, bullet points, or any special formatting — "
        "your response will be spoken aloud. Be direct and concise."
    )
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{system_prompt}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{question}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )

    print(f"[LLM] Question : {question}")
    buffer = ""
    total_tokens = 0
    t0 = time.time()

    for token in llm(
        prompt,
        max_tokens=LLM_MAX_TOKENS,
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1,
        stop=["<|eot_id|>", "<|end_of_text|>"],
        stream=True,
    ):
        text = token["choices"][0]["text"]
        buffer += text
        total_tokens += 1

        # flush to TTS at sentence boundaries
        for char in SENTENCE_END:
            if char in buffer:
                idx = buffer.rindex(char)
                sentence = buffer[:idx+1].strip()
                if sentence:
                    print(f"[LLM] → TTS: {sentence}")
                    sentence_queue.put(sentence)
                buffer = buffer[idx+1:]
                break

    # flush any remaining text
    if buffer.strip():
        print(f"[LLM] → TTS: {buffer.strip()}")
        sentence_queue.put(buffer.strip())

    elapsed = time.time() - t0
    print(f"[LLM] Done. {total_tokens} tokens in {elapsed:.2f}s "
          f"({total_tokens/elapsed:.1f} tok/s)\n")
    sentence_queue.put(None)  # signal done


# ══════════════════════════════════════════════════════════════════════════════
# MODULE 3 — TTS (streaming consumer)
# ══════════════════════════════════════════════════════════════════════════════

def load_tts_model():
    print("[TTS] Loading Chatterbox Turbo...")
    t0 = time.time()
    model = ChatterboxTurboTTS.from_pretrained(device="cuda")
    print(f"[TTS] Model loaded in {time.time()-t0:.2f}s\n")
    return model


def speak_streaming(tts_model: ChatterboxTurboTTS, sentence_queue: queue.Queue,
                    ref_audio: str = TTS_REF_AUDIO):
    """
    Consumes sentences from sentence_queue, synthesizes them, and plays them back.
    Playback runs in a separate thread so synthesis of the next sentence starts
    immediately without waiting for the current one to finish playing.
    Returns (total_synth_time, time_to_first_audio) for accurate benchmarking.
    """
    use_clone        = ref_audio and os.path.exists(ref_audio)
    total_synth_time = 0.0
    first_sentence   = True
    time_to_first    = 0.0
    playback_thread  = None
    t_start          = time.time()

    def play(audio_np, sr):
        sd.play(audio_np, sr)
        sd.wait()

    while True:
        sentence = sentence_queue.get()
        if sentence is None:
            break

        t0 = time.time()
        wav = tts_model.generate(
            sentence,
            audio_prompt_path=ref_audio if use_clone else None,
            exaggeration=TTS_EXAGGERATION,
            cfg_weight=TTS_CFG_WEIGHT,
        )
        synth_time = time.time() - t0
        total_synth_time += synth_time

        if first_sentence:
            time_to_first = time.time() - t_start
            print(f"[TTS] ⚡ Time to first audio: {time_to_first:.2f}s")
            first_sentence = False

        audio_np = wav.squeeze().cpu().numpy()
        duration = len(audio_np) / tts_model.sr
        print(f"[TTS] Synth: {synth_time:.2f}s | Audio: {duration:.2f}s | "
              f"RTF: {synth_time/duration:.3f}")

        if playback_thread is not None:
            playback_thread.join()

        playback_thread = threading.Thread(
            target=play, args=(audio_np, tts_model.sr), daemon=True
        )
        playback_thread.start()

    # Record processing end time before waiting for last playback
    t_processing_end = time.time()

    if playback_thread is not None:
        playback_thread.join()

    return total_synth_time, time_to_first, t_processing_end



if __name__ == "__main__":
    ref_audio = select_voice()
    if ref_audio:
        print(f"[TTS] Voice cloning enabled: {os.path.basename(ref_audio)}")
    else:
        print("[TTS] Using default voice.")

    asr_model = load_asr_model()
    vad_model = load_vad_model()
    llm       = load_llm()
    tts_model = load_tts_model()

    print("=" * 50)
    print("🎙️  Voice Q&A System — Ready!")
    print("Speak to ask a question. Ctrl+C to exit.")
    print("=" * 50 + "\n")

    while True:
        try:
            # Module 1: record + transcribe
            audio = record_audio(vad_model)
            if len(audio) == 0:
                continue

            t_start = time.time()
            transcript = transcribe(asr_model, audio)
            t_asr = time.time()

            if not transcript.strip():
                print("[!] Could not understand, please try again.\n")
                continue

            # Modules 2 + 3: LLM streams sentences → TTS plays them in parallel
            sentence_queue = queue.Queue()
            t_llm_start = time.time()

            # LLM runs in background thread
            llm_thread = threading.Thread(
                target=stream_sentences,
                args=(llm, transcript, sentence_queue),
                daemon=True,
            )
            llm_thread.start()

            # TTS runs in main thread (audio playback requires main thread on Windows)
            tts_synth_time, time_to_first, t_processing_end = speak_streaming(tts_model, sentence_queue, ref_audio = ref_audio)
            llm_thread.join()

            #timing summary
            asr_time     = t_asr - t_start
            llm_tts_wall = t_processing_end - t_llm_start
            total_proc   = t_processing_end - t_start
            print("─" * 50)
            print(f"  ASR                  : {asr_time:.2f}s")
            print(f"  LLM+TTS (wall time)  : {llm_tts_wall:.2f}s")
            print(f"  TTS synth total      : {tts_synth_time:.2f}s")
            print(f"  Time to first audio  : {asr_time + time_to_first:.2f}s")
            print(f"  Total processing     : {total_proc:.2f}s  (excl. playback)")
            print("─" * 50 + "\n")

        except KeyboardInterrupt:
            print("\n[System] Shutting down. Goodbye!")
            break