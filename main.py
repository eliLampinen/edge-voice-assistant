import os
import time
import wave
import winsound
import subprocess
from pathlib import Path

import sounddevice as sd
import soundfile as sf
from openai import OpenAI
from piper import PiperVoice

from config import (
    OPENAI_API_KEY,
    WHISPER_EXE,
    WHISPER_MODEL,
    PIPER_MODEL,
    WORKDIR,
    PIPER_PLAYBACK_WAV,
    SAMPLE_RATE,
    RECORD_SECONDS,
    SESSION_TIMEOUT,
    SYSTEM_PROMPT,
)

# Initialize OpenAI client
client = OpenAI()

# Session handling
last_input_time = 0.0
previous_response_id = None

# Load Piper once
voice = PiperVoice.load(str(PIPER_MODEL))


# ----------------------------
# Audio capture
# ----------------------------
def record_wav(out_path: Path, duration_seconds: int = RECORD_SECONDS) -> Path:
    """Record audio from microphone and save to WAV file."""
    print("\nSpeak now...")
    audio = sd.rec(
        int(duration_seconds * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    sf.write(str(out_path), audio, SAMPLE_RATE)
    return out_path


# ----------------------------
# Whisper.cpp transcription
# ----------------------------
def transcribe_with_whisper(wav_path: Path) -> str:
    """Transcribe audio using Whisper.cpp."""
    ts = int(time.time())
    out_prefix = WORKDIR / f"whisper_{ts}"

    cmd = [
        str(WHISPER_EXE),
        "-m", str(WHISPER_MODEL),
        "-f", str(wav_path),
        "-l", "fi",
        "-otxt",
        "-of", str(out_prefix),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"Whisper failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )

    txt_path = out_prefix.with_suffix(".txt")
    if not txt_path.exists():
        raise FileNotFoundError(f"Whisper did not create text file: {txt_path}")

    text = txt_path.read_text(encoding="utf-8").strip()
    return text


# ----------------------------
# OpenAI response
# ----------------------------
def ask_gpt54(user_text: str) -> str:
    """Get response from OpenAI GPT-5.4 with optional session context."""
    global last_input_time, previous_response_id

    now = time.time()
    if now - last_input_time > SESSION_TIMEOUT:
        previous_response_id = None
        print("[New session]")

    last_input_time = now

    try:
        kwargs = {
            "model": "gpt-5.4",
            "instructions": SYSTEM_PROMPT,
            "input": user_text,
            "tools": [{"type": "web_search"}],
            "tool_choice": "auto",
        }

        # Continue conversation if within session timeout
        if previous_response_id is not None:
            kwargs["previous_response_id"] = previous_response_id

        response = client.responses.create(**kwargs)
        previous_response_id = response.id

        answer = (response.output_text or "").strip()
        if not answer:
            return "No response received."

        return answer

    except Exception as e:
        return f"[API error] {e}"


# ----------------------------
# TTS with Piper
# ----------------------------
def speak_text(text: str) -> None:
    """Synthesize and play text using Piper TTS."""
    if not text.strip():
        return

    with wave.open(str(PIPER_PLAYBACK_WAV), "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)

    winsound.PlaySound(str(PIPER_PLAYBACK_WAV), winsound.SND_FILENAME)


# ----------------------------
# Main loop
# ----------------------------
def main() -> None:
    """Main application loop."""
    print("Ready. Press Enter, speak for 6 seconds, and the machine responds.")
    print("Exit by typing q and Enter.\n")

    while True:
        cmd = input("Enter = speak | q = exit: ").strip().lower()
        if cmd == "q":
            break

        try:
            wav_path = WORKDIR / "input.wav"
            record_wav(wav_path, RECORD_SECONDS)

            user_text = transcribe_with_whisper(wav_path)
            if not user_text:
                print("Whisper did not hear anything.")
                continue

            print(f"You: {user_text}")

            answer = ask_gpt54(user_text)
            print(f"Chat: {answer}")

            speak_text(answer)

        except Exception as e:
            print(f"[Error] {e}")


if __name__ == "__main__":
    main()
