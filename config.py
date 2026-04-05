import os
import platform
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    PROJECT_ROOT_TEMP = Path(__file__).parent
    load_dotenv(PROJECT_ROOT_TEMP / ".env")
except ImportError:
    pass

# ----------------------------
# API Configuration
# ----------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing.")

# ----------------------------
# Paths (relative to project root)
# ----------------------------
PROJECT_ROOT = Path(__file__).parent

WHISPER_DIR = PROJECT_ROOT.parent / "whisper.cpp"

if platform.system().lower() == "windows":
    WHISPER_EXE = WHISPER_DIR / "whisper-cli.exe"
else:
    WHISPER_EXE = WHISPER_DIR / "whisper-cli"

WHISPER_MODEL = PROJECT_ROOT.parent / "whisper.cpp" / "models" / "ggml-small-q5_1.bin" # Adjust model as needed

PIPER_MODEL = PROJECT_ROOT / "models" / "fi_FI-harri-medium.onnx" # Adjust model as needed
SYSTEM_PROMPT_FILE = PROJECT_ROOT / "prompts" / "system_prompt.txt"

WORKDIR = PROJECT_ROOT / "temp"
WORKDIR.mkdir(exist_ok=True)

PIPER_PLAYBACK_WAV = WORKDIR / "reply.wav"

# ----------------------------
# Audio Settings
# ----------------------------
SAMPLE_RATE = 16000
RECORD_SECONDS = 6

# ----------------------------
# Session Handling for invividual conversations
# ----------------------------
SESSION_TIMEOUT = 60

# ----------------------------
# Load system prompt from file
# ----------------------------
def load_system_prompt() -> str:
    """Load system prompt from file."""
    if SYSTEM_PROMPT_FILE.exists():
        return SYSTEM_PROMPT_FILE.read_text(encoding="utf-8").strip()
    else:
        # Fallback prompt if file doesn't exist
        return """
You are a helpful Finnish voice assistant.

Context:
- The user is located in Kittilä, Finland.
"""


SYSTEM_PROMPT = load_system_prompt()