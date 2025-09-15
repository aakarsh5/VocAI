# asr.py
from faster_whisper import WhisperModel

model = WhisperModel("small", device="cpu")  # "tiny" for faster, less accurate

def transcribe_file(wav_path: str) -> str:
    segments, _ = model.transcribe(wav_path)
    text = " ".join([seg.text for seg in segments])
    return text.strip()
