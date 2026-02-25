"""
🎙️ MODULE 1: TRANSCRIBER
Uses OpenAI Whisper to transcribe audio into word-level timestamped segments.
Word-level timestamps are critical for karaoke subtitle sync.
"""

import whisper
import json
from pathlib import Path
from config import CONFIG


def transcribe_audio(audio_path: str) -> dict:
    """
    Transcribe audio with word-level timestamps.
    Returns structured transcript with segments and full text.
    """
    model_name = CONFIG["whisper"]["model"]
    print(f"      Loading Whisper model: {model_name}")
    model = whisper.load_model(model_name)

    # word_timestamps=True gives us per-word timing for karaoke subs
    result = model.transcribe(
        audio_path,
        language=CONFIG["whisper"]["language"],
        word_timestamps=True,
        verbose=False,
    )

    # Build clean segment list
    segments = []
    for seg in result["segments"]:
        words = []
        if "words" in seg:
            for w in seg["words"]:
                words.append({
                    "word": w["word"].strip(),
                    "start": round(w["start"], 3),
                    "end": round(w["end"], 3),
                })
        segments.append({
            "id": seg["id"],
            "start": round(seg["start"], 3),
            "end": round(seg["end"], 3),
            "text": seg["text"].strip(),
            "words": words,
        })

    # Compute total duration from last segment
    duration = segments[-1]["end"] if segments else 0

    return {
        "full_text": result["text"].strip(),
        "segments": segments,
        "duration": duration,
        "language": result.get("language", "en"),
        "word_count": sum(len(s["words"]) for s in segments),
    }


def get_word_chunks(transcript: dict, words_per_chunk: int = 4) -> list:
    """
    Regroup words into karaoke-style display chunks.
    Each chunk has a start/end time and text to display.

    Used by the subtitle animator to display 4 words at a time,
    synced precisely to audio timing.
    """
    all_words = []
    for seg in transcript["segments"]:
        all_words.extend(seg["words"])

    chunks = []
    for i in range(0, len(all_words), words_per_chunk):
        group = all_words[i:i + words_per_chunk]
        if not group:
            continue
        chunks.append({
            "text": " ".join(w["word"] for w in group),
            "start": group[0]["start"],
            "end": group[-1]["end"],
            "is_hook": i == 0,  # Mark first chunk as hook
        })
    return chunks
