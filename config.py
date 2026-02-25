"""
⚙️ CONFIG — All global settings for the Reel Engine
Edit this file to customize output behavior.
"""
from dotenv import load_dotenv
import os

load_dotenv()

CONFIG = {
    # ── Video Settings ──────────────────────────────────────────────
    "video": {
        "width": 1080,
        "height": 1920,
        "fps": 30,
        "format": "mp4",
        "codec": "libx264",
        "audio_codec": "aac",
        "bitrate": "8000k",
        "master_volume": 1.2,        # Boost audio by 20% by default
    },

    # ── Whisper Settings ────────────────────────────────────────────
    "whisper": {
        "model": "base",          # tiny / base / small / medium / large
        "language": "en",
    },

    # ── Emotion → Visual Style Mapping ──────────────────────────────
    "emotion_styles": {
        "motivation": {
            "bg_color": (10, 10, 30),        # Deep navy
            "text_color": (255, 215, 0),      # Gold
            "accent_color": (255, 140, 0),
            "effect": "slow_zoom",
            "cut_speed": "medium",
            "has_cinematic_bars": True,
            "subtitle_style": "bold_center",
            "sfx": "cinematic_rise",
        },
        "anxiety": {
            "bg_color": (20, 0, 0),
            "text_color": (255, 60, 60),
            "accent_color": (200, 0, 0),
            "effect": "glitch",
            "cut_speed": "fast",
            "has_cinematic_bars": False,
            "subtitle_style": "shake",
            "sfx": "heartbeat",
        },
        "deep": {
            "bg_color": (5, 5, 15),
            "text_color": (180, 180, 255),
            "accent_color": (100, 100, 200),
            "effect": "subtle_drift",
            "cut_speed": "slow",
            "has_cinematic_bars": True,
            "subtitle_style": "fade_center",
            "sfx": "ambient_low",
        },
        "funny": {
            "bg_color": (20, 20, 20),
            "text_color": (255, 255, 255),
            "accent_color": (255, 220, 0),
            "effect": "meme_zoom",
            "cut_speed": "fast",
            "has_cinematic_bars": False,
            "subtitle_style": "pop",
            "sfx": "comedic_sting",
        },
        "neutral": {
            "bg_color": (15, 15, 25),
            "text_color": (220, 220, 255),
            "accent_color": (150, 150, 200),
            "effect": "slow_zoom",
            "cut_speed": "medium",
            "has_cinematic_bars": True,
            "subtitle_style": "bold_center",
            "sfx": None,
        },
    },

    # ── Format Templates ────────────────────────────────────────────
    "formats": {
        "cinematic_trailer": "Cinematic mini-trailer with dramatic reveals",
        "thought_vs_reality": "Split-screen: what you think vs what's real",
        "brain_simulation": "Simulates a thought being processed",
        "deep_arc": "Emotional arc storytelling with visual metaphors",
        "dialogue_mode": "You vs Your Brain internal dialogue",
    },

    # ── Subtitle Settings ────────────────────────────────────────────
    "subtitles": {
        "font_size_normal": 52,
        "font_size_hook": 80,
        "font": "assets/fonts/Montserrat-Bold.ttf",  # fallback to default
        "padding": 40,
        "words_per_chunk": 4,       # karaoke style: 4 words at a time
        "hook_words": 3,               # first line = 3 power words max
        "max_width_ratio": 0.88,       # subtitle can use 88% of frame width
    },

    # ── Pattern Interruption ─────────────────────────────────────────
    "retention": {
        "hook_duration_sec": 2.5,
        "fast_cut_window_sec": 5,
        "pattern_interrupt_interval_sec": 3.5,
        "min_clip_duration_sec": 1.2,
        "max_clip_duration_sec": 4.0,
    },

    # ── APIs (Free Tiers) ────────────────────────────────────────────
    "apis": {
        # Gemini: free tier at https://aistudio.google.com/
        "gemini_api_key": os.getenv("GEMINI_API_KEY"),
        "gemini_model": "gemini-2.5-flash",   # Free tier model

        # Pexels: free API at https://www.pexels.com/api/
        "pexels_api_key": os.getenv("PEXELS_API_KEY"),

        # Unsplash: free API at https://unsplash.com/developers
        "unsplash_api_key": os.getenv("UNSPLASH_API_KEY"),

        # Pixabay: free at https://pixabay.com/api/docs/
        "pixabay_api_key": os.getenv("PIXABAY_API_KEY"),
    },
}
