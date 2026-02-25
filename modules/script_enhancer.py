"""
✍️ MODULE 3: SCRIPT ENHANCER
Uses Gemini (free tier) to rewrite and optimize the script for maximum
Instagram retention. Rewrites the hook, adds a punchline, reorders
for emotional arc, and generates scene direction hints.

Falls back to rule-based enhancement if no API key is set.
"""

import json
import re
import requests
from config import CONFIG


GEMINI_PROMPT_TEMPLATE = """You are a viral Instagram Reels scriptwriter. 
Your job is to transform a raw voice transcript into a psychologically optimized script for a 
faceless creator. The reel must make people STOP scrolling in the first 2 seconds.

TRANSCRIPT:
{transcript}

DETECTED EMOTION: {emotion}
DETECTED INTENSITY: {intensity}/1.0
FORMAT: {format}

Your task:
1. Write a HOOK (max 8 words) that creates immediate curiosity or emotional punch. Make it feel surprising.
2. Rewrite the transcript into 3 parts: HOOK → BUILD-UP → PUNCHLINE
3. For each part, write a [SCENE NOTE] describing what visual should appear
4. Keep total script under the original duration
5. Make it feel human, NOT like AI

Respond in this EXACT JSON format:
{{
  "hook": "...",
  "hook_visual_note": "...",
  "buildup": "...",
  "buildup_visual_note": "...",
  "punchline": "...",
  "punchline_visual_note": "...",
  "full_rewritten": "...",
  "subtitle_emphasis_words": ["word1", "word2", "word3"],
  "search_query_visuals": ["query1", "query2", "query3"]
}}"""


def enhance_script(transcript: dict, analysis: dict) -> dict:
    """
    Enhance script using Gemini API.
    Falls back to rule-based enhancement if API key missing.
    """
    api_key = CONFIG["apis"]["gemini_api_key"]

    if api_key and api_key != "YOUR_GEMINI_API_KEY":
        try:
            return _enhance_with_gemini(transcript, analysis, api_key)
        except Exception as e:
            print(f"      ⚠️  Gemini failed ({e}), using rule-based fallback...")

    return _rule_based_enhancement(transcript, analysis)


def _enhance_with_gemini(transcript: dict, analysis: dict, api_key: str) -> dict:
    """Call Gemini 1.5 Flash (free tier) for script optimization."""
    prompt = GEMINI_PROMPT_TEMPLATE.format(
        transcript=transcript["full_text"],
        emotion=analysis["dominant_emotion"],
        intensity=analysis["intensity"],
        format=analysis["format"],
    )

    url = f"https://generativelanguage.googleapis.com/v1/models/{CONFIG['apis']['gemini_model']}:generateContent"
    headers = {"Content-Type": "application/json"}
    params = {"key": api_key}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.8,
            "maxOutputTokens": 1024,
        },
    }

    resp = requests.post(url, headers=headers, params=params, json=payload, timeout=30)
    resp.raise_for_status()
    raw = resp.json()["candidates"][0]["content"]["parts"][0]["text"]

    # Parse JSON from response (strip markdown fences if present)
    json_str = re.sub(r"```json|```", "", raw).strip()
    data = json.loads(json_str)

    # Ensure all required fields exist
    return _normalize_enhanced(data, transcript, analysis)


def _rule_based_enhancement(transcript: dict, analysis: dict) -> dict:
    """
    Rule-based fallback when no Gemini key is available.
    Uses the detected hook segment + pacing map to restructure.
    """
    segments = transcript["segments"]
    hook_seg = analysis["hook_segment"]
    emotion = analysis["dominant_emotion"]

    # Find the hook sentence
    hook_text = hook_seg["text"] if hook_seg else segments[0]["text"]

    # Remaining sentences (excluding hook)
    remaining = [s["text"] for s in segments if s.get("id") != hook_seg.get("id")]

    # Split into buildup (first 60%) and punchline (last 40%)
    split = max(1, int(len(remaining) * 0.6))
    buildup = " ".join(remaining[:split])
    punchline = " ".join(remaining[split:]) or hook_text

    # Add emotion-specific hook prefix
    hook_prefixes = {
        "motivation": "Nobody tells you this about",
        "anxiety": "This is what your brain does when",
        "deep": "The thing nobody admits is",
        "funny": "My brain at 2am:",
        "neutral": "Here's what I figured out about",
    }
    prefix = hook_prefixes.get(emotion, "")
    hook = f"{prefix} {hook_text[:50]}..." if prefix else hook_text[:60]

    # Emphasis words = all-caps words + EMOTION_LEXICONS top words
    words = transcript["full_text"].split()
    emphasis = [w for w in words if w.isupper() and len(w) > 3][:5]

    return {
        "hook": hook,
        "hook_visual_note": f"dark atmospheric background, text slam animation",
        "buildup": buildup,
        "buildup_visual_note": f"{emotion} themed visuals, slow motion",
        "punchline": punchline,
        "punchline_visual_note": "zoom in, high contrast, dramatic pause",
        "full_rewritten": f"{hook}\n\n{buildup}\n\n{punchline}",
        "subtitle_emphasis_words": emphasis,
        "search_query_visuals": analysis["keywords"][:3],
        "source": "rule_based",
    }


def _normalize_enhanced(data: dict, transcript: dict, analysis: dict) -> dict:
    """Ensure all fields exist with fallbacks."""
    defaults = {
        "hook": transcript["segments"][0]["text"][:60] if transcript["segments"] else "",
        "hook_visual_note": "atmospheric dark background",
        "buildup": transcript["full_text"],
        "buildup_visual_note": "relevant visuals",
        "punchline": "",
        "punchline_visual_note": "dramatic close-up",
        "full_rewritten": transcript["full_text"],
        "subtitle_emphasis_words": [],
        "search_query_visuals": analysis["keywords"][:3],
        "source": "gemini",
    }
    for k, v in defaults.items():
        if k not in data or not data[k]:
            data[k] = v
    return data
