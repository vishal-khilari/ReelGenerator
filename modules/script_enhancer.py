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
6. Keep each field SHORT: hook ≤ 10 words, hook_visual_note ≤ 20 words, buildup ≤ 60 words, punchline ≤ 30 words, buildup_visual_note ≤ 15 words, punchline_visual_note ≤ 15 words. Be concise.

IMPORTANT: Respond ONLY with raw JSON. Do NOT wrap in markdown or code blocks. Do NOT add any explanation before or after. Start your response directly with {{ and end with }}

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
            "temperature": 0.6,
            "maxOutputTokens": 8192,  # Increased from 2048 — old value caused truncated JSON
            # NOTE: stopSequences removed — it was causing Gemini to stop
            # immediately when it tried to wrap output in ```json blocks,
            # resulting in an empty response with finishReason: STOP.
        },
    }

    def _call_gemini():
        resp = requests.post(url, headers=headers, params=params, json=payload, timeout=30)
        resp.raise_for_status()
        response_json = resp.json()

        # Surface real API errors (invalid key, quota exceeded, etc.)
        if "error" in response_json:
            raise ValueError(f"Gemini API error: {response_json['error']}")

        candidates = response_json.get("candidates", [])
        if not candidates:
            raise ValueError(f"Gemini returned no candidates. Full response: {response_json}")

        candidate = candidates[0]

        # Check for safety filters or other blocking reasons
        finish_reason = candidate.get("finishReason", "")
        if finish_reason not in ("STOP", "MAX_TOKENS", ""):
            raise ValueError(
                f"Gemini candidate blocked. finishReason: {finish_reason}. "
                f"Full candidate: {candidate}"
            )

        parts = candidate.get("content", {}).get("parts", [])
        if not parts:
            raise ValueError(
                f"Gemini returned empty parts. This usually means stopSequences "
                f"triggered immediately or a silent safety block. "
                f"Full candidate: {candidate}"
            )

        return parts[0]["text"]

    # ===== FIRST ATTEMPT =====
    raw = _call_gemini()

    # ===== DEBUG RAW OUTPUT =====
    print("\n===== GEMINI RAW RESPONSE =====")
    print(raw)
    print("================================\n")

    # ===== CLEAN MARKDOWN (safety net in case model still wraps) =====
    clean = re.sub(r"```json|```", "", raw).strip()

    # ===== EXTRACT JSON SAFELY =====
    start = clean.find("{")
    end = clean.rfind("}")

    if start == -1 or end == -1:
        print("⚠️  Gemini returned no JSON object. Retrying once...")
        raw = _call_gemini()
        clean = re.sub(r"```json|```", "", raw).strip()
        start = clean.find("{")
        end = clean.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(
                f"Gemini returned invalid JSON after retry. Raw output:\n{raw}"
            )

    json_str = clean[start:end + 1]

    # ===== PARSE JSON =====
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as e:
        print("⚠️  JSON parsing failed. Raw Gemini output:")
        print(raw)
        raise e

    # Ensure all required fields exist with fallbacks
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
        "hook_visual_note": "dark atmospheric background, text slam animation",
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