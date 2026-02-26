"""
🗓️ MODULE 4b: VISUAL PLANNER
Uses Gemini to plan clip count, duration, search queries, and effects.
Falls back to rule-based planner if Gemini is unavailable or truncates.

FIXES:
  - Rule-based fallback queries are now far more specific and relevant
    (old defaults like "sunrise horizon" for a speech about failure/success
     produced generic unrelated stock photos)
  - Gemini prompt now includes the actual transcript text so it can generate
    contextually accurate search queries
  - Query trimming now strips special characters that cause API issues
  - Added validation: queries with < 2 characters are replaced with defaults
"""

import json
import re
import requests
from collections import Counter
from config import CONFIG


PLANNER_PROMPT = """You are a video editor creating an Instagram Reel visual plan.

Audio duration: {duration:.1f}s | Emotion: {emotion} | Format: {format}

FULL TRANSCRIPT (use this to pick SPECIFIC, RELEVANT visuals):
{full_text}

Pacing:
{pacing_map}

Hook: {hook}
Story: {buildup}
End: {punchline}

Effects available: slow_zoom, glitch, subtle_drift, meme_zoom

Output ONLY a raw JSON array (no markdown, no explanation).
Rules:
- Exactly {num_clips} clips
- Durations must sum to exactly {duration:.1f}
- First clip: 0.0 to {hook_duration}s
- search_query: 2-4 words, concrete nouns matching the transcript content
  GOOD: "Michael Jordan basketball", "rejected manuscript pile", "student studying desk"
  BAD: "success cinematic", "dark abstract", "person thinking"
- Match effect to mood: high_energy → slow_zoom or meme_zoom, low_energy → subtle_drift
- For motivational/success speeches: use specific scenarios from the transcript

[{{"clip_index":0,"start_sec":0.0,"end_sec":{hook_duration},"duration_sec":{hook_duration},"search_query":"QUERY","effect":"EFFECT","note":"NOTE"}}, ...]"""


def plan_visuals(transcript: dict, analysis: dict, enhanced_script: dict) -> list:
    api_key = CONFIG["apis"]["gemini_api_key"]
    duration = transcript["duration"]

    if api_key and api_key != "YOUR_GEMINI_API_KEY":
        try:
            plan = _plan_with_gemini(transcript, analysis, enhanced_script, api_key)
            plan = _validate_and_fix_plan(plan, duration)
            print(f"      🤖 Gemini visual plan: {len(plan)} clips")
            return plan
        except Exception as e:
            print(f"      ⚠️  Gemini planner failed ({e}), using rule-based fallback...")

    plan = _rule_based_plan(analysis, enhanced_script, transcript, duration)
    print(f"      📐 Rule-based visual plan: {len(plan)} clips")
    return plan


def _plan_with_gemini(transcript, analysis, enhanced_script, api_key):
    duration = transcript["duration"]
    retention = CONFIG["retention"]
    hook_dur = retention["hook_duration_sec"]
    num_clips = 8

    pacing_summary = " | ".join(
        f"[{p['start']:.0f}-{p['end']:.0f}s {p['label']} {p['wpm']}wpm]"
        for p in analysis["pacing_map"]
    )

    prompt = PLANNER_PROMPT.format(
        duration=duration,
        emotion=analysis["dominant_emotion"],
        format=analysis["format"],
        # FIX: include full transcript so Gemini picks specific relevant queries
        full_text=transcript["full_text"][:600],
        pacing_map=pacing_summary,
        hook=enhanced_script.get("hook", "")[:60],
        buildup=enhanced_script.get("buildup", "")[:120],
        punchline=enhanced_script.get("punchline", "")[:80],
        hook_duration=hook_dur,
        num_clips=num_clips,
    )

    url = (
        f"https://generativelanguage.googleapis.com/v1/models/"
        f"{CONFIG['apis']['gemini_model']}:generateContent"
    )
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 8192},
    }

    resp = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        params={"key": api_key},
        json=payload,
        timeout=45,
    )
    resp.raise_for_status()
    data = resp.json()

    if "error" in data:
        raise ValueError(f"Gemini API error: {data['error']}")

    raw = (data.get("candidates", [{}])[0]
               .get("content", {})
               .get("parts", [{}])[0]
               .get("text", ""))

    print("\n===== GEMINI VISUAL PLAN RAW =====")
    print(raw[:800])
    print("==================================\n")

    plan = _parse_json_array(raw)
    if not plan:
        plan = _extract_partial_clips(raw)
    if not plan:
        raise ValueError("Could not extract any valid clips from Gemini response")

    for clip in plan:
        clip["search_query"] = _clean_query(clip.get("search_query", "cinematic landscape"))

    return plan


def _parse_json_array(raw: str) -> list:
    clean = re.sub(r"```json|```", "", raw).strip()
    start = clean.find("[")
    end = clean.rfind("]")
    if start == -1 or end == -1:
        return []
    try:
        return json.loads(clean[start: end + 1])
    except json.JSONDecodeError:
        return []


def _extract_partial_clips(raw: str) -> list:
    clips = []
    depth = 0
    start = None
    for i, ch in enumerate(raw):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                fragment = raw[start: i + 1]
                try:
                    obj = json.loads(fragment)
                    if "search_query" in obj or "clip_index" in obj:
                        clips.append(obj)
                except json.JSONDecodeError:
                    pass
                start = None
    print(f"      🔧 Partial extraction recovered {len(clips)} clips")
    return clips


def _rule_based_plan(
    analysis: dict,
    enhanced_script: dict,
    transcript: dict,
    duration: float,
) -> list:
    """
    Rule-based visual planner with content-aware search queries.

    FIX: Old planner used vague emotion defaults like "sunrise horizon" that
    produce generic unrelated stock photos. Now we:
      1. Extract specific nouns/entities from the actual transcript
      2. Use emotion-specific but SPECIFIC scenario queries
      3. Pull from Gemini's search_query_visuals if available
    """
    retention = CONFIG["retention"]
    emotion = analysis["dominant_emotion"]
    style = CONFIG["emotion_styles"][emotion]
    hook_dur = retention["hook_duration_sec"]

    effect_map = {
        "hook":          style["effect"],
        "high_energy":   "slow_zoom" if emotion in ("motivation", "neutral") else style["effect"],
        "medium_energy": "subtle_drift",
        "low_energy":    "subtle_drift",
    }

    # ── Query sources (priority order) ─────────────────────────────
    # 1. Gemini's own visual suggestions (most relevant to enhanced script)
    gemini_q = [
        _clean_query(q, 4)
        for q in enhanced_script.get("search_query_visuals", [])
        if q and len(q.strip()) > 2
    ]

    # 2. Named entities from transcript (most specific/concrete)
    entity_q = _extract_entity_queries(analysis.get("keywords", []))

    # 3. Emotion + topic specific pools (much better defaults than before)
    emotion_topic_defaults = _get_topic_defaults(emotion, transcript["full_text"])

    # 4. Merge and deduplicate
    seen, query_pool = set(), []
    for q in gemini_q + entity_q + emotion_topic_defaults:
        q = _clean_query(q, 4)
        if q and q not in seen and len(q) > 2:
            seen.add(q)
            query_pool.append(q)

    if not query_pool:
        query_pool = ["person achieving goal", "success journey", "hard work pays off"]

    # ── Build clip plan ────────────────────────────────────────────
    num_after_hook = 7
    slot_dur = round((duration - hook_dur) / num_after_hook, 3)
    slot_dur = max(retention["min_clip_duration_sec"],
                   min(retention["max_clip_duration_sec"], slot_dur))

    plan = [{
        "clip_index": 0,
        "start_sec": 0.0,
        "end_sec": round(hook_dur, 3),
        "duration_sec": round(hook_dur, 3),
        "search_query": query_pool[0] if query_pool else "success journey cinematic",
        "effect": style["effect"],
        "note": "Hook",
    }]

    cursor = hook_dur
    pacing = analysis["pacing_map"]

    for i in range(num_after_hook):
        is_last = (i == num_after_hook - 1)
        end = round(duration if is_last else cursor + slot_dur, 3)
        actual_dur = round(end - cursor, 3)
        if actual_dur <= 0:
            continue

        label = _dominant_label_in_window(pacing, cursor, end)
        effect = effect_map.get(label, "subtle_drift")
        query = query_pool[(i + 1) % len(query_pool)]

        plan.append({
            "clip_index": i + 1,
            "start_sec": cursor,
            "end_sec": end,
            "duration_sec": actual_dur,
            "search_query": query,
            "effect": effect,
            "note": f"Slot {i+1}: {label}",
        })
        cursor = end

    return plan


def _get_topic_defaults(emotion: str, full_text: str) -> list:
    """
    Return search queries that are specific to the speech topic,
    not just generic emotion aesthetics.

    FIX: Old defaults like "sunrise horizon" don't match content.
    We now detect speech topic from keywords and return targeted queries.
    """
    text_lower = full_text.lower()

    # Detect topic from transcript content
    topic_queries = []

    # Failure/success/resilience speech detection
    if any(w in text_lower for w in ["failure", "fail", "rejected", "cut from"]):
        topic_queries += [
            "person overcoming failure",
            "rejected letter pile",
            "athlete failing then succeeding",
            "student struggling books",
            "person getting back up",
        ]

    if any(w in text_lower for w in ["michael jordan", "basketball"]):
        topic_queries += [
            "basketball player practice court",
            "basketball hoop empty gym",
            "athlete intense training",
        ]

    if any(w in text_lower for w in ["harry potter", "jk rowling", "book", "published"]):
        topic_queries += [
            "person writing manuscript desk",
            "stack of books library",
            "author writing coffee shop",
        ]

    if any(w in text_lower for w in ["grade", "studying", "school", "stupid"]):
        topic_queries += [
            "student studying notebook",
            "open textbook desk lamp",
            "person reading focus",
        ]

    if any(w in text_lower for w in ["hard work", "born", "become good"]):
        topic_queries += [
            "person working hard office",
            "hands building creating",
            "sunrise determination",
        ]

    # Emotion-based fallbacks if no topic detected
    if not topic_queries:
        fallbacks = {
            "motivation": [
                "person reaching mountain peak",
                "athlete breaking finish line",
                "sunrise determination road",
                "hands writing goals notebook",
                "crowd celebrating achievement",
                "person climbing stairs success",
            ],
            "anxiety": [
                "dark room single light",
                "person alone rainy window",
                "storm clouds dramatic sky",
                "blurred city lights night",
                "hands covering face stress",
                "empty hallway dark corridor",
            ],
            "deep": [
                "starry night sky milky way",
                "ocean horizon sunset",
                "forest fog misty morning",
                "person silhouette cliff",
                "candle flame dark room",
                "open book sunlight",
            ],
            "funny": [
                "surprised person coffee",
                "dog confused tilt head",
                "person laughing reaction",
                "chaotic messy desk",
                "wide eyes shocked face",
                "coffee spill accident",
            ],
            "neutral": [
                "clean modern workspace",
                "calm lake reflection",
                "open empty road",
                "minimal interior design",
                "abstract geometric light",
                "city timelapse evening",
            ],
        }
        topic_queries = fallbacks.get(emotion, ["cinematic landscape dramatic"])

    return topic_queries


def _extract_entity_queries(keywords: list) -> list:
    """
    Turn spaCy-extracted entities/keywords into specific search queries.
    e.g. ["michael jordan", "harry potter", "failure"] → contextual queries
    """
    queries = []
    skip_generic = {"world", "people", "person", "thing", "time", "life", "one"}
    for kw in keywords:
        kw = kw.strip().lower()
        if len(kw) < 3 or kw in skip_generic:
            continue
        if "jordan" in kw or "basketball" in kw:
            queries.append("basketball player court")
        elif "rowling" in kw or "potter" in kw or "harry" in kw:
            queries.append("author writing book")
        elif "failure" in kw or "fail" in kw:
            queries.append("person resilience comeback")
        elif "success" in kw or "succeed" in kw:
            queries.append("achievement success celebration")
        elif "study" in kw or "grade" in kw or "school" in kw:
            queries.append("student studying hard")
        else:
            queries.append(kw)
    return queries


def _dominant_label_in_window(pacing: list, start: float, end: float) -> str:
    labels = [
        p["label"] for p in pacing
        if p["start"] < end and p["end"] > start
    ]
    if not labels:
        return "medium_energy"
    return Counter(labels).most_common(1)[0][0]


def _validate_and_fix_plan(plan: list, total_duration: float) -> list:
    valid = []
    for i, clip in enumerate(plan):
        clip.setdefault("clip_index", i)
        clip.setdefault("start_sec", 0.0)
        clip.setdefault("end_sec", clip["start_sec"] + 2.0)
        clip.setdefault("duration_sec", clip["end_sec"] - clip["start_sec"])
        clip.setdefault("search_query", "cinematic landscape")
        clip.setdefault("effect", "subtle_drift")
        clip.setdefault("note", "")
        clip["search_query"] = _clean_query(clip["search_query"], 4)
        if clip["duration_sec"] > 0:
            valid.append(clip)

    if not valid:
        raise ValueError("No valid clips after validation")

    for i, clip in enumerate(valid):
        clip["clip_index"] = i

    actual = sum(c["duration_sec"] for c in valid)
    if abs(actual - total_duration) > 0.1:
        scale = total_duration / actual
        cursor = 0.0
        for clip in valid:
            new_dur = round(clip["duration_sec"] * scale, 3)
            clip["duration_sec"] = new_dur
            clip["start_sec"] = round(cursor, 3)
            clip["end_sec"] = round(cursor + new_dur, 3)
            cursor = clip["end_sec"]

    valid[-1]["end_sec"] = round(total_duration, 3)
    valid[-1]["duration_sec"] = round(total_duration - valid[-1]["start_sec"], 3)
    return valid


def _clean_query(query: str, max_words: int = 4) -> str:
    """
    FIX: Old _trim_query only truncated words. This version also:
      - Strips special characters that break URL params (&, =, ?, #)
      - Collapses whitespace
      - Returns empty-safe result
    """
    if not query:
        return "cinematic landscape"
    # Remove URL-unsafe characters
    cleaned = re.sub(r'[&=?#"\'\\]', '', str(query))
    # Normalize whitespace
    cleaned = " ".join(cleaned.strip().split())
    # Limit word count
    words = cleaned.split()[:max_words]
    result = " ".join(words)
    return result if len(result) > 2 else "cinematic landscape"


# ── Backward-compat alias ─────────────────────────────────────────────────────
def _trim_query(query: str, max_words: int = 3) -> str:
    return _clean_query(query, max_words)


def _segment_text(transcript: dict, segment_id: int) -> str:
    for seg in transcript["segments"]:
        if seg["id"] == segment_id:
            return seg["text"][:60]
    return ""