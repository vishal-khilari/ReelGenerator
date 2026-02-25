"""
🧠 MODULE 2: ANALYZER
Uses spaCy + lexicon scoring to detect:
  - Dominant emotion (motivation / anxiety / deep / funny / neutral)
  - Emotional intensity (0.0 → 1.0)
  - Keywords for visual search
  - Best hook sentence (highest emotional density in first 3 seconds)
  - Format recommendation
  - Pacing map (which segments are high/low energy)
"""

import re
import spacy
from collections import Counter
from config import CONFIG

# ── Emotion Lexicons (expandable) ────────────────────────────────────────────
EMOTION_LEXICONS = {
    "motivation": [
        "dream", "goal", "achieve", "success", "win", "rise", "grind", "hustle",
        "believe", "grow", "build", "create", "fight", "never", "give", "up",
        "strong", "power", "vision", "focus", "discipline", "unstoppable",
        "potential", "greatness", "purpose", "commit", "earn", "deserve",
    ],
    "anxiety": [
        "fear", "panic", "worry", "stress", "anxious", "nervous", "overwhelm",
        "scared", "doubt", "fail", "lost", "dark", "alone", "tired", "broken",
        "stuck", "trap", "suffocate", "spiral", "collapse", "dread", "terror",
        "exhausted", "numb", "empty", "hollow", "pressure",
    ],
    "deep": [
        "truth", "reality", "exist", "meaning", "purpose", "soul", "conscious",
        "universe", "human", "time", "moment", "death", "life", "wonder",
        "silence", "void", "infinite", "illusion", "perception", "mind",
        "aware", "feeling", "subconscious", "thought", "identity",
    ],
    "funny": [
        "literally", "actually", "basically", "absolutely", "seriously",
        "ridiculous", "insane", "crazy", "wild", "brain", "wait", "okay",
        "imagine", "except", "plot", "twist", "nobody", "everyone", "always",
        "never", "why", "huh", "lol", "funny", "joke", "ironic",
    ],
}

# ── Format Heuristics ─────────────────────────────────────────────────────────
FORMAT_RULES = {
    "anxiety": "brain_simulation",           # glitchy internal monologue
    "motivation": "cinematic_trailer",       # slow burn + powerful reveal
    "deep": "deep_arc",                      # philosophical emotional journey
    "funny": "dialogue_mode",               # you vs your brain comedy
    "neutral": "cinematic_trailer",
}

# ── Intensity Amplifiers ──────────────────────────────────────────────────────
INTENSITY_WORDS = [
    "never", "always", "every", "entire", "completely", "literally",
    "absolutely", "extremely", "impossible", "devastating", "incredible",
    "insane", "wildly", "deeply", "profoundly", "desperately",
]


def analyze_script(transcript: dict) -> dict:
    """
    Full analysis pipeline. Returns emotion, intensity, keywords, format.
    """
    nlp = _load_spacy()
    text = transcript["full_text"]
    segments = transcript["segments"]

    # 1. Emotion scoring
    emotion_scores = _score_emotions(text)
    dominant_emotion = max(emotion_scores, key=emotion_scores.get)

    # 2. Intensity score (0.0 → 1.0)
    intensity = _compute_intensity(text, emotion_scores)

    # 3. Keyword extraction via spaCy
    keywords = _extract_keywords(nlp, text)

    # 4. Hook detection — highest emotion density sentence near start
    hook_segment = _detect_hook_segment(segments, emotion_scores, dominant_emotion)

    # 5. Pacing map — label each segment as high/low energy
    pacing_map = _build_pacing_map(segments)

    # 6. Format recommendation
    format_choice = FORMAT_RULES.get(dominant_emotion, "cinematic_trailer")

    # 7. Sentence-level emotion breakdown for dynamic styling
    sentence_emotions = _analyze_sentences(text)

    return {
        "dominant_emotion": dominant_emotion,
        "emotion_scores": emotion_scores,
        "intensity": intensity,
        "keywords": keywords[:8],          # Top 8 for visual search
        "hook_segment": hook_segment,
        "pacing_map": pacing_map,
        "format": format_choice,
        "sentence_emotions": sentence_emotions,
        "style": CONFIG["emotion_styles"][dominant_emotion],
        "total_segments": len(segments),
        "duration": transcript["duration"],
    }


def _load_spacy():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        import subprocess
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")


def _score_emotions(text: str) -> dict:
    """Score each emotion category by keyword presence + weighting."""
    words = re.findall(r'\b\w+\b', text.lower())
    word_freq = Counter(words)
    scores = {}
    for emotion, lexicon in EMOTION_LEXICONS.items():
        score = sum(word_freq.get(w, 0) for w in lexicon)
        # Normalize by lexicon size so smaller lexicons aren't penalized
        scores[emotion] = score / max(len(lexicon), 1)
    return scores


def _compute_intensity(text: str, emotion_scores: dict) -> float:
    """
    Intensity = (max emotion score) × (amplifier count) × (capitalization ratio)
    Clamped to 0.0–1.0
    """
    words = text.lower().split()
    amplifier_count = sum(1 for w in words if w in INTENSITY_WORDS)
    caps_words = [w for w in text.split() if w.isupper() and len(w) > 2]

    base = max(emotion_scores.values()) if emotion_scores else 0
    amplifier_boost = min(amplifier_count * 0.05, 0.3)
    caps_boost = min(len(caps_words) * 0.05, 0.2)

    raw = base + amplifier_boost + caps_boost
    return round(min(raw, 1.0), 3)


def _extract_keywords(nlp, text: str) -> list:
    """
    Extract concrete, visual keywords using spaCy POS tagging.
    Prioritize NOUNs and ADJs that map to searchable images.
    """
    doc = nlp(text)
    keywords = []

    # Named entities first (most visually concrete)
    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG", "GPE", "NORP", "LOC", "PRODUCT", "EVENT"):
            keywords.append(ent.text.lower())

    # Key nouns and adjective-noun pairs
    for token in doc:
        if token.pos_ in ("NOUN", "PROPN") and not token.is_stop and len(token.text) > 3:
            keywords.append(token.lemma_.lower())
        if token.pos_ == "ADJ" and not token.is_stop:
            # Look for following noun
            if token.i + 1 < len(doc) and doc[token.i + 1].pos_ in ("NOUN", "PROPN"):
                keywords.append(f"{token.text} {doc[token.i+1].text}".lower())

    # Deduplicate preserving order
    seen = set()
    unique = []
    for k in keywords:
        if k not in seen and len(k) > 2:
            seen.add(k)
            unique.append(k)

    return unique


def _detect_hook_segment(segments: list, emotion_scores: dict, dominant_emotion: str) -> dict:
    """
    Find the single most emotionally dense sentence.
    This will be MOVED to the front of the reel as the HOOK.
    """
    lexicon = EMOTION_LEXICONS.get(dominant_emotion, [])
    best_seg = segments[0] if segments else None
    best_score = 0

    for seg in segments:
        words = seg["text"].lower().split()
        score = sum(1 for w in words if w in lexicon)
        density = score / max(len(words), 1)
        if density > best_score:
            best_score = density
            best_seg = seg

    return best_seg


def _build_pacing_map(segments: list) -> list:
    """
    Label each segment as: hook / high_energy / medium_energy / low_energy
    Used by video assembler to determine cut speed and effect intensity.
    """
    pacing = []
    for i, seg in enumerate(segments):
        words = seg["text"].split()
        duration = max(seg["end"] - seg["start"], 0.1)
        wpm = (len(words) / duration) * 60  # words per minute speaking rate

        if i == 0:
            label = "hook"
        elif wpm > 160:
            label = "high_energy"
        elif wpm > 100:
            label = "medium_energy"
        else:
            label = "low_energy"

        pacing.append({
            "segment_id": i,
            "start": seg["start"],
            "end": seg["end"],
            "label": label,
            "wpm": round(wpm),
        })
    return pacing


def _analyze_sentences(text: str) -> list:
    """Per-sentence emotion tags for dynamic style switching mid-video."""
    sentences = re.split(r'[.!?]+', text)
    result = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        scores = _score_emotions(sent)
        dominant = max(scores, key=scores.get)
        result.append({"sentence": sent, "emotion": dominant, "scores": scores})
    return result
