"""
🧠 MODULE 2: ANALYZER
Uses spaCy + lexicon scoring to detect:
  - Dominant emotion (motivation / anxiety / deep / funny / neutral)
  - Emotional intensity (0.0 → 1.0)
  - Keywords for visual search
  - Best hook sentence (highest emotional density in first 3 seconds)
  - Format recommendation
  - Pacing map (which segments are high/low energy)

FIXES:
  - Expanded emotion lexicons with root forms, inflections, and common variants
    (e.g. "success" alone missed "succeed", "succeeded", "successful")
  - Lexicon scoring now uses lemmatized words via spaCy for better matching
  - Added fallback to "motivation" when all scores are zero (prevents wrong defaults)
"""

import re
import spacy
from collections import Counter
from config import CONFIG

# ── Emotion Lexicons (expanded with word variants) ────────────────────────────
#
# ROOT CAUSE OF MISDETECTION:
#   Old lexicon had "success" but transcript contains "successful", "succeed",
#   "succeeded" → zero score for motivation on a motivation speech!
#   Fix: added inflected forms + semantically close words for each category.
#
EMOTION_LEXICONS = {
    "motivation": [
        # Core success words — all inflections
        "dream", "dreams", "dreamer",
        "goal", "goals",
        "achieve", "achieved", "achievement", "achiever",
        "success", "successful", "succeed", "succeeds", "succeeded",
        "win", "wins", "winner", "winning",
        "rise", "rises", "rising",
        "grind", "hustle",
        "believe", "belief", "believed",
        "grow", "grows", "growth", "grew",
        "build", "builds", "built",
        "create", "creates", "created",
        "fight", "fights",
        "never",
        "strong", "strength", "stronger",
        "power", "powerful",
        "vision",
        "focus", "focused",
        "discipline",
        "unstoppable",
        "potential",
        "greatness", "great",
        "purpose",
        "commit", "commitment",
        "earn", "earned",
        "deserve", "deserves",
        # Added for common motivational speeches
        "failure", "failures",       # reframed as fuel → motivation context
        "lesson", "lessons",
        "hard", "work", "worked",
        "keep", "going",
        "define", "defined",
        "teach", "teaches", "taught",
        "legend", "legends",
        "born",
        "become", "becomes",
        "overcome", "overcame",
        "inspire", "inspired", "inspiration",
        "persist", "persistence",
        "resilience", "resilient",
        "try", "tries", "tried",
        "improve", "improved",
        "learn", "learns", "learned",
    ],
    "anxiety": [
        "fear", "fears",
        "panic",
        "worry", "worries", "worried",
        "stress", "stressed",
        "anxious", "anxiety",
        "nervous",
        "overwhelm", "overwhelmed",
        "scared", "scare",
        "doubt", "doubts",
        "fail",                        # raw fail (not failure-as-lesson)
        "lost",
        "dark", "darkness",
        "alone", "lonely",
        "tired", "exhausted",
        "broken",
        "stuck",
        "trap", "trapped",
        "suffocate", "suffocating",
        "spiral",
        "collapse", "collapsed",
        "dread",
        "terror", "terrified",
        "numb", "empty", "hollow",
        "pressure",
        "helpless",
        "hopeless",
        "shame", "ashamed",
    ],
    "deep": [
        "truth", "truths",
        "reality",
        "exist", "existence",
        "meaning",
        "purpose",
        "soul",
        "conscious", "consciousness",
        "universe",
        "human", "humanity",
        "time",
        "moment", "moments",
        "death",
        "life",
        "wonder",
        "silence",
        "void",
        "infinite", "infinity",
        "illusion",
        "perception",
        "mind",
        "aware", "awareness",
        "feeling", "feelings",
        "subconscious",
        "thought", "thoughts",
        "identity",
        "understand", "understood", "understanding",
        "question", "questions",
        "answer", "answers",
        "wisdom",
        "reflection",
        "journey",
        "choice", "choices",
        "change",
        "perspective",
    ],
    "funny": [
        "literally", "actually", "basically",
        "absolutely", "seriously",
        "ridiculous", "insane", "crazy", "wild",
        "brain",
        "wait", "okay",
        "imagine",
        "except",
        "plot", "twist",
        "nobody", "everyone",
        "always", "never",
        "why", "huh",
        "lol",
        "funny", "joke",
        "ironic", "irony",
        "awkward",
        "random",
        "relatable",
        "same",
        "mood",
        "vibe",
    ],
}

# ── Format Heuristics ─────────────────────────────────────────────────────────
FORMAT_RULES = {
    "anxiety": "brain_simulation",
    "motivation": "cinematic_trailer",
    "deep": "deep_arc",
    "funny": "dialogue_mode",
    "neutral": "cinematic_trailer",
}

# ── Intensity Amplifiers ──────────────────────────────────────────────────────
INTENSITY_WORDS = [
    "never", "always", "every", "entire", "completely", "literally",
    "absolutely", "extremely", "impossible", "devastating", "incredible",
    "insane", "wildly", "deeply", "profoundly", "desperately",
    "hundreds", "thousands", "over",           # added for scale words
]


def analyze_script(transcript: dict) -> dict:
    """Full analysis pipeline. Returns emotion, intensity, keywords, format."""
    nlp = _load_spacy()
    text = transcript["full_text"]
    segments = transcript["segments"]

    # 1. Emotion scoring — now uses lemmatized tokens for better matching
    emotion_scores = _score_emotions_with_lemma(nlp, text)

    # 2. Fallback: if all scores are 0 or tied, check sentence structure
    dominant_emotion = _pick_dominant_emotion(emotion_scores, text)

    # 3. Intensity score (0.0 → 1.0)
    intensity = _compute_intensity(text, emotion_scores)

    # 4. Keyword extraction via spaCy
    keywords = _extract_keywords(nlp, text)

    # 5. Hook detection
    hook_segment = _detect_hook_segment(segments, emotion_scores, dominant_emotion)

    # 6. Pacing map
    pacing_map = _build_pacing_map(segments)

    # 7. Format recommendation
    format_choice = FORMAT_RULES.get(dominant_emotion, "cinematic_trailer")

    # 8. Sentence-level emotion breakdown
    sentence_emotions = _analyze_sentences(nlp, text)

    return {
        "dominant_emotion": dominant_emotion,
        "emotion_scores": emotion_scores,
        "intensity": intensity,
        "keywords": keywords[:8],
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


def _score_emotions_with_lemma(nlp, text: str) -> dict:
    """
    Score emotions using both raw words AND lemmatized tokens.
    This fixes the core bug where "successful" didn't match "success".
    """
    doc = nlp(text)

    # Build a frequency counter of both raw lower words AND lemmas
    word_tokens = set()
    for token in doc:
        word_tokens.add(token.text.lower())
        word_tokens.add(token.lemma_.lower())

    # Also do raw word frequency (for per-occurrence weighting)
    raw_words = re.findall(r'\b\w+\b', text.lower())
    word_freq = Counter(raw_words)

    # Add lemma frequencies
    for token in doc:
        lemma = token.lemma_.lower()
        if lemma != token.text.lower():
            word_freq[lemma] = word_freq.get(lemma, 0) + 1

    scores = {}
    for emotion, lexicon in EMOTION_LEXICONS.items():
        score = sum(word_freq.get(w, 0) for w in lexicon)
        scores[emotion] = score / max(len(lexicon), 1)

    return scores


def _pick_dominant_emotion(emotion_scores: dict, text: str) -> str:
    """
    Pick the dominant emotion with smart fallback logic.
    If all scores are very low (< 0.01), use heuristics on the text.
    """
    max_score = max(emotion_scores.values()) if emotion_scores else 0
    dominant = max(emotion_scores, key=emotion_scores.get)

    # If the winning margin is very thin or all near-zero, apply heuristics
    if max_score < 0.01:
        text_lower = text.lower()
        # Simple keyword presence check for common speech types
        if any(w in text_lower for w in ["succeed", "success", "failure teaches", "hard work", "never give"]):
            return "motivation"
        if any(w in text_lower for w in ["anxiety", "panic", "fear", "dark", "alone"]):
            return "anxiety"
        if any(w in text_lower for w in ["universe", "consciousness", "existence", "meaning of"]):
            return "deep"
        return "motivation"  # default for speech content

    return dominant


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

    for ent in doc.ents:
        if ent.label_ in ("PERSON", "ORG", "GPE", "NORP", "LOC", "PRODUCT", "EVENT"):
            keywords.append(ent.text.lower())

    for token in doc:
        if token.pos_ in ("NOUN", "PROPN") and not token.is_stop and len(token.text) > 3:
            keywords.append(token.lemma_.lower())
        if token.pos_ == "ADJ" and not token.is_stop:
            if token.i + 1 < len(doc) and doc[token.i + 1].pos_ in ("NOUN", "PROPN"):
                keywords.append(f"{token.text} {doc[token.i+1].text}".lower())

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
        wpm = (len(words) / duration) * 60

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


def _analyze_sentences(nlp, text: str) -> list:
    """Per-sentence emotion tags for dynamic style switching mid-video."""
    sentences = re.split(r'[.!?]+', text)
    result = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        scores = _score_emotions_with_lemma(nlp, sent)
        dominant = max(scores, key=scores.get)
        result.append({"sentence": sent, "emotion": dominant, "scores": scores})
    return result


# ── Backward-compatible wrapper for old raw-word scoring ─────────────────────
def _score_emotions(text: str) -> dict:
    """Legacy raw-word scorer kept for any external callers."""
    words = re.findall(r'\b\w+\b', text.lower())
    word_freq = Counter(words)
    scores = {}
    for emotion, lexicon in EMOTION_LEXICONS.items():
        score = sum(word_freq.get(w, 0) for w in lexicon)
        scores[emotion] = score / max(len(lexicon), 1)
    return scores