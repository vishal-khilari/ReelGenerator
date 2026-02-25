# 🎬 AI REEL ENGINE
### Voice In → Instagram Reel Out. Fully Automated.

---

## 🏗️ SYSTEM ARCHITECTURE

```
YOUR VOICE (.mp3/.wav)
        │
        ▼
┌─────────────────┐
│  1. TRANSCRIBER │  Whisper → word-level timestamps
└────────┬────────┘
         │ transcript.json
         ▼
┌─────────────────┐
│  2. ANALYZER    │  spaCy + lexicon → emotion, keywords, pacing, hook
└────────┬────────┘
         │ analysis.json
         ▼
┌──────────────────────┐
│  3. SCRIPT ENHANCER  │  Gemini API → rewrites hook, buildup, punchline
└────────┬─────────────┘
         │ enhanced_script.json
         ▼
┌──────────────────┐
│  4. VISUAL       │  Pexels / Pixabay / Unsplash → vertical images
│     FETCHER      │  Fallback: NumPy procedural dark backgrounds
└────────┬─────────┘
         │ visuals_raw/
         ▼
┌──────────────────┐
│  5. EFFECTS      │  OpenCV → glitch / slow_zoom / drift / meme_zoom
│     ENGINE       │  Renders per-frame JPEGs for each clip
└────────┬─────────┘
         │ frames/
         ▼
┌──────────────────────────────────────────┐
│  6. VIDEO ASSEMBLER (MoviePy + PIL)       │
│     • Builds video timeline              │
│     • Cross-dissolve transitions         │
│     • Karaoke subtitle overlay           │
│     • Hook text slam (first 2.5s)        │
│     • Composites all layers              │
└────────┬─────────────────────────────────┘
         │ raw_assembled.mp4
         ▼
┌──────────────────┐
│  7. EXPORTER     │  ffmpeg → H.264 + AAC + loudnorm + faststart
└────────┬─────────┘
         │
         ▼
  output/{name}_reel.mp4
  1080×1920 | 30fps | 9:16 | Instagram Ready ✅
```

---

## 📁 FOLDER STRUCTURE

```
reel_engine/
├── main.py                    ← Entry point. Run this.
├── config.py                  ← All settings (API keys, style config)
├── modules/
│   ├── transcriber.py         ← Whisper audio → timestamped transcript
│   ├── analyzer.py            ← Emotion detection + keyword extraction
│   ├── script_enhancer.py     ← Gemini API script rewriting
│   ├── visual_fetcher.py      ← Image APIs + procedural backgrounds
│   ├── effects_engine.py      ← OpenCV cinematic effects
│   ├── video_assembler.py     ← MoviePy assembly + subtitles
│   └── exporter.py            ← ffmpeg final export
├── assets/
│   ├── fonts/                 ← Place Montserrat-Bold.ttf here
│   ├── overlays/              ← Optional PNG overlays
│   └── sfx/                   ← Optional sound effects
├── input/                     ← Drop your voice files here
└── output/                    ← Final reels saved here
    └── {session_name}/        ← Per-session working files
        ├── transcript.json
        ├── analysis.json
        ├── enhanced_script.json
        ├── visuals_raw/
        ├── frames/
        └── raw_assembled.mp4
```

---

## ⚡ QUICK START

### 1. Install dependencies
```bash
pip install openai-whisper moviepy==1.0.3 spacy opencv-python pillow numpy requests
python -m spacy download en_core_web_sm
```

### 2. Add API keys to config.py
```python
"apis": {
    "gemini_api_key": "YOUR_KEY",     # https://aistudio.google.com (FREE)
    "pexels_api_key": "YOUR_KEY",     # https://pexels.com/api (FREE)
    "pixabay_api_key": "YOUR_KEY",    # https://pixabay.com/api/docs (FREE)
}
```

### 3. (Optional) Add a better font
Download Montserrat-Bold.ttf from Google Fonts and place it in `assets/fonts/`.

### 4. Run the pipeline
```bash
python main.py input/my_voice.mp3
```

That's it. Find your reel in `output/my_voice_reel.mp4`

---

## 🧠 EMOTION DETECTION LOGIC

The system scores each emotion using weighted keyword lexicons:

```
raw_score(emotion) = Σ(keyword_frequency) / lexicon_size

intensity = raw_score_max + (amplifier_words × 0.05) + (ALL_CAPS_words × 0.05)
intensity = clamp(0.0, 1.0)
```

**Detected Emotions and Their Visual Style:**

| Emotion    | Effect      | Color         | Subtitle Style | Use Case             |
|------------|-------------|---------------|----------------|----------------------|
| motivation | slow_zoom   | Gold on Navy  | Bold Center    | Growth, success talk |
| anxiety    | glitch      | Red on Black  | Shake          | Mental health, fear  |
| deep       | subtle_drift| Blue on Black | Fade Center    | Philosophy, truth    |
| funny      | meme_zoom   | White on Dark | Pop            | Comedy, irony        |
| neutral    | slow_zoom   | White on Dark | Bold Center    | General content      |

---

## 🎯 RETENTION ALGORITHM

```
HOOK SELECTION:
  hook_segment = argmax(Σ emotion_keywords / word_count) for each segment

STRUCTURE:
  - 0.0s → 2.5s:  Hook (strongest sentence) + BIG text slam
  - 2.5s → 70%:   Build-up (Gemini rewritten for pacing)
  - 70% → end:    Punchline + final visual

PATTERN INTERRUPTION:
  Every 3.5 seconds → visual cut + optional style shift

CUT SPEED BY EMOTION:
  anxiety  → 1.5s clips  (fast cuts = nervous energy)
  funny    → 1.5s clips  (fast cuts = comedic timing)
  neutral  → 2.5s clips  (medium pace)
  deep     → 3.5s clips  (slow cuts = weight, contemplation)
  motivation → 2.5s clips
```

---

## 🎬 CONTENT FORMATS

The system auto-selects format based on emotion:

| Format               | Trigger Emotion | What It Looks Like                          |
|----------------------|-----------------|---------------------------------------------|
| cinematic_trailer    | motivation      | Slow burns, dramatic reveals, gold text     |
| brain_simulation     | anxiety         | Glitchy cuts, red tones, fragmented text    |
| deep_arc             | deep            | Dark backgrounds, drifting visuals          |
| dialogue_mode        | funny           | Rapid cuts, pop text, meme zoom             |

---

## 💸 ZERO COST BREAKDOWN

| Tool          | Cost   | What It Does               |
|---------------|--------|----------------------------|
| Whisper       | FREE   | Transcription (runs local) |
| spaCy         | FREE   | NLP / keyword extraction   |
| OpenCV        | FREE   | All video effects          |
| MoviePy       | FREE   | Video assembly             |
| ffmpeg        | FREE   | Export encoding            |
| PIL/Pillow    | FREE   | Subtitle rendering         |
| NumPy         | FREE   | Procedural backgrounds     |
| Pexels API    | FREE   | 200 req/hr image fetching  |
| Pixabay API   | FREE   | Unlimited image fetching   |
| Unsplash API  | FREE   | 50 req/hr image fetching   |
| Gemini Flash  | FREE   | 15 RPM / 1M TPM free tier  |

**Total: $0/month** (within free tier limits)

---

## 🔧 CUSTOMIZATION TIPS

**Change Whisper quality:**
```python
# config.py
"whisper": {"model": "medium"}   # tiny→base→small→medium→large
```

**Add your own emotion lexicon:**
```python
# analyzer.py → EMOTION_LEXICONS
"sadness": ["loss", "miss", "gone", "empty", ...]
```

**Change subtitle style:**
```python
# config.py
"subtitles": {
    "words_per_subtitle": 3,     # 3 words at a time
    "font_size_normal": 60,      # bigger subs
}
```

**Force a specific format:**
```python
# In main.py, override analysis:
analysis["format"] = "brain_simulation"
```
