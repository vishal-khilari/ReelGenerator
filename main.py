"""
🎬 AI REEL ENGINE — Main Pipeline
Voice In → Instagram Reel Out. Fully Automated.
"""

import sys
import os
import json
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.transcriber import transcribe_audio
from modules.analyzer import analyze_script
from modules.script_enhancer import enhance_script
from modules.visual_planner import plan_visuals
from modules.visual_fetcher import fetch_visuals
from modules.effects_engine import apply_effects_to_visuals
from modules.video_assembler import assemble_reel
from modules.exporter import export_final_reel
from config import CONFIG

def run_pipeline(audio_path: str):
    print("\n" + "="*60)
    print("🎬  AI REEL ENGINE — Starting Pipeline")
    print("="*60)

    audio_path = Path(audio_path)
    if not audio_path.exists():
        print(f"❌ Audio file not found: {audio_path}")
        sys.exit(1)

    stem = audio_path.stem
    session_dir = Path("output") / stem
    session_dir.mkdir(parents=True, exist_ok=True)

    # ─── STEP 1: TRANSCRIBE ────────────────────────────────────────
    print("\n[1/7] 🎙️  Transcribing audio with Whisper...")
    transcript = transcribe_audio(str(audio_path))
    _save(session_dir / "transcript.json", transcript)
    print(f"    ✅ {len(transcript['segments'])} segments detected")

    # ─── STEP 2: ANALYZE ──────────────────────────────────────────
    print("\n[2/7] 🧠  Analyzing emotion, keywords, pacing...")
    analysis = analyze_script(transcript)
    _save(session_dir / "analysis.json", analysis)
    print(f"    ✅ Dominant emotion: {analysis['dominant_emotion'].upper()}")
    print(f"    ✅ Format selected: {analysis['format']}")
    print(f"    ✅ Intensity score: {analysis['intensity']:.2f}")

    # ─── STEP 3: ENHANCE SCRIPT ───────────────────────────────────
    print("\n[3/7] ✍️  Enhancing script with AI (hook + pacing)...")
    enhanced = enhance_script(transcript, analysis)
    _save(session_dir / "enhanced_script.json", enhanced)
    print(f"    ✅ Hook: \"{enhanced['hook'][:60]}...\"")

    # ─── STEP 3b: PLAN VISUALS ────────────────────────────────────
    print("\n[3b/7] 🗓️  Planning visual structure with Gemini...")
    visual_plan = plan_visuals(transcript, analysis, enhanced)
    _save(session_dir / "visual_plan.json", visual_plan)
    print(f"    ✅ {len(visual_plan)} clips planned "
          f"({sum(c['duration_sec'] for c in visual_plan):.1f}s total)")

    # ─── STEP 4: FETCH VISUALS ────────────────────────────────────
    print("\n[4/7] 🖼️  Fetching visuals for each planned clip slot...")
    visuals = fetch_visuals(visual_plan, analysis['dominant_emotion'], session_dir)
    print(f"    ✅ {len(visuals)} visuals collected")

    # ─── STEP 5: APPLY EFFECTS ────────────────────────────────────
    print("\n[5/7] ⚡  Applying per-clip effects...")
    processed_visuals = apply_effects_to_visuals(
        visuals, analysis['dominant_emotion'], analysis['format'],
        session_dir, visual_plan=visual_plan
    )
    print(f"    ✅ Effects applied: {analysis['dominant_emotion']} style")

    # ─── STEP 6: ASSEMBLE VIDEO ───────────────────────────────────
    print("\n[6/7] 🎬  Assembling reel with subtitles + transitions...")
    raw_video = assemble_reel(
        audio_path=str(audio_path),
        visuals=processed_visuals,
        enhanced_script=enhanced,
        transcript=transcript,
        analysis=analysis,
        session_dir=session_dir
    )
    print(f"    ✅ Raw reel assembled")

    # ─── STEP 7: EXPORT ───────────────────────────────────────────
    print("\n[7/7] 📤  Exporting final 1080x1920 Instagram Reel...")
    final_path = export_final_reel(raw_video, session_dir, stem)
    print(f"\n{'='*60}")
    print(f"🚀  REEL READY: {final_path}")
    print(f"{'='*60}\n")
    return final_path


def _save(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Reel Engine")
    parser.add_argument("audio", help="Path to your voice audio file (.mp3 or .wav)")
    args = parser.parse_args()
    run_pipeline(args.audio)