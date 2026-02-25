"""
🎬 MODULE 6: VIDEO ASSEMBLER
The heart of the engine. Combines:
  - Processed visual clips (from effects engine)
  - Audio (original voice)
  - Animated subtitles (karaoke-style, emotion-aware)
  - Hook text overlay (first 2.5s big text slam)
  - Pattern interruptions (every 3.5s)
  - Pacing-aware cut timing

Uses MoviePy for clip assembly and PIL for subtitle frame rendering.
"""

import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from moviepy.editor import (
    VideoFileClip, ImageSequenceClip, AudioFileClip,
    CompositeVideoClip, concatenate_videoclips, ImageClip,
)
import moviepy.editor as mpy
from modules.transcriber import get_word_chunks
from modules.effects_engine import generate_transition_frame
from config import CONFIG

W, H = CONFIG["video"]["width"], CONFIG["video"]["height"]
FPS = CONFIG["video"]["fps"]
RETENTION = CONFIG["retention"]
SUB_CFG = CONFIG["subtitles"]


def assemble_reel(
    audio_path: str,
    visuals: list,
    enhanced_script: dict,
    transcript: dict,
    analysis: dict,
    session_dir: Path,
) -> str:
    """
    Full assembly pipeline. Returns path to raw combined video.
    """
    style = analysis["style"]
    emotion = analysis["dominant_emotion"]
    duration = transcript["duration"]

    # 1. Build video timeline from processed visual clips
    video_clip = _build_video_timeline(visuals, duration, session_dir)

    # 2. Load audio
    audio_clip = AudioFileClip(audio_path)

    # 3. Generate subtitle frames and overlay them
    word_chunks = get_word_chunks(transcript, words_per_chunk=SUB_CFG["words_per_chunk"])
    subtitle_overlay = _build_subtitle_overlay(word_chunks, enhanced_script, style, duration, session_dir)

    # 4. Generate hook overlay (first 2.5 seconds big text)
    hook_overlay = _build_hook_overlay(enhanced_script["hook"], style, session_dir)

    # 5. Composite all layers
    layers = [video_clip, subtitle_overlay]
    if hook_overlay:
        layers.append(hook_overlay)

    final = CompositeVideoClip(layers, size=(W, H))
    final = final.set_audio(audio_clip)
    final = final.set_duration(min(duration, audio_clip.duration))

    # 6. Save raw assembled video
    raw_path = str(session_dir / "raw_assembled.mp4")
    final.write_videofile(
        raw_path,
        fps=FPS,
        codec="libx264",
        audio_codec="aac",
        verbose=False,
        logger=None,
        temp_audiofile=str(session_dir / "temp_audio.m4a"),
    )
    return raw_path


def _build_video_timeline(visuals: list, total_duration: float, session_dir: Path) -> mpy.VideoClip:
    """
    Build a continuous video from processed visual clip frame sequences.
    Adjusts clip durations to fill the total audio duration.
    Adds cross-dissolve transitions between clips.
    """
    clips = []
    n = len(visuals)
    clip_dur = total_duration / max(n, 1)
    clip_dur = np.clip(clip_dur, RETENTION["min_clip_duration_sec"], RETENTION["max_clip_duration_sec"])

    transition_dur = 0.3  # seconds for transition between clips

    for i, vis in enumerate(visuals):
        frames_dir = Path(vis["frames_dir"])
        frame_files = sorted(frames_dir.glob("frame_*.jpg"))

        if not frame_files:
            continue

        # Load frames as numpy array
        frame_paths = [str(f) for f in frame_files]

        # Trim/extend to match clip_dur
        needed = int(clip_dur * FPS)
        while len(frame_paths) < needed:
            frame_paths += frame_paths   # loop frames if too short
        frame_paths = frame_paths[:needed]

        # Write transition frames (cross-dissolve with next clip)
        if i < len(visuals) - 1:
            next_frames_dir = Path(visuals[i+1]["frames_dir"])
            next_frames = sorted(next_frames_dir.glob("frame_*.jpg"))
            if next_frames:
                trans_count = int(transition_dur * FPS)
                trans_dir = session_dir / "frames" / f"trans_{i:03d}"
                trans_dir.mkdir(exist_ok=True)
                _write_transition_frames(
                    frame_paths[-trans_count:],
                    [str(f) for f in next_frames[:trans_count]],
                    trans_dir, trans_count, "dissolve"
                )
                # Replace tail with transition
                frame_paths = frame_paths[:-trans_count]
                frame_paths += [str(f) for f in sorted(trans_dir.glob("trans_*.jpg"))]

        clip = ImageSequenceClip(frame_paths, fps=FPS)
        clips.append(clip)

    if not clips:
        # Fallback: black video
        black = np.zeros((H, W, 3), dtype=np.uint8)
        return ImageClip(black).set_duration(total_duration)

    return concatenate_videoclips(clips, method="compose")


def _write_transition_frames(frames_a: list, frames_b: list, out_dir: Path, n: int, style: str):
    """Write n transition frames blending the last frames of A into the first frames of B."""
    for i in range(n):
        t = i / max(n - 1, 1)
        fa = cv2.imread(frames_a[min(i, len(frames_a)-1)])
        fb = cv2.imread(frames_b[min(i, len(frames_b)-1)])
        if fa is None or fb is None:
            continue
        fa = cv2.resize(fa, (W, H))
        fb = cv2.resize(fb, (W, H))
        trans = generate_transition_frame(fa, fb, t, style)
        cv2.imwrite(str(out_dir / f"trans_{i:05d}.jpg"), trans, [cv2.IMWRITE_JPEG_QUALITY, 85])


def _build_subtitle_overlay(
    word_chunks: list, enhanced_script: dict, style: dict, duration: float, session_dir: Path
) -> mpy.VideoClip:
    """
    Render karaoke-style subtitles synchronized to word timing.
    Each chunk = 4 words displayed at their exact timestamp.
    Emphasis words get a different color/size.
    """
    emphasis_words = set(w.lower() for w in enhanced_script.get("subtitle_emphasis_words", []))
    frames_dir = session_dir / "subtitle_frames"
    frames_dir.mkdir(exist_ok=True)

    text_color = style["text_color"]
    accent_color = style["accent_color"]
    total_frames = int(duration * FPS)

    # Map frame → subtitle text
    subtitle_map = {}  # frame_idx → (text, is_hook, is_emphasis)
    for chunk in word_chunks:
        start_f = int(chunk["start"] * FPS)
        end_f = int(chunk["end"] * FPS)
        for f in range(start_f, min(end_f + 1, total_frames)):
            subtitle_map[f] = {
                "text": chunk["text"],
                "is_hook": chunk["is_hook"],
            }

    # Render subtitle frames
    sub_frame_paths = []
    last_text = None
    last_path = None

    for fi in range(total_frames):
        info = subtitle_map.get(fi)
        text = info["text"] if info else ""
        is_hook = info["is_hook"] if info else False

        # Only re-render if text changed (performance optimization)
        if text != last_text:
            frame_arr = _render_subtitle_frame(
                text, text_color, accent_color, emphasis_words, is_hook, style
            )
            path = str(frames_dir / f"sub_{fi:06d}.png")
            Image.fromarray(frame_arr).save(path, "PNG")
            last_path = path
            last_text = text
        else:
            # Symlink reuse won't work cross-platform; just write same frame
            import shutil
            path = str(frames_dir / f"sub_{fi:06d}.png")
            if last_path:
                shutil.copy2(last_path, path)

        sub_frame_paths.append(path)

    subtitle_clip = ImageSequenceClip(sub_frame_paths, fps=FPS)
    return subtitle_clip


def _render_subtitle_frame(
    text: str, text_color: tuple, accent_color: tuple,
    emphasis_words: set, is_hook: bool, style: dict
) -> np.ndarray:
    """
    Render one subtitle frame (transparent background PNG).
    The subtitle area sits at bottom 30% of frame.
    """
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))  # Transparent

    if not text.strip():
        return np.array(img)

    draw = ImageDraw.Draw(img)

    # Font size
    font_size = SUB_CFG["font_size_hook"] if is_hook else SUB_CFG["font_size_normal"]

    # Try to load custom font, fall back to default
    font = _get_font(font_size)

    # Wrap text if too long
    lines = _wrap_text(text, draw, font, int(W * SUB_CFG["max_width_ratio"]))

    # Position: bottom 25% of frame
    total_text_h = len(lines) * (font_size + 8)
    y_start = H - 380 - total_text_h  # above bottom edge

    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        line_w = bbox[2] - bbox[0]
        x = (W - line_w) // 2

        words_in_line = line.split()
        # Render word-by-word for emphasis highlighting
        curr_x = x
        for word in words_in_line:
            word_bbox = draw.textbbox((0, 0), word + " ", font=font)
            word_w = word_bbox[2] - word_bbox[0]

            is_emphasis = word.lower().strip(".,!?") in emphasis_words
            color = (*accent_color, 255) if is_emphasis else (*text_color, 255)

            # Shadow for legibility
            draw.text((curr_x + 2, y_start + 2), word, font=font, fill=(0, 0, 0, 180))
            draw.text((curr_x, y_start), word, font=font, fill=color)
            curr_x += word_w

        y_start += font_size + 8

    return np.array(img)


def _build_hook_overlay(hook_text: str, style: dict, session_dir: Path) -> mpy.ImageClip:
    """
    First 2.5 seconds: BIG hook text slams onto screen.
    Creates a dramatic attention-grabbing intro overlay.
    """
    hook_dur = RETENTION["hook_duration_sec"]
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Large font for hook
    font = _get_font(SUB_CFG["font_size_hook"] + 20)
    text_color = style["text_color"]
    accent_color = style["accent_color"]

    # Split hook into 2 lines max
    words = hook_text.split()
    if len(words) <= 3:
        lines = [hook_text]
    else:
        mid = len(words) // 2
        lines = [" ".join(words[:mid]), " ".join(words[mid:])]

    # Vertical center placement for hook
    line_h = SUB_CFG["font_size_hook"] + 20
    total_h = len(lines) * line_h
    y = (H - total_h) // 2 - 50  # Slightly above center

    for i, line in enumerate(lines):
        bbox = draw.textbbox((0, 0), line, font=font)
        lw = bbox[2] - bbox[0]
        x = (W - lw) // 2
        ly = y + i * line_h

        # Thick shadow
        for dx, dy in [(-3,-3),(3,-3),(-3,3),(3,3),(0,4),(4,0)]:
            draw.text((x+dx, ly+dy), line, font=font, fill=(0,0,0,200))

        color = (*accent_color, 255) if i == 0 else (*text_color, 255)
        draw.text((x, ly), line, font=font, fill=color)

    hook_arr = np.array(img)
    hook_clip = ImageClip(hook_arr, ismask=False).set_duration(hook_dur)
    return hook_clip


def _get_font(size: int) -> ImageFont.FreeTypeFont:
    """Load font with fallback chain."""
    font_path = SUB_CFG["font"]
    candidates = [
        font_path,
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
    ]
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except (IOError, OSError):
            continue
    return ImageFont.load_default()


def _wrap_text(text: str, draw: ImageDraw.Draw, font: ImageFont.FreeTypeFont, max_width: int) -> list:
    """Wrap text into lines that fit within max_width pixels."""
    words = text.split()
    lines = []
    current = ""

    for word in words:
        test = f"{current} {word}".strip()
        bbox = draw.textbbox((0, 0), test, font=font)
        if bbox[2] - bbox[0] <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines
