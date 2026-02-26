"""
⚡ MODULE 5: EFFECTS ENGINE
Uses OpenCV + NumPy to apply cinematic effects to each visual frame.

Available effects:
  - slow_zoom        → Ken Burns style zoom in
  - glitch           → RGB channel shift + scanlines + pixelation
  - subtle_drift     → Slow pan/drift (parallax feel)
  - meme_zoom        → Aggressive zoom punch + color pop
  - cinematic_bars   → Top + bottom letterbox bars

Each effect returns a list of frames (numpy arrays) for that image's duration.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image
from config import CONFIG

W, H = CONFIG["video"]["width"], CONFIG["video"]["height"]
FPS = CONFIG["video"]["fps"]


def apply_effects_to_visuals(
    visual_paths: list,
    emotion: str,
    format_name: str,
    session_dir: Path,
    visual_plan: list = None,       # ← new param: per-clip plan from visual_planner
) -> list:
    """
    Apply cinematic effects to each image.
    When visual_plan is provided, uses per-clip effect and duration from the plan.
    Falls back to emotion style defaults when plan is absent.
    Returns list of dicts: {path, frames_dir, effect, duration_sec}
    """
    style = CONFIG["emotion_styles"][emotion]
    has_bars = style["has_cinematic_bars"]
    processed = []

    for i, img_path in enumerate(visual_paths):
        out_dir = session_dir / "frames" / f"clip_{i:03d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        # Use plan values when available, otherwise fall back to style defaults
        if visual_plan and i < len(visual_plan):
            slot = visual_plan[i]
            clip_effect = slot.get("effect", style["effect"])
            dur = slot["duration_sec"]
        else:
            clip_effect = style["effect"]
            # Every 4th clip gets a subtle_drift for variety (legacy behaviour)
            if i % 4 == 3:
                clip_effect = "subtle_drift"
            cut_speed = style["cut_speed"]
            dur = {"fast": 1.5, "medium": 2.5, "slow": 3.5}[cut_speed]

        frame_count = int(dur * FPS)
        _render_effect_frames(img_path, clip_effect, frame_count, has_bars, out_dir)

        processed.append({
            "path": img_path,
            "frames_dir": str(out_dir),
            "effect": clip_effect,
            "duration_sec": dur,
            "frame_count": frame_count,
            "has_bars": has_bars,
        })

    return processed


def _render_effect_frames(
    img_path: str, effect: str, n_frames: int, has_bars: bool, out_dir: Path
):
    """Render all frames for a single clip to disk."""
    img = cv2.imread(img_path)
    if img is None:
        img = np.zeros((H, W, 3), dtype=np.uint8)
    img = cv2.resize(img, (W, H))

    for f in range(n_frames):
        t = f / max(n_frames - 1, 1)  # 0.0 → 1.0 progress

        if effect == "slow_zoom":
            frame = _slow_zoom(img, t, zoom_from=1.0, zoom_to=1.12)
        elif effect == "glitch":
            frame = _glitch_effect(img, t, intensity=0.6)
        elif effect == "subtle_drift":
            frame = _subtle_drift(img, t)
        elif effect == "meme_zoom":
            frame = _meme_zoom(img, t)
        else:
            frame = img.copy()

        if has_bars:
            frame = _add_cinematic_bars(frame)

        cv2.imwrite(str(out_dir / f"frame_{f:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])


def _slow_zoom(img: np.ndarray, t: float, zoom_from: float = 1.0, zoom_to: float = 1.12) -> np.ndarray:
    """
    Ken Burns zoom: slightly scale up the image over time.
    t=0 → zoom_from, t=1 → zoom_to
    """
    zoom = zoom_from + (zoom_to - zoom_from) * t
    new_w = int(W / zoom)
    new_h = int(H / zoom)
    x1 = (W - new_w) // 2
    y1 = (H - new_h) // 2
    cropped = img[y1:y1+new_h, x1:x1+new_w]
    return cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LINEAR)


def _glitch_effect(img: np.ndarray, t: float, intensity: float = 0.5) -> np.ndarray:
    """
    Glitch effect: RGB channel offset + scanlines + random block corruption.
    Intensity oscillates — strongest at start/peak moments.
    """
    glitch_strength = intensity * (0.5 + 0.5 * np.sin(t * np.pi * 6))
    frame = img.copy().astype(np.float32)

    # 1. RGB channel shift
    shift_x = int(glitch_strength * 15)
    shift_y = int(glitch_strength * 5)

    b, g, r = cv2.split(frame)
    rows, cols = r.shape

    M_r = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    M_b = np.float32([[1, 0, -shift_x], [0, 1, -shift_y]])
    r = cv2.warpAffine(r, M_r, (cols, rows))
    b = cv2.warpAffine(b, M_b, (cols, rows))
    frame = cv2.merge([b, g, r])

    # 2. Scanlines (horizontal dark lines every N pixels)
    line_spacing = 4
    for y in range(0, rows, line_spacing):
        frame[y, :] = frame[y, :] * 0.7

    # 3. Random block corruption (glitch blocks)
    if glitch_strength > 0.3 and np.random.rand() > 0.5:
        block_h = np.random.randint(5, 30)
        block_y = np.random.randint(0, rows - block_h)
        shift_amount = np.random.randint(-40, 40)
        block = frame[block_y:block_y + block_h, :]
        frame[block_y:block_y + block_h, :] = np.roll(block, shift_amount, axis=1)

    return np.clip(frame, 0, 255).astype(np.uint8)


def _subtle_drift(img: np.ndarray, t: float) -> np.ndarray:
    """
    Parallax drift: slowly pan across oversized image for depth.
    """
    src = cv2.resize(img, (int(W * 1.15), int(H * 1.05)))
    max_x = src.shape[1] - W
    max_y = src.shape[0] - H

    x = int(max_x * 0.5 * (1 + np.sin(t * np.pi)))
    y = int(max_y * t * 0.3)
    x = min(x, max_x)
    y = min(y, max_y)

    return src[y:y+H, x:x+W]


def _meme_zoom(img: np.ndarray, t: float) -> np.ndarray:
    """
    Aggressive punch zoom: fast zoom in with slight rotation wobble.
    """
    ease = 1 - (1 - t) ** 3  # cubic ease out
    zoom = 1.0 + ease * 0.35  # zoom up to 1.35x

    new_w = int(W / zoom)
    new_h = int(H / zoom)
    x1 = (W - new_w) // 2
    y1 = (H - new_h) // 2

    cropped = img[y1:y1+new_h, x1:x1+new_w]
    zoomed = cv2.resize(cropped, (W, H), interpolation=cv2.INTER_LINEAR)

    hsv = cv2.cvtColor(zoomed, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * (1 + ease * 0.4), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _add_cinematic_bars(frame: np.ndarray, bar_ratio: float = 0.075) -> np.ndarray:
    """
    Add black letterbox bars at top and bottom (cinematic 2.35:1 feel).
    bar_ratio: fraction of frame height for each bar.
    """
    bar_h = int(H * bar_ratio)
    frame = frame.copy()
    frame[:bar_h, :] = 0
    frame[H - bar_h:, :] = 0
    return frame


def generate_transition_frame(
    frame_a: np.ndarray, frame_b: np.ndarray, t: float, transition_type: str = "dissolve"
) -> np.ndarray:
    """
    Generate a single transition frame between two clips.
    t: 0.0 (full A) → 1.0 (full B)
    Types: dissolve, wipe_left, flash_white
    """
    if transition_type == "dissolve":
        return cv2.addWeighted(frame_a, 1 - t, frame_b, t, 0)

    elif transition_type == "wipe_left":
        out = frame_a.copy()
        cut_x = int(W * t)
        out[:, :cut_x] = frame_b[:, :cut_x]
        return out

    elif transition_type == "flash_white":
        white = np.full_like(frame_a, 255)
        if t < 0.5:
            alpha = t * 2
            return cv2.addWeighted(frame_a, 1 - alpha, white, alpha, 0)
        else:
            alpha = (t - 0.5) * 2
            return cv2.addWeighted(white, 1 - alpha, frame_b, alpha, 0)

    return frame_a