"""
📤 MODULE 7: EXPORTER
Uses ffmpeg to take the raw assembled video and produce an
Instagram-optimized 1080x1920 reel with:

  - H.264 video codec (Instagram compatible)
  - AAC audio at 192kbps
  - Moov atom moved to front (fast start for streaming)
  - Color profile: yuv420p (maximum compatibility)
  - Bitrate: 8Mbps (Instagram recommended)
  - Pixel format preserved
  - Optional: loudness normalization for audio (LUFS -14)
"""

import os
import subprocess
import shutil
from pathlib import Path
from config import CONFIG


def export_final_reel(raw_video_path: str, session_dir: Path, stem: str) -> str:
    """
    Export final Instagram-ready reel using ffmpeg.
    Returns path to final .mp4 file.
    """
    cfg = CONFIG["video"]
    output_dir = Path("output")
    output_dir.mkdir(exist_ok=True)
    final_path = str(output_dir / f"{stem}_reel.mp4")
    temp_normalized = str(session_dir / "temp_loudnorm.mp4")

    # Step 1: Loudness normalize audio to -14 LUFS (Instagram/Spotify standard)
    loudnorm_success = _loudness_normalize(raw_video_path, temp_normalized)
    source = temp_normalized if loudnorm_success else raw_video_path

    # Step 2: Final encode with Instagram optimization
    cmd = [
        "ffmpeg", "-y",
        "-i", source,
        "-vf", f"scale={cfg['width']}:{cfg['height']}:force_original_aspect_ratio=decrease,"
               f"pad={cfg['width']}:{cfg['height']}:(ow-iw)/2:(oh-ih)/2:black,"
               "format=yuv420p",
        "-c:v", cfg["codec"],           # libx264
        "-preset", "slow",              # Better compression, worth the wait
        "-crf", "18",                   # High quality (18 = near-lossless visually)
        "-b:v", cfg["bitrate"],
        "-maxrate", "10000k",
        "-bufsize", "20000k",
        "-c:a", cfg["audio_codec"],     # aac
        "-b:a", "192k",
        "-ar", "44100",
        "-movflags", "+faststart",      # Moov atom first = instant play on Instagram
        "-metadata", f"comment=Created by AI Reel Engine",
        final_path,
    ]

    print(f"      Running ffmpeg export...")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"      ⚠️  ffmpeg warning: {result.stderr[-500:]}")
        # Try simpler fallback
        _simple_export(raw_video_path, final_path, cfg)

    # Cleanup temp
    for temp_file in [temp_normalized]:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception:
            pass

    file_size_mb = os.path.getsize(final_path) / (1024 * 1024)
    print(f"      ✅ Final size: {file_size_mb:.1f} MB")
    _print_video_info(final_path)

    return final_path


def _loudness_normalize(input_path: str, output_path: str) -> bool:
    """
    Two-pass loudness normalization to -14 LUFS (Instagram standard).
    Returns True if successful.
    """
    try:
        # Pass 1: Analyze
        analyze_cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-af", "loudnorm=I=-14:TP=-1.5:LRA=11:print_format=json",
            "-f", "null", "-",
        ]
        result = subprocess.run(analyze_cmd, capture_output=True, text=True, timeout=120)

        # Parse LUFS values from output
        import re, json
        match = re.search(r'\{[^}]+\}', result.stderr, re.DOTALL)
        if not match:
            return False

        stats = json.loads(match.group())

        # Pass 2: Apply normalization with measured values
        normalize_cmd = [
            "ffmpeg", "-y",
            "-i", input_path,
            "-af", (
                f"loudnorm=I=-14:TP=-1.5:LRA=11"
                f":measured_I={stats['input_i']}"
                f":measured_TP={stats['input_tp']}"
                f":measured_LRA={stats['input_lra']}"
                f":measured_thresh={stats['input_thresh']}"
                f":offset={stats['target_offset']}"
                f":linear=true:print_format=summary"
            ),
            "-c:v", "copy",
            output_path,
        ]
        result2 = subprocess.run(normalize_cmd, capture_output=True, text=True, timeout=120)
        return result2.returncode == 0

    except Exception as e:
        print(f"      ⚠️  Loudnorm skipped: {e}")
        return False


def _simple_export(input_path: str, output_path: str, cfg: dict):
    """Simplified ffmpeg export as fallback."""
    cmd = [
        "ffmpeg", "-y",
        "-i", input_path,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        "-movflags", "+faststart",
        output_path,
    ]
    subprocess.run(cmd, check=True)


def _print_video_info(video_path: str):
    """Print key video metadata using ffprobe."""
    try:
        cmd = [
            "ffprobe", "-v", "quiet",
            "-print_format", "json",
            "-show_streams", video_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            import json
            data = json.loads(result.stdout)
            for stream in data.get("streams", []):
                if stream.get("codec_type") == "video":
                    w = stream.get("width")
                    h = stream.get("height")
                    fps = stream.get("r_frame_rate", "?")
                    dur = float(stream.get("duration", 0))
                    print(f"      📐 {w}×{h} | {fps}fps | {dur:.1f}s")
    except Exception:
        pass
