"""
🖼️ MODULE 4: VISUAL FETCHER
Fetches high-quality vertical images/videos from free APIs:
  - Pexels (photos + videos)
  - Pixabay (photos + videos)
  - Unsplash (photos)

Fallback: Generates dark gradient backgrounds procedurally using numpy.
All visuals are resized to 1080x1920 (9:16 vertical).

FIXES:
  - Pixabay per_page minimum is 3 (was 2 → caused 400 errors)
  - Removed double && bug by cleaning params before passing to requests
  - Increased timeouts (Pexels was timing out at 10s)
  - Added retry logic for transient 503 errors
"""

import os
import time
import requests
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
from config import CONFIG


W, H = CONFIG["video"]["width"], CONFIG["video"]["height"]

# Retry settings
MAX_RETRIES = 2
RETRY_DELAY = 1.5  # seconds between retries


def fetch_visuals(visual_plan: list, emotion: str, session_dir: Path) -> list:
    """
    Fetch one image per clip slot defined by the visual plan.
    Returns list of local image paths, in the same order as visual_plan.
    """
    visuals_dir = session_dir / "visuals_raw"
    visuals_dir.mkdir(exist_ok=True)
    style = CONFIG["emotion_styles"][emotion]
    collected = []

    for slot in visual_plan:
        query = slot["search_query"].strip()
        if not query:
            query = f"{emotion} cinematic"

        images = (
            _fetch_pexels_photos(query, visuals_dir)
            or _fetch_pixabay(query, visuals_dir)
            or _fetch_unsplash(query, visuals_dir)
        )
        if images:
            collected.append(images[0])
        else:
            path = _generate_bg(
                style["bg_color"], style["accent_color"],
                len(collected), visuals_dir
            )
            collected.append(path)

    return collected


def _build_queries(keywords: list, emotion: str) -> list:
    """Augment keywords with emotion-specific modifiers for better imagery."""
    emotion_modifiers = {
        "motivation": ["cinematic", "dramatic", "powerful", "sunrise", "victory"],
        "anxiety": ["dark", "storm", "chaos", "urban night", "abstract"],
        "deep": ["cosmos", "philosophy", "minimal", "abstract", "silhouette"],
        "funny": ["reaction", "funny moment", "surprised", "meme aesthetic"],
        "neutral": ["cinematic", "minimal", "clean"],
    }
    mods = emotion_modifiers.get(emotion, ["cinematic"])
    queries = []
    for i, kw in enumerate(keywords[:3]):
        mod = mods[i % len(mods)]
        queries.append(f"{kw} {mod}")
    queries.append(f"{emotion} dark cinematic aesthetic")
    queries.append("dark minimalist vertical background")
    return queries


def _fetch_pexels_photos(query: str, out_dir: Path) -> list:
    """
    Fetch from Pexels free API.
    FIX: Increased timeout to 20s, added retry logic for 503 errors.
    """
    api_key = CONFIG["apis"]["pexels_api_key"]
    if not api_key or api_key == "YOUR_PEXELS_API_KEY":
        return []

    for attempt in range(MAX_RETRIES):
        try:
            url = "https://api.pexels.com/v1/search"
            headers = {"Authorization": api_key}
            params = {
                "query": query,
                "per_page": 3,           # FIX: was 2, increased for better results
                "orientation": "portrait",
                "size": "medium",
            }
            resp = requests.get(url, headers=headers, params=params, timeout=20)

            # Retry on 503
            if resp.status_code == 503 and attempt < MAX_RETRIES - 1:
                print(f"      ⚠️  Pexels 503, retrying ({attempt+1}/{MAX_RETRIES})...")
                time.sleep(RETRY_DELAY)
                continue

            resp.raise_for_status()
            photos = resp.json().get("photos", [])
            paths = []
            for i, photo in enumerate(photos[:2]):
                img_url = photo["src"]["large"]
                safe_query = query[:20].replace(" ", "_").replace("/", "_")
                path = out_dir / f"pexels_{safe_query}_{i}.jpg"
                _download_image(img_url, path)
                paths.append(str(path))
            return paths

        except requests.exceptions.Timeout:
            if attempt < MAX_RETRIES - 1:
                print(f"      ⚠️  Pexels timeout, retrying ({attempt+1}/{MAX_RETRIES})...")
                time.sleep(RETRY_DELAY)
            else:
                print(f"      ⚠️  Pexels timed out for '{query}' after {MAX_RETRIES} attempts")
        except Exception as e:
            print(f"      ⚠️  Pexels failed for '{query}': {e}")
            break

    return []


def _fetch_pixabay(query: str, out_dir: Path) -> list:
    """
    Fetch from Pixabay free API.
    FIX: per_page minimum is 3 (was 2 → caused 400 Bad Request).
    FIX: Clean query to prevent double && in URL.
    FIX: Increased timeout to 20s.
    """
    api_key = CONFIG["apis"]["pixabay_api_key"]
    if not api_key or api_key == "YOUR_PIXABAY_API_KEY":
        return []

    try:
        url = "https://pixabay.com/api/"
        # FIX: Build params carefully — no empty strings, no trailing spaces
        clean_query = " ".join(query.strip().split())  # normalize whitespace
        params = {
            "key": api_key,
            "q": clean_query,
            "image_type": "photo",
            "orientation": "vertical",
            "per_page": 3,            # FIX: was 2 → 400 error (min is 3)
            "safesearch": "true",
            "order": "popular",
        }
        # FIX: Remove any None/empty values that create double && in URL
        params = {k: v for k, v in params.items() if v is not None and v != ""}

        resp = requests.get(url, params=params, timeout=20)
        resp.raise_for_status()
        hits = resp.json().get("hits", [])
        paths = []
        for i, hit in enumerate(hits[:2]):
            img_url = hit["largeImageURL"]
            safe_query = clean_query[:20].replace(" ", "_").replace("/", "_")
            path = out_dir / f"pixabay_{safe_query}_{i}.jpg"
            _download_image(img_url, path)
            paths.append(str(path))
        return paths
    except Exception as e:
        print(f"      ⚠️  Pixabay failed for '{query}': {e}")
        return []


def _fetch_unsplash(query: str, out_dir: Path) -> list:
    """
    Fetch from Unsplash free API.
    FIX: Increased timeout to 20s.
    """
    api_key = CONFIG["apis"]["unsplash_api_key"]
    if not api_key or api_key == "YOUR_UNSPLASH_API_KEY":
        return []

    try:
        url = "https://api.unsplash.com/search/photos"
        headers = {"Authorization": f"Client-ID {api_key}"}
        params = {
            "query": query.strip(),
            "per_page": 3,
            "orientation": "portrait",
        }
        resp = requests.get(url, headers=headers, params=params, timeout=20)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        paths = []
        for i, photo in enumerate(results[:2]):
            img_url = photo["urls"]["regular"]
            safe_query = query[:20].replace(" ", "_").replace("/", "_")
            path = out_dir / f"unsplash_{safe_query}_{i}.jpg"
            _download_image(img_url, path)
            paths.append(str(path))
        return paths
    except Exception as e:
        print(f"      ⚠️  Unsplash failed for '{query}': {e}")
        return []


def _download_image(url: str, save_path: Path):
    """Download, crop center, resize to 1080x1920."""
    resp = requests.get(url, timeout=20)
    resp.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(resp.content)
    _resize_to_vertical(save_path)


def _resize_to_vertical(img_path: Path):
    """
    Smart crop: resize image to fill 1080x1920 without distortion.
    Center-crops to maintain aspect ratio.
    """
    img = Image.open(img_path).convert("RGB")
    iw, ih = img.size
    target_ratio = W / H  # 9/16 = 0.5625
    img_ratio = iw / ih

    if img_ratio > target_ratio:
        new_w = int(ih * target_ratio)
        left = (iw - new_w) // 2
        img = img.crop((left, 0, left + new_w, ih))
    else:
        new_h = int(iw / target_ratio)
        top = int(ih * 0.1)
        top = min(top, ih - new_h)
        img = img.crop((0, top, iw, top + new_h))

    img = img.resize((W, H), Image.LANCZOS)
    img.save(img_path, "JPEG", quality=90)


def _generate_bg(bg_color: tuple, accent_color: tuple, index: int, out_dir: Path) -> str:
    """
    Procedurally generate a cinematic dark background with:
    - Radial gradient
    - Noise texture
    - Light leak overlay
    """
    arr = np.zeros((H, W, 3), dtype=np.uint8)
    arr[:, :] = bg_color

    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    max_dist = np.sqrt(cx**2 + cy**2)
    glow = 1.0 - (dist / max_dist) * 0.6
    glow = np.clip(glow, 0, 1)

    for c in range(3):
        mixed = bg_color[c] * (1 - glow * 0.4) + accent_color[c] * glow * 0.2
        arr[:, :, c] = np.clip(mixed, 0, 255)

    noise = np.random.randint(-12, 13, (H, W, 3), dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr)
    img = img.filter(ImageFilter.GaussianBlur(radius=3))
    path = out_dir / f"gen_bg_{index}.jpg"
    img.save(str(path), "JPEG", quality=88)
    return str(path)