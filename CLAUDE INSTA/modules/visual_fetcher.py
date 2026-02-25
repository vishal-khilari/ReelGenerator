"""
🖼️ MODULE 4: VISUAL FETCHER
Fetches high-quality vertical images/videos from free APIs:
  - Pexels (photos + videos)
  - Pixabay (photos + videos)
  - Unsplash (photos)

Fallback: Generates dark gradient backgrounds procedurally using numpy.
All visuals are resized to 1080x1920 (9:16 vertical).
"""

import os
import requests
import numpy as np
from pathlib import Path
from PIL import Image, ImageFilter, ImageEnhance
from config import CONFIG


W, H = CONFIG["video"]["width"], CONFIG["video"]["height"]


def fetch_visuals(keywords: list, emotion: str, session_dir: Path) -> list:
    """
    Try to fetch visuals from APIs using keywords.
    Falls back to procedural generation if all APIs fail.
    Returns list of local image paths.
    """
    visuals_dir = session_dir / "visuals_raw"
    visuals_dir.mkdir(exist_ok=True)

    # Build emotion-augmented queries
    queries = _build_queries(keywords, emotion)
    collected = []

    # Try APIs in order
    for query in queries[:4]:  # 4 queries × up to 2 images = ~8 images
        images = (
            _fetch_pexels_photos(query, visuals_dir)
            or _fetch_pixabay(query, visuals_dir)
            or _fetch_unsplash(query, visuals_dir)
        )
        collected.extend(images)
        if len(collected) >= 8:
            break

    # Always add procedurally generated backgrounds as fallback/filler
    style = CONFIG["emotion_styles"][emotion]
    gen_dir = session_dir / "visuals_raw"
    for i in range(max(0, 8 - len(collected))):
        path = _generate_bg(style["bg_color"], style["accent_color"], i, gen_dir)
        collected.append(path)

    return collected[:10]  # Cap at 10 visuals


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
    # Add pure emotion queries
    queries.append(f"{emotion} dark cinematic aesthetic")
    queries.append("dark minimalist vertical background")
    return queries


def _fetch_pexels_photos(query: str, out_dir: Path) -> list:
    """Fetch from Pexels free API. Returns list of saved image paths."""
    api_key = CONFIG["apis"]["pexels_api_key"]
    if not api_key or api_key == "YOUR_PEXELS_API_KEY":
        return []

    try:
        url = "https://api.pexels.com/v1/search"
        headers = {"Authorization": api_key}
        params = {
            "query": query,
            "per_page": 2,
            "orientation": "portrait",  # Vertical images only
            "size": "medium",
        }
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        photos = resp.json().get("photos", [])
        paths = []
        for i, photo in enumerate(photos):
            img_url = photo["src"]["large"]
            path = out_dir / f"pexels_{query[:20].replace(' ','_')}_{i}.jpg"
            _download_image(img_url, path)
            paths.append(str(path))
        return paths
    except Exception as e:
        print(f"      ⚠️  Pexels failed for '{query}': {e}")
        return []


def _fetch_pixabay(query: str, out_dir: Path) -> list:
    """Fetch from Pixabay free API."""
    api_key = CONFIG["apis"]["pixabay_api_key"]
    if not api_key or api_key == "YOUR_PIXABAY_API_KEY":
        return []

    try:
        url = "https://pixabay.com/api/"
        params = {
            "key": api_key,
            "q": query,
            "image_type": "photo",
            "orientation": "vertical",
            "per_page": 2,
            "safesearch": "true",
            "order": "popular",
        }
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        hits = resp.json().get("hits", [])
        paths = []
        for i, hit in enumerate(hits[:2]):
            img_url = hit["largeImageURL"]
            path = out_dir / f"pixabay_{query[:20].replace(' ','_')}_{i}.jpg"
            _download_image(img_url, path)
            paths.append(str(path))
        return paths
    except Exception as e:
        print(f"      ⚠️  Pixabay failed for '{query}': {e}")
        return []


def _fetch_unsplash(query: str, out_dir: Path) -> list:
    """Fetch from Unsplash free API."""
    api_key = CONFIG["apis"]["unsplash_api_key"]
    if not api_key or api_key == "YOUR_UNSPLASH_API_KEY":
        return []

    try:
        url = "https://api.unsplash.com/search/photos"
        headers = {"Authorization": f"Client-ID {api_key}"}
        params = {
            "query": query,
            "per_page": 2,
            "orientation": "portrait",
        }
        resp = requests.get(url, headers=headers, params=params, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        paths = []
        for i, photo in enumerate(results[:2]):
            img_url = photo["urls"]["regular"]
            path = out_dir / f"unsplash_{query[:20].replace(' ','_')}_{i}.jpg"
            _download_image(img_url, path)
            paths.append(str(path))
        return paths
    except Exception as e:
        print(f"      ⚠️  Unsplash failed for '{query}': {e}")
        return []


def _download_image(url: str, save_path: Path):
    """Download, crop center, resize to 1080x1920."""
    resp = requests.get(url, timeout=15)
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
        # Image is wider — crop sides
        new_w = int(ih * target_ratio)
        left = (iw - new_w) // 2
        img = img.crop((left, 0, left + new_w, ih))
    else:
        # Image is taller — crop top/bottom (keep top third for portraits)
        new_h = int(iw / target_ratio)
        top = int(ih * 0.1)  # Slight top-biased crop
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

    # Base color fill
    arr[:, :] = bg_color

    # Radial gradient (lighter in center, darker at edges)
    cy, cx = H // 2, W // 2
    Y, X = np.ogrid[:H, :W]
    dist = np.sqrt((X - cx)**2 + (Y - cy)**2)
    max_dist = np.sqrt(cx**2 + cy**2)
    glow = 1.0 - (dist / max_dist) * 0.6  # 0.4 at edges, 1.0 at center
    glow = np.clip(glow, 0, 1)

    for c in range(3):
        # Mix bg_color toward accent_color using glow map
        mixed = bg_color[c] * (1 - glow * 0.4) + accent_color[c] * glow * 0.2
        arr[:, :, c] = np.clip(mixed, 0, 255)

    # Film grain noise
    noise = np.random.randint(-12, 13, (H, W, 3), dtype=np.int16)
    arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Save
    img = Image.fromarray(arr)
    # Light gaussian blur for smoothness
    img = img.filter(ImageFilter.GaussianBlur(radius=3))
    path = out_dir / f"gen_bg_{index}.jpg"
    img.save(str(path), "JPEG", quality=88)
    return str(path)
