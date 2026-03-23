#!/usr/bin/env python3
"""
Bulk Image Generator — OpenRouter + Google Imagen 4
----------------------------------------------------
1. Edit prompts.yaml with your prompts (see example file)
2. Run: python bulk_image_gen.py
3. Paste your OpenRouter API key when prompted
4. Images are saved to the output folder defined in prompts.yaml (or ./output)
"""

import base64
import getpass
import json
import os
import sys
import time
from pathlib import Path

import requests
import yaml


# ── Config ────────────────────────────────────────────────────────────────────

MODEL = "google/imagen-4"
API_URL = "https://openrouter.ai/api/v1/images/generations"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_PROMPTS_FILE = "prompts.yaml"

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_prompts(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def sanitize_filename(text: str, max_len: int = 40) -> str:
    """Turn a prompt snippet into a safe filename fragment."""
    keep = []
    for ch in text.lower():
        if ch.isalnum():
            keep.append(ch)
        elif ch in (" ", "_", "-"):
            keep.append("_")
    cleaned = "".join(keep).strip("_")
    # collapse repeated underscores
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned[:max_len]


def save_image(data: dict, output_dir: Path, prefix: str, index: int) -> Path:
    """
    OpenRouter image responses come back as either:
      - b64_json  →  base64-encoded image bytes
      - url       →  a hosted URL to download from
    """
    img_obj = data["data"][0]

    if "b64_json" in img_obj and img_obj["b64_json"]:
        img_bytes = base64.b64decode(img_obj["b64_json"])
        ext = "png"
    elif "url" in img_obj and img_obj["url"]:
        resp = requests.get(img_obj["url"], timeout=60)
        resp.raise_for_status()
        img_bytes = resp.content
        # Guess extension from content-type or URL
        ct = resp.headers.get("content-type", "")
        if "jpeg" in ct or "jpg" in ct:
            ext = "jpg"
        elif "webp" in ct:
            ext = "webp"
        else:
            ext = "png"
    else:
        raise ValueError("Response contained neither b64_json nor url.")

    filename = f"{index:03d}_{prefix}.{ext}"
    out_path = output_dir / filename
    out_path.write_bytes(img_bytes)
    return out_path


def generate_image(api_key: str, prompt_cfg: dict, global_cfg: dict) -> dict:
    """Send one image generation request to OpenRouter."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://localhost",   # required by OpenRouter
        "X-Title": "bulk-image-gen",
    }

    # Merge global defaults with per-prompt overrides
    size = prompt_cfg.get("size", global_cfg.get("size", "1024x1024"))
    n = prompt_cfg.get("n", global_cfg.get("n", 1))

    payload = {
        "model": MODEL,
        "prompt": prompt_cfg["prompt"],
        "n": n,
        "size": size,
        "response_format": "b64_json",   # prefer inline bytes; fall back to url
    }

    # Optional: negative prompt (passed as extra_body if supported)
    if "negative_prompt" in prompt_cfg:
        payload["negative_prompt"] = prompt_cfg["negative_prompt"]

    response = requests.post(API_URL, headers=headers, json=payload, timeout=120)

    if response.status_code != 200:
        raise RuntimeError(
            f"API error {response.status_code}: {response.text[:400]}"
        )

    return response.json()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Locate prompts file ─────────────────────────────────────────────
    prompts_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROMPTS_FILE

    if not os.path.exists(prompts_file):
        print(f"[ERROR] Prompts file not found: {prompts_file}")
        print(f"        Create '{prompts_file}' next to this script (see prompts_example.yaml).")
        sys.exit(1)

    data = load_prompts(prompts_file)
    prompts: list = data.get("prompts", [])
    global_cfg: dict = data.get("global", {})

    if not prompts:
        print("[ERROR] No prompts found in the YAML file.")
        sys.exit(1)

    # ── 2. Output folder ───────────────────────────────────────────────────
    output_dir = Path(data.get("output_dir", DEFAULT_OUTPUT_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 3. API key ─────────────────────────────────────────────────────────
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        api_key = getpass.getpass("Paste your OpenRouter API key: ").strip()
    if not api_key:
        print("[ERROR] No API key provided.")
        sys.exit(1)

    # ── 4. Run ─────────────────────────────────────────────────────────────
    total = len(prompts)
    print(f"\n🖼  Generating {total} image(s) → {output_dir}/\n")

    successes, failures = 0, 0

    for i, prompt_cfg in enumerate(prompts, start=1):
        # Allow a prompt entry to be a plain string for convenience
        if isinstance(prompt_cfg, str):
            prompt_cfg = {"prompt": prompt_cfg}

        label = prompt_cfg.get("label") or sanitize_filename(prompt_cfg["prompt"])
        print(f"  [{i}/{total}] {label[:60]}", end="", flush=True)

        try:
            result = generate_image(api_key, prompt_cfg, global_cfg)
            out_path = save_image(result, output_dir, label, i)
            print(f"  ✓  saved → {out_path}")
            successes += 1
        except Exception as exc:
            print(f"  ✗  FAILED: {exc}")
            failures += 1

        # Small pause to be kind to rate limits
        if i < total:
            time.sleep(0.5)

    print(f"\nDone. {successes} succeeded, {failures} failed.\n")


if __name__ == "__main__":
    main()
