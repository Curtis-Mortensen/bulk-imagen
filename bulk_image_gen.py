#!/usr/bin/env python3
"""
Bulk Image Generator — Google Imagen 4 via AI Studio (Gemini API)
-----------------------------------------------------------------
1. Edit prompts.yaml with your prompts
2. Run: python bulk_image_gen.py
3. Paste your Google AI Studio API key when prompted
4. Images are saved to the output folder defined in prompts.yaml (or ./output)

Install deps:
    uv pip install google-genai pyyaml
"""

import getpass
import os
import sys
import time
from pathlib import Path

import yaml

# ── Config ────────────────────────────────────────────────────────────────────

MODEL = "imagen-4.0-generate-001"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_PROMPTS_FILE = "prompts.yaml"

# Aspect ratio map: human-friendly → Imagen 4 accepted values
ASPECT_RATIOS = {
    "1:1":  "1:1",
    "16:9": "16:9",
    "9:16": "9:16",
    "4:3":  "4:3",
    "3:4":  "3:4",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_prompts(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data


def sanitize_filename(text: str, max_len: int = 40) -> str:
    keep = []
    for ch in text.lower():
        if ch.isalnum():
            keep.append(ch)
        elif ch in (" ", "_", "-"):
            keep.append("_")
    cleaned = "".join(keep).strip("_")
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned[:max_len]


def generate_and_save(client, prompt_cfg: dict, global_cfg: dict,
                      output_dir: Path, index: int) -> list:
    """Generate image(s) for one prompt entry and save them. Returns saved paths."""
    from google.genai import types

    prompt_text = prompt_cfg["prompt"]
    n = int(prompt_cfg.get("n", global_cfg.get("n", 1)))
    aspect_ratio = prompt_cfg.get("aspect_ratio",
                                  global_cfg.get("aspect_ratio", "1:1"))
    aspect_ratio = ASPECT_RATIOS.get(aspect_ratio, "1:1")

    label = prompt_cfg.get("label") or sanitize_filename(prompt_text)

    config_kwargs = dict(
        number_of_images=n,
        aspect_ratio=aspect_ratio,
        output_mime_type="image/png",
    )

    # Optional: negative prompt
    if "negative_prompt" in prompt_cfg:
        config_kwargs["negative_prompt"] = prompt_cfg["negative_prompt"]

    response = client.models.generate_images(
        model=MODEL,
        prompt=prompt_text,
        config=types.GenerateImagesConfig(**config_kwargs),
    )

    saved = []
    for img_index, generated in enumerate(response.generated_images):
        suffix = f"_{img_index + 1}" if n > 1 else ""
        filename = f"{index:03d}_{label}{suffix}.png"
        out_path = output_dir / filename
        out_path.write_bytes(generated.image.image_bytes)
        saved.append(out_path)

    return saved


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── 1. Locate prompts file ─────────────────────────────────────────────
    prompts_file = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PROMPTS_FILE

    if not os.path.exists(prompts_file):
        print(f"[ERROR] Prompts file not found: {prompts_file}")
        print(f"        Create '{prompts_file}' next to this script.")
        sys.exit(1)

    data = load_prompts(prompts_file)
    prompts = data.get("prompts", [])
    global_cfg = data.get("global", {})

    if not prompts:
        print("[ERROR] No prompts found in the YAML file.")
        sys.exit(1)

    # ── 2. Output folder ───────────────────────────────────────────────────
    output_dir = Path(data.get("output_dir", DEFAULT_OUTPUT_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 3. API key + client ────────────────────────────────────────────────
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        api_key = getpass.getpass("Paste your Google AI Studio API key: ").strip()
    if not api_key:
        print("[ERROR] No API key provided.")
        sys.exit(1)

    try:
        from google import genai
    except ImportError:
        print("[ERROR] google-genai is not installed.")
        print("        Run: uv pip install google-genai")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # ── 4. Run ─────────────────────────────────────────────────────────────
    total = len(prompts)
    print(f"\n  Generating {total} prompt(s) -> {output_dir}/\n")

    successes, failures = 0, 0

    for i, prompt_cfg in enumerate(prompts, start=1):
        if isinstance(prompt_cfg, str):
            prompt_cfg = {"prompt": prompt_cfg}

        label = prompt_cfg.get("label") or sanitize_filename(prompt_cfg["prompt"])
        print(f"  [{i}/{total}] {label[:60]}", end="", flush=True)

        try:
            saved_paths = generate_and_save(client, prompt_cfg, global_cfg,
                                            output_dir, i)
            for p in saved_paths:
                print(f"\n          OK  {p}", end="")
            print()
            successes += 1
        except Exception as exc:
            print(f"  FAILED: {exc}")
            failures += 1

        if i < total:
            time.sleep(0.3)

    print(f"\nDone. {successes} succeeded, {failures} failed.\n")


if __name__ == "__main__":
    main()
