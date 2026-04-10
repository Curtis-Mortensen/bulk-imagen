#!/usr/bin/env python3
"""
Bulk Image Generator — Gemini 3.1 Flash (Nano Banana 2) via Google AI Studio
-----------------------------------------------------------------------------
Usage:
    python bulk_image_gen.py [prompts.yaml] [--prefix prefix.yaml]

Install deps:
    uv pip install google-genai pyyaml
"""

dependencies = [
  "google-genai",
  "pyyaml",
  "requests",
]

import argparse
import getpass
import os
import sys
import time
from pathlib import Path

import yaml

# ── Config ────────────────────────────────────────────────────────────────────

MODEL = "gemini-3.1-flash-image-preview"
DEFAULT_OUTPUT_DIR = "output"
DEFAULT_PROMPTS_FILE = "prompts.yaml"

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prefix(path: str) -> str:
    """Load prefix text from a .txt or .yaml file."""
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] Prefix file not found: {path}")
        sys.exit(1)
    raw = p.read_text(encoding="utf-8")
    if p.suffix.lower() in (".yaml", ".yml"):
        data = yaml.safe_load(raw)
        if isinstance(data, str):
            return data.strip()
        if isinstance(data, dict):
            text = data.get("prefix") or data.get("text") or ""
            return str(text).strip()
        raise ValueError(f"Prefix YAML must contain a 'prefix:' string, got: {type(data)}")
    return raw.strip()


def build_full_prompt(prefix: str, subject: str) -> str:
    subject = subject.strip()
    if not subject.lower().startswith("subject:"):
        subject = f"Subject: {subject}"
    return f"{prefix}\n\n{subject}" if prefix else subject


def sanitize_filename(text: str, max_len: int = 40) -> str:
    if text.lower().startswith("subject:"):
        text = text[len("subject:"):].strip()
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


def generate_and_save(client, full_prompt: str, output_dir: Path,
                      index: int, label: str) -> list:
    from google.genai import types

    response = client.models.generate_content(
        model=MODEL,
        contents=full_prompt,
        config=types.GenerateContentConfig(
            response_modalities=["TEXT", "IMAGE"],
        ),
    )

    saved = []
    img_count = 0
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            img_count += 1
            suffix = f"_{img_count}" if img_count > 1 else ""
            mime = getattr(part.inline_data, "mime_type", "image/png")
            ext = "jpg" if "jpeg" in mime else "webp" if "webp" in mime else "png"
            out_path = output_dir / f"{index:03d}_{label}{suffix}.{ext}"
            out_path.write_bytes(part.inline_data.data)
            saved.append(out_path)

    return saved


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Bulk image generator — Gemini Flash")
    parser.add_argument("prompts_file", nargs="?", default=DEFAULT_PROMPTS_FILE,
                        help="Path to prompts YAML (default: prompts.yaml)")
    parser.add_argument("--prefix", metavar="FILE",
                        help="YAML or .txt file prepended to every prompt")
    args = parser.parse_args()

    # ── 1. Load prompts ────────────────────────────────────────────────────
    if not os.path.exists(args.prompts_file):
        print(f"[ERROR] Prompts file not found: {args.prompts_file}")
        sys.exit(1)

    data = load_yaml(args.prompts_file)
    prompts = data.get("prompts", [])

    if not prompts:
        print("[ERROR] No prompts found in the YAML file.")
        sys.exit(1)

    # ── 2. Load prefix ─────────────────────────────────────────────────────
    prefix_file = args.prefix or data.get("prefix_file")
    prefix_text = ""
    if prefix_file:
        prefix_text = load_prefix(prefix_file)
        preview = prefix_text[:120].replace("\n", " ")
        print(f"\n  Prefix: {preview}{'...' if len(prefix_text) > 120 else ''}")

    # ── 3. Output folder ───────────────────────────────────────────────────
    output_dir = Path(data.get("output_dir", DEFAULT_OUTPUT_DIR))
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── 4. API key ─────────────────────────────────────────────────────────
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        api_key = getpass.getpass("Paste your Google AI Studio API key: ").strip()
    if not api_key:
        print("[ERROR] No API key provided.")
        sys.exit(1)

    try:
        from google import genai
    except ImportError:
        print("[ERROR] google-genai is not installed. Run: uv pip install google-genai")
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # ── 5. Run ─────────────────────────────────────────────────────────────
    total = len(prompts)
    print(f"\n  Generating {total} image(s) -> {output_dir}/\n")

    successes, failures = 0, 0

    for i, prompt_cfg in enumerate(prompts, start=1):
        if isinstance(prompt_cfg, str):
            prompt_cfg = {"prompt": prompt_cfg}

        subject = prompt_cfg["prompt"]
        label = prompt_cfg.get("label") or sanitize_filename(subject)
        full_prompt = build_full_prompt(prefix_text, subject)

        print(f"  [{i}/{total}] {label[:60]}", end="", flush=True)

        try:
            saved_paths = generate_and_save(client, full_prompt, output_dir, i, label)
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
