#!/usr/bin/env python3
"""
Bulk Image Generator — Google Imagen 4 via AI Studio (Gemini API)
-----------------------------------------------------------------
Usage:
    python bulk_image_gen.py [prompts.yaml] [--prefix prefix.yaml]

    --prefix  Optional YAML or .txt file whose text is prepended to every prompt.
              Can also be set inside prompts.yaml as: prefix_file: "prefix.yaml"

Each final prompt sent to Imagen is assembled as:
    <prefix text>

    Subject: <your prompt>

Install deps:
    uv pip install google-genai pyyaml
"""

import argparse
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

ASPECT_RATIOS = {
    "1:1":  "1:1",
    "16:9": "16:9",
    "9:16": "9:16",
    "4:3":  "4:3",
    "3:4":  "3:4",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_prefix(path: str) -> str:
    """Load prefix text from either a .txt file or a YAML file with a 'prefix' key."""
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
            # Accept either a 'prefix' key or a 'text' key
            text = data.get("prefix") or data.get("text") or ""
            return str(text).strip()
        raise ValueError(f"Prefix YAML must contain a 'prefix:' string, got: {type(data)}")

    # Plain text file
    return raw.strip()


def build_full_prompt(prefix: str, subject: str) -> str:
    """Combine prefix block with the per-image subject line."""
    subject = subject.strip()
    # If the subject already starts with "Subject:" leave it alone,
    # otherwise wrap it so the format is consistent.
    if not subject.lower().startswith("subject:"):
        subject = f"Subject: {subject}"
    if prefix:
        return f"{prefix}\n\n{subject}"
    return subject


def sanitize_filename(text: str, max_len: int = 40) -> str:
    # Strip a leading "Subject:" if present before making the filename
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


def generate_and_save(client, full_prompt: str, prompt_cfg: dict,
                      global_cfg: dict, output_dir: Path, index: int) -> list:
    from google.genai import types

    n = int(prompt_cfg.get("n", global_cfg.get("n", 1)))
    aspect_ratio = prompt_cfg.get("aspect_ratio",
                                  global_cfg.get("aspect_ratio", "1:1"))
    aspect_ratio = ASPECT_RATIOS.get(aspect_ratio, "1:1")

    label = prompt_cfg.get("label") or sanitize_filename(prompt_cfg["prompt"])

    config_kwargs = dict(
        number_of_images=n,
        aspect_ratio=aspect_ratio,
        output_mime_type="image/png",
    )

    if "negative_prompt" in prompt_cfg:
        config_kwargs["negative_prompt"] = prompt_cfg["negative_prompt"]

    response = client.models.generate_images(
        model=MODEL,
        prompt=full_prompt,
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
    parser = argparse.ArgumentParser(description="Bulk Imagen 4 image generator")
    parser.add_argument("prompts_file", nargs="?", default=DEFAULT_PROMPTS_FILE,
                        help="Path to prompts YAML file (default: prompts.yaml)")
    parser.add_argument("--prefix", metavar="FILE",
                        help="YAML or .txt file whose text is prepended to every prompt")
    args = parser.parse_args()

    # ── 1. Load prompts ────────────────────────────────────────────────────
    if not os.path.exists(args.prompts_file):
        print(f"[ERROR] Prompts file not found: {args.prompts_file}")
        sys.exit(1)

    data = load_yaml(args.prompts_file)
    prompts = data.get("prompts", [])
    global_cfg = data.get("global", {})

    if not prompts:
        print("[ERROR] No prompts found in the YAML file.")
        sys.exit(1)

    # ── 2. Load prefix (CLI flag > prompts.yaml key > none) ───────────────
    prefix_file = args.prefix or data.get("prefix_file")
    prefix_text = ""
    if prefix_file:
        prefix_text = load_prefix(prefix_file)
        print(f"\n  Prefix loaded from: {prefix_file}")
        # Show a short preview
        preview = prefix_text[:120].replace("\n", " ")
        print(f"  Preview: {preview}{'...' if len(prefix_text) > 120 else ''}")

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
        full_prompt = build_full_prompt(prefix_text, subject)

        label = prompt_cfg.get("label") or sanitize_filename(subject)
        print(f"  [{i}/{total}] {label[:60]}", end="", flush=True)

        try:
            saved_paths = generate_and_save(client, full_prompt, prompt_cfg,
                                            global_cfg, output_dir, i)
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
