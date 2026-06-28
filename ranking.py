#!/usr/bin/env python3
"""
One-Page Dungeon Ranker — DeepSeek V4 Flash via OpenRouter
------------------------------------------------------------
Rates transcribed dungeon Markdown files and writes a standardized ## Ranking
block that can be compiled into a summary HTML document.

Usage:
    export OPENROUTER_API_KEY="your-key-here"

    # Rate all .md files in a year folder
    python ranking.py MD-OPDC/2010

    # Rate every year folder under MD-OPDC
    python ranking.py MD-OPDC

    # Re-rate files that already have rankings
    python ranking.py MD-OPDC/2010 --force

    # Compile rankings into a self-contained HTML summary
    python ranking.py MD-OPDC/2010 --compile
    python ranking.py MD-OPDC --compile --output MD-OPDC/rankings.html

Install deps:
    pip install pyyaml
"""

dependencies = [
    "pyyaml",
]

import argparse
import getpass
import html
import json
import os
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib import error, request

try:
    import yaml
except ImportError:
    yaml = None  # type: ignore

# ── Config ────────────────────────────────────────────────────────────────────

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-v4-flash"
RANKING_BEGIN = "<!-- RANKING:BEGIN -->"
RANKING_END = "<!-- RANKING:END -->"
RANKING_HEADING = "## Ranking"
DEFAULT_OUTPUT_HTML = "rankings.html"
ERRORS_FILENAME = "errors.md"
PARALLEL_THRESHOLD = 4
MAX_PARALLEL_WORKERS = 4

MAP_SECTION_RE = re.compile(
    r"^###\s+.*(?:map|Map|play area|Play Area|navigation|Navigation|accessibility|Accessibility).*",
    re.MULTILINE,
)
THINKING_RE = re.compile(r"^## Thinking\s*$", re.MULTILINE)
TRANSCRIPTION_RE = re.compile(r"^## Transcription\s*$", re.MULTILINE)
RANKING_BLOCK_RE = re.compile(
    rf"{re.escape(RANKING_BEGIN)}\s*\n(.*?)\n{re.escape(RANKING_END)}",
    re.DOTALL,
)
EXISTING_RANKING_RE = re.compile(
    rf"^{re.escape(RANKING_HEADING)}\s*\n{RANKING_BEGIN}.*?(?:\n{RANKING_END}\s*)?",
    re.DOTALL | re.MULTILINE,
)

RATING_FIELDS = (
    "concept_originality",
    "mechanics_originality",
    "interesting_details",
    "map_quality",
)
RATING_LABELS = {
    "concept_originality": "Concept Originality",
    "mechanics_originality": "Mechanics Originality",
    "interesting_details": "Interesting Details",
    "map_quality": "Map Quality",
}
LEGACY_RATING_ALIASES = {
    "originality": "concept_originality",
    "mechanics": "mechanics_originality",
}
TEXT_FIELDS = ("title", "summary_1", "summary_2", "resolutions")
ALL_FIELDS = TEXT_FIELDS + ("rooms",) + RATING_FIELDS

RATING_PROMPT = """You are rating a one-page D&D dungeon based on its Markdown transcript.

Return ONLY a YAML block (no markdown fences, no commentary) with exactly these fields:

title: <dungeon title>
summary_1: <one sentence describing the dungeon concept>
summary_2: <one sentence on standout mechanics, hook, or twist>
rooms: <integer count of keyed rooms/locations; best estimate if unclear>
resolutions: <comma-separated list drawn only from: Combat, Diplomacy, Puzzles, Fetch Quests, Stealth, Roleplay, Traps, Exploration, Skill Challenges, Social. Include only resolutions explicitly supported by the text.>
concept_originality: <integer 1-5, originality of the overall dungeon concept>
mechanics_originality: <integer 1-5, originality of the mechanics, puzzles, and encounter design>
interesting_details: <integer 1-5, density of memorable flavor, NPCs, set dressing, and surprising touches>
map_quality: <integer 1-5, how impressive the map sounds versus a plain grid>

Scoring guide:
- concept_originality: 5 = highly unique premise; 1 = generic trope dungeon
- mechanics_originality: 5 = clever or novel systems; 1 = combat-only room clearing
- interesting_details: 5 = rich, distinctive details throughout; 1 = bare rooms with little flavor
- map_quality: 5 = multi-level, unusual topology, ship, planet, etc.; 1 = plain grid corridors

Transcript:
"""


# ── Helpers ───────────────────────────────────────────────────────────────────


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ts(dt: Optional[datetime] = None) -> str:
    dt = dt or utc_now()
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def log(message: str) -> None:
    print(f"[{ts()}] {message}", flush=True)


def ensure_yaml() -> None:
    if yaml is None:
        print("Missing dependency: pyyaml\nInstall with: pip install pyyaml", file=sys.stderr)
        sys.exit(1)


def resolve_api_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if key:
        return key
    key = getpass.getpass("OpenRouter API key: ").strip()
    if not key:
        print("OPENROUTER_API_KEY is required.", file=sys.stderr)
        sys.exit(1)
    return key


def discover_md_files(folder: Path) -> list[Path]:
    return sorted(folder.glob("*.md"), key=lambda p: p.name.lower())


def discover_year_folders(parent: Path) -> list[Path]:
    return sorted(
        (p for p in parent.iterdir() if p.is_dir() and not p.name.startswith(".")),
        key=lambda p: p.name,
    )


def is_year_batch_root(folder: Path) -> bool:
    if not folder.is_dir():
        return False
    subdirs = discover_year_folders(folder)
    if not subdirs:
        return False
    md_here = discover_md_files(folder)
    return len(md_here) == 0 and len(subdirs) > 0


def has_transcription(content: str) -> bool:
    return TRANSCRIPTION_RE.search(content) is not None


def has_ranking(content: str) -> bool:
    return RANKING_BEGIN in content and RANKING_END in content


def strip_thinking_section(content: str) -> str:
    if not THINKING_RE.search(content):
        return content
    if not has_transcription(content):
        return content
    return re.sub(
        r"^## Thinking\s*\n.*?(?=^## Transcription\s*\n)",
        "",
        content,
        count=1,
        flags=re.MULTILINE | re.DOTALL,
    )


def extract_transcription_body(content: str) -> str:
    match = TRANSCRIPTION_RE.search(content)
    if not match:
        return content
    return content[match.end() :].strip()


def find_map_section_end(content: str) -> int:
    """Return index after the last map-related ### section, or end of file."""
    matches = list(MAP_SECTION_RE.finditer(content))
    if not matches:
        return len(content.rstrip())

    last = matches[-1]
    rest = content[last.end() :]
    next_heading = re.search(r"^##\s+", rest, flags=re.MULTILINE)
    if next_heading:
        return last.end() + next_heading.start()
    return len(content.rstrip())


def normalize_ranking_fields(data: dict) -> dict:
    normalized = dict(data)
    for legacy, current in LEGACY_RATING_ALIASES.items():
        if current not in normalized and legacy in normalized:
            normalized[current] = normalized[legacy]
    return normalized


def row_average(row: dict) -> float:
    return sum(int(row[field]) for field in RATING_FIELDS) / len(RATING_FIELDS)


def format_ranking_block(data: dict, rated_at: str, model: str) -> str:
    lines = [
        RANKING_HEADING,
        "",
        RANKING_BEGIN,
        f"title: {data['title']}",
        f"summary_1: {data['summary_1']}",
        f"summary_2: {data['summary_2']}",
        f"rooms: {data['rooms']}",
        f"resolutions: {data['resolutions']}",
    ]
    for field in RATING_FIELDS:
        lines.append(f"{field}: {data[field]}")
    lines.extend(
        [
            f"rated_at: {rated_at}",
            f"model: {model}",
            RANKING_END,
            "",
        ]
    )
    return "\n".join(lines)


def remove_existing_ranking(content: str) -> str:
    content = EXISTING_RANKING_RE.sub("", content)
    return content.rstrip() + "\n"


def insert_ranking(content: str, ranking_block: str) -> str:
    content = remove_existing_ranking(content)
    insert_at = find_map_section_end(content)
    before = content[:insert_at].rstrip()
    after = content[insert_at:].lstrip()
    if after:
        return f"{before}\n\n{ranking_block}\n{after}"
    return f"{before}\n\n{ranking_block}"


def parse_ranking_yaml(yaml_text: str) -> dict:
    ensure_yaml()
    data = yaml.safe_load(yaml_text)
    if not isinstance(data, dict):
        raise ValueError("Ranking response was not a YAML mapping")

    data = normalize_ranking_fields(data)

    missing = [field for field in ALL_FIELDS if field not in data]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")

    for field in RATING_FIELDS:
        value = data[field]
        if isinstance(value, str) and value.isdigit():
            value = int(value)
        if not isinstance(value, int) or value < 1 or value > 5:
            raise ValueError(f"{field} must be an integer from 1 to 5")

    rooms = data["rooms"]
    if isinstance(rooms, str) and rooms.isdigit():
        rooms = int(rooms)
    if not isinstance(rooms, int) or rooms < 0:
        raise ValueError("rooms must be a non-negative integer")

    cleaned = {}
    for field in TEXT_FIELDS:
        cleaned[field] = str(data[field]).strip()
    cleaned["rooms"] = rooms
    for field in RATING_FIELDS:
        cleaned[field] = int(data[field])
    return cleaned


def extract_yaml_from_response(text: str) -> str:
    fenced = re.search(r"```(?:ya?ml)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    return text.strip()


def call_openrouter(api_key: str, transcript: str, timeout: int = 120) -> str:
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": RATING_PROMPT + transcript,
            }
        ],
        "temperature": 0.2,
    }
    req = request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/cursor-agent/dungeon-ranking",
            "X-Title": "OPDC Dungeon Ranker",
        },
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"Unexpected OpenRouter response: {body}") from exc


def parse_ranking_block(block_text: str) -> Optional[dict]:
    ensure_yaml()
    try:
        data = yaml.safe_load(block_text)
    except yaml.YAMLError:
        return None
    if not isinstance(data, dict):
        return None
    try:
        return parse_ranking_yaml(yaml.dump(data, sort_keys=False))
    except ValueError:
        return None


def extract_ranking_from_md(path: Path) -> Optional[dict]:
    content = path.read_text(encoding="utf-8")
    match = RANKING_BLOCK_RE.search(content)
    if not match:
        return None
    data = parse_ranking_block(match.group(1))
    if data is None:
        return None
    data["source_file"] = path.name
    data["source_path"] = str(path)
    data["batch"] = path.parent.name
    return data


def title_from_filename(path: Path) -> str:
    stem = path.stem
    parts = stem.split("_", 2)
    if len(parts) >= 3:
        return parts[2].replace("_", " ")
    return stem


# ── Errors log ────────────────────────────────────────────────────────────────


@dataclass
class ErrorEntry:
    file: str
    reason: str


def write_errors_file(folder: Path, entries: list[ErrorEntry]) -> None:
    if not entries:
        if (folder / ERRORS_FILENAME).exists():
            (folder / ERRORS_FILENAME).unlink()
        return

    lines = [
        "# Ranking Errors",
        "",
        f"_Generated: {ts()}_",
        "",
        "These Markdown files were not sent to the rating model.",
        "",
        "| File | Reason |",
        "| --- | --- |",
    ]
    for entry in entries:
        lines.append(f"| `{entry.file}` | {entry.reason} |")
    lines.append("")
    (folder / ERRORS_FILENAME).write_text("\n".join(lines), encoding="utf-8")


# ── Rating workflow ───────────────────────────────────────────────────────────


@dataclass
class RatingResult:
    path: Path
    status: str
    message: str = ""


def rate_file(
    path: Path,
    api_key: str,
    force: bool = False,
    dry_run: bool = False,
) -> RatingResult:
    content = path.read_text(encoding="utf-8")

    if not has_transcription(content):
        return RatingResult(path, "error", "Missing ## Transcription section (stuck at Thinking stage)")

    if has_ranking(content) and not force:
        return RatingResult(path, "skipped", "Ranking already present")

    cleaned = strip_thinking_section(content)
    transcript = extract_transcription_body(cleaned)
    if not transcript:
        return RatingResult(path, "error", "Transcription section is empty")

    if dry_run:
        return RatingResult(path, "dry_run", "Would rate")

    raw = call_openrouter(api_key, transcript)
    yaml_text = extract_yaml_from_response(raw)
    data = parse_ranking_yaml(yaml_text)
    ranking_block = format_ranking_block(data, ts(), MODEL)
    updated = insert_ranking(cleaned, ranking_block)
    path.write_text(updated, encoding="utf-8")
    return RatingResult(path, "rated", data["title"])


def process_folder(
    folder: Path,
    api_key: str,
    force: bool = False,
    delay: float = 0.5,
    workers: int = MAX_PARALLEL_WORKERS,
    no_parallel: bool = False,
    dry_run: bool = False,
) -> list[RatingResult]:
    md_files = discover_md_files(folder)
    if not md_files:
        log(f"No .md files in {folder}")
        return []

    log(f"Processing {len(md_files)} file(s) in {folder}")

    errors: list[ErrorEntry] = []
    to_rate: list[Path] = []

    for path in md_files:
        content = path.read_text(encoding="utf-8")
        if not has_transcription(content):
            errors.append(
                ErrorEntry(path.name, "Missing ## Transcription section (stuck at Thinking stage)")
            )
            continue
        if has_ranking(content) and not force:
            continue
        to_rate.append(path)

    write_errors_file(folder, errors)

    results: list[RatingResult] = []
    for entry in errors:
        results.append(RatingResult(folder / entry.file, "error", entry.reason))

    skipped = len(md_files) - len(to_rate) - len(errors)
    for _ in range(skipped):
        pass

    for path in md_files:
        content = path.read_text(encoding="utf-8")
        if has_transcription(content) and has_ranking(content) and not force:
            results.append(RatingResult(path, "skipped", "Ranking already present"))

    if not to_rate:
        log(f"Nothing to rate in {folder} ({len(errors)} error(s), {skipped} skipped)")
        return results

    use_parallel = not no_parallel and len(to_rate) > PARALLEL_THRESHOLD
    worker_count = min(workers, len(to_rate)) if use_parallel else 1

    def work(path: Path) -> RatingResult:
        try:
            return rate_file(path, api_key, force=force, dry_run=dry_run)
        except Exception as exc:
            return RatingResult(path, "failed", str(exc))

    if use_parallel:
        log(f"Rating {len(to_rate)} file(s) with {worker_count} worker(s)")
        with ThreadPoolExecutor(max_workers=worker_count) as pool:
            futures = {pool.submit(work, path): path for path in to_rate}
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                log(f"{result.status.upper():7} {result.path.name} {result.message}".rstrip())
    else:
        log(f"Rating {len(to_rate)} file(s) sequentially")
        for index, path in enumerate(to_rate):
            result = work(path)
            results.append(result)
            log(f"{result.status.upper():7} {result.path.name} {result.message}".rstrip())
            if delay and index < len(to_rate) - 1:
                time.sleep(delay)

    return results


# ── HTML compile ──────────────────────────────────────────────────────────────


def collect_rankings(folders: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for folder in folders:
        for path in discover_md_files(folder):
            ranking = extract_ranking_from_md(path)
            if ranking:
                rows.append(ranking)
    rows.sort(key=lambda row: (row.get("batch", ""), row.get("source_file", "")))
    return rows


def average_score(rows: list[dict], field: str) -> float:
    if not rows:
        return 0.0
    return sum(int(row[field]) for row in rows) / len(rows)


def render_stars(score: int) -> str:
    return "★" * score + "☆" * (5 - score)


def resolution_tags(resolutions: str) -> str:
    tags = [tag.strip() for tag in resolutions.split(",") if tag.strip()]
    return "".join(f'<span class="tag">{html.escape(tag)}</span>' for tag in tags)


def compile_html(rows: list[dict], title: str, batches: list[str]) -> str:
    total = len(rows)
    category_avgs = {field: average_score(rows, field) for field in RATING_FIELDS}
    avg_overall = (
        sum(row_average(row) for row in rows) / len(rows) if rows else 0.0
    )
    batch_options = "".join(
        f'<option value="{html.escape(batch)}">{html.escape(batch)}</option>' for batch in batches
    )

    table_rows = []
    cards = []

    for index, row in enumerate(rows, start=1):
        avg_score = row_average(row)
        batch = row.get("batch", "")
        source_file = row.get("source_file", "")
        title_text = row.get("title", title_from_filename(Path(source_file)))
        search_blob = " ".join(
            [
                batch,
                source_file,
                title_text,
                row.get("summary_1", ""),
                row.get("summary_2", ""),
                row.get("resolutions", ""),
            ]
        ).lower()

        score_cells = "".join(
            f"""
              <td class="score" data-sort="{int(row[field])}">{render_stars(int(row[field]))}</td>"""
            for field in RATING_FIELDS
        )

        metric_cards = "".join(
            f"""
                <div><span>{html.escape(RATING_LABELS[field])}</span><strong>{render_stars(int(row[field]))}</strong></div>"""
            for field in RATING_FIELDS
        )

        table_rows.append(
            f"""
            <tr data-batch="{html.escape(batch)}" data-search="{html.escape(search_blob)}">
              <td class="num">{index}</td>
              <td class="batch">{html.escape(batch)}</td>
              <td><strong>{html.escape(title_text)}</strong><div class="file">{html.escape(source_file)}</div></td>
              <td>{int(row['rooms'])}</td>
              <td class="resolutions">{resolution_tags(row['resolutions'])}</td>{score_cells}
              <td class="score average" data-sort="{avg_score:.4f}">{avg_score:.2f}</td>
            </tr>"""
        )

        cards.append(
            f"""
            <article class="card" data-batch="{html.escape(batch)}" data-search="{html.escape(search_blob)}">
              <div class="card-head">
                <div>
                  <div class="eyebrow">{html.escape(batch)} · {int(row['rooms'])} rooms</div>
                  <h2>{html.escape(title_text)}</h2>
                </div>
                <div class="total-badge">{avg_score:.2f} avg</div>
              </div>
              <p class="summary">{html.escape(row.get('summary_1', ''))}</p>
              <p class="summary">{html.escape(row.get('summary_2', ''))}</p>
              <div class="metrics">{metric_cards}
              </div>
              <div class="resolutions">{resolution_tags(row['resolutions'])}</div>
              <div class="file">{html.escape(source_file)}</div>
            </article>"""
        )

    stat_cards = "".join(
        f"""
      <div class="stat"><span>Avg {html.escape(RATING_LABELS[field].lower())}</span><strong>{category_avgs[field]:.2f}</strong></div>"""
        for field in RATING_FIELDS
    )

    table_headers = "".join(
        f"""
            <th data-key="{field}">{html.escape(RATING_LABELS[field])}</th>"""
        for field in RATING_FIELDS
    )

    sort_keys_js = ", ".join(f"'{field}'" for field in RATING_FIELDS) + ", 'average'"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #0f1419;
      --panel: #171d25;
      --panel-2: #1f2731;
      --text: #e8edf2;
      --muted: #9aa7b5;
      --accent: #d4a55a;
      --accent-2: #7ec8c8;
      --border: #2c3642;
      --tag-bg: #243040;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Iowan Old Style", "Palatino Linotype", serif;
      background: linear-gradient(180deg, #0b1015 0%, var(--bg) 220px, var(--bg) 100%);
      color: var(--text);
      line-height: 1.5;
    }}
    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 2rem 1.25rem 4rem;
    }}
    header {{
      margin-bottom: 2rem;
    }}
    h1 {{
      margin: 0 0 0.35rem;
      font-size: clamp(2rem, 4vw, 3rem);
      letter-spacing: 0.02em;
    }}
    .lede {{
      color: var(--muted);
      max-width: 70ch;
      margin: 0;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 1rem;
      margin: 1.5rem 0 2rem;
    }}
    .stat {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1rem 1.1rem;
    }}
    .stat span {{
      display: block;
      color: var(--muted);
      font-size: 0.9rem;
      margin-bottom: 0.35rem;
    }}
    .stat strong {{
      font-size: 1.5rem;
      color: var(--accent);
    }}
    .toolbar {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.75rem;
      align-items: center;
      margin-bottom: 1rem;
    }}
    .toolbar input, .toolbar select, .toolbar button {{
      background: var(--panel);
      color: var(--text);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 0.65rem 0.85rem;
      font: inherit;
    }}
    .toolbar button {{
      cursor: pointer;
    }}
    .toolbar button.active {{
      border-color: var(--accent);
      color: var(--accent);
    }}
    .view-toggle {{
      margin-left: auto;
      display: flex;
      gap: 0.5rem;
    }}
    .table-panel, .cards {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      overflow: hidden;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
    }}
    th, td {{
      padding: 0.85rem 0.9rem;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
      text-align: left;
    }}
    th {{
      background: var(--panel-2);
      color: var(--muted);
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      cursor: pointer;
      user-select: none;
    }}
    tr:hover td {{
      background: rgba(255,255,255,0.02);
    }}
    .num, .score, .average {{
      white-space: nowrap;
      font-variant-numeric: tabular-nums;
    }}
    .score {{
      color: var(--accent-2);
      letter-spacing: 0.08em;
    }}
    .batch, .file {{
      color: var(--muted);
      font-size: 0.9rem;
    }}
    .resolutions {{
      display: flex;
      flex-wrap: wrap;
      gap: 0.35rem;
    }}
    .tag {{
      display: inline-block;
      background: var(--tag-bg);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 0.15rem 0.55rem;
      font-size: 0.78rem;
      color: #d7e2ec;
    }}
    .cards {{
      display: none;
      padding: 1rem;
      gap: 1rem;
    }}
    .cards.active {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    }}
    .card {{
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1rem;
    }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      gap: 1rem;
      align-items: start;
      margin-bottom: 0.75rem;
    }}
    .card h2 {{
      margin: 0.2rem 0 0;
      font-size: 1.2rem;
    }}
    .eyebrow {{
      color: var(--muted);
      font-size: 0.85rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .summary {{
      margin: 0 0 0.65rem;
      color: #d9e0e7;
    }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 0.5rem;
      margin: 0.75rem 0;
      font-size: 0.92rem;
    }}
    .metrics span {{
      display: block;
      color: var(--muted);
      font-size: 0.78rem;
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }}
    .total-badge {{
      background: #2d2418;
      color: var(--accent);
      border: 1px solid #4a3a24;
      border-radius: 999px;
      padding: 0.35rem 0.7rem;
      font-weight: bold;
      white-space: nowrap;
    }}
    .hidden {{ display: none !important; }}
    @media (max-width: 900px) {{
      .table-panel {{ overflow-x: auto; }}
      table {{ min-width: 1100px; }}
      .view-toggle {{ margin-left: 0; width: 100%; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>{html.escape(title)}</h1>
      <p class="lede">Compiled from standardized <code>## Ranking</code> blocks in dungeon Markdown transcripts. Each category is scored out of 5: concept originality, mechanics originality, interesting details, and map quality. The average is the mean of those four scores.</p>
    </header>

    <section class="stats">
      <div class="stat"><span>Dungeons ranked</span><strong>{total}</strong></div>
      <div class="stat"><span>Avg overall</span><strong>{avg_overall:.2f}</strong></div>{stat_cards}
    </section>

    <div class="toolbar">
      <input id="search" type="search" placeholder="Search title, summary, resolutions…" aria-label="Search">
      <select id="batch-filter" aria-label="Filter by batch">
        <option value="">All batches</option>
        {batch_options}
      </select>
      <div class="view-toggle">
        <button id="table-view" class="active" type="button">Table</button>
        <button id="card-view" type="button">Cards</button>
      </div>
    </div>

    <section id="table-panel" class="table-panel">
      <table>
        <thead>
          <tr>
            <th data-key="index">#</th>
            <th data-key="batch">Batch</th>
            <th data-key="title">Dungeon</th>
            <th data-key="rooms">Rooms</th>
            <th>Resolutions</th>{table_headers}
            <th data-key="average">Average</th>
          </tr>
        </thead>
        <tbody>
          {''.join(table_rows)}
        </tbody>
      </table>
    </section>

    <section id="cards" class="cards">
      {''.join(cards)}
    </section>
  </div>

  <script>
    const searchInput = document.getElementById('search');
    const batchFilter = document.getElementById('batch-filter');
    const tablePanel = document.getElementById('table-panel');
    const cards = document.getElementById('cards');
    const tableView = document.getElementById('table-view');
    const cardView = document.getElementById('card-view');
    const tableRows = Array.from(document.querySelectorAll('tbody tr'));
    const cardNodes = Array.from(document.querySelectorAll('.card'));

    function applyFilters() {{
      const q = searchInput.value.trim().toLowerCase();
      const batch = batchFilter.value;
      const match = (el) => {{
        const blob = el.dataset.search || '';
        const okSearch = !q || blob.includes(q);
        const okBatch = !batch || el.dataset.batch === batch;
        return okSearch && okBatch;
      }};
      tableRows.forEach((row) => row.classList.toggle('hidden', !match(row)));
      cardNodes.forEach((card) => card.classList.toggle('hidden', !match(card)));
    }}

    searchInput.addEventListener('input', applyFilters);
    batchFilter.addEventListener('change', applyFilters);

    tableView.addEventListener('click', () => {{
      tableView.classList.add('active');
      cardView.classList.remove('active');
      tablePanel.classList.remove('hidden');
      cards.classList.remove('active');
    }});

    cardView.addEventListener('click', () => {{
      cardView.classList.add('active');
      tableView.classList.remove('active');
      tablePanel.classList.add('hidden');
      cards.classList.add('active');
    }});

    document.querySelectorAll('th[data-key]').forEach((th) => {{
      th.addEventListener('click', () => {{
        const key = th.dataset.key;
        const tbody = th.closest('table').querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr')).filter((row) => !row.classList.contains('hidden'));
        const idx = Array.from(th.parentNode.children).indexOf(th);
        const asc = th.dataset.asc !== 'true';
        th.dataset.asc = asc ? 'true' : 'false';
        rows.sort((a, b) => {{
          let av, bv;
          if (key === 'title') {{
            av = a.children[2].innerText.toLowerCase();
            bv = b.children[2].innerText.toLowerCase();
          }} else if (key === 'index') {{
            av = Number(a.children[0].innerText);
            bv = Number(b.children[0].innerText);
          }} else if (key === 'batch') {{
            av = a.children[1].innerText.toLowerCase();
            bv = b.children[1].innerText.toLowerCase();
          }} else if (key === 'rooms') {{
            av = Number(a.children[3].innerText);
            bv = Number(b.children[3].innerText);
          }} else {{
            const scoreKeys = [{sort_keys_js}];
            const cell = scoreKeys.indexOf(key) + 5;
            av = Number(a.children[cell].dataset.sort || 0);
            bv = Number(b.children[cell].dataset.sort || 0);
          }}
          if (av < bv) return asc ? -1 : 1;
          if (av > bv) return asc ? 1 : -1;
          return 0;
        }});
        rows.forEach((row) => tbody.appendChild(row));
      }});
    }});
  </script>
</body>
</html>
"""


def resolve_compile_targets(folder: Path) -> tuple[list[Path], str, Path]:
    if is_year_batch_root(folder):
        year_folders = discover_year_folders(folder)
        title = f"One-Page Dungeon Rankings — {folder.name}"
        output = folder / DEFAULT_OUTPUT_HTML
        return year_folders, title, output
    title = f"One-Page Dungeon Rankings — {folder.name}"
    output = folder / DEFAULT_OUTPUT_HTML
    return [folder], title, output


def run_compile(folder: Path, output: Optional[Path] = None) -> Path:
    ensure_yaml()
    folders, title, default_output = resolve_compile_targets(folder)
    out_path = output or default_output
    rows = collect_rankings(folders)
    batches = sorted({row.get("batch", "") for row in rows if row.get("batch")})
    if not rows:
        log(f"No rankings found under {folder}")
    html_doc = compile_html(rows, title, batches)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc, encoding="utf-8")
    log(f"Wrote {len(rows)} ranking(s) to {out_path}")
    return out_path


# ── CLI ───────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Rate one-page dungeon Markdown transcripts via OpenRouter and compile HTML summaries.",
    )
    parser.add_argument(
        "folder",
        type=Path,
        help="Folder containing .md transcripts (e.g. MD-OPDC/2010) or a parent folder of year batches (e.g. MD-OPDC)",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Extract existing ## Ranking blocks and write a self-contained HTML summary",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=f"HTML output path for --compile (default: <folder>/{DEFAULT_OUTPUT_HTML})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-rate files even when a ranking block already exists",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=MAX_PARALLEL_WORKERS,
        help=f"Maximum parallel workers (default: {MAX_PARALLEL_WORKERS})",
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="Disable parallel rating",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        help="Delay between files in sequential mode (default: 0.5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate files and report actions without calling the API or writing rankings",
    )
    return parser


def main() -> int:
    ensure_yaml()
    parser = build_parser()
    args = parser.parse_args()
    folder = args.folder.resolve()

    if not folder.exists() or not folder.is_dir():
        print(f"Folder not found: {folder}", file=sys.stderr)
        return 1

    if args.compile:
        try:
            run_compile(folder, args.output.resolve() if args.output else None)
            return 0
        except Exception:
            traceback.print_exc()
            return 1

    api_key = "" if args.dry_run else resolve_api_key()

    if is_year_batch_root(folder):
        targets = discover_year_folders(folder)
        log(f"Batch root detected: {len(targets)} year folder(s) under {folder}")
    else:
        targets = [folder]

    exit_code = 0
    for target in targets:
        try:
            results = process_folder(
                target,
                api_key,
                force=args.force,
                delay=args.delay,
                workers=args.workers,
                no_parallel=args.no_parallel,
                dry_run=args.dry_run,
            )
            failed = sum(1 for result in results if result.status in {"error", "failed"})
            if failed:
                exit_code = 1
        except Exception:
            traceback.print_exc()
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
