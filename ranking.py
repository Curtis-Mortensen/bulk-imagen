#!/usr/bin/env python3
"""
One-Page Dungeon Ranker — DeepSeek V4 Flash (thinking: high) via OpenRouter
--------------------------------------------------------------------------
Rates transcribed dungeon Markdown files and appends a standardized ranking
block to the end of each file. Rankings can be compiled into HTML summaries.

Usage:
    export OPENROUTER_API_KEY="your-key-here"

    python ranking.py MD-OPDC/2010
    python ranking.py MD-OPDC
    python ranking.py MD-OPDC/2010 --force
    python ranking.py MD-OPDC/2010 --clear-ratings
    python ranking.py MD-OPDC/2010 --compile
    python ranking.py MD-OPDC --compile --output MD-OPDC/rankings.html
"""

import argparse
import getpass
import html
import json
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib import error, request

# ── Config ────────────────────────────────────────────────────────────────────

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-v4-flash"
SCORE_MIN = 1
SCORE_MAX = 10
HUMOR_MIN = 1
HUMOR_MAX = 5
CONTENT_RATINGS = ("G", "PG", "PG-13", "R", "NC-17")
DEFAULT_OUTPUT_HTML = "rankings.html"
ERRORS_FILENAME = "errors.md"
DEFAULT_LOG_DIR_NAME = "logs"
PARALLEL_THRESHOLD = 4
MAX_PARALLEL_WORKERS = 4

THINKING_RE = re.compile(r"^## Thinking\s*$", re.MULTILINE)
TRANSCRIPTION_RE = re.compile(r"^## Transcription\s*$", re.MULTILINE)

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
    "humor": "Humor",
    "content_rating": "Content Rating",
}
EXTRA_FIELDS = ("humor", "content_rating")
ALL_FIELDS = ("title", "summary", "rooms", "resolutions") + RATING_FIELDS + EXTRA_FIELDS
RANKING_FIELD_ORDER = ALL_FIELDS + ("rated_at", "model")
HALF_POINT_RE = r"\d+(?:\.5)?"
CONTENT_RATING_RE = "|".join(re.escape(rating) for rating in CONTENT_RATINGS)

RANKING_TAIL_RE = re.compile(
    r"(?:^|\n)"
    r"title: .+\n"
    r"summary: .+\n"
    r"rooms: \d+\n"
    r"resolutions: .+\n"
    rf"concept_originality: {HALF_POINT_RE}\n"
    rf"mechanics_originality: {HALF_POINT_RE}\n"
    rf"interesting_details: {HALF_POINT_RE}\n"
    rf"map_quality: {HALF_POINT_RE}\n"
    rf"humor: {HALF_POINT_RE}\n"
    rf"content_rating: (?:{CONTENT_RATING_RE})\n"
    r"rated_at: .+\n"
    r"model: .+\n?\Z",
    re.DOTALL,
)

OLD_RANKING_BLOCK_RE = re.compile(
    r"\n## Ranking\s*\n\s*<!-- RANKING:BEGIN -->.*?<!-- RANKING:END -->\s*",
    re.DOTALL,
)

RATING_PROMPT = """You are rating one-page D&D dungeons from their Markdown transcripts.

Return ONLY valid JSON (no markdown fences, no commentary) with exactly these keys:

{
  "title": "<dungeon title>",
  "summary": "<at most 2 sentences total describing the dungeon; be concise>",
  "rooms": <integer count of keyed rooms/locations; best estimate if unclear>,
  "resolutions": "<comma-separated list drawn only from: Combat, Diplomacy, Puzzles, Fetch Quests, Stealth, Roleplay, Traps, Exploration, Skill Challenges, Social. Include only resolutions explicitly supported by the text.>",
  "concept_originality": <number from 1 to 10 in 0.5 increments, e.g. 3.5>,
  "mechanics_originality": <number from 1 to 10 in 0.5 increments>,
  "interesting_details": <number from 1 to 10 in 0.5 increments>,
  "map_quality": <number from 1 to 10 in 0.5 increments>,
  "humor": <number from 1 to 5 in 0.5 increments; how funny, playful, or comedic the dungeon is; does not count toward the overall average>,
  "content_rating": "<one of: G, PG, PG-13, R, NC-17 based on violence, horror, gore, and mature themes in the text>"
}

Scoring guide:
- concept_originality: how far the premise departs from a traditional dungeon crawl (room-by-room combat in underground halls). 10 = highly unconventional setting, structure, or frame; 1 = textbook dungeon crawl.
- mechanics_originality: originality of mechanics, puzzles, and encounter design. 10 = clever or novel systems; 1 = combat-only room clearing.
- interesting_details: density of memorable flavor, NPCs, set dressing, and surprising touches. 10 = rich distinctive details throughout; 1 = bare rooms with little flavor.
- map_quality: how impressive the map sounds versus a plain grid. 10 = multi-level, unusual topology, ship, planet, etc.; 1 = plain grid corridors.
- humor: comedic intent, jokes, absurdity, and playful tone. 5 = consistently funny or delightfully silly; 1 = entirely serious.
- content_rating: G = family-friendly; PG = mild peril; PG-13 = moderate violence or frightening content; R = strong violence, horror, or mature themes; NC-17 = extreme gore, cruelty, or explicit mature content.

The summary MUST be no longer than 2 sentences.

Transcript:
"""

_log_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ts(dt: Optional[datetime] = None) -> str:
    dt = dt or utc_now()
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def log(message: str) -> None:
    print(f"[{ts()}] {message}", flush=True)


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
    return len(discover_md_files(folder)) == 0


def has_transcription(content: str) -> bool:
    return TRANSCRIPTION_RE.search(content) is not None


def has_ranking(content: str) -> bool:
    return RANKING_TAIL_RE.search(content) is not None


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
    body = content[match.end() :]
    body = strip_ranking_tail(body)
    return body.strip()


def strip_old_ranking_blocks(content: str) -> str:
    return OLD_RANKING_BLOCK_RE.sub("\n", content)


def strip_ranking_tail(content: str) -> str:
    content = strip_old_ranking_blocks(content)
    match = RANKING_TAIL_RE.search(content)
    if not match:
        return content
    return content[: match.start()].rstrip() + "\n"


def parse_half_score(value, min_val: float, max_val: float, field: str) -> float:
    if isinstance(value, str):
        value = value.strip()
        if not value:
            raise ValueError(f"{field} is required")
        value = float(value)
    elif isinstance(value, int):
        value = float(value)
    elif not isinstance(value, float):
        raise ValueError(f"{field} must be a number from {min_val} to {max_val} in 0.5 increments")

    if value < min_val or value > max_val:
        raise ValueError(f"{field} must be a number from {min_val} to {max_val} in 0.5 increments")
    if abs(value * 2 - round(value * 2)) > 1e-9:
        raise ValueError(f"{field} must use 0.5 increments")
    return value


def row_average(row: dict) -> float:
    return sum(float(row[field]) for field in RATING_FIELDS) / len(RATING_FIELDS)


def format_ranking_block(data: dict, rated_at: str, model: str) -> str:
    lines = []
    for field in RANKING_FIELD_ORDER:
        if field in ("rated_at", "model"):
            value = rated_at if field == "rated_at" else model
        else:
            value = data[field]
        lines.append(f"{field}: {value}")
    return "\n".join(lines) + "\n"


def append_ranking(content: str, ranking_block: str) -> str:
    base = strip_ranking_tail(content).rstrip()
    return f"{base}\n{ranking_block}"


def parse_ranking_lines(block: str) -> dict:
    data: dict[str, str] = {}
    for line in block.strip().splitlines():
        if ": " not in line:
            continue
        key, value = line.split(": ", 1)
        data[key.strip()] = value.strip()
    return parse_ranking_data(data)


def parse_ranking_data(data: dict) -> dict:
    if not isinstance(data, dict):
        raise ValueError("Ranking response was not a JSON object")

    missing = [field for field in ALL_FIELDS if field not in data]
    if missing:
        raise ValueError(f"Missing fields: {', '.join(missing)}")

    rooms = data["rooms"]
    if isinstance(rooms, str) and rooms.isdigit():
        rooms = int(rooms)
    if not isinstance(rooms, int) or rooms < 0:
        raise ValueError("rooms must be a non-negative integer")

    cleaned: dict = {}
    cleaned["title"] = str(data["title"]).strip()
    cleaned["summary"] = str(data["summary"]).strip()
    cleaned["rooms"] = rooms
    cleaned["resolutions"] = str(data["resolutions"]).strip()
    for field in RATING_FIELDS:
        cleaned[field] = parse_half_score(data[field], SCORE_MIN, SCORE_MAX, field)

    cleaned["humor"] = parse_half_score(data["humor"], HUMOR_MIN, HUMOR_MAX, "humor")

    content_rating = str(data["content_rating"]).strip()
    if content_rating not in CONTENT_RATINGS:
        raise ValueError(f"content_rating must be one of: {', '.join(CONTENT_RATINGS)}")
    cleaned["content_rating"] = content_rating
    return cleaned


def extract_json_from_response(text: str) -> str:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def append_log(log_file: Path, line: str) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with _log_lock:
        with log_file.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


def stream_openrouter(api_key: str, transcript: str, log_file: Path, timeout: int = 300) -> str:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": RATING_PROMPT + transcript}],
        "stream": True,
        "reasoning": {"effort": "high"},
        "response_format": {"type": "json_object"},
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

    append_log(log_file, f"[{ts()}] REQUEST_START model={MODEL} reasoning=high")
    reasoning_parts: list[str] = []
    content_parts: list[str] = []

    try:
        with request.urlopen(req, timeout=timeout) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str == "[DONE]":
                    append_log(log_file, f"[{ts()}] STREAM_DONE")
                    break
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    append_log(log_file, f"[{ts()}] CHUNK_PARSE_ERROR {data_str[:200]}")
                    continue

                choice = (chunk.get("choices") or [{}])[0]
                delta = choice.get("delta") or {}
                message = choice.get("message") or {}

                for key in ("reasoning", "reasoning_content"):
                    piece = delta.get(key) or message.get(key)
                    if piece:
                        reasoning_parts.append(piece)
                        append_log(log_file, f"[{ts()}] REASONING {piece}")

                for detail in delta.get("reasoning_details") or message.get("reasoning_details") or []:
                    if not isinstance(detail, dict):
                        continue
                    piece = detail.get("text") or detail.get("content") or ""
                    if piece:
                        reasoning_parts.append(piece)
                        append_log(log_file, f"[{ts()}] REASONING_DETAIL {piece}")

                piece = delta.get("content") or message.get("content")
                if piece:
                    content_parts.append(piece)
                    append_log(log_file, f"[{ts()}] CONTENT {piece}")

                usage = chunk.get("usage")
                if usage:
                    append_log(log_file, f"[{ts()}] USAGE {json.dumps(usage)}")

    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        append_log(log_file, f"[{ts()}] HTTP_ERROR {exc.code} {detail}")
        raise RuntimeError(f"OpenRouter HTTP {exc.code}: {detail}") from exc
    except error.URLError as exc:
        append_log(log_file, f"[{ts()}] URL_ERROR {exc}")
        raise RuntimeError(f"OpenRouter request failed: {exc}") from exc

    reasoning_text = "".join(reasoning_parts)
    content_text = "".join(content_parts)
    append_log(log_file, f"[{ts()}] REQUEST_END reasoning_chars={len(reasoning_text)} content_chars={len(content_text)}")
    if not content_text:
        raise RuntimeError("OpenRouter returned no content")
    return content_text


def extract_ranking_from_md(path: Path) -> Optional[dict]:
    content = path.read_text(encoding="utf-8")
    match = RANKING_TAIL_RE.search(content)
    if not match:
        return None
    try:
        data = parse_ranking_lines(match.group(0))
    except ValueError:
        return None
    data["source_file"] = path.name
    data["source_path"] = str(path)
    data["batch"] = path.parent.name
    author_first, author_last = author_from_filename(path)
    data["author_first"] = author_first
    data["author_last"] = author_last
    return data


def author_from_filename(path: Path) -> tuple[str, str]:
    parts = path.stem.split("_", 2)
    if len(parts) < 2:
        return "", path.stem
    author_part = parts[1]
    spaced = re.sub(r"([a-z])([A-Z])", r"\1 \2", author_part)
    spaced = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", spaced)
    words = [word for word in spaced.replace("and", " ").split() if word]
    if not words:
        return "", author_part
    if len(words) == 1:
        return "", words[0]
    return words[0], words[-1]


def title_from_filename(path: Path) -> str:
    stem = path.stem
    parts = stem.split("_", 2)
    if len(parts) >= 3:
        return parts[2].replace("_", " ")
    return stem


def ranking_log_path(log_dir: Path, md_path: Path) -> Path:
    return log_dir / f"ranking_{md_path.stem}.log"


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
    log_dir: Path,
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

    log_file = ranking_log_path(log_dir, path)
    if log_file.exists():
        log_file.unlink()

    raw = stream_openrouter(api_key, transcript, log_file)
    data = parse_ranking_data(json.loads(extract_json_from_response(raw)))
    ranking_block = format_ranking_block(data, ts(), MODEL)
    updated = append_ranking(cleaned, ranking_block)
    path.write_text(updated, encoding="utf-8")
    return RatingResult(path, "rated", data["title"])


def clear_ratings_in_folder(folder: Path) -> int:
    cleared = 0
    for path in discover_md_files(folder):
        content = path.read_text(encoding="utf-8")
        stripped = strip_ranking_tail(content)
        if stripped != content:
            path.write_text(stripped, encoding="utf-8")
            cleared += 1
            log(f"CLEARED {path.name}")
    return cleared


def clear_ratings(folder: Path) -> int:
    if is_year_batch_root(folder):
        total = 0
        for target in discover_year_folders(folder):
            count = clear_ratings_in_folder(target)
            log(f"Cleared {count} ranking(s) in {target}")
            total += count
        return total
    count = clear_ratings_in_folder(folder)
    log(f"Cleared {count} ranking(s) in {folder}")
    return count


def process_folder(
    folder: Path,
    api_key: str,
    log_dir: Path,
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
            return rate_file(path, api_key, log_dir, force=force, dry_run=dry_run)
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
    return sum(float(row[field]) for row in rows) / len(rows)


def render_score(score: float, max_score: int = SCORE_MAX) -> str:
    if float(score).is_integer():
        return f"{int(score)}/{max_score}"
    return f"{score:g}/{max_score}"


def content_rating_sort_value(rating: str) -> int:
    order = {rating: index for index, rating in enumerate(CONTENT_RATINGS, start=1)}
    return order.get(rating, 0)


def format_author_name(row: dict) -> str:
    first = row.get("author_first", "")
    last = row.get("author_last", "")
    if first and last:
        return f"{first} {last}"
    return last or first or "Unknown"


def resolution_tags(resolutions: str) -> str:
    tags = [tag.strip() for tag in resolutions.split(",") if tag.strip()]
    return "".join(f'<span class="tag">{html.escape(tag)}</span>' for tag in tags)


def compile_html(rows: list[dict], title: str, batches: list[str]) -> str:
    total = len(rows)
    category_avgs = {field: average_score(rows, field) for field in RATING_FIELDS}
    avg_humor = average_score(rows, "humor") if rows else 0.0
    avg_overall = sum(row_average(row) for row in rows) / len(rows) if rows else 0.0
    batch_options = "".join(
        f'<option value="{html.escape(batch)}">{html.escape(batch)}</option>' for batch in batches
    )

    sort_options = [
        ("average:desc", "Average (high to low)"),
        ("average:asc", "Average (low to high)"),
        ("concept_originality:desc", "Concept originality (high to low)"),
        ("mechanics_originality:desc", "Mechanics originality (high to low)"),
        ("interesting_details:desc", "Interesting details (high to low)"),
        ("map_quality:desc", "Map quality (high to low)"),
        ("humor:desc", "Humor (high to low)"),
        ("humor:asc", "Humor (low to high)"),
        ("content_rating:asc", "Content rating (mildest first)"),
        ("content_rating:desc", "Content rating (strongest first)"),
        ("author_last:asc", "Author last name (A–Z)"),
        ("author_last:desc", "Author last name (Z–A)"),
        ("title:asc", "Title (A–Z)"),
        ("title:desc", "Title (Z–A)"),
        ("rooms:desc", "Rooms (most first)"),
        ("rooms:asc", "Rooms (fewest first)"),
        ("batch:asc", "Batch"),
    ]
    sort_options_html = "".join(
        f'<option value="{html.escape(value)}"{" selected" if value == "average:desc" else ""}>'
        f"{html.escape(label)}</option>"
        for value, label in sort_options
    )

    table_rows = []
    cards = []

    for index, row in enumerate(rows, start=1):
        avg_score = row_average(row)
        batch = row.get("batch", "")
        source_file = row.get("source_file", "")
        title_text = row.get("title", title_from_filename(Path(source_file)))
        author_name = format_author_name(row)
        author_last = row.get("author_last", "")
        humor_score = float(row["humor"])
        content_rating = row["content_rating"]
        search_blob = " ".join(
            [
                batch,
                source_file,
                title_text,
                author_name,
                author_last,
                row.get("summary", ""),
                row.get("resolutions", ""),
                content_rating,
            ]
        ).lower()

        sort_values = {
            "average": f"{avg_score:.4f}",
            "title": title_text.lower(),
            "author_last": author_last.lower(),
            "summary": row.get("summary", "").lower(),
            "batch": batch.lower(),
            "rooms": str(int(row["rooms"])),
            "humor": f"{humor_score:.4f}",
            "content_rating": str(content_rating_sort_value(content_rating)),
        }
        for field in RATING_FIELDS:
            sort_values[field] = f"{float(row[field]):.4f}"

        sort_attrs = " ".join(
            f'data-sort-{key.replace("_", "-")}="{html.escape(value, quote=True)}"'
            for key, value in sort_values.items()
        )

        score_cells = "".join(
            f"""
              <td class="score" data-sort="{float(row[field]):.4f}">{render_score(float(row[field]))}</td>"""
            for field in RATING_FIELDS
        )

        metric_cards = "".join(
            f"""
                <div><span>{html.escape(RATING_LABELS[field])}</span><strong>{render_score(float(row[field]))}</strong></div>"""
            for field in RATING_FIELDS
        )

        table_rows.append(
            f"""
            <tr data-row-id="{index}" data-batch="{html.escape(batch)}" data-search="{html.escape(search_blob)}" {sort_attrs}>
              <td class="num">{index}</td>
              <td class="batch">{html.escape(batch)}</td>
              <td><strong>{html.escape(title_text)}</strong><div class="file">{html.escape(source_file)}</div></td>
              <td>{html.escape(author_name)}</td>
              <td class="summary">{html.escape(row.get('summary', ''))}</td>
              <td>{int(row['rooms'])}</td>
              <td class="resolutions">{resolution_tags(row['resolutions'])}</td>{score_cells}
              <td class="score" data-sort="{humor_score:.4f}">{render_score(humor_score, HUMOR_MAX)}</td>
              <td class="rating" data-sort="{content_rating_sort_value(content_rating)}"><span class="tag rating-tag">{html.escape(content_rating)}</span></td>
              <td class="score average" data-sort="{avg_score:.4f}">{avg_score:.2f}</td>
            </tr>"""
        )

        cards.append(
            f"""
            <article class="card" data-row-id="{index}" data-batch="{html.escape(batch)}" data-search="{html.escape(search_blob)}" {sort_attrs}>
              <div class="card-head">
                <div>
                  <div class="eyebrow">{html.escape(batch)} · {int(row['rooms'])} rooms · {html.escape(author_name)}</div>
                  <h2>{html.escape(title_text)}</h2>
                </div>
                <div class="total-badge">{avg_score:.2f} avg</div>
              </div>
              <p class="summary">{html.escape(row.get('summary', ''))}</p>
              <div class="metrics">{metric_cards}
                <div><span>Humor</span><strong>{render_score(humor_score, HUMOR_MAX)}</strong></div>
                <div><span>Content rating</span><strong><span class="tag rating-tag">{html.escape(content_rating)}</span></strong></div>
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
            <th data-key="{field}" data-sort-field="{field}">{html.escape(RATING_LABELS[field])}</th>"""
        for field in RATING_FIELDS
    )

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
    .wrap {{ max-width: 1280px; margin: 0 auto; padding: 2rem 1.25rem 4rem; }}
    h1 {{ margin: 0 0 0.35rem; font-size: clamp(2rem, 4vw, 3rem); }}
    .lede {{ color: var(--muted); max-width: 75ch; margin: 0; }}
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
    .stat span {{ display: block; color: var(--muted); font-size: 0.9rem; margin-bottom: 0.35rem; }}
    .stat strong {{ font-size: 1.5rem; color: var(--accent); }}
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
    .toolbar button {{ cursor: pointer; }}
    .toolbar button.active {{ border-color: var(--accent); color: var(--accent); }}
    .view-toggle {{ margin-left: auto; display: flex; gap: 0.5rem; }}
    .table-panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
      overflow-x: auto;
    }}
    .cards {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 16px;
    }}
    table {{ width: 100%; border-collapse: collapse; }}
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
    tr:hover td {{ background: rgba(255,255,255,0.02); }}
    .num, .score, .average {{ white-space: nowrap; font-variant-numeric: tabular-nums; }}
    .score {{ color: var(--accent-2); }}
    .batch, .file {{ color: var(--muted); font-size: 0.9rem; }}
    .summary {{ color: #d9e0e7; max-width: 36ch; }}
    .resolutions {{ display: flex; flex-wrap: wrap; gap: 0.35rem; }}
    .tag {{
      display: inline-block;
      background: var(--tag-bg);
      border: 1px solid var(--border);
      border-radius: 999px;
      padding: 0.15rem 0.55rem;
      font-size: 0.78rem;
      color: #d7e2ec;
    }}
    .rating-tag {{ color: #f0d7a8; border-color: #4a3a24; background: #2d2418; }}
    .cards {{ display: none; padding: 1rem; gap: 1rem; }}
    .cards.active {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); }}
    .card {{
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 1rem;
    }}
    .card-head {{ display: flex; justify-content: space-between; gap: 1rem; align-items: start; margin-bottom: 0.75rem; }}
    .card h2 {{ margin: 0.2rem 0 0; font-size: 1.2rem; }}
    .eyebrow {{ color: var(--muted); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.05em; }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(3, 1fr);
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
      table {{ min-width: 1680px; }}
      .view-toggle {{ margin-left: 0; width: 100%; }}
      .metrics {{ grid-template-columns: repeat(2, 1fr); }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <header>
      <h1>{html.escape(title)}</h1>
      <p class="lede">Compiled from trailing ranking blocks in dungeon Markdown transcripts. Scored categories use a 1–10 half-point scale; humor is scored 1–5 and does not count toward the average. Content ratings follow the movie scale (G, PG, PG-13, R, NC-17).</p>
    </header>

    <section class="stats">
      <div class="stat"><span>Dungeons ranked</span><strong>{total}</strong></div>
      <div class="stat"><span>Avg overall</span><strong>{avg_overall:.2f}</strong></div>
      <div class="stat"><span>Avg humor</span><strong>{avg_humor:.2f}</strong></div>{stat_cards}
    </section>

    <div class="toolbar">
      <input id="search" type="search" placeholder="Search title, author, summary, resolutions…" aria-label="Search">
      <select id="batch-filter" aria-label="Filter by batch">
        <option value="">All batches</option>
        {batch_options}
      </select>
      <select id="sort-by" aria-label="Sort by">
        {sort_options_html}
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
            <th data-key="index" data-sort-field="index">#</th>
            <th data-key="batch" data-sort-field="batch">Batch</th>
            <th data-key="title" data-sort-field="title">Dungeon</th>
            <th data-key="author_last" data-sort-field="author_last">Author</th>
            <th data-key="summary" data-sort-field="summary">Summary</th>
            <th data-key="rooms" data-sort-field="rooms">Rooms</th>
            <th>Resolutions</th>{table_headers}
            <th data-key="humor" data-sort-field="humor">Humor</th>
            <th data-key="content_rating" data-sort-field="content_rating">Rating</th>
            <th data-key="average" data-sort-field="average">Average</th>
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
    const sortSelect = document.getElementById('sort-by');
    const tablePanel = document.getElementById('table-panel');
    const cardsPanel = document.getElementById('cards');
    const tableView = document.getElementById('table-view');
    const cardView = document.getElementById('card-view');
    const tbody = document.querySelector('tbody');
    let tableRows = Array.from(document.querySelectorAll('tbody tr'));
    let cardNodes = Array.from(document.querySelectorAll('.card'));
    let sortField = 'average';
    let sortAscending = false;

    const STRING_SORT_FIELDS = new Set(['title', 'author_last', 'batch', 'summary']);

    function parseSortValue(value) {{
      const [field, direction] = value.split(':');
      return [field, direction === 'asc'];
    }}

    function sortDatasetKey(key) {{
      return 'sort' + key.split('_').map((part) => part.charAt(0).toUpperCase() + part.slice(1)).join('');
    }}

    function readSortValue(el, key) {{
      if (key === 'summary') {{
        return (el.querySelector('.summary')?.textContent || '').toLowerCase();
      }}
      const datasetKey = sortDatasetKey(key);
      const raw = el.dataset[datasetKey];
      if (STRING_SORT_FIELDS.has(key)) {{
        return (raw || '').toLowerCase();
      }}
      return Number(raw || 0);
    }}

    function compareValues(av, bv, asc) {{
      if (av < bv) return asc ? -1 : 1;
      if (av > bv) return asc ? 1 : -1;
      return 0;
    }}

    function renumberRows() {{
      let position = 1;
      tableRows.forEach((row) => {{
        if (!row.classList.contains('hidden')) {{
          row.querySelector('.num').textContent = String(position++);
        }}
      }});
    }}

    function applySort() {{
      const asc = sortAscending;
      const field = sortField;
      const visibleRows = tableRows.filter((row) => !row.classList.contains('hidden'));
      const visibleCards = cardNodes.filter((card) => !card.classList.contains('hidden'));

      visibleRows.sort((a, b) => compareValues(readSortValue(a, field), readSortValue(b, field), asc));
      visibleCards.sort((a, b) => compareValues(readSortValue(a, field), readSortValue(b, field), asc));

      visibleRows.forEach((row) => tbody.appendChild(row));
      visibleCards.forEach((card) => cardsPanel.appendChild(card));
      renumberRows();
    }}

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
      applySort();
    }}

    function setSortFromHeader(field, asc) {{
      sortField = field;
      sortAscending = asc;
      const preferred = `${{field}}:${{asc ? 'asc' : 'desc'}}`;
      const fallback = `${{field}}:desc`;
      const option = Array.from(sortSelect.options).find((entry) => entry.value === preferred)
        || Array.from(sortSelect.options).find((entry) => entry.value === fallback);
      if (option) {{
        sortSelect.value = option.value;
      }}
      applySort();
    }}

    searchInput.addEventListener('input', applyFilters);
    batchFilter.addEventListener('change', applyFilters);
    sortSelect.addEventListener('change', () => {{
      [sortField, sortAscending] = parseSortValue(sortSelect.value);
      applySort();
    }});

    tableView.addEventListener('click', () => {{
      tableView.classList.add('active');
      cardView.classList.remove('active');
      tablePanel.classList.remove('hidden');
      cardsPanel.classList.remove('active');
    }});

    cardView.addEventListener('click', () => {{
      cardView.classList.add('active');
      tableView.classList.remove('active');
      tablePanel.classList.add('hidden');
      cardsPanel.classList.add('active');
    }});

    document.querySelectorAll('th[data-sort-field]').forEach((th) => {{
      th.addEventListener('click', () => {{
        const field = th.dataset.sortField;
        if (field === 'index') {{
          return;
        }}
        const asc = th.dataset.asc !== 'true';
        document.querySelectorAll('th[data-sort-field]').forEach((header) => {{
          header.dataset.asc = header === th && asc ? 'true' : 'false';
        }});
        setSortFromHeader(field, asc);
      }});
    }});

    applySort();
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


def resolve_log_dir(folder: Path, log_dir: Optional[Path]) -> Path:
    if log_dir:
        return log_dir
    if is_year_batch_root(folder):
        return folder / DEFAULT_LOG_DIR_NAME
    return folder / DEFAULT_LOG_DIR_NAME


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
        help="Extract trailing ranking blocks and write a self-contained HTML summary",
    )
    parser.add_argument(
        "--clear-ratings",
        action="store_true",
        help="Remove trailing ranking blocks from .md files in the target folder(s)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help=f"HTML output path for --compile (default: <folder>/{DEFAULT_OUTPUT_HTML})",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help=f"Directory for per-file OpenRouter logs (default: <folder>/{DEFAULT_LOG_DIR_NAME})",
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

    if args.clear_ratings:
        try:
            clear_ratings(folder)
            return 0
        except Exception:
            traceback.print_exc()
            return 1

    api_key = "" if args.dry_run else resolve_api_key()
    log_dir = args.log_dir.resolve() if args.log_dir else resolve_log_dir(folder, None)

    if is_year_batch_root(folder):
        targets = discover_year_folders(folder)
        log(f"Batch root detected: {len(targets)} year folder(s) under {folder}")
    else:
        targets = [folder]

    exit_code = 0
    for target in targets:
        target_log_dir = args.log_dir.resolve() if args.log_dir else target / DEFAULT_LOG_DIR_NAME
        try:
            results = process_folder(
                target,
                api_key,
                target_log_dir,
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
