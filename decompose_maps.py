#!/usr/bin/env python3
"""
D&D Map Decomposer — Gemini 3.1 Flash-Lite via Google AI Studio
-----------------------------------------------------------------
Batch-transcribes PDF/PNG/JPEG dungeon maps into Markdown using Gemini 3.1
Flash-Lite with medium reasoning and streamed thinking summaries.

Usage:
    python decompose_maps.py MAPS_FOLDER [--log-dir LOG_DIR] [--force] [--delay SECONDS]

Install deps:
    pip install google-genai
"""

dependencies = [
    "google-genai",
]

import argparse
import getpass
import json
import os
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# ── Config ────────────────────────────────────────────────────────────────────

MODEL = "gemini-3.1-flash-lite"
PARALLEL_THRESHOLD = 4
MAX_PARALLEL_WORKERS = 4
SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}
MIME_TYPES = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
}

DND_PROMPT = (
    "I have here a one page D&D dungeon concept. Start by describing the page, "
    "then transcribe all the written text, then describe the map with enough "
    "detail for someone who can't see it to play."
)

BLOCKED_FINISH_REASONS = {
    "SAFETY",
    "RECITATION",
    "LANGUAGE",
    "BLOCKLIST",
    "PROHIBITED_CONTENT",
    "SPII",
    "IMAGE_SAFETY",
    "IMAGE_PROHIBITED_CONTENT",
    "IMAGE_RECITATION",
    "IMAGE_OTHER",
    "MAX_TOKENS",
}


# ── Logging ───────────────────────────────────────────────────────────────────

def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def ts(dt: Optional[datetime] = None) -> str:
    dt = dt or utc_now()
    return dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


class DecomposeLogger:
    """Thread-safe session logger that mirrors messages to stdout and a session log."""

    def __init__(self, session_log: Path):
        self.session_log = session_log
        self.session_log.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._write_session(f"SESSION START {ts()}")
        self._write_session(f"log_file={session_log}")

    def _append(self, path: Path, line: str) -> None:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def _write_session(self, message: str) -> None:
        line = f"[{ts()}] {message}"
        with self._lock:
            print(line, flush=True)
            self._append(self.session_log, line)

    def file_logger(self, file_log: Path) -> "FileLogger":
        return FileLogger(self, file_log)

    def info(self, event: str, **details) -> None:
        detail_str = ""
        if details:
            detail_str = " " + json.dumps(details, default=str, sort_keys=True)
        self._write_session(f"{event}{detail_str}")

    def close(self) -> None:
        self._write_session(f"SESSION END {ts()}")


class FileLogger:
    """Per-map logger that writes to both the session log and a dedicated file log."""

    def __init__(self, session: DecomposeLogger, file_log: Path):
        self.session = session
        self.file_log = file_log
        self.file_log.parent.mkdir(parents=True, exist_ok=True)

    def _emit(self, line: str) -> None:
        with self.session._lock:
            print(line, flush=True)
            self.session._append(self.session.session_log, line)
        self.session._append(self.file_log, line)

    def info(self, event: str, **details) -> None:
        detail_str = ""
        if details:
            detail_str = " " + json.dumps(details, default=str, sort_keys=True)
        self._emit(f"[{ts()}] {event}{detail_str}")


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class StreamResult:
    thinking_text: str = ""
    response_text: str = ""
    finish_reason: Optional[str] = None
    prompt_block_reason: Optional[str] = None
    prompt_block_message: Optional[str] = None
    chunk_count: int = 0
    thinking_chunks: int = 0
    response_chunks: int = 0
    usage_metadata: Optional[dict] = None
    stream_started_at: Optional[str] = None
    stream_ended_at: Optional[str] = None
    errors: list[str] = field(default_factory=list)

    @property
    def was_blocked(self) -> bool:
        if self.prompt_block_reason:
            return True
        if self.finish_reason and self.finish_reason in BLOCKED_FINISH_REASONS:
            return True
        return False

    @property
    def block_description(self) -> Optional[str]:
        if self.prompt_block_reason:
            parts = [f"Prompt blocked: {self.prompt_block_reason}"]
            if self.prompt_block_message:
                parts.append(self.prompt_block_message)
            return " — ".join(parts)
        if self.finish_reason and self.finish_reason in BLOCKED_FINISH_REASONS:
            return f"Output blocked: {self.finish_reason}"
        return None


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_api_key() -> str:
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        api_key = getpass.getpass("Paste your Google AI Studio API key: ").strip()
    if not api_key:
        print("[ERROR] No API key provided.")
        sys.exit(1)
    return api_key


def discover_maps(folder: Path) -> list[Path]:
    maps = [
        p for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    return maps


def build_contents(map_path: Path):
    from google.genai import types

    mime = MIME_TYPES[map_path.suffix.lower()]
    file_bytes = map_path.read_bytes()
    return [
        types.Part.from_bytes(data=file_bytes, mime_type=mime),
        types.Part.from_text(text=DND_PROMPT),
    ]


def extract_chunk_parts(chunk) -> list:
    if not chunk.candidates:
        return []
    candidate = chunk.candidates[0]
    if not candidate.content or not candidate.content.parts:
        return []
    return candidate.content.parts


def serialize_usage(usage) -> Optional[dict]:
    if usage is None:
        return None
    if hasattr(usage, "model_dump"):
        return usage.model_dump(exclude_none=True)
    return str(usage)


def stream_decompose(client, map_path: Path, logger: FileLogger) -> StreamResult:
    from google.genai import types

    result = StreamResult()
    contents = build_contents(map_path)
    config = types.GenerateContentConfig(
        thinking_config=types.ThinkingConfig(
            thinking_level="medium",
            include_thoughts=True,
        ),
    )

    logger.info(
        "REQUEST SEND",
        model=MODEL,
        map_path=str(map_path),
        map_size_bytes=map_path.stat().st_size,
        mime_type=MIME_TYPES[map_path.suffix.lower()],
        thinking_level="medium",
        include_thoughts=True,
        prompt=DND_PROMPT,
    )

    send_at = ts()
    result.stream_started_at = send_at
    logger.info("STREAM START", sent_at=send_at)

    try:
        stream = client.models.generate_content_stream(
            model=MODEL,
            contents=contents,
            config=config,
        )

        for chunk in stream:
            result.chunk_count += 1

            if chunk.prompt_feedback:
                pf = chunk.prompt_feedback
                if pf.block_reason:
                    result.prompt_block_reason = str(pf.block_reason)
                if pf.block_reason_message:
                    result.prompt_block_message = pf.block_reason_message
                logger.info(
                    "PROMPT_FEEDBACK",
                    chunk=result.chunk_count,
                    block_reason=result.prompt_block_reason,
                    block_reason_message=result.prompt_block_message,
                )

            if chunk.candidates:
                candidate = chunk.candidates[0]
                if candidate.finish_reason:
                    result.finish_reason = str(candidate.finish_reason)
                    logger.info(
                        "FINISH_REASON",
                        chunk=result.chunk_count,
                        finish_reason=result.finish_reason,
                    )

            for part in extract_chunk_parts(chunk):
                if not part.text:
                    continue
                if part.thought:
                    result.thinking_text += part.text
                    result.thinking_chunks += 1
                    logger.info(
                        "STREAM RECEIVED",
                        chunk=result.chunk_count,
                        kind="thinking",
                        chars=len(part.text),
                        preview=part.text[:120],
                    )
                else:
                    result.response_text += part.text
                    result.response_chunks += 1
                    logger.info(
                        "STREAM RECEIVED",
                        chunk=result.chunk_count,
                        kind="response",
                        chars=len(part.text),
                        preview=part.text[:120],
                    )

            if chunk.usage_metadata:
                result.usage_metadata = serialize_usage(chunk.usage_metadata)
                logger.info(
                    "USAGE_METADATA",
                    chunk=result.chunk_count,
                    usage=result.usage_metadata,
                )

    except Exception as exc:
        err = f"{type(exc).__name__}: {exc}"
        result.errors.append(err)
        logger.info("STREAM ERROR", error=err, traceback=traceback.format_exc())

    result.stream_ended_at = ts()
    logger.info(
        "STREAM END",
        ended_at=result.stream_ended_at,
        chunk_count=result.chunk_count,
        thinking_chars=len(result.thinking_text),
        response_chars=len(result.response_text),
        finish_reason=result.finish_reason,
        blocked=result.was_blocked,
        block_description=result.block_description,
    )
    return result


def build_markdown(map_path: Path, result: StreamResult) -> str:
    lines = [
        f"# {map_path.stem}",
        "",
        f"_Source: `{map_path.name}`_",
        f"_Processed: {ts()}_",
        f"_Model: `{MODEL}` (medium reasoning)_",
        "",
    ]

    if result.was_blocked:
        lines.extend([
            "> **Blocked output**",
            f"> {result.block_description}",
            "",
        ])

    if result.errors:
        lines.extend([
            "## Errors",
            "",
        ])
        for err in result.errors:
            lines.append(f"- {err}")
        lines.append("")

    if result.thinking_text.strip():
        lines.extend([
            "## Thinking",
            "",
            result.thinking_text.strip(),
            "",
        ])

    if result.response_text.strip():
        lines.extend([
            "## Transcription",
            "",
            result.response_text.strip(),
            "",
        ])
    elif not result.thinking_text.strip():
        lines.extend([
            "## Transcription",
            "",
            "_No model output was received._",
            "",
        ])

    if result.was_blocked or result.errors:
        lines.extend([
            "## Processing Notes",
            "",
        ])
        if result.block_description:
            lines.append(f"- {result.block_description}")
        if result.finish_reason:
            lines.append(f"- Finish reason: `{result.finish_reason}`")
        if result.prompt_block_reason:
            lines.append(f"- Prompt block reason: `{result.prompt_block_reason}`")
        if result.stream_started_at:
            lines.append(f"- Stream started: `{result.stream_started_at}`")
        if result.stream_ended_at:
            lines.append(f"- Stream ended: `{result.stream_ended_at}`")
        if result.chunk_count:
            lines.append(
                f"- Chunks received: {result.chunk_count} "
                f"({result.thinking_chunks} thinking, {result.response_chunks} response)"
            )
        if result.usage_metadata:
            lines.append(f"- Usage: `{json.dumps(result.usage_metadata, sort_keys=True)}`")
        for err in result.errors:
            lines.append(f"- Error: {err}")
        lines.append("")

    return "\n".join(lines)


def process_map(
    client,
    map_path: Path,
    output_dir: Path,
    log_dir: Path,
    session_logger: DecomposeLogger,
    force: bool,
) -> str:
    md_path = output_dir / f"{map_path.stem}.md"
    file_log = log_dir / f"{map_path.stem}.log"

    if md_path.exists() and not force:
        session_logger.info("SKIP", map_path=map_path, reason="markdown_exists", md_path=md_path)
        return "skipped"

    logger = session_logger.file_logger(file_log)
    logger.info(
        "FILE START",
        map_path=map_path,
        md_path=md_path,
        file_log=file_log,
    )

    try:
        result = stream_decompose(client, map_path, logger)
        markdown = build_markdown(map_path, result)
        md_path.write_text(markdown, encoding="utf-8")
        logger.info("MD WRITTEN", md_path=md_path, bytes=len(markdown.encode("utf-8")))

        if result.errors:
            status = "error"
        elif result.was_blocked:
            status = "blocked"
        elif result.response_text.strip() or result.thinking_text.strip():
            status = "ok"
        else:
            status = "empty"

        logger.info("FILE END", status=status)
        return status
    except Exception as exc:
        logger.info("FILE FAILED", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
        fallback = build_markdown(
            map_path,
            StreamResult(
                errors=[f"{type(exc).__name__}: {exc}"],
                stream_started_at=ts(),
                stream_ended_at=ts(),
            ),
        )
        md_path.write_text(fallback, encoding="utf-8")
        logger.info("MD WRITTEN", md_path=md_path, note="error_fallback")
        logger.info("FILE END", status="failed")
        return "failed"


def process_map_job(
    api_key: str,
    map_path: Path,
    output_dir: Path,
    log_dir: Path,
    session_logger: DecomposeLogger,
    force: bool,
    index: int,
    total: int,
) -> tuple[Path, str]:
    from google import genai

    session_logger.info("BATCH ITEM", index=index, total=total, map_path=map_path)
    client = genai.Client(api_key=api_key)
    status = process_map(
        client=client,
        map_path=map_path,
        output_dir=output_dir,
        log_dir=log_dir,
        session_logger=session_logger,
        force=force,
    )
    session_logger.info("BATCH ITEM DONE", index=index, total=total, map_path=map_path, status=status)
    return map_path, status


def run_sequential(
    client,
    maps: list[Path],
    output_dir: Path,
    log_dir: Path,
    session_logger: DecomposeLogger,
    force: bool,
    delay: float,
) -> dict[str, int]:
    counts = {"ok": 0, "blocked": 0, "skipped": 0, "failed": 0, "error": 0, "empty": 0}
    total = len(maps)

    for i, map_path in enumerate(maps):
        session_logger.info("BATCH ITEM", index=i + 1, total=total, map_path=map_path)
        status = process_map(
            client=client,
            map_path=map_path,
            output_dir=output_dir,
            log_dir=log_dir,
            session_logger=session_logger,
            force=force,
        )
        counts[status] = counts.get(status, 0) + 1
        session_logger.info(
            "BATCH ITEM DONE",
            index=i + 1,
            total=total,
            map_path=map_path,
            status=status,
        )

        if i < total - 1 and status != "skipped":
            session_logger.info("DELAY", seconds=delay)
            time.sleep(delay)

    return counts


def run_parallel(
    api_key: str,
    maps: list[Path],
    output_dir: Path,
    log_dir: Path,
    session_logger: DecomposeLogger,
    force: bool,
    workers: int,
) -> dict[str, int]:
    counts = {"ok": 0, "blocked": 0, "skipped": 0, "failed": 0, "error": 0, "empty": 0}
    total = len(maps)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                process_map_job,
                api_key,
                map_path,
                output_dir,
                log_dir,
                session_logger,
                force,
                i + 1,
                total,
            ): map_path
            for i, map_path in enumerate(maps)
        }

        for future in as_completed(futures):
            map_path = futures[future]
            try:
                _, status = future.result()
            except Exception as exc:
                session_logger.info(
                    "BATCH ITEM FAILED",
                    map_path=map_path,
                    error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc(),
                )
                status = "failed"
            counts[status] = counts.get(status, 0) + 1

    return counts


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Decompose D&D dungeon maps to Markdown via Gemini 3.1 Flash-Lite",
    )
    parser.add_argument(
        "maps_folder",
        help="Folder containing PDF/PNG/JPEG dungeon maps",
    )
    parser.add_argument(
        "--log-dir",
        metavar="DIR",
        help="Directory for log files (default: <maps_folder>/logs)",
    )
    parser.add_argument(
        "--output-dir",
        metavar="DIR",
        help="Directory for Markdown output (default: same as maps_folder)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess maps even if a matching .md file already exists",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.5,
        metavar="SECONDS",
        help="Seconds to wait between API calls (default: 0.5)",
    )
    args = parser.parse_args()

    maps_folder = Path(args.maps_folder)
    if not maps_folder.is_dir():
        print(f"[ERROR] Maps folder not found: {maps_folder}")
        sys.exit(1)

    output_dir = Path(args.output_dir) if args.output_dir else maps_folder
    output_dir.mkdir(parents=True, exist_ok=True)

    run_stamp = utc_now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir) if args.log_dir else maps_folder / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    session_log = log_dir / f"decompose_session_{run_stamp}.log"

    maps = discover_maps(maps_folder)
    if not maps:
        print(f"[ERROR] No PDF/PNG/JPEG files found in {maps_folder}")
        sys.exit(1)

    api_key = get_api_key()

    try:
        from google import genai
    except ImportError:
        print("[ERROR] google-genai is not installed. Run: pip install google-genai")
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    session_logger = DecomposeLogger(session_log)

    use_parallel = len(maps) > PARALLEL_THRESHOLD
    workers = MAX_PARALLEL_WORKERS if use_parallel else 1

    session_logger.info(
        "BATCH START",
        maps_folder=str(maps_folder),
        output_dir=str(output_dir),
        log_dir=str(log_dir),
        map_count=len(maps),
        model=MODEL,
        force=args.force,
        delay=args.delay,
        parallel=use_parallel,
        workers=workers,
    )

    if use_parallel:
        counts = run_parallel(
            api_key=api_key,
            maps=maps,
            output_dir=output_dir,
            log_dir=log_dir,
            session_logger=session_logger,
            force=args.force,
            workers=workers,
        )
    else:
        counts = run_sequential(
            client=client,
            maps=maps,
            output_dir=output_dir,
            log_dir=log_dir,
            session_logger=session_logger,
            force=args.force,
            delay=args.delay,
        )

    session_logger.info("BATCH END", counts=counts, session_log=str(session_log))
    session_logger.close()

    print(
        f"\nDone. {counts['ok']} ok, {counts['blocked']} blocked, "
        f"{counts['skipped']} skipped, {counts['failed']} failed, "
        f"{counts['error']} errors, {counts['empty']} empty.\n"
        f"Session log: {session_log}\n"
    )


if __name__ == "__main__":
    main()
