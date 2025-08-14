#!/usr/bin/env python3
"""
csvai.py — Apply an AI prompt to each row in a CSV file and write enriched results.

This script reads an input CSV, renders a Jinja prompt for each row (you can use raw
CSV column names like {{ Org Name }}; they’ll be auto-sanitized to {{ Org_Name }}),
calls an OpenAI model via the Responses API, extracts JSON fields according to an
optional JSON schema, and writes the original columns plus AI-generated fields to an
output CSV.

Key features:
- Async + concurrent: processes rows in batches with configurable concurrency.
- Resumable: skips rows already present in the output (based on 'id').
- SIGINT (Ctrl+C): finishes the current batch, then exits cleanly.

Configuration:
- All tunables (model, concurrency, timeouts, batch size, etc.) can be set via a `.env` file.
- CLI flags `--model`, `--limit`, `--prompt`, `--output`, `--schema` override `.env` values.
- JSON schema (if provided) enforces structured output and validates AI responses.

Usage:
    python csvai.py input.csv [--prompt PROMPT_FILE] [--output OUTPUT_FILE]
                              [--limit N] [--model MODEL] [--schema SCHEMA_FILE]

Example:
    python csvai.py address.csv --prompt address.prompt.txt --schema address.schema.json
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import os
import signal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from dotenv import load_dotenv
from jinja2 import Environment, StrictUndefined
from openai import AsyncOpenAI

# =============================
# Configuration
# =============================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-4o-mini")
MAX_OUTPUT_TOKENS = int(os.getenv("MAX_OUTPUT_TOKENS", os.getenv("MAX_TOKENS", 800)))
TEMPERATURE = float(os.getenv("TEMPERATURE", 0.2))
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", 10))
PROCESSING_BATCH_SIZE = int(os.getenv("PROCESSING_BATCH_SIZE", 50))
REQUEST_TIMEOUT = float(os.getenv("REQUEST_TIMEOUT", 45.0))
MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", 4))
INITIAL_BACKOFF = float(os.getenv("INITIAL_BACKOFF", 2.0))
BACKOFF_FACTOR = float(os.getenv("BACKOFF_FACTOR", 1.7))
DEFAULT_PROMPT_FILENAME = os.getenv("DEFAULT_PROMPT_FILENAME", "prompt.txt")
ALT_PROMPT_SUFFIX = os.getenv("ALT_PROMPT_SUFFIX", ".prompt.txt")
OUTPUT_FILE_SUFFIX = os.getenv("OUTPUT_FILE_SUFFIX", "_enriched.csv")

# =============================
# Logging
# =============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
for noisy in ("httpx", "openai", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# =============================
# Signals / graceful shutdown
# =============================
SHUTDOWN_REQUESTED = False

def _handle_sigint(sig, frame):
    global SHUTDOWN_REQUESTED
    if SHUTDOWN_REQUESTED:
        logging.warning("Force exit")
        raise SystemExit(130)
    SHUTDOWN_REQUESTED = True
    logging.warning("Finishing current batch and then exiting. CTRL+C to exit now.")

signal.signal(signal.SIGINT, _handle_sigint)

# =============================
# CSV & prompt helpers
# =============================

def read_csv(filename: str) -> List[Dict[str, str]]:
    with open(filename, mode="r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def read_prompt(filename: str) -> str:
    with open(filename, "r", encoding="utf-8") as f:
        return f.read()


def sanitize_key_name(key: str) -> str:
    return key.strip().replace('"', "").replace("'", "").replace(" ", "_")


def sanitize_keys(row: Dict[str, Any]) -> Dict[str, str]:
    return {sanitize_key_name(k): ("" if v is None else str(v)).strip()
            for k, v in (row or {}).items() if k is not None}


def sanitize_prompt_placeholders(prompt_template: str, raw_keys: List[str]) -> str:
    """Map {{ Raw Header }} → {{ Raw_Header }} only for simple identifiers (leave complex Jinja intact)."""
    import re
    _CURLY_VAR_RE = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")
    key_map = {k: sanitize_key_name(k) for k in raw_keys}

    def _replace(m: "re.Match[str]") -> str:
        expr = m.group(1).strip()
        # Leave complex Jinja untouched
        if any(ch in expr for ch in ('"', "'", '[', ']', '.', '|', '(', ')', ':')):
            return m.group(0)
        if expr in key_map and key_map[expr] != expr:
            return "{{ " + key_map[expr] + " }}"
        return m.group(0)

    return _CURLY_VAR_RE.sub(_replace, prompt_template)


def render_prompt(prompt_template: str, row: Dict[str, str], raw_row: Dict[str, str]) -> str:
    env = Environment(undefined=StrictUndefined)
    context = dict(row, raw=raw_row)
    prompt_text = sanitize_prompt_placeholders(prompt_template, list(raw_row.keys()))
    return env.from_string(prompt_text).render(**context)

# =============================
# OpenAI async client & calls (Responses API only)
# =============================

def get_async_client() -> AsyncOpenAI:
    if not OPENAI_API_KEY:
        logging.error("OPENAI_API_KEY missing. Set it in your environment or .env file.")
        raise SystemExit(2)
    return AsyncOpenAI(api_key=OPENAI_API_KEY, timeout=REQUEST_TIMEOUT)


def _pick_text_from_response(resp: Any) -> str:
    """Extract text from a Responses API result."""
    text = getattr(resp, "output_text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    output = getattr(resp, "output", None)
    if isinstance(output, list) and output:
        for item in output:
            content = getattr(item, "content", None)
            if isinstance(content, list):
                for part in content:
                    t = getattr(part, "text", None)
                    if isinstance(t, str) and t.strip():
                        return t.strip()
                    if isinstance(part, dict):
                        val = part.get("text")
                        if isinstance(val, str) and val.strip():
                            return val.strip()
    return ""


async def call_openai_responses(
    prompt: str,
    client: AsyncOpenAI,
    model: str,
    schema: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Call Responses API with Structured Outputs via `text.format`.
    Returns the raw JSON string on success, or None after retries.
    """
    backoff = INITIAL_BACKOFF
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            # Build `text.format` per current docs
            if schema:
                text_cfg: Dict[str, Any] = {
                    "format": {
                        "type": "json_schema",
                        "name": "row_schema",
                        "schema": schema,
                        "strict": True,
                    }
                }
            else:
                text_cfg = {"format": {"type": "json_object"}}

            resp = await client.responses.create(
                model=model,
                input=prompt,
                temperature=TEMPERATURE,
                max_output_tokens=MAX_OUTPUT_TOKENS,
                text=text_cfg,
            )
            return _pick_text_from_response(resp) or None
        except Exception as e:
            if attempt == MAX_ATTEMPTS:
                logging.error(f"Responses API error (final): {e}")
                return None
            logging.warning(
                f"Responses API error (attempt {attempt}): {e} → retrying in {backoff:.1f}s"
            )
            await asyncio.sleep(backoff)
            backoff *= BACKOFF_FACTOR
    return None

# =============================
# Schema helpers
# =============================

def load_schema(schema_path: Optional[str], prompt_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    """Load a JSON Schema if provided. Otherwise auto-discover beside the prompt.

    Auto-discovery rule:
      - If prompt ends with ".prompt.txt": strip the ".prompt" and add ".schema.json"
          e.g., "address.prompt.txt" → "address.schema.json"
      - Else: replace the last extension with ".schema.json"
          e.g., "test.txt" → "test.schema.json"

    Also normalizes the schema to meet Responses API structured-output requirements:
    - Ensures `required` exists and includes **every** key in `properties` when `type: object`.
    """
    path: Optional[Path] = None

    # Explicit schema path wins
    if schema_path:
        path = Path(schema_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Schema file not found: {path}")
        logging.info("Using schema file: %s", path)

    # Auto-discovery from prompt filename
    elif prompt_path is not None:
        name = prompt_path.name
        if name.endswith(".prompt.txt"):
            # address.prompt.txt → address.schema.json
            cand = prompt_path.with_name(name[:-len(".prompt.txt")] + ".schema.json")
        else:
            # test.txt → test.schema.json
            cand = prompt_path.with_name(prompt_path.stem + ".schema.json")

        if cand.exists():
            path = cand
            logging.info("Auto-discovered schema file: %s", path)

    if not path:
        return None

    # Load & normalize schema
    with open(path, "r", encoding="utf-8") as f:
        schema = json.load(f)
        if not isinstance(schema, dict):
            raise ValueError("Schema root must be a JSON object")

        # Normalize: ensure `required` includes *all* properties for type=object
        try:
            if schema.get("type") == "object" and isinstance(schema.get("properties"), dict):
                props = list(schema["properties"].keys())
                req = schema.get("required")
                if not isinstance(req, list):
                    schema["required"] = props
                else:
                    missing = [k for k in props if k not in req]
                    if missing:
                        schema["required"] = req + missing
        except Exception:
            # Keep schema as-is if normalization fails; caller will surface API errors
            pass

    return schema

# =============================
# File helpers
# =============================

def choose_prompt_file(input_path: Path, user_prompt: Optional[str]) -> Path:
    """Return the prompt file to use, with auto-discovery if none provided.

    Auto-discovery tries, in order (first existing wins):
      1) <input>.prompt.txt              (e.g., address.csv → address.prompt.txt)
      2) DEFAULT_PROMPT_FILENAME         (e.g., prompt.txt)
    """
    path: Optional[Path] = None
    if user_prompt:
        path = Path(user_prompt).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        logging.info("Using prompt file: %s", path)
        return path

    # 1) address.csv → address.prompt.txt
    c1 = input_path.with_suffix(ALT_PROMPT_SUFFIX)
    if c1.exists():
        logging.info("Auto-discovered prompt file: %s", c1)
        return c1

    # 2) prompt.txt (DEFAULT_PROMPT_FILENAME)
    c2 = Path(DEFAULT_PROMPT_FILENAME)
    if c2.exists():
        logging.info("Auto-discovered prompt file: %s", c2)
        return c2

    raise FileNotFoundError(
        f"No prompt file supplied and neither '{c1.name}' nor '{c2.name}' exist."
    )


def default_output_file(input_path: Path, user_output: Optional[str]) -> Path:
    return Path(user_output) if user_output else input_path.with_name(f"{input_path.stem}{OUTPUT_FILE_SUFFIX}")


def collect_existing_ids_and_header(output_file: Path) -> Tuple[Set[str], List[str]]:
    """Return the set of IDs already in output and the existing header (if any)."""
    if not (output_file.exists() and output_file.stat().st_size > 0):
        return set(), []
    with open(output_file, "r", encoding="utf-8", newline="") as f:
        rdr = csv.DictReader(f)
        header = rdr.fieldnames or []
        ids: Set[str] = set()
        if "id" in (header or []):
            for row in rdr:
                rid = (row.get("id") or "").strip()
                if rid:
                    ids.add(rid)
        else:
            for i, _ in enumerate(rdr):
                ids.add(str(i))
        return ids, header


def batched(items: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), max(1, size)):
        yield items[i : i + size]

# =============================
# Row processing
# =============================

async def process_row(
    row_idx: int,
    raw_row: Dict[str, Any],
    client: AsyncOpenAI,
    prompt_template: str,
    model: str,
    schema: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    row_id = (raw_row.get("id") or str(row_idx)).strip()
    sanitized = sanitize_keys(raw_row)
    sanitized["id"] = row_id
    try:
        prompt = render_prompt(prompt_template, sanitized, raw_row)
    except Exception as e:
        return {"id": row_id, "_error": f"prompt_error: {e}"}

    raw = await call_openai_responses(prompt, client, model, schema)
    if not raw:
        return {"id": row_id, "_error": "api_empty"}

    try:
        enriched = json.loads(raw)
        if isinstance(enriched, list):
            enriched = (enriched[0] if enriched else {})  # normalize list → first obj
        if not isinstance(enriched, dict):
            return {"id": row_id, "_error": "json_type"}
    except Exception as e:
        return {"id": row_id, "_error": f"json_parse: {e}"}

    out = dict(sanitized)
    out.update(sanitize_keys(enriched))
    return out

# =============================
# Main routine
# =============================

async def main_process(args: argparse.Namespace) -> None:
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        raise SystemExit(1)

    output_path = default_output_file(input_path, args.output)
    prompt_path = choose_prompt_file(input_path, args.prompt)

    all_rows = read_csv(str(input_path))
    if not all_rows:
        logging.info("No rows to process.")
        return

    prompt_template = read_prompt(str(prompt_path))
    schema = load_schema(args.schema, prompt_path)

    client = get_async_client()

    # Build candidate list & resume set
    existing_ids, existing_header = collect_existing_ids_and_header(output_path)

    def src_id(i: int, r: Dict[str, str]) -> str:
        return (r.get("id") or str(i)).strip()

    pending: List[Tuple[int, Dict[str, str]]] = [
        (i, r) for i, r in enumerate(all_rows) if src_id(i, r) not in existing_ids
    ]

    limit = args.limit if args.limit is not None and args.limit >= 0 else None
    if limit is not None:
        pending = pending[: limit]

    total_rows = len(all_rows)
    scheduled = len(pending)

    if scheduled == 0:
        if len(existing_ids) == total_rows:
            logging.info("Everything is already processed. Nothing to do.")
        else:
            logging.info("No rows scheduled (limit=0 or nothing new).")
        await client.close()
        return

    logging.info(
        "Plan → already_enriched=%d | pending_new=%d | scheduled_now=%d",
        len(existing_ids), len(all_rows) - len(existing_ids), scheduled,
    )
    logging.info("Using model=%s, concurrency=%d, batch_size=%d",
                 args.model, MAX_CONCURRENT_REQUESTS, PROCESSING_BATCH_SIZE)

    # Determine base/header keys
    base_keys = [sanitize_key_name(k) for k in all_rows[0].keys()]
    if "id" not in base_keys:
        base_keys = ["id"] + base_keys

    tentative_header: List[str] = list(dict.fromkeys(base_keys))

    # Concurrency control
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    processed, failed = 0, 0
    written_ids_this_run: Set[str] = set()

    # Open output once. If this is a resume, do not write header again.
    is_resume = output_path.exists() and output_path.stat().st_size > 0 and bool(existing_header)

    try:
        with open(output_path, "a", encoding="utf-8", newline="") as f_out:
            writer: Optional[csv.DictWriter] = None

            for bi, batch in enumerate(batched(pending, PROCESSING_BATCH_SIZE), start=1):

                async def run_one(idx: int, raw: Dict[str, Any]) -> Dict[str, Any]:
                    async with sem:
                        return await process_row(idx, raw, client, prompt_template, args.model, schema)

                results = await asyncio.gather(*(run_one(idx, raw) for idx, raw in batch))

                # Filter successes and update dynamic keys (before header is fixed for new runs)
                successes: List[Dict[str, Any]] = []
                batch_keys: List[str] = []
                for res in results:
                    if res.get("_error"):
                        failed += 1
                        continue
                    successes.append(res)
                    for k in res.keys():
                        if k not in batch_keys:
                            batch_keys.append(k)

                if not successes:
                    logging.info("Batch %d: no successful rows.", bi)
                    if SHUTDOWN_REQUESTED:
                        logging.warning("Stopping after current batch. Ctrl+C to exit now.")
                        break
                    continue

                # Initialize writer/header
                if writer is None:
                    if is_resume and existing_header:
                        writer = csv.DictWriter(f_out, fieldnames=existing_header, extrasaction="ignore")
                    else:
                        final_header = list(dict.fromkeys(tentative_header + batch_keys))
                        writer = csv.DictWriter(f_out, fieldnames=final_header, extrasaction="ignore")
                        writer.writeheader()

                # Write successes using fixed header
                writer.writerows({k: row.get(k, "") for k in writer.fieldnames} for row in successes)

                # Stats
                processed += len(successes)
                for s in successes:
                    if "id" in s and s["id"] is not None:
                        written_ids_this_run.add(str(s["id"]).strip())

                logging.info("Batch %d done | wrote=%d | total_this_run=%d/%d | failed=%d",
                             bi, len(successes), processed, scheduled, failed)

                if SHUTDOWN_REQUESTED:
                    logging.warning("Stopping after current batch. Ctrl+C to exit now.")
                    break

    finally:
        overall_done = len(existing_ids | written_ids_this_run)
        remaining = max(0, total_rows - overall_done)

        logging.info("Summary: output=%s", output_path)
        logging.info("Totals: input=%d | processed_this_run=%d | processed_overall=%d | remaining=%d | failed=%d",
                     total_rows, processed, overall_done, remaining, failed)
        await client.close()

# =============================
# CLI
# =============================

def main():
    parser = argparse.ArgumentParser(description="Enrich CSV rows using the OpenAI Responses API (async, JSON-only).")
    parser.add_argument("input", help="Input CSV file path")
    parser.add_argument("--prompt", "-p", help="Prompt text file path")
    parser.add_argument("--output", "-o", help="Output CSV file path")
    parser.add_argument("--schema", help="JSON schema file path (strict). If omitted, uses json_object.")
    parser.add_argument("--limit", type=int, help="Limit number of new rows to process")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Model to use (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    try:
        asyncio.run(main_process(args))
    except SystemExit as e:
        raise e
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        raise SystemExit(1)

if __name__ == "__main__":
    main()

