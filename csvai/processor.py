"""Asynchronous CSV row processor using OpenAI models."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from jinja2 import Environment, StrictUndefined
from openai import AsyncOpenAI

from .io_utils import (
    read_rows,
    read_prompt,
    choose_prompt_file,
    default_output_file,
    collect_existing_ids_and_header,
    RowWriter,
)
from .settings import Settings

# =============================
# Logging
# =============================
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
for noisy in ("httpx", "openai", "urllib3"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

# =============================
# Helpers
# =============================


def sanitize_key_name(key: str) -> str:
    return key.strip().replace('"', "").replace("'", "").replace(" ", "_")


def sanitize_keys(row: Dict[str, Any]) -> Dict[str, str]:
    return {sanitize_key_name(k): ("" if v is None else str(v)).strip() for k, v in (row or {}).items() if k is not None}


_CURLY_VAR_RE = re.compile(r"\{\{\s*([^}]+?)\s*\}\}")


def sanitize_prompt_placeholders(prompt_template: str, raw_keys: List[str]) -> str:
    """Map {{ Raw Header }} → {{ Raw_Header }} only for simple identifiers."""
    key_map = {k: sanitize_key_name(k) for k in raw_keys}

    def _replace(m: "re.Match[str]") -> str:
        expr = m.group(1).strip()
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
# OpenAI client helpers
# =============================


def get_async_client(settings: Settings) -> AsyncOpenAI:
    if not settings.openai_api_key:
        logging.error("OPENAI_API_KEY missing. Set it in your environment or .env file.")
        raise SystemExit(2)
    return AsyncOpenAI(api_key=settings.openai_api_key, timeout=settings.request_timeout)


def _pick_text_from_response(resp: Any) -> str:
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
    settings: Settings,
) -> Optional[str]:
    backoff = settings.initial_backoff
    for attempt in range(1, settings.max_attempts + 1):
        try:
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
                temperature=settings.temperature,
                max_output_tokens=settings.max_output_tokens,
                text=text_cfg,
            )
            return _pick_text_from_response(resp) or None
        except Exception as e:
            if attempt == settings.max_attempts:
                logging.error(f"Responses API error (final): {e}")
                return None
            logging.warning(
                f"Responses API error (attempt {attempt}): {e} → retrying in {backoff:.1f}s"
            )
            await asyncio.sleep(backoff)
            backoff *= settings.backoff_factor
    return None


# =============================
# Schema helpers
# =============================


def load_schema(schema_path: Optional[str], prompt_path: Optional[Path]) -> Optional[Dict[str, Any]]:
    path: Optional[Path] = None
    if schema_path:
        path = Path(schema_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Schema file not found: {path}")
        logging.info("Using schema file: %s", path)
    elif prompt_path is not None:
        name = prompt_path.name
        if name.endswith(".prompt.txt"):
            cand = prompt_path.with_name(name[:-len(".prompt.txt")] + ".schema.json")
        else:
            cand = prompt_path.with_name(prompt_path.stem + ".schema.json")
        if cand.exists():
            path = cand
            logging.info("Auto-discovered schema file: %s", path)
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        schema = json.load(f)
        if not isinstance(schema, dict):
            raise ValueError("Schema root must be a JSON object")
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
            pass
    return schema


# =============================
# File helpers
# =============================


def batched(items: List[Any], size: int) -> Iterable[List[Any]]:
    for i in range(0, len(items), max(1, size)):
        yield items[i : i + size]


# =============================
# Row processing
# =============================


class RowResult:
    def __init__(self, id: str, data: Optional[Dict[str, Any]] = None, error: Optional[str] = None):
        self.id = id
        self.data = data or {}
        self.error = error

    @property
    def ok(self) -> bool:
        return self.error is None


async def process_row(
    row_idx: int,
    raw_row: Dict[str, Any],
    client: AsyncOpenAI,
    prompt_template: str,
    model: str,
    schema: Optional[Dict[str, Any]],
    settings: Settings,
) -> RowResult:
    row_id = (raw_row.get("id") or str(row_idx)).strip()
    sanitized = sanitize_keys(raw_row)
    sanitized["id"] = row_id
    try:
        prompt = render_prompt(prompt_template, sanitized, raw_row)
    except Exception as e:
        return RowResult(id=row_id, error=f"prompt_error: {e}")
    raw = await call_openai_responses(prompt, client, model, schema, settings)
    if not raw:
        return RowResult(id=row_id, error="api_empty")
    try:
        enriched = json.loads(raw)
        if isinstance(enriched, list):
            enriched = (enriched[0] if enriched else {})
        if not isinstance(enriched, dict):
            return RowResult(id=row_id, error="json_type")
    except Exception as e:
        return RowResult(id=row_id, error=f"json_parse: {e}")
    out = dict(sanitized)
    out.update(sanitize_keys(enriched))
    return RowResult(id=row_id, data=out)


# =============================
# Core processor class
# =============================


@dataclass
class ProcessorConfig:
    input: str
    prompt: Optional[str] = None
    output: Optional[str] = None
    schema: Optional[str] = None
    limit: Optional[int] = None
    model: Optional[str] = None


class CSVAIProcessor:
    def __init__(self, config: ProcessorConfig, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.config = config
        if not self.config.model:
            self.config.model = self.settings.default_model
        self.pause_event = asyncio.Event()
        self.pause_event.set()
        self.shutdown_event = asyncio.Event()

    def pause(self) -> None:
        self.pause_event.clear()

    def resume(self) -> None:
        self.pause_event.set()

    def stop(self) -> None:
        self.shutdown_event.set()

    async def run(self) -> None:
        args = self.config
        input_path = Path(args.input).expanduser().resolve()
        if not input_path.exists():
            logging.error(f"Input file not found: {input_path}")
            raise SystemExit(1)

        output_path = default_output_file(input_path, args.output)
        prompt_path = choose_prompt_file(input_path, args.prompt)

        all_rows = read_rows(str(input_path))
        if not all_rows:
            logging.info("No rows to process.")
            return

        prompt_template = read_prompt(str(prompt_path))
        schema = load_schema(args.schema, prompt_path)

        client = get_async_client(self.settings)

        existing_ids, existing_header = collect_existing_ids_and_header(output_path)

        def src_id(i: int, r: Dict[str, str]) -> str:
            return (r.get("id") or str(i)).strip()

        pending: List[Tuple[int, Dict[str, str]]] = [
            (i, r) for i, r in enumerate(all_rows) if src_id(i, r) not in existing_ids
        ]
        limit = args.limit if args.limit is not None and args.limit >= 0 else None
        if limit is not None:
            pending = pending[:limit]

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
        logging.info(
            "Using model=%s, concurrency=%d, batch_size=%d",
            args.model,
            self.settings.max_concurrent_requests,
            self.settings.processing_batch_size,
        )

        base_keys = [sanitize_key_name(k) for k in all_rows[0].keys()]
        if "id" not in base_keys:
            base_keys = ["id"] + base_keys
        tentative_header: List[str] = list(dict.fromkeys(base_keys))

        sem = asyncio.Semaphore(self.settings.max_concurrent_requests)
        processed, failed = 0, 0
        written_ids_this_run: Set[str] = set()
        header: Optional[List[str]] = existing_header if existing_header else None
        writer: Optional[RowWriter] = RowWriter(output_path, header) if header else None

        try:
            for bi, batch in enumerate(batched(pending, self.settings.processing_batch_size), start=1):
                await self.pause_event.wait()
                if self.shutdown_event.is_set():
                    break

                async def run_one(idx: int, raw: Dict[str, Any]) -> RowResult:
                    async with sem:
                        return await process_row(idx, raw, client, prompt_template, args.model, schema, self.settings)

                results = await asyncio.gather(
                    *(run_one(idx, raw) for idx, raw in batch), return_exceptions=True
                )

                successes: List[Dict[str, Any]] = []
                batch_keys: List[str] = []
                for res in results:
                    if isinstance(res, Exception):
                        failed += 1
                        logging.error("Unexpected error: %s", res)
                        continue
                    if not res.ok:
                        failed += 1
                        continue
                    successes.append(res.data)
                    for k in res.data.keys():
                        if k not in batch_keys:
                            batch_keys.append(k)

                if not successes:
                    logging.info("Batch %d: no successful rows.", bi)
                    if self.shutdown_event.is_set():
                        logging.warning("Stopping after current batch.")
                        break
                    continue

                if header is None:
                    header = list(dict.fromkeys(tentative_header + batch_keys))
                    writer = RowWriter(output_path, header)

                if writer is not None:
                    writer.append(successes)

                processed += len(successes)
                for s in successes:
                    if "id" in s and s["id"] is not None:
                        written_ids_this_run.add(str(s["id"]).strip())

                logging.info(
                    "Batch %d done | wrote=%d | total_this_run=%d/%d | failed=%d",
                    bi,
                    len(successes),
                    processed,
                    scheduled,
                    failed,
                )

                if self.shutdown_event.is_set():
                    logging.warning("Stopping after current batch.")
                    break

        finally:
            overall_done = len(existing_ids | written_ids_this_run)
            remaining = max(0, total_rows - overall_done)

            logging.info("Summary: output=%s", output_path)
            logging.info(
                "Totals: input=%d | processed_this_run=%d | processed_overall=%d | remaining=%d | failed=%d",
                total_rows,
                processed,
                overall_done,
                remaining,
                failed,
            )
            if writer is not None:
                writer.close()
            await client.close()

