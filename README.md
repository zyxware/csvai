# csvai — Apply an AI prompt to each row in a CSV and write enriched results

`csvai.py` reads an input CSV, renders a prompt for each row (you can use raw
CSV column names like `{{ Org Name }}`, calls an **OpenAI model via the Responses API**, and writes the original columns plus
AI-generated fields to an output CSV.

The tool is **async + concurrent**, **resumable**, and **crash-safe**. It supports
**Structured Outputs** with a **JSON Schema** for reliable JSON, or **JSON mode**
(without a schema) if you prefer a lighter setup.

---

## Features

* **Structured Outputs**: enforce exact JSON with a schema.
* **JSON mode**: force a single JSON object without defining a schema.
* **Async & concurrent**: process many rows in parallel.
* **Resumable**: rows already written (by `id`) are skipped on re-run.
* **Crash-safe CSV**: header is fixed after the first successful batch; rows are flushed per batch.

---

## Requirements

* Python **3.9+**
* Packages: `openai` (latest, Responses API), `jinja2`, `python-dotenv`
* An **OpenAI API key** in your environment

```bash
pip install -U openai jinja2 python-dotenv
export OPENAI_API_KEY=sk-...
```

> **Note on determinism:** the Responses API does **not** support `seed`. If you need stability,
> keep `temperature` low and use a clear schema.

---

## Quick Start

```bash
# With a prompt and a strict schema (best reliability)
python csvai.py address.csv --prompt address.prompt.txt --schema address.schema.json

# Or JSON mode (no schema; still a single JSON object)
python csvai.py address.csv --prompt address.prompt.txt
```

### Example prompt (`address.prompt.txt`)

```text
Extract city, state, and country from the given address.

Rules:
- city: city/town/locality (preserve accents, proper case)
- state: ISO-standard name of the state/region/province or "" if none
- country: ISO 3166 English short name of the country; infer if obvious, else ""
- Ignore descriptors like "(EU)"
- Do not guess street-level info

Return ONLY the fields required by the tool schema.

Address: {{ Address }}
```

### Example schema (`address.schema.json`)

```json
{
  "type": "object",
  "properties": {
    "city":    { "type": "string" },
    "state":   { "type": "string" },
    "country": { "type": "string" }
  },
  "required": ["city", "state", "country"],
  "additionalProperties": false
}
```

> If your schema includes `properties`, **Responses Structured Outputs requires** that
> `required` includes **every** property. If you want a property to be optional, either
> remove it from `properties` entirely or use JSON mode instead of a schema.

---

## CLI

```bash
python csvai.py INPUT.csv [--prompt PROMPT_FILE] [--output OUTPUT_FILE]
                          [--limit N] [--model MODEL] [--schema SCHEMA_FILE]
```

**Flags**

* `--prompt, -p` — path to a plaintext prompt file (Jinja template).
* `--output, -o` — output CSV path (default: `<input>_enriched.csv`).
* `--limit` — process only the first `N` new/pending rows.
* `--model` — model name (default from `.env`, falls back to `gpt-4o-mini`).
* `--schema` — path to a JSON Schema for structured outputs.

---

## Environment Variables (`.env`)

You can set defaults in a `.env` file (overridden by CLI flags):

* `OPENAI_API_KEY` — **required**
* `DEFAULT_MODEL` — default model (e.g., `gpt-4o-mini`)
* `MAX_OUTPUT_TOKENS` — Responses `max_output_tokens` cap (default: `800`)
* `TEMPERATURE` — sampling temperature (default: `0.2`)
* `MAX_CONCURRENT_REQUESTS` — concurrency (default: `10`)
* `PROCESSING_BATCH_SIZE` — batch size written between flushes (default: `50`)
* `REQUEST_TIMEOUT` — request timeout seconds (default: `45.0`)
* `DEFAULT_PROMPT_FILENAME` — default prompt filename if not supplied (default: `prompt.txt`)
* `ALT_PROMPT_SUFFIX` — automatic prompt suffix checked beside input (default: `.prompt.txt`)
* `OUTPUT_FILE_SUFFIX` — suffix for the output file (default: `_enriched.csv`)

Example `.env`:

```ini
OPENAI_API_KEY=sk-...
DEFAULT_MODEL=gpt-4o-mini
MAX_OUTPUT_TOKENS=600
TEMPERATURE=0.2
MAX_CONCURRENT_REQUESTS=12
PROCESSING_BATCH_SIZE=100
REQUEST_TIMEOUT=45
ALT_PROMPT_SUFFIX=.prompt.txt
OUTPUT_FILE_SUFFIX=_enriched.csv
```

---

## Input/Output Behavior

* **Input CSV**: the script reads all rows. If an `id` column exists, it’s used to resume.
  If not, rows are indexed `0..N-1` internally for this run.
* **Prompt rendering**: every row is sanitized so `{{ Raw Header }}` becomes `{{ Raw_Header }}`.
  You can also reference the raw values as `{{ raw["Raw Header"] }}` if needed.
* **Output CSV**: contains the original columns plus AI-generated fields. The **header is fixed**
  after the first successful batch; later rows are written with the same header order.
* **Resume**: rerunning skips rows whose `id` is already present in the output file.

---

## Structured Outputs vs JSON Mode

### Structured Outputs (recommended)

* Add `--schema address.schema.json`.
* The request includes:

  ```python
  text={"format": {
      "type": "json_schema",
      "name": "row_schema",
      "schema": schema,
      "strict": True
  }}
  ```
* Guarantees the model returns **exactly** the keys/types you expect.

### JSON Mode (no schema)

* The request includes: `text={"format": {"type": "json_object"}}`.
* The model must still return a single JSON object, but no exact schema is enforced.
* Simpler to set up, but occasionally the model may add an extra key/token.
* Need to mention the keyword JSON in the prompt to get this working.

---

## Performance & Concurrency

* Concurrency is controlled by `MAX_CONCURRENT_REQUESTS`.
* Increase gradually; too high can trigger API rate limits.
* `PROCESSING_BATCH_SIZE` controls how many results are written per batch.
* `REQUEST_TIMEOUT` guards slow requests; the script retries with backoff.

---

## Troubleshooting

**“Missing required parameter: `text.format.name`”**
You used structured outputs but didn’t include `name` **alongside** `type` and `schema`.
The script already sends this correctly; check that you’re running the latest version and
that `--schema` points to the right file.

**“Invalid schema … `required` must include every key”**
The Responses structured-outputs path expects `required` to include **all** keys in `properties`.
Either (a) add them all to `required`, (b) remove non-required keys from `properties`, or (c) use JSON mode.

---

## FAQ

**Q: Does it run concurrently?**
Yes. Concurrency is controlled via `MAX_CONCURRENT_REQUESTS` (default 10).

**Q: Can I rely on an `id` column?**
Yes. If present in the input CSV, it’s used for resumability. Otherwise rows are indexed for the session.

**Q: Can I output nested JSON?**
The schema can be nested, but CSV is flat. If you want nested data, extend the script with a flattener
(e.g., convert `address.street` → `address_street`).

**Q: Which models work?**
Recent `gpt-4o*` models support Responses + Structured Outputs. If a model doesn’t support it, use JSON mode.

**Q: Do I need a JSON schema?**
No, but it’s strongly recommended for stable columns and fewer parse failures.


---

## License

GNU GPL v2 or above.
