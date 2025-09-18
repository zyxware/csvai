# CSVAI — Apply an AI prompt to each row in a CSV or Excel file and write enriched results

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy?repo=https://github.com/zyxware/csvai)

The `csvai` library reads an input CSV or Excel file, renders a prompt for each row (you can use raw column names like `{{ Address }}`), calls an **OpenAI model via the Responses API**, and writes the original columns plus AI-generated fields to an output CSV or Excel file. It also support **image analysis** (vision) when enabled.

The tool is **async + concurrent**, **resumable**, and **crash-safe**. It supports **Structured Outputs** with a **JSON Schema** for reliable JSON, or **JSON mode** (without a schema) if you prefer a lighter setup.

We also have a **CSV AI Prompt Builder** (a Custom GPT) to help you generate prompts and JSON Schemas tailored to your CSVs.

---

## Features

* **Structured Outputs**: enforce exact JSON with a schema for consistent, validated results.
* **JSON mode**: force a single JSON object without defining a schema.
* **Async & concurrent**: process many rows in parallel for faster throughput.
* **Resumable**: rows already written (by `id`) are skipped on re-run.
* **CSV or Excel**: handle `.csv` and `.xlsx` inputs and outputs.
* **image analysis**: add `--process-image` to attach an image per row (via URL or local file) to multimodal models like `gpt-4o-mini`.

---

## Installation

Requires Python **3.9+**.
OpenAI API Key: Create a key - https://platform.openai.com/api-keys and use it in the .env file with OPENAI_API_KEY=
See example.env in the [project repo](https://github.com/zyxware/csvai).

### From PyPI

```bash
pip install csvai
# Include Streamlit UI dependencies
pip install "csvai[ui]"
```

### From GitHub

```bash
# Install directly from the repository
pip install git+https://github.com/zyxware/csvai.git
# With Streamlit UI dependencies
pip install "csvai[ui] @ git+https://github.com/zyxware/csvai.git"
```

### For local development

```bash
git clone https://github.com/zyxware/csvai
cd csvai
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp example.env .env
# Edit .env and set OPENAI_API_KEY=sk-...
```

Installing the package exposes the `csvai` CLI and the `csvai-ui` command.

---

## Usage

### CLI

#### Auto-discovery

If you name your files like this:

```
input.csv      # or input.xlsx
input.prompt.txt
input.schema.json   # optional
```

Run:

```bash
csvai input.csv      # or input.xlsx
```

#### Or specify prompt & schema explicitly

```bash
# With a prompt and a strict schema (best reliability)
csvai address.xlsx --prompt address.prompt.txt --schema address.schema.json

# Or JSON mode (no schema; still a single JSON object)
csvai address.xlsx --prompt address.prompt.txt
```

Sample datasets (`address.csv` and `address.xlsx`) with the matching prompt and schema live in the `example/` directory.

### Streamlit UI

After installing with the `ui` extra, launch the web interface:

```bash
csvai-ui
```

The UI lets you upload a CSV/Excel file, provide a prompt and optional schema.
A "Process images" toggle is available to attach an image per row; you can set the image column (default `image`) and the image root directory (default `./images`).

---

## Example Prompt & Schema

### Prompt (`address.prompt.txt`)

```text
Extract city, state, and country from the given address.

Rules:
- city: city/town/locality (preserve accents, proper case)
- state: ISO-standard name of the state/region/province or "" if none
- country: ISO 3166 English short name of the country; infer if obvious, else ""
- Ignore descriptors like "(EU)"
- Do not guess street-level info

Inputs:
Address: {{Address}}
```

### Schema (`address.schema.json`)

```json
{
  "type": "object",
  "properties": {
    "city": {
      "type": "string",
      "description": "City, town, or locality name with correct casing and accents preserved"
    },
    "state": {
      "type": "string",
      "description": "ISO-standard name of the state, region, or province, or empty string if none"
    },
    "country": {
      "type": "string",
      "description": "ISO 3166 English short name of the country, inferred if obvious, else empty string"
    }
  },
  "required": ["city", "state", "country"],
  "additionalProperties": false
}
```
---

## Creating Prompts & Schemas with the CSV AI Prompt Builder

You can use the **[CSV AI Prompt Builder](https://chat.openai.com/g/g-689d8067bd888191a896d2cfdab27a39-csv-ai-prompt-builder)** custom GPT to:

* Quickly design a **prompt** tailored to your CSV data.
* Generate a **JSON Schema** that matches your desired structured output.

**Example input to the builder:**

```
File: reviews.csv. Inputs: title,body,stars. Output: sentiment,summary.
```

**Example result:**

**Prompt**

```
Analyze each review and produce sentiment and a concise summary.

Rules:
- sentiment: one of positive, neutral, negative.
- Star mapping: stars ≤ 2 ⇒ negative; 3 ⇒ neutral; ≥ 4 ⇒ positive. If stars is blank or invalid, infer from tone.
- summary: 1–2 sentences, factual, include key pros/cons, no emojis, no first person, no marketing fluff.
- Use the same language as the Body.
- Return only the fields required by the tool schema.

Inputs:
Title: {{title}}
Body: {{body}}
Stars: {{stars}}
```

**Schema**

```json
{
  "type": "object",
  "properties": {
    "sentiment": {
      "type": "string",
      "description": "Overall sentiment derived from stars and/or tone; one of positive, neutral, negative",
      "enum": ["positive", "neutral", "negative"]
    },
    "summary": {
      "type": "string",
      "description": "Concise 1–2 sentence summary capturing key pros/cons without opinionated fluff"
    }
  },
  "required": ["sentiment", "summary"],
  "additionalProperties": false
}
```

**Command to execute**

```bash
python -m csvai.cli reviews.csv --prompt reviews.prompt.txt --schema reviews.schema.json
```
**Tip — have the builder generate a schema for you**
* **I have `products.csv` with Product Title, Product Description, Category, and Sub Category. Help me enrich with SEO meta fields.**
* **I have `reviews.csv` with Title, Body, and Stars. Help me extract sentiment and generate a short summary.**
* **I have `address.csv` with an Address field. Help me extract City, State, and Country using ISO-standard names.**
* **I have `tickets.csv` with Subject and Description. Help me classify each ticket into predefined support categories.**
* **I have `posts.csv` with Title, Body, URL, Image URL, Brand, and Platform. Help me generate social media captions, hashtags, emojis, CTAs, and alt text.**
* **I have `jobs.csv` with Job Title and Description. Help me categorize jobs into sectors and identify the level of seniority.**


---

## CLI

```bash
csvai INPUT.csv [--prompt PROMPT_FILE] [--output OUTPUT_FILE]
                          [--limit N] [--model MODEL] [--schema SCHEMA_FILE]
                          [--process-image] [--image-col COL] [--image-root DIR]
```

**Flags**

* `--prompt, -p` — path to a plaintext prompt file (Jinja template).
* `--output, -o` — output CSV path (default: `<input>_enriched.csv`).
* `--limit` — process only the first `N` new/pending rows.
* `--model` — model name (default from `.env`, falls back to `gpt-4o-mini`).
* `--schema` — path to a JSON Schema for structured outputs (optional).
* `--process-image` — enable image analysis; when set, attaches an image per row if available.
* `--image-col` — name of the image column (default: `image`).
* `--image-root` — directory to resolve local image filenames (default: `./images`).

Notes on images:
- If the image cell is blank, the row is processed as text-only.
- If the cell is a full URL (`http(s)://...`), the model fetches it.
- Otherwise the value is treated as a filename: resolved as an absolute/relative path first, then `./images/<filename>`.
- If a referenced file is missing/unreadable, the tool logs a warning and proceeds text-only.

---

## Environment Variables (`.env`)

See [`example.env`](example.env) for all configurable variables.

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

* **Input CSV**: the script reads all rows. If an `id` column exists, it’s used to resume. If not, rows are indexed `0..N-1` internally for this run.
* **Prompt rendering**: every row is sanitized so `{{ Raw Header }}` becomes `{{ Raw_Header }}`. You can also reference the raw values as `{{ raw["Raw Header"] }}` if needed.
* **Output CSV**: contains the original columns plus AI-generated fields. The **header is fixed** after the first successful batch; later rows are written with the same header order.
* **Resume**: rerunning skips rows whose `id` is already present in the output file.

---

## Image Analysis Example

Files in `examples/`:

- `image.csv` — demo rows with an image URL, a local filename, and a blank image.
- `image.prompt.txt` — prompt to produce a one-sentence `description`.
- `image.schema.json` — schema requiring the `description` field.

Local image (for row 2): place a file at `./images/sample.jpg` (relative to your current working directory). For convenience you can download a sample image, for example:

```bash
mkdir -p images
curl -L -o images/sample.jpg https://upload.wikimedia.org/wikipedia/commons/3/3f/JPEG_example_flower.jpg
```

Run the example (multimodal enabled):

```bash
csvai examples/image.csv \
  --prompt examples/image.prompt.txt \
  --schema examples/image.schema.json \
  --process-image
```

Notes:
- The image column defaults to `image`; override with `--image-col` if needed.
- Local filenames are resolved as-is first; if not found, `./images/<filename>` is tried.
- If an image is missing or unreadable, the row is processed as text-only and a warning is logged.

### Sports Alt Text Sample (10 images)

1) Run CSVAI with the alt text prompt and schema:

```bash
csvai examples/sample-images.csv \
  --prompt examples/sample-images.prompt.txt \
  --schema examples/sample-images.schema.json \
  --process-image
```

The output file will be `examples/sample-images_enriched.csv` and include the generated `alt_text` field.


## Structured Outputs vs JSON Mode

### Structured Outputs (recommended)

When you pass `--schema`, the request includes:

```python
text={
  "format": {
    "type": "json_schema",
    "name": "row_schema",
    "schema": schema,
    "strict": true
  }
}
```

This guarantees the model returns **exactly** the keys/types you expect.

### JSON Mode (no schema)

When no schema is provided, the request includes:

```python
text={"format": {"type": "json_object"}}
```

The model must still return a single JSON object, but no exact schema is enforced.

> **Prompting tip:** mention the word **JSON** in your prompt and explicitly list the expected fields to improve compliance in JSON mode.

---

## Performance & Concurrency

* Concurrency is controlled by `MAX_CONCURRENT_REQUESTS`.
* Increase gradually; too high can trigger API rate limits.
* `PROCESSING_BATCH_SIZE` controls how many results are written per batch.
* `REQUEST_TIMEOUT` guards slow requests; the script retries with backoff.

---

## Troubleshooting

**“Missing required parameter: `text.format.name`”**
You used structured outputs but didn’t include `name` **alongside** `type` and `schema`. The script already sends this correctly; ensure you’re on the latest version and that `--schema` points to the right file.

**“Invalid schema … `required` must include every key”**
The Responses structured-outputs path expects `required` to include **all** keys in `properties`. Either (a) add them all to `required`, (b) remove non-required keys from `properties`, or (c) use JSON mode.

**Rows not resuming**
Ensure there’s an `id` column in both input and output. If not present, the script uses positional IDs for the current run only.

---

## FAQ

**Q: Does it run concurrently?**
Yes. Concurrency is controlled via `MAX_CONCURRENT_REQUESTS` (default 10).

**Q: Can I rely on an `id` column?**
Yes. If present in the input CSV, it’s used for resumability. Otherwise rows are indexed for the session.

**Q: Can I output nested JSON?**
The schema can be nested, but CSV is flat. If you want nested data, extend the script with a flattener (e.g., convert `address.street` → `address_street`).

**Q: Which models work?**
Recent `gpt-4o*` models support Responses + Structured Outputs. If a model doesn’t support it, use JSON mode.

**Q: Do I need a JSON schema?**
No, but it’s strongly recommended for stable columns and fewer parse failures.

---

## Support

This application was developed as an internal tool and we will continue to improve and optimize it as long as we use it. If you would like us to customize this or build a similar or related system to automate your tasks with AI, we are available for **commercial support**.

---

### About Zyxware Technologies

At **Zyxware Technologies**, our mission is to help organizations harness the power of technology to solve real-world problems. Guided by our founding values of honesty and fairness, we are committed to delivering genuine value to our clients and the free and open-source community.

**CSVAI** is a direct result of this philosophy. We originally developed it to automate and streamline our own internal data-enrichment tasks. Realizing its potential to help others, we are sharing it as a free tool in the spirit of our commitment to Free Software.

Our expertise is centered around our **AI & Automation Services**. We specialize in building intelligent solutions that reduce manual effort, streamline business operations, and unlock data-driven insights. While we provide powerful free tools like this one, we also offer **custom development and commercial support** for businesses that require tailored AI solutions.

If you're looking to automate a unique business process or build a similar system, we invite you to [**reach out to us**](https://www.zyxware.com/contact-us) to schedule a free discovery call.

---

## Updates

For updates and new versions, visit: [Project Page @ Zyxware](https://www.zyxware.com/article/6935/csvai-automate-data-enrichment-any-csv-or-excel-file-generative-ai)

---

## Contact

[https://www.zyxware.com/contact-us](https://www.zyxware.com/contact-us)

---

## Source Repository

[https://github.com/zyxware/csvai](https://github.com/zyxware/csvai)

---

## Reporting Issues

[https://github.com/zyxware/csvai/issues](https://github.com/zyxware/csvai/issues)

---

## License and Disclaimer

**GPL v2** – Free to use & modify. Use it at your own risk. We are not collecting any user data.

---

## Need Help or Commercial Support?

If you have any questions, feel free to [contact us](https://www.zyxware.com/contact-us).
