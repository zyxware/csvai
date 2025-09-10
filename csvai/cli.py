"""Command-line interface for CSVAI."""

import argparse
import asyncio
import logging
import signal

from .processor import CSVAIProcessor, ProcessorConfig
from .settings import Settings


def main() -> None:
    """Run the CSVAI processor via the command line."""
    settings = Settings()

    parser = argparse.ArgumentParser(
        description="Enrich CSV/Excel rows.",
    )
    parser.add_argument("input", help="Input CSV or Excel file path")
    parser.add_argument("--prompt", "-p", help="Prompt text file path")
    parser.add_argument("--output", "-o", help="Output CSV or Excel file path")
    parser.add_argument(
        "--schema",
        help="JSON schema file path (strict). If omitted, uses json_object.",
    )
    parser.add_argument("--limit", type=int, help="Limit number of new rows to process")
    parser.add_argument("--model", default=settings.default_model, help="Model to use")
    # Image processing options (Option A)
    parser.add_argument(
        "--process-image",
        action="store_true",
        help="Enable image processing; attaches image from column (default 'image') or URL",
    )
    parser.add_argument(
        "--image-col",
        help="Name of the image column (default: 'image')",
    )
    parser.add_argument(
        "--image-root",
        help="Directory to resolve local image filenames (default: ./images)",
    )
    args = parser.parse_args()

    config = ProcessorConfig(
        input=args.input,
        prompt=args.prompt,
        output=args.output,
        schema=args.schema,
        limit=args.limit,
        model=args.model,
        process_image=args.process_image,
        image_col=args.image_col,
        image_root=args.image_root,
    )
    processor = CSVAIProcessor(config, settings=settings)

    def _handle_signal(sig, frame):
        logging.warning("Signal %s received: shutting down after current batch", sig)
        processor.stop()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    try:
        asyncio.run(processor.run())
    except SystemExit as e:
        raise e
    except Exception as e:
        logging.error(f"Fatal error: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
