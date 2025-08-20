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
    args = parser.parse_args()

    config = ProcessorConfig(
        input=args.input,
        prompt=args.prompt,
        output=args.output,
        schema=args.schema,
        limit=args.limit,
        model=args.model,
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
