"""CSVAI package providing CSV enrichment tools."""

from .processor import CSVAIProcessor, ProcessorConfig
from .settings import Settings

__all__ = ["CSVAIProcessor", "ProcessorConfig", "Settings", "__version__"]

__version__ = "0.1.0"
