"""
CLI entry point for the AI Document Classifier.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.logger import logger


def main():
    """Main CLI entry point."""
    logger.info("AI Document Classifier - CLI Interface")
    logger.info("This is a placeholder. CLI functionality will be implemented in Phase 8.")
    print("AI Document Classifier v1.0.0")
    print("CLI interface coming soon...")


if __name__ == "__main__":
    main()

