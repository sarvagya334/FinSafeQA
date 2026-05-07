"""
convert_docling.py
------------------
Converts raw source files (PDF, CSV, TXT) in DATA_RAW into Markdown files
stored in DATA_PROCESSED.

- PDFs  → converted via Docling (IBM's document-understanding library)
- CSVs  → converted to Markdown tables via pandas
- TXTs  → copied as-is into a markdown fence

Already-converted files are skipped unless the source has been modified
(checked by comparing mtime).
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _out_path(raw_file: Path, processed_dir: str) -> Path:
    """Return the destination .md path for a given raw file."""
    return Path(processed_dir) / (raw_file.stem + ".md")


def _is_stale(src: Path, dst: Path) -> bool:
    """Return True if dst doesn't exist or is older than src."""
    if not dst.exists():
        return True
    return src.stat().st_mtime > dst.stat().st_mtime


# ---------------------------------------------------------------------------
# Per-format converters
# ---------------------------------------------------------------------------

def _convert_pdf(src: Path, dst: Path) -> None:
    """Convert a PDF to Markdown using Docling."""
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as e:
        raise ImportError(
            "docling is not installed. Run: pip install docling"
        ) from e

    logger.info("Converting PDF: %s", src.name)
    converter = DocumentConverter()
    result = converter.convert(str(src))
    md_text = result.document.export_to_markdown()

    dst.write_text(md_text, encoding="utf-8")
    logger.info("  → %s (%d chars)", dst.name, len(md_text))


def _convert_csv(src: Path, dst: Path) -> None:
    """Convert a CSV file to a Markdown table."""
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "pandas is not installed. Run: pip install pandas"
        ) from e

    logger.info("Converting CSV: %s", src.name)
    df = pd.read_csv(src)

    # Markdown header derived from filename
    title = src.stem.replace("_", " ").strip()
    md_lines = [f"# {title}\n", df.to_markdown(index=False)]
    md_text = "\n\n".join(md_lines)

    dst.write_text(md_text, encoding="utf-8")
    logger.info("  → %s (%d rows)", dst.name, len(df))


def _convert_txt(src: Path, dst: Path) -> None:
    """Wrap a plain-text file in a markdown code block."""
    logger.info("Converting TXT: %s", src.name)
    raw = src.read_text(encoding="utf-8", errors="replace")
    title = src.stem.replace("_", " ").strip()
    md_text = f"# {title}\n\n```\n{raw}\n```\n"
    dst.write_text(md_text, encoding="utf-8")
    logger.info("  → %s", dst.name)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {
    ".pdf": _convert_pdf,
    ".csv": _convert_csv,
    ".txt": _convert_txt,
}


def convert_all_raw_to_markdown(
    raw_dir: str,
    processed_dir: str,
    *,
    force: bool = False,
) -> List[str]:
    """
    Walk *raw_dir*, convert every supported file to Markdown and save it
    under *processed_dir*.

    Parameters
    ----------
    raw_dir : str
        Directory containing raw source files.
    processed_dir : str
        Directory where converted Markdown files will be written.
    force : bool
        If True, re-convert even if the destination is up to date.

    Returns
    -------
    List[str]
        Absolute paths of all (converted + pre-existing) Markdown files
        in *processed_dir*.
    """
    raw_path = Path(raw_dir)
    proc_path = Path(processed_dir)
    proc_path.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists():
        logger.warning("DATA_RAW directory does not exist: %s", raw_dir)
        return []

    converted: List[str] = []
    skipped: List[str] = []

    for src in sorted(raw_path.iterdir()):
        if src.is_dir() or src.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        dst = _out_path(src, processed_dir)
        converter_fn = SUPPORTED_EXTENSIONS[src.suffix.lower()]

        if not force and not _is_stale(src, dst):
            logger.debug("Up to date, skipping: %s", src.name)
            skipped.append(str(dst))
            continue

        try:
            converter_fn(src, dst)
            converted.append(str(dst))
        except Exception as exc:
            logger.error("Failed to convert %s: %s", src.name, exc)

    if converted:
        print(f"[convert_docling] Converted {len(converted)} file(s).")
    if skipped:
        print(f"[convert_docling] Skipped {len(skipped)} up-to-date file(s).")

    # Return all markdown files in processed_dir (converted + pre-existing)
    all_md = sorted(proc_path.glob("*.md"))
    return [str(p) for p in all_md]
