"""Project paths, relative to the app-build root."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "config"
DATA_PRIVATE = ROOT / "data_private"
DATA_DERIVED = ROOT / "data_derived"
ANALYSIS = ROOT / "analysis"
REPORTS = ROOT / "reports"
EXPORT_PUBLIC = ROOT / "export_public"
EXPORT_FULL = ROOT / "export_full"

RAW_PDFS = DATA_PRIVATE / "raw_pdfs"
TEXT = DATA_PRIVATE / "text"
PAGES = DATA_PRIVATE / "pages"
CROPS = DATA_PRIVATE / "crops"
LLM_LOG = DATA_PRIVATE / "llm_log"
MANIFEST = DATA_DERIVED / "manifest.csv"
