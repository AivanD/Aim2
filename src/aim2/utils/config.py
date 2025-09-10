from pathlib import Path
import os
from dotenv import load_dotenv

def find_project_root(current_path, marker='pyproject.toml'):
    """
    Locates the project root by searching upwards for a marker file.
    """
    path = Path(current_path).resolve()
    while path.parent != path:
        if (path / marker).exists():
            return path
        path = path.parent
    raise FileNotFoundError(f"Project root with marker '{marker}' not found from '{current_path}'.")

PROJECT_ROOT = find_project_root(__file__)
# Load environment variables from .env file at the project root
load_dotenv(dotenv_path=PROJECT_ROOT / '.env')

INPUT_DIR       = PROJECT_ROOT / "input"
OUTPUT_DIR      = PROJECT_ROOT / "output"
DATA_DIR        = PROJECT_ROOT / "data"
MODELS_DIR      = PROJECT_ROOT / "models"

RAW_OUTPUT_DIR  = OUTPUT_DIR / "raw"
PROCESSED_OUTPUT_DIR = OUTPUT_DIR / "processed"

PO_OBO          = DATA_DIR / "plant-ontology.obo"
TO_OBO          = DATA_DIR / "to.obo"
PECO_OBO        = DATA_DIR / "peco.obo"
GO_OBO          = DATA_DIR / "go.obo"

HF_TOKEN        = os.getenv("HF_TOKEN")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY")

GROQ_MODEL      = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# VLLM
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

def ensure_dirs():
    """
    Ensures that the input and output directories exist.
    """
    for directory in [INPUT_DIR, OUTPUT_DIR, RAW_OUTPUT_DIR, PROCESSED_OUTPUT_DIR, MODELS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)