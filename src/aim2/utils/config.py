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
TARDC_INPUT_DIR   = PROJECT_ROOT / "tardc/input"
TARDC_FINISHED_INPUTS = PROJECT_ROOT / "tardc/finished_inputs"
TARDC_INPUT_MALFORMED_DIR = PROJECT_ROOT / "tardc/input_malformed"
TARDC_INPUT_REALLY_ODD = PROJECT_ROOT / 'tardc/really_odd'
OUTPUT_DIR      = PROJECT_ROOT / "output"
TARDC_OUTPUT_DIR  = PROJECT_ROOT / "tardc/output"
DATA_DIR        = PROJECT_ROOT / "data"
MODELS_DIR      = PROJECT_ROOT / "models"

# log folder
LOGS_DIR        = PROJECT_ROOT / "logs"

# NER-specific output directories
NER_OUTPUT_DIR = OUTPUT_DIR / "ner"
RAW_NER_OUTPUT_DIR  = NER_OUTPUT_DIR / "raw"
EVAL_NER_OUTPUT_DIR = NER_OUTPUT_DIR / "evaluation" # For per-passage results used in evaluation
PROCESSED_NER_OUTPUT_DIR = NER_OUTPUT_DIR / "processed"

# RE-specific output directories (for future use)
RE_OUTPUT_DIR = OUTPUT_DIR / "re"
RAW_RE_OUTPUT_DIR = RE_OUTPUT_DIR / "raw"
PROCESSED_RE_OUTPUT_DIR = RE_OUTPUT_DIR / "processed"

# tardc-specific directories
TARDC_NER_OUTPUT_DIR = TARDC_OUTPUT_DIR / "ner"
TARDC_RAW_NER_OUTPUT_DIR  = TARDC_NER_OUTPUT_DIR / "raw"
TARDC_EVAL_NER_OUTPUT_DIR = TARDC_NER_OUTPUT_DIR / "evaluation" # For per-passage results used in evaluation
TARDC_PROCESSED_NER_OUTPUT_DIR = TARDC_NER_OUTPUT_DIR / "processed"
TARDC_RE_OUTPUT_DIR = TARDC_OUTPUT_DIR / "re"
TARDC_RAW_RE_OUTPUT_DIR = TARDC_RE_OUTPUT_DIR / "raw"
TARDC_PROCESSED_RE_OUTPUT_DIR = TARDC_RE_OUTPUT_DIR / "processed"


PO_OBO          = DATA_DIR / "plant-ontology.obo"
TO_OBO          = DATA_DIR / "to.obo"
PECO_OBO        = DATA_DIR / "peco.obo"
GO_OBO          = DATA_DIR / "go.obo"
CHEMONT_OBO     = DATA_DIR / "ChemOnt_2_1.obo"
PATHWAYS_PMN    = DATA_DIR / "All-pathways-of-PlantCyc.txt"
DATABASE_FILE   = DATA_DIR / "references.sqlite"

HF_TOKEN        = os.getenv("HF_TOKEN", "")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
NCBI_API_KEY    = os.getenv("NCBI_API_KEY", "")

GROQ_MODEL              = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_MODEL_RE_EVAL      = os.getenv("GROQ_MODEL_RE_EVAL", "openai/gpt-oss-120b")
GPT_MODEL_NER           = os.getenv("GPT_MODEL_NER", "gpt-4.1")
GPT_MODEL_RE_EVAL       = os.getenv("GPT_MODEL_RE_EVAL", "gpt-5-mini")

# VLLM
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# local databases raw files links
REFERENCE_FILES_URLS = {
    # Pubchem reference files
    "CID-SMILES.gz": "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz",
    "CID-InChI-Key.gz": "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-InChI-Key.gz",
    "CID-Synonym-filtered.gz": "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-Synonym-filtered.gz",

    # NCBI Taxonomy dump
    "taxdump.tar.gz": "https://ftp.ncbi.nlm.nih.gov/pub/taxonomy/taxdump.tar.gz"
}

def ensure_dirs():
    """
    Ensures that the input and output directories exist.
    """
    for directory in [INPUT_DIR, OUTPUT_DIR, RAW_NER_OUTPUT_DIR, EVAL_NER_OUTPUT_DIR, PROCESSED_NER_OUTPUT_DIR, MODELS_DIR, NER_OUTPUT_DIR, RE_OUTPUT_DIR, RAW_RE_OUTPUT_DIR, PROCESSED_RE_OUTPUT_DIR, LOGS_DIR,
                      TARDC_INPUT_DIR, 
                      TARDC_FINISHED_INPUTS, TARDC_INPUT_MALFORMED_DIR, TARDC_INPUT_REALLY_ODD,
                      TARDC_OUTPUT_DIR, TARDC_RAW_NER_OUTPUT_DIR, TARDC_EVAL_NER_OUTPUT_DIR, TARDC_PROCESSED_NER_OUTPUT_DIR,
                      TARDC_RE_OUTPUT_DIR, TARDC_RAW_RE_OUTPUT_DIR, TARDC_PROCESSED_RE_OUTPUT_DIR]:
        directory.mkdir(parents=True, exist_ok=True)