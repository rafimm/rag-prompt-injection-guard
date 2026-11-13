"""All settings in one place, so they are easy to find and change.

In one line: thresholds, model names, dataset name, seed and paths used by the other modules.

If EMBED_MODEL or the stored attacks change, re-tune PASS_THRESHOLD and BLOCK_THRESHOLD.
"""
from pathlib import Path

# --- Data ---------------------------------------------------------------
DATASET_NAME = "deepset/prompt-injections"  # https://huggingface.co/datasets/deepset/prompt-injections
SEED = 42  # fixed seed -> the index/tuning split is identical on every run

# --- Layer 1: similarity to known attacks -------------------------------
EMBED_MODEL = "intfloat/multilingual-e5-small"  # turns text into a list of 384 numbers
PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = PROJECT_ROOT / "chroma_db"  # where Chroma saves the index on disk
COLLECTION_NAME = "known_attacks"
TOP_K = 3  # how many similar attacks to retrieve (Layer 2 receives these as evidence)

# Thresholds, chosen by hand from the tuning report (scripts/build_index.py) on the 110 tuning prompts.
#   score <= PASS_THRESHOLD   -> "safe"      (no LLM call)
#   score >= BLOCK_THRESHOLD  -> "injection" (no LLM call)
#   anything in between       -> "unsure"    (Layer 2 decides)
PASS_THRESHOLD = 0.85   # on the tuning prompts: 0 of 41 attacks scored this low, 33 of 69 safe prompts did
BLOCK_THRESHOLD = 0.98  # on the tuning prompts: 0 of 69 safe prompts scored this high, 1 of 41 attacks did

# False: store only the 162 index-part attacks, so tuning-pile scores stay honest.
# True:  store all 203 train attacks (do this AFTER tuning, see docs/ARCHITECTURE.md).
INDEX_ALL_TRAIN_ATTACKS = False

# --- LLM (Layer 2 classifier and the main chain) ---------------------------
LLM_MODEL = "gemma-4-26b-a4b-it"  # Gemma 4 via the Gemini API: free tier, supports structured output
