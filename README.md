# Prompt Injection Guard

A guard that checks every input **before** it reaches an LLM chain and blocks likely
prompt injections. It uses two layers:

1. **Layer 1: similarity check (fast, no LLM).** Compares the input with known attacks
   stored in a vector database.
2. **Layer 2: retrieval-augmented LLM classifier (Gemini).** Called only when Layer 1 is
   unsure. It receives the top 3 most similar known attacks as context and returns
   `is_injection`, `confidence` and `attack_type`.

```
guard | RunnableBranch(safe → main_chain, unsafe → refusal)
```

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design, including a
[step-by-step walkthrough](docs/ARCHITECTURE.md#21-how-a-prompt-gets-checked-walkthrough)
of how one prompt is checked.

## Dataset

[`deepset/prompt-injections`](https://huggingface.co/datasets/deepset/prompt-injections):
662 prompts (English and German) labeled `0` = safe, `1` = injection.
- **Train-split injections** are indexed as known attacks.
- **Test split** is used for evaluation.

## Tech stack

LangChain · Gemma 4 (via the Gemini API) · ChromaDB · `intfloat/multilingual-e5-small` · Streamlit · Python 3.12

## Project structure

The project is built one step at a time. Steps 1 to 4 are done; steps 5 and 6 are still to come.
Settings (thresholds, model names, paths) are all in `src/prompt_injection_guard/config.py`.

| Step | File | What it does |
|---|---|---|
| 1 | `src/prompt_injection_guard/data.py` | Load the dataset; split train into index and tuning parts |
| 2 | `src/prompt_injection_guard/layer1.py` | Index known attacks in Chroma; score inputs by similarity |
| 3 | `src/prompt_injection_guard/layer2.py` | Gemini classifier with structured output, given the top-3 retrieved attacks as context |
| 4 | `src/prompt_injection_guard/guard.py` | Combine the layers; `RunnableBranch` to main chain or refusal |
| 5 | `scripts/evaluate.py` | Precision, recall and FPR for Layer 1, Layer 2 and combined |
| 6 | `app/streamlit_app.py` | Streamlit chat UI (optional) |

## Setup

```bash
cd code/prompt-injection-guard
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"           # installs the package and its dependencies, plus pytest
cp .env.example .env              # then put your Gemini API key (from Google AI Studio) in .env
```

## Usage

Run these from the project folder, in this order:

```bash
python scripts/check_data.py      # optional: load the dataset and show the split sizes
python scripts/build_index.py     # build the known-attack index (chroma_db/) and print the tuning report
python scripts/try_layer2.py      # try the Gemini classifier on a few messages
python scripts/try_guard.py       # run the guard on a few messages: shows which layer decided, then the reply
pytest                            # run the tests (test_layer1.py needs the index from build_index.py)
```

Coming in later steps:

```bash
python scripts/evaluate.py          # precision / recall / FPR for each mode (step 5)
streamlit run app/streamlit_app.py  # chat UI (step 6)
```

**Using the guard in your own code:**

```python
from prompt_injection_guard.guard import build_main_chain, guarded_answer
from prompt_injection_guard.layer1 import load_index
from prompt_injection_guard.layer2 import build_classifier

# Build once (needs the index from build_index.py and GOOGLE_API_KEY in .env).
store, classifier, main_chain = load_index(), build_classifier(), build_main_chain()

reply, verdict = guarded_answer("How do I return an order?", store, classifier, main_chain)
print(reply)                              # the assistant's answer, or a refusal
print(verdict["safe"], verdict["details"]["decided_by"])   # e.g. True layer2
```

`guarded_answer()` is the same flow written as plain `if/else`: Layer 1, then Layer 2 only if
Layer 1 is unsure, then answer or refuse. If you prefer a LangChain chain, `build_guarded_chain()`
in `guard.py` builds `guard | RunnableBranch(safe -> main_chain, unsafe -> refusal)` from the same pieces.

**Note on rate limits:** the Gemini free tier allows only about 5 requests per minute, so
the `try_*` scripts pause between calls. If you see a `429 RESOURCE_EXHAUSTED` error, wait a
minute and run again. The guard fails closed, so an API error blocks the message.

## Results

_To be filled in after `evaluate.py` runs._

| Mode | Precision | Recall | False-positive rate | LLM calls |
|---|---|---|---|---|
| Layer 1 only | | | | 0 |
| Layer 2 only (no retrieval) | | | | |
| Layer 2 + retrieval | | | | |
| Combined | | | | |

## Planned enhancement

Replace the placeholder `main_chain` with a **document RAG chain** (`rag_chain.py`) over
the fictional e-commerce support docs in
`RAG-Assignment/rag-domain-datasets/data/05_ecommerce_customer_support/`. The guard and
evaluation stay the same. See section 8 of [ARCHITECTURE.md](docs/ARCHITECTURE.md).
