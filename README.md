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

See [ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design.

## Dataset

[`deepset/prompt-injections`](https://huggingface.co/datasets/deepset/prompt-injections):
662 prompts (English and German) labeled `0` = safe, `1` = injection.
- **Train-split injections** are indexed as known attacks.
- **Test split** is used for evaluation.

## Tech stack

LangChain · Gemini Flash · ChromaDB · `intfloat/multilingual-e5-small` · Streamlit · Python 3.12

## Project structure

We build these one at a time, and run and check each before moving on.
Current status and next steps are tracked in [HANDOFF.md](HANDOFF.md).

| Step | File | What it does |
|---|---|---|
| 1 | `src/prompt_injection_guard/data.py` | Load the dataset; split train into index and tuning parts |
| 2 | `src/prompt_injection_guard/layer1.py` | Index known attacks in Chroma; score inputs by similarity |
| 3 | `src/prompt_injection_guard/layer2.py` | Gemini classifier with structured output, given the top-3 retrieved attacks as context |
| 4 | `guard.py` | Combine the layers; `RunnableBranch` to main chain or refusal |
| 5 | `scripts/evaluate.py` | Precision, recall and FPR for Layer 1, Layer 2 and combined |
| 6 | `app/streamlit_app.py` | Streamlit chat UI (optional) |

## Setup

```bash
cd code/prompt-injection-guard
python3.12 -m venv .venv          # already created
source .venv/bin/activate
pip install -e ".[dev]"           # installs the package and its dependencies, plus pytest
cp .env.example .env              # then put your Gemini API key (from Google AI Studio) in .env
```

## Usage (once all steps are built)

```bash
python scripts/build_index.py     # build the known-attack index and print the tuning report
python scripts/try_layer2.py      # try the Gemini classifier live
python scripts/evaluate.py        # print precision / recall / FPR for each mode (step 5)
streamlit run app/streamlit_app.py  # chat UI (step 6)
pytest                            # run the tests (build the index first)
```

## Results

_To be filled in after `evaluate.py` runs._

| Mode | Precision | Recall | False-positive rate | LLM calls |
|---|---|---|
| Layer 1 only | | | | 0 |
| Layer 2 only (no retrieval) | | | | |
| Layer 2 + retrieval | | | | |
| Combined | | | | |

## Planned enhancement

Replace the placeholder `main_chain` with a **document RAG chain** (`rag_chain.py`) over
the fictional e-commerce support docs in
`RAG-Assignment/rag-domain-datasets/data/05_ecommerce_customer_support/`. The guard and
evaluation stay the same. See section 8 of [ARCHITECTURE.md](docs/ARCHITECTURE.md).
