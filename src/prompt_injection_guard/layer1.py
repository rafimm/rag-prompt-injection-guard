"""Step 2: Layer 1 - similarity to known attacks (fast, no LLM call).

In one line: stores known attacks in a vector database and scores a new message by how
closely it matches the nearest one, then gives a verdict (safe / injection / unsure).

How it works
    1. Embed the known attacks (injections from the index pile) and store them in Chroma.
    2. To check a new message: embed it, ask Chroma for the closest known attacks,
       and use the similarity of the closest one as its SCORE (0 to 1, higher = more attack-like).
    3. Thresholds turn the score into a verdict (added after we look at the tuning scores).

Build the index and print the tuning report with:   python scripts/build_index.py
"""
import shutil

import numpy as np

# Chroma is the vector database. HuggingFaceEmbeddings runs the embedding model locally.
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from prompt_injection_guard.config import (
    CHROMA_DIR,
    COLLECTION_NAME,
    BLOCK_THRESHOLD,
    EMBED_MODEL,
    PASS_THRESHOLD,
    TOP_K,
)


# ---------------------------------------------------------------------------
# Part 1: build the index and score a message
# ---------------------------------------------------------------------------
def get_embeddings():
    """Load the embedding model.

    e5 models were trained with a "query: " label in front of every text, so we add it
    to everything we embed (this is how the model expects to be used).
    normalize_embeddings=True makes every vector length 1, which makes
    "closeness" equal to cosine similarity.
    """
    settings = {"prompt": "query: ", "normalize_embeddings": True}
    return HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        encode_kwargs=settings,         # used when embedding the stored attacks
        query_encode_kwargs=settings,   # used when embedding a message we are checking
    )


def open_store():
    """Open (or create) the Chroma collection on disk. build_index() and load_index() both use this."""
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=get_embeddings(),
        persist_directory=str(CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},  # measure closeness with cosine similarity
    )


def build_index(known_attacks):
    """Embed the known attacks and save them in Chroma. Any old index is deleted first."""
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)  # start clean so we never mix old and new attacks

    vectorstore = open_store()
    # add_texts embeds each attack and stores text + vector together.
    vectorstore.add_texts(known_attacks)
    return vectorstore


def load_index():
    """Open the index that build_index() saved earlier (no re-embedding needed)."""
    return open_store()


def find_similar(vectorstore, text, k=TOP_K):
    """Return the k most similar known attacks as [(attack_text, similarity), ...], best first."""
    # similarity_search_with_relevance_scores returns (Document, score) pairs.
    # For cosine, score = 1 - distance, so 1.0 means identical and lower means less similar.
    hits = vectorstore.similarity_search_with_relevance_scores(text, k=k)
    results = [(doc.page_content, float(score)) for doc, score in hits]
    return sorted(results, key=lambda pair: pair[1], reverse=True)


def score(vectorstore, text):
    """The Layer 1 score: similarity to the single closest known attack."""
    return find_similar(vectorstore, text, k=1)[0][1]


def verdict_from_score(s):
    """Turn a score into Layer 1's verdict using the two thresholds."""
    if s >= BLOCK_THRESHOLD:
        return "injection"  # almost a copy of a known attack
    if s <= PASS_THRESHOLD:
        return "safe"       # nothing like any known attack
    return "unsure"         # in between: Layer 2 (Gemini) will decide


def check(vectorstore, text):
    """Layer 1 check of one message.

    Returns:
        score:           similarity to the closest known attack
        verdict:         "safe", "injection" or "unsure"
        similar_attacks: the top-3 closest known attacks [(text, similarity), ...],
                         which Layer 2 will receive as evidence when the verdict is "unsure"
    """
    similar = find_similar(vectorstore, text, k=TOP_K)  # one search gives both score and evidence
    best_score = similar[0][1]
    return {"score": best_score, "verdict": verdict_from_score(best_score), "similar_attacks": similar}


# ---------------------------------------------------------------------------
# Part 2: tuning report (we read this, then pick PASS and BLOCK by hand)
# ---------------------------------------------------------------------------
def tuning_report(vectorstore, tuning_part):
    """Score every tuning prompt and print how safe vs attack prompts are spread out."""
    texts = [r["text"] for r in tuning_part]
    labels = np.array([r["label"] for r in tuning_part])
    scores = np.array([score(vectorstore, t) for t in texts])

    safe, attack = scores[labels == 0], scores[labels == 1]

    # Summary of each group: if attacks score clearly higher, Layer 1 is useful.
    print(f"\n{'':<10}{'count':>6}{'min':>8}{'5%':>8}{'median':>8}{'95%':>8}{'max':>8}")
    for name, group in (("safe", safe), ("attack", attack)):
        p5, p50, p95 = np.percentile(group, [5, 50, 95])
        print(f"{name:<10}{len(group):>6}{group.min():>8.3f}{p5:>8.3f}{p50:>8.3f}{p95:>8.3f}{group.max():>8.3f}")

    # Text histogram: how many prompts fall in each score range (S = safe, A = attack).
    lo, hi = np.floor(scores.min() * 50) / 50, np.ceil(scores.max() * 50) / 50
    edges = np.arange(lo, hi + 0.02, 0.02)
    safe_counts, _ = np.histogram(safe, bins=edges)
    attack_counts, _ = np.histogram(attack, bins=edges)
    print("\nScore range      safe | attack")
    for i in range(len(edges) - 1):
        print(f"{edges[i]:.2f}-{edges[i+1]:.2f}   {'S' * safe_counts[i]:<28}| {'A' * attack_counts[i]}")

    # What the chosen thresholds do on these 110 prompts.
    verdicts = np.array([verdict_from_score(s) for s in scores])
    print(f"\nWith PASS={PASS_THRESHOLD} and BLOCK={BLOCK_THRESHOLD}:")
    for v in ("safe", "unsure", "injection"):
        n_s, n_a = ((verdicts == v) & (labels == 0)).sum(), ((verdicts == v) & (labels == 1)).sum()
        print(f"  {v:<10} {n_s + n_a:>3} prompts  ({n_s} safe, {n_a} attacks)")
    print(f"  -> {(verdicts == 'unsure').mean():.0%} of prompts would go to Layer 2")

    # The extremes tell us the most about where the thresholds can go.
    order = np.argsort(scores)
    print("\nHighest-scoring SAFE prompts (risk of wrongly blocking):")
    for i in [j for j in order[::-1] if labels[j] == 0][:3]:
        print(f"  {scores[i]:.3f}  {texts[i][:80]}")
    print("\nLowest-scoring ATTACK prompts (risk of wrongly passing):")
    for i in [j for j in order if labels[j] == 1][:3]:
        print(f"  {scores[i]:.3f}  {texts[i][:80]}")
