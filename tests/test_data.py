"""Tests for data.py.   Run:  pytest tests/test_data.py

Each test checks one thing we rely on later. If a check fails, Python prints
which one and what was expected.
"""
from prompt_injection_guard.data import load_data, split_train

# Load once and reuse in every test (downloading/splitting is the slow part).
train, test = load_data()
index_part, tuning_part = split_train(train)


def count_injections(rows):
    # Labels are 0 or 1, so the sum is the number of injections.
    return sum(r["label"] for r in rows)


def test_dataset_sizes():
    """The dataset should have the sizes stated on its Hugging Face page."""
    assert len(train) == 546
    assert len(test) == 116
    assert len(train) + len(test) == 662


def test_row_format():
    """Every row needs text (a non-empty string) and a label that is 0 or 1."""
    for r in train + test:
        assert isinstance(r["text"], str) and r["text"].strip() != ""
        assert r["label"] in (0, 1)


def test_split_sizes():
    """Index + tuning must add up to the full train set, with a 80/20 cut."""
    assert len(index_part) + len(tuning_part) == len(train)
    assert len(index_part) == 436
    assert len(tuning_part) == 110


def test_known_attack_count():
    """The index pile must contain exactly the 162 injections we plan to put in Chroma."""
    known_attacks = [r["text"] for r in index_part if r["label"] == 1]
    assert len(known_attacks) == 162


def test_split_is_stratified():
    """Both piles should have about the same share of injections as the full train set."""
    full = count_injections(train) / len(train)
    for part in (index_part, tuning_part):
        share = count_injections(part) / len(part)
        assert abs(share - full) < 0.02, f"injection share {share:.3f} vs {full:.3f}"


def test_no_overlap_between_piles():
    """No prompt may be in both index and tuning (otherwise tuning scores are inflated)."""
    index_texts = {r["text"] for r in index_part}
    tuning_texts = {r["text"] for r in tuning_part}
    assert index_texts.isdisjoint(tuning_texts)


def test_test_split_is_untouched():
    """Test prompts must not appear in train (otherwise the final evaluation is not honest)."""
    train_texts = {r["text"] for r in train}
    overlap = [r["text"] for r in test if r["text"] in train_texts]
    # Report how many, so a failure is easy to understand.
    assert overlap == [], f"{len(overlap)} test prompts also appear in train: {overlap[:2]}"


def test_split_is_reproducible():
    """Running the split again must give the same prompts in the same order."""
    again_index, again_tuning = split_train(train)
    assert [r["text"] for r in again_index] == [r["text"] for r in index_part]
    assert [r["text"] for r in again_tuning] == [r["text"] for r in tuning_part]
