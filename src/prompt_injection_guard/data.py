"""Step 1: Load the deepset/prompt-injections dataset and split it.

In one line: gets the labeled prompts and divides them into index, tuning and test pieces.

Each row is one short prompt with a label:  0 = safe, 1 = injection.

    train (546) ──┬── index part  (80%) -> its injections become the "known attacks" in Chroma
                  └── tuning part (20%) -> used to pick Layer 1 thresholds
    test  (116) ───── used ONLY for the final evaluation

Check it with:   python scripts/check_data.py
"""

# `datasets` is Hugging Face's library for downloading and working with datasets.
#   - load_dataset: downloads a dataset from the Hugging Face Hub (cached locally after the first run)
#   - Dataset:      the table-like object the library uses; we need it to do a stratified split
from datasets import Dataset, load_dataset

from prompt_injection_guard.config import DATASET_NAME, SEED  # see config.py for what these mean


def load_data():
    """Download the dataset and return (train, test) as plain Python lists.

    Each item looks like:  {"text": "Ignore all previous instructions ...", "label": 1}
    """
    # The dataset already comes split into "train" and "test" by its authors.
    ds = load_dataset(DATASET_NAME)

    # list(...) turns each Hugging Face split into a list of dicts,
    # which is easier to read and loop over than the Dataset object.
    return list(ds["train"]), list(ds["test"])


def split_train(train, tuning_fraction=0.2):
    """Split the train list into (index_part, tuning_part).

    - index_part  (80%): its injections are the known attacks we store in Chroma.
    - tuning_part (20%): prompts Layer 1 has NOT seen, used to choose its thresholds.

    Why not tune on the test split?  Then the test results would be biased: we'd be
    picking thresholds that suit the test data, and the scores would look better than
    they really are.  The test split must stay untouched until the final evaluation.
    """
    # Convert our list back into a Dataset so we can use its built-in splitting.
    # "stratify_by_column" needs the label column to be a special "class label" type,
    # so class_encode_column converts it. The values stay 0 and 1.
    ds = Dataset.from_list(train).class_encode_column("label")

    # Stratified split: both parts keep the same safe/injection ratio as the full train set
    # (about 37% injections). Without stratifying, one part could end up with
    # too few injections by chance.
    parts = ds.train_test_split(
        test_size=tuning_fraction,  # 20% goes to the tuning part
        seed=SEED,                  # same split every run
        stratify_by_column="label", # keep the 0/1 ratio balanced across both parts
    )

    # train_test_split names its outputs "train" and "test". Here they mean
    # "index part" and "tuning part" (not the dataset's real test split).
    return list(parts["train"]), list(parts["test"])


def summarize(name, rows):
    """Print how many prompts, injections and safe prompts a split has."""
    # Labels are 0 or 1, so adding them up counts the injections.
    n_inj = sum(r["label"] for r in rows)
    print(f"{name:<12} {len(rows):>4} prompts  ({n_inj} injections, {len(rows) - n_inj} safe)")
