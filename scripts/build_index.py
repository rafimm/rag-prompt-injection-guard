"""Build the Chroma index of known attacks and print the tuning report.   Run: python scripts/build_index.py"""
from prompt_injection_guard.config import CHROMA_DIR, INDEX_ALL_TRAIN_ATTACKS
from prompt_injection_guard.data import load_data, split_train
from prompt_injection_guard.layer1 import build_index, tuning_report

train, _test = load_data()
index_part, tuning_part = split_train(train)

# Which attacks go into Chroma? (see INDEX_ALL_TRAIN_ATTACKS in config.py)
source = train if INDEX_ALL_TRAIN_ATTACKS else index_part
known_attacks = [r["text"] for r in source if r["label"] == 1]
assert len(known_attacks) == sum(r["label"] for r in source)  # sanity check
print(f"Indexing {len(known_attacks)} known attacks...")

vectorstore = build_index(known_attacks)
print(f"Index built: {vectorstore._collection.count()} vectors stored in {CHROMA_DIR.name}/")

tuning_report(vectorstore, tuning_part)
