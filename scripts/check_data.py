"""Load the dataset, split it, and print the size and balance of every split.   Run: python scripts/check_data.py"""
from prompt_injection_guard.data import load_data, split_train, summarize

train, test = load_data()
index_part, tuning_part = split_train(train)

summarize("train", train)
summarize("  index", index_part)
summarize("  tuning", tuning_part)
summarize("test", test)

# Show a few real examples: the first 3 rows and the first 3 injections.
print("\nExamples:")
first_injections = [r for r in train if r["label"] == 1][:3]
for r in train[:3] + first_injections:
    print(f"  [{r['label']}] {r['text'][:90]}")  # [:90] trims long prompts for display
