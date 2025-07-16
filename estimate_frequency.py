# Author: @jessiclassy
# Use spaCy dependency parser to estimate the impact of ABCD on the provided dataset
import spacy
from collections import defaultdict
from datasets import load_dataset
from tqdm.auto import tqdm
import argparse

# Lightweight spaCy pipeline
nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer", "textcat"])

# Detection based on certain dependency labels, words, or conditions
CLAUSE_RULES = {
    "discourse_connectives": {
        "deps": ["advmod", "discourse"],
        "words": {"however", "therefore"},
    },
    "vp_conjunction": {
        "deps": ["conj"],
        "conditions": [lambda t: t.head.pos_ == "VERB"],
    },
    "wh_relative_clause": {"deps": ["relcl"]},
    "restrictive_relative_clause": {
        "deps": ["acl"],
        "conditions": [lambda t: t.head.pos_ == "NOUN"],
    },
    "clausal_argument_that": {
        "deps": ["ccomp", "xcomp"],
        "conditions": [lambda t: any(c.lower_ == "that" for c in t.head.children)],
    },
}


def analyze_frequency(dataset_name, split="train", text_field="text", sample_size=None):
    # Store sentences with target clauses
    targets = []

    # Load dataset on the fly - sampling done later
    dataset = load_dataset(dataset_name, split=split, streaming=True)

    # Initialize counters for target clauses and total counts
    counts = defaultdict(int)
    total_sents = 0
    total_tokens = 0

    # Visualize progress
    progress = tqdm(total=sample_size, desc="Analyzing frequency", unit="examples")

    # Process dataset
    for i, example in enumerate(dataset):
        # Break loop if sample size is not None and the current index exceeds that
        if sample_size and i >= sample_size:
            break
        
        # spaCy pipeline over the current text field's contents
        doc = nlp(example[text_field])
        total_sents += len(list(doc.sents))
        total_tokens += len(doc)

        for sentence in doc.sents:
            if len(sentence) == 0 or not sentence.text.strip():
                continue
            # Switch off until rules are met
            match = False
            for token in sentence:
                for clause, rules in CLAUSE_RULES.items():
                    # If a dependency label is found
                    if token.dep_ in rules["deps"]:
                        # all([]) returns True
                        if all(cond(token) for cond in rules.get("conditions", [])):
                            # For discourse connectives, check for specific words, otherwise increment
                            if "words" not in rules or token.lower_ in rules["words"]:
                                counts[clause] += 1
                                # Turn on the switch if needed
                                if not match:
                                    match = True
            # If switch is on, append the sentence before resetting
            if match:
                targets.append(sentence)
        # Increment progress
        progress.update(1)
    progress.close()

    # Calculate frequency metrics
    results = {}

    # Process the clause counts for analysis
    for clause, count in counts.items():
        results[clause] = {
            "absolute_count": count,
            "per_million_tokens": (count / total_tokens) * 1_000_000,
            "per_sentence": (count / total_sents) * 100 if total_sents > 0 else 0,
        }

    return results, total_tokens, total_sents, targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="HuggingFace dataset name")
    parser.add_argument("--split", default="train", help="Dataset split")
    parser.add_argument("--text_field", default="text", help="Field containing text")
    parser.add_argument(
        "--sample", type=int, default=None, help="Process only N examples"
    )
    args = parser.parse_args()

    results, total_tokens, total_sents, target_sentences = analyze_frequency(
        args.dataset, split=args.split, text_field=args.text_field, sample_size=args.sample
    )

    # LOG RESULTS
    print(f"\nAnalyzed {total_tokens:,} tokens across {total_sents:,} sentences")
    print("\nFrequency metrics:")
    for clause, metrics in sorted(
        results.items(), key=lambda x: x[1]["per_million_tokens"]
    ):
        print(f"{clause.replace('_', ' ').title():>28}:")
        print(f"{'Absolute count':>20}: {metrics['absolute_count']:>6,}")
        print(f"{'Per million tokens':>20}: {metrics['per_million_tokens']:>6.1f}")
        print(f"{'% of sentences':>20}: {metrics['per_sentence']:>6.2f}%")

    # WRITE OUTPUT
    with open("target_sentences.txt", mode="w+") as file:
        for ts in target_sentences:
            file.write(f"{ts}\n")
