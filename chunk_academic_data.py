"""
Parse academic datasets (Cochrane, PLABA, Med-EASi) into aligned chunks.
These high-quality human-written pairs should be broken into:
1. Sentence-aligned pairs
2. Phrase-aligned pairs (extract medical term → plain English mappings)

This multiplies our training data significantly from the existing resources.
"""

import json
import os
import re
from datasets import load_from_disk, load_dataset
from collections import defaultdict

OUTPUT_DIR = "./medclear_results/training_v2"

def split_sentences(text):
    """Split text into sentences, protecting abbreviations."""
    protected = text
    abbrevs = ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'vs.', 'etc.', 'i.e.', 'e.g.',
               'approx.', 'Fig.', 'No.', 'Vol.', 'al.', 'et al.']
    for a in abbrevs:
        protected = protected.replace(a, a.replace('.', '<DOT>'))

    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', protected)
    return [s.replace('<DOT>', '.').strip() for s in sentences if len(s.strip()) > 15]


def align_by_overlap(src_sents, tgt_sents):
    """Align source/target sentences by word overlap (better than position-based)."""
    pairs = []

    for tgt in tgt_sents:
        tgt_words = set(tgt.lower().split())
        if len(tgt_words) < 5:
            continue

        best_src = None
        best_overlap = 0

        for src in src_sents:
            src_words = set(src.lower().split())
            overlap = len(tgt_words & src_words) / max(len(tgt_words | src_words), 1)
            if overlap > best_overlap:
                best_overlap = overlap
                best_src = src

        # Only keep if reasonable overlap (>20% Jaccard similarity)
        if best_src and best_overlap > 0.20 and len(best_src) > 20:
            pairs.append((best_src, tgt))

    return pairs


def extract_term_mappings(source, target):
    """Extract medical term → plain English mappings from a source/target pair.
    Looks for patterns where a medical term in the source is explained in the target."""
    mappings = []

    # Pattern: "X (also known as Y)" or "X, which is Y"
    patterns = [
        r'(\w[\w\s]{2,30}?)\s*\((?:also (?:known|called) as|or|that is|i\.e\.|meaning)\s+([^)]+)\)',
        r'(\w[\w\s]{2,30}?),?\s+which (?:is|are|means?)\s+([^,.]{10,80})',
        r'(\w[\w\s]{2,30}?),?\s+(?:also (?:known|called) as)\s+([^,.]{5,60})',
    ]

    for pat in patterns:
        for match in re.finditer(pat, target, re.IGNORECASE):
            term = match.group(1).strip()
            definition = match.group(2).strip()
            if 5 < len(term) < 50 and 10 < len(definition) < 100:
                mappings.append((term, definition))

    return mappings


def process_cochrane():
    """Process Cochrane pairs into chunks."""
    print("\n=== Processing Cochrane ===")
    cochrane_dir = "./medclear_results/raw_data"

    sentence_pairs = []
    term_pairs = []

    for split_file in ["train.json", "validation.json", "test.json"]:
        path = os.path.join(cochrane_dir, split_file)
        if not os.path.exists(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            src = item.get("source", "")
            tgt = item.get("target", "")

            if not src or not tgt:
                continue

            # Sentence alignment
            src_sents = split_sentences(src)
            tgt_sents = split_sentences(tgt)
            aligned = align_by_overlap(src_sents, tgt_sents)

            for s, t in aligned:
                sentence_pairs.append({
                    "input_text": f"simplify: {s}",
                    "target": t,
                    "level": "sentence",
                    "source_dataset": "cochrane",
                })

            # Term extraction
            for term, defn in extract_term_mappings(src + " " + tgt, tgt):
                term_pairs.append({
                    "input_text": f"define: {term}",
                    "target": defn,
                    "level": "term",
                    "source_dataset": "cochrane",
                })

    print(f"  Sentence pairs: {len(sentence_pairs)}")
    print(f"  Term pairs: {len(term_pairs)}")
    return sentence_pairs, term_pairs


def process_plaba():
    """Process PLABA pairs into chunks."""
    print("\n=== Processing PLABA ===")
    plaba_dir = "./medclear_results/datasets/plaba"

    sentence_pairs = []
    term_pairs = []

    import csv
    for fname in ["train.csv", "val.csv", "test.csv"]:
        path = os.path.join(plaba_dir, fname)
        if not os.path.exists(path):
            continue

        with open(path, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                src = row.get("input_text", "")
                tgt = row.get("target_text", "")

                if not src or not tgt:
                    continue

                src_sents = split_sentences(src)
                tgt_sents = split_sentences(tgt)
                aligned = align_by_overlap(src_sents, tgt_sents)

                for s, t in aligned:
                    sentence_pairs.append({
                        "input_text": f"simplify: {s}",
                        "target": t,
                        "level": "sentence",
                        "source_dataset": "plaba",
                    })

                for term, defn in extract_term_mappings(src + " " + tgt, tgt):
                    term_pairs.append({
                        "input_text": f"define: {term}",
                        "target": defn,
                        "level": "term",
                        "source_dataset": "plaba",
                    })

    print(f"  Sentence pairs: {len(sentence_pairs)}")
    print(f"  Term pairs: {len(term_pairs)}")
    return sentence_pairs, term_pairs


def process_medeasi():
    """Process Med-EASi pairs into chunks — already at sentence level!"""
    print("\n=== Processing Med-EASi ===")
    medeasi_dir = "./medclear_results/datasets/medeasi"

    sentence_pairs = []

    for split in ["train", "validation", "test"]:
        split_dir = os.path.join(medeasi_dir, split)
        if not os.path.exists(split_dir):
            continue

        ds = load_from_disk(split_dir)
        for ex in ds:
            expert = ex.get("Expert", "")
            simple = ex.get("Simple", "")

            if expert and simple and len(expert) > 20 and len(simple) > 20:
                sentence_pairs.append({
                    "input_text": f"simplify: {expert}",
                    "target": simple,
                    "level": "sentence",
                    "source_dataset": "medeasi",
                })

    print(f"  Sentence pairs: {len(sentence_pairs)}")
    return sentence_pairs


def main():
    print("Chunking academic datasets into aligned training pairs...")

    # Process each dataset
    cochrane_sents, cochrane_terms = process_cochrane()
    plaba_sents, plaba_terms = process_plaba()
    medeasi_sents = process_medeasi()

    # Combine
    all_sentence_pairs = cochrane_sents + plaba_sents + medeasi_sents
    all_term_pairs = cochrane_terms + plaba_terms

    # Deduplicate term pairs
    seen = set()
    unique_terms = []
    for t in all_term_pairs:
        key = t["input_text"].lower()
        if key not in seen:
            seen.add(key)
            unique_terms.append(t)

    print(f"\n=== ACADEMIC CHUNKS SUMMARY ===")
    print(f"Total sentence pairs: {len(all_sentence_pairs)}")
    print(f"  Cochrane: {len(cochrane_sents)}")
    print(f"  PLABA: {len(plaba_sents)}")
    print(f"  Med-EASi: {len(medeasi_sents)}")
    print(f"Total term pairs: {len(unique_terms)} (deduplicated from {len(all_term_pairs)})")

    # Save
    out_file = os.path.join(OUTPUT_DIR, "academic_chunks.jsonl")
    with open(out_file, "w", encoding="utf-8") as f:
        for p in all_sentence_pairs + unique_terms:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nSaved to {out_file}")
    print(f"Total academic chunk pairs: {len(all_sentence_pairs) + len(unique_terms)}")

    # Show samples
    print("\n--- Cochrane sentence sample ---")
    if cochrane_sents:
        s = cochrane_sents[0]
        print(f"  IN:  {s['input_text'][:100]}...")
        print(f"  OUT: {s['target'][:100]}...")

    print("\n--- PLABA sentence sample ---")
    if plaba_sents:
        s = plaba_sents[0]
        print(f"  IN:  {s['input_text'][:100]}...")
        print(f"  OUT: {s['target'][:100]}...")

    print("\n--- Med-EASi sentence sample ---")
    if medeasi_sents:
        s = medeasi_sents[0]
        print(f"  IN:  {s['input_text'][:100]}...")
        print(f"  OUT: {s['target'][:100]}...")

    if unique_terms:
        print("\n--- Extracted term samples ---")
        for t in unique_terms[:5]:
            print(f"  {t['input_text']} → {t['target']}")


if __name__ == "__main__":
    main()
