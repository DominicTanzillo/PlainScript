"""
Build the FINAL training dataset for MedClear V2.

Combines all sources with proper weighting to teach the model
to be a medical jargon translator at every level of granularity.

Target distribution:
  - Terms/definitions: 30% (the foundation — learn the vocabulary)
  - Sentences: 40% (learn to compose translations)
  - Phrases: 15% (clinical shorthand → plain English)
  - Paragraphs: 10% (learn coherent output)
  - RAG-augmented: 5% (learn to use MedlinePlus context)
"""

import json
import os
import random

OUTPUT_DIR = "./medclear_results/training_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(42)


def load_jsonl(path):
    pairs = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                pairs.append(json.loads(line))
    return pairs


def main():
    print("Building FINAL training dataset...\n")

    # ── 1. TERM PAIRS (hand-written + academic extracted + flashcards) ──
    # Hand-written terms from build_training_data_v2.py
    from build_training_data_v2 import build_term_pairs, build_phrase_pairs
    hand_terms = build_term_pairs()

    # Academic extracted terms
    academic_chunks = load_jsonl("./medclear_results/training_v2/academic_chunks.jsonl")
    academic_terms = [p for p in academic_chunks if p.get("level") == "term"]

    # Flashcards (cap at 3000 most relevant)
    flashcard_pairs = []
    fc_dir = "./medclear_results/datasets/medical_flashcards/train"
    if os.path.exists(fc_dir):
        from datasets import load_from_disk
        ds = load_from_disk(fc_dir)
        for i in range(len(ds)):
            inp = ds[i]["input"]
            out = ds[i]["output"]
            if any(inp.startswith(q) for q in ["What is ", "What are ", "Define "]):
                if len(out) < 150:
                    flashcard_pairs.append({
                        "input_text": inp,
                        "target": out,
                        "level": "flashcard",
                    })
            if len(flashcard_pairs) >= 3000:
                break

    all_terms = hand_terms + academic_terms + flashcard_pairs
    # Oversample hand-written terms (highest quality)
    all_terms += hand_terms * 3  # 4x total for hand-written
    random.shuffle(all_terms)
    print(f"Terms: {len(all_terms)} (hand: {len(hand_terms)}, academic: {len(academic_terms)}, flashcard: {len(flashcard_pairs)})")

    # ── 2. PHRASE PAIRS ──
    phrase_pairs = build_phrase_pairs()
    phrase_pairs_oversampled = phrase_pairs * 10  # Heavy oversampling
    print(f"Phrases: {len(phrase_pairs_oversampled)} ({len(phrase_pairs)} base x10)")

    # ── 3. SENTENCE PAIRS (academic + synthetic) ──
    academic_sents = [p for p in academic_chunks if p.get("level") == "sentence"]
    random.shuffle(academic_sents)
    academic_sents_capped = academic_sents[:8000]  # Cap academic sentences

    synthetic_sents = load_jsonl("./medclear_results/synthetic_data/aligned_training_data.jsonl")
    synthetic_sents = [p for p in synthetic_sents if p.get("granularity") == "sentence"]
    for p in synthetic_sents:
        p["input_text"] = f"simplify: {p.get('source', p.get('input_text', ''))}"
        p["target"] = p.get("target", "")
        p["level"] = "sentence"

    all_sents = academic_sents_capped + synthetic_sents[:2000]
    print(f"Sentences: {len(all_sents)} (academic: {len(academic_sents_capped)}, synthetic: {min(2000, len(synthetic_sents))})")

    # ── 4. PARAGRAPH PAIRS (synthetic clinical) ──
    paragraphs = load_jsonl("./medclear_results/synthetic_data/claude_generated_pairs.jsonl")
    for p in paragraphs:
        p["input_text"] = f"simplify clinical note: {p['source']}"
        p["level"] = "paragraph"
    print(f"Paragraphs: {len(paragraphs)}")

    # ── 5. RAG-AUGMENTED PAIRS ──
    rag_pairs = load_jsonl("./medclear_results/synthetic_data/claude_generated_pairs_rag.jsonl")
    rag_with_context = [p for p in rag_pairs if "REFERENCE INFORMATION:" in p.get("source", "")]
    for p in rag_with_context:
        p["input_text"] = f"simplify with context: {p['source']}"
        p["level"] = "rag"
    print(f"RAG-augmented: {len(rag_with_context)}")

    # ── COMBINE ──
    all_data = (
        all_terms +
        phrase_pairs_oversampled +
        all_sents +
        paragraphs +
        rag_with_context
    )
    random.shuffle(all_data)

    # Split
    n = len(all_data)
    train_end = int(n * 0.9)
    val_end = int(n * 0.95)

    splits = {
        "train": all_data[:train_end],
        "validation": all_data[train_end:val_end],
        "test": all_data[val_end:],
    }

    # Save
    for split_name, data in splits.items():
        path = os.path.join(OUTPUT_DIR, f"{split_name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for p in data:
                # Ensure consistent format
                entry = {
                    "input_text": p.get("input_text", ""),
                    "target": p.get("target", ""),
                    "level": p.get("level", "unknown"),
                }
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"\n=== FINAL TRAINING DATA ===")
    print(f"Total: {n:,}")
    print(f"  Train: {len(splits['train']):,}")
    print(f"  Val:   {len(splits['validation']):,}")
    print(f"  Test:  {len(splits['test']):,}")

    from collections import Counter
    level_counts = Counter(p.get("level", "?") for p in all_data)
    print(f"\nBy level:")
    for level, count in level_counts.most_common():
        pct = count / n * 100
        print(f"  {level}: {count:,} ({pct:.1f}%)")


if __name__ == "__main__":
    main()
