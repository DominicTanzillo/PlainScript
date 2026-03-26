"""
=============================================================================
Synthetic Plain-Language Training Data Generator
=============================================================================
Uses Claude Opus (via Anthropic API) to generate plain-language versions
of clinical discharge summaries and operative notes.

Sources:
  - MTSamples (~5,000 medical transcriptions)
  - Asclepius Synthetic Clinical Notes (~158K notes)

Output: Paired (clinical_text, plain_language) examples for fine-tuning.

Usage:
    # Set your API key
    export ANTHROPIC_API_KEY=sk-ant-...

    # Generate synthetic pairs (default: 500 from MTSamples)
    python generate_synthetic.py --source mtsamples --count 500

    # Generate from Asclepius discharge summaries
    python generate_synthetic.py --source asclepius --count 1000

    # Use all sources
    python generate_synthetic.py --source all --count 2000
=============================================================================
"""

import argparse
import json
import os
import time
from pathlib import Path

import httpx
import textstat

# -- Configuration ----------------------------------------------------------
OUTPUT_DIR = "./medclear_results/synthetic_data"
DATASETS_DIR = "./medclear_results/datasets"

# Anthropic API settings
API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-opus-4-20250514"
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are an expert medical communicator who translates clinical text into plain language that patients and their families can understand.

Your task: Given a clinical note (discharge summary, operative report, or visit summary), rewrite it in plain English that:
1. An 8th-grader could understand (Flesch-Kincaid grade level 6-8)
2. Preserves ALL key medical facts, findings, and instructions
3. Explains medical terms when they must be used
4. Uses short sentences and common words
5. Organizes information clearly with headings if helpful
6. Maintains the clinical accuracy - never change the meaning

CRITICAL: Do NOT add information not in the original. Do NOT change diagnoses, medications, or instructions. If something is uncertain in the original, keep it uncertain in the simplified version.

Output ONLY the plain-language version. No preamble, no commentary."""

USER_PROMPT_TEMPLATE = """Rewrite the following clinical text in plain language that a patient can easily understand:

---
{clinical_text}
---

Plain language version:"""


def call_claude(clinical_text: str, api_key: str) -> str | None:
    """Call Claude API directly via httpx to generate plain-language version."""
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": MODEL,
        "max_tokens": MAX_TOKENS,
        "system": SYSTEM_PROMPT,
        "messages": [
            {
                "role": "user",
                "content": USER_PROMPT_TEMPLATE.format(
                    clinical_text=clinical_text),
            }
        ],
    }

    try:
        with httpx.Client(timeout=120.0) as client:
            response = client.post(API_URL, headers=headers, json=payload)
            response.raise_for_status()
            data = response.json()
            return data["content"][0]["text"]
    except httpx.HTTPStatusError as e:
        print(f"  API error {e.response.status_code}: {e.response.text[:200]}")
        if e.response.status_code == 429:
            print("  Rate limited. Waiting 60s...")
            time.sleep(60)
            return call_claude(clinical_text, api_key)
        return None
    except Exception as e:
        print(f"  Error: {e}")
        return None


def score_quality(source: str, simplified: str) -> dict:
    """Score the quality of a simplification pair."""
    src_fk = textstat.flesch_kincaid_grade(source) if len(source) > 50 else None
    tgt_fk = textstat.flesch_kincaid_grade(simplified) if len(simplified) > 50 else None
    tgt_ease = textstat.flesch_reading_ease(simplified) if len(simplified) > 50 else None
    tgt_smog = textstat.smog_index(simplified) if len(simplified) > 50 else None

    return {
        "source_fk_grade": src_fk,
        "target_fk_grade": tgt_fk,
        "target_reading_ease": tgt_ease,
        "target_smog": tgt_smog,
        "source_word_count": len(source.split()),
        "target_word_count": len(simplified.split()),
        "grade_reduction": (src_fk - tgt_fk) if src_fk and tgt_fk else None,
    }


def load_mtsamples() -> list[dict]:
    """Load MTSamples dataset."""
    from datasets import load_from_disk
    path = os.path.join(DATASETS_DIR, "mtsamples", "train")
    data = load_from_disk(path)

    # Prioritize discharge summaries and surgical notes
    priority_specialties = [
        "Discharge Summary", "Surgery", "General Medicine",
        "Consult - History and Phy.", "Cardiovascular / Pulmonary",
        "Orthopedic", "Neurology", "Gastroenterology",
        "SOAP / Chart / Progress Notes",
    ]

    examples = []
    for ex in data:
        text = ex.get("transcription", "")
        if not text or len(text.strip()) < 100:
            continue
        specialty = ex.get("medical_specialty", "").strip()
        priority = 0 if specialty in priority_specialties else 1
        examples.append({
            "text": text.strip(),
            "specialty": specialty,
            "description": ex.get("description", ""),
            "source_dataset": "mtsamples",
            "priority": priority,
        })

    # Sort by priority (discharge/surgery first)
    examples.sort(key=lambda x: (x["priority"], x["specialty"]))
    return examples


def load_asclepius() -> list[dict]:
    """Load Asclepius Synthetic Clinical Notes (discharge summaries only)."""
    from datasets import load_from_disk
    path = os.path.join(DATASETS_DIR, "asclepius", "train")
    data = load_from_disk(path)

    examples = []
    for ex in data:
        note = ex.get("note", "")
        if not note or len(note.strip()) < 100:
            continue
        # Only take discharge summaries and hospital course summaries
        note_lower = note[:200].lower()
        if any(kw in note_lower for kw in
               ["discharge summary", "hospital course", "admission date",
                "discharge date", "chief complaint"]):
            examples.append({
                "text": note.strip(),
                "specialty": "Discharge Summary (Synthetic)",
                "description": "",
                "source_dataset": "asclepius",
                "priority": 0,
            })

    return examples


def generate_pairs(source: str, count: int, api_key: str):
    """Generate synthetic plain-language training pairs."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load source data
    print(f"\nLoading {source} data...")
    if source == "mtsamples":
        examples = load_mtsamples()
    elif source == "asclepius":
        examples = load_asclepius()
    elif source == "all":
        examples = load_mtsamples() + load_asclepius()
    else:
        raise ValueError(f"Unknown source: {source}")

    print(f"Loaded {len(examples)} candidate notes")

    # Limit to requested count
    examples = examples[:count]
    print(f"Will process {len(examples)} notes")

    # Output file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(
        OUTPUT_DIR, f"synthetic_pairs_{source}_{timestamp}.jsonl")
    stats_file = os.path.join(
        OUTPUT_DIR, f"synthetic_stats_{source}_{timestamp}.json")

    pairs = []
    failed = 0
    filtered = 0

    print(f"\nGenerating plain-language versions with {MODEL}...")
    print(f"Output: {output_file}\n")

    with open(output_file, "w", encoding="utf-8") as f:
        for i, ex in enumerate(examples):
            # Truncate very long notes to ~2000 chars to save API cost
            clinical_text = ex["text"][:3000]

            print(f"[{i+1}/{len(examples)}] {ex['specialty'][:40]}... ",
                  end="", flush=True)

            simplified = call_claude(clinical_text, api_key)

            if not simplified:
                print("FAILED")
                failed += 1
                continue

            # Quality check
            quality = score_quality(clinical_text, simplified)

            # Filter: simplified should be easier than source
            if (quality["target_fk_grade"] is not None and
                    quality["target_fk_grade"] > 12):
                print(f"FILTERED (FK grade {quality['target_fk_grade']:.1f})")
                filtered += 1
                continue

            pair = {
                "source": clinical_text,
                "target": simplified,
                "source_dataset": ex["source_dataset"],
                "specialty": ex["specialty"],
                "quality_scores": quality,
            }
            pairs.append(pair)
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

            fk = quality["target_fk_grade"]
            fk_str = f"FK={fk:.1f}" if fk else "FK=N/A"
            print(f"OK ({fk_str}, {quality['target_word_count']} words)")

            # Rate limiting: ~1 request per second
            time.sleep(1.0)

    # Save stats
    stats = {
        "source": source,
        "total_processed": len(examples),
        "successful": len(pairs),
        "failed": failed,
        "filtered": filtered,
        "model": MODEL,
        "output_file": output_file,
    }

    if pairs:
        fk_grades = [p["quality_scores"]["target_fk_grade"]
                     for p in pairs
                     if p["quality_scores"]["target_fk_grade"] is not None]
        if fk_grades:
            stats["avg_target_fk_grade"] = sum(fk_grades) / len(fk_grades)
            stats["min_target_fk_grade"] = min(fk_grades)
            stats["max_target_fk_grade"] = max(fk_grades)

    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"  Successful pairs: {len(pairs)}")
    print(f"  Failed:           {failed}")
    print(f"  Filtered:         {filtered}")
    if "avg_target_fk_grade" in stats:
        print(f"  Avg FK grade:     {stats['avg_target_fk_grade']:.1f}")
    print(f"  Output:           {output_file}")
    print(f"  Stats:            {stats_file}")

    return pairs


def convert_to_training_format():
    """Convert all generated JSONL files into a combined dataset for training."""
    from datasets import Dataset, DatasetDict
    import glob

    print("\nConverting synthetic data to training format...")

    all_pairs = []
    for jsonl_file in glob.glob(os.path.join(OUTPUT_DIR, "*.jsonl")):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                pair = json.loads(line)
                all_pairs.append({
                    "source": pair["source"],
                    "target": pair["target"],
                    "dataset": f"synthetic_{pair['source_dataset']}",
                    "input_text": "simplify medical text: " + pair["source"],
                })

    if not all_pairs:
        print("No synthetic data found!")
        return None

    print(f"Total synthetic pairs: {len(all_pairs)}")

    # Split 90/5/5
    import random
    random.seed(42)
    random.shuffle(all_pairs)
    n = len(all_pairs)
    train_end = int(n * 0.9)
    val_end = int(n * 0.95)

    combined = DatasetDict({
        "train": Dataset.from_list(all_pairs[:train_end]),
        "validation": Dataset.from_list(all_pairs[train_end:val_end]),
        "test": Dataset.from_list(all_pairs[val_end:]),
    })

    out_dir = os.path.join(OUTPUT_DIR, "training_ready")
    combined.save_to_disk(out_dir)
    print(f"Saved to {out_dir}")
    for split in combined:
        print(f"  {split}: {len(combined[split])} examples")

    return combined


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate synthetic plain-language training data")
    parser.add_argument(
        "--source", choices=["mtsamples", "asclepius", "all"],
        default="mtsamples",
        help="Source dataset for clinical notes",
    )
    parser.add_argument(
        "--count", type=int, default=500,
        help="Number of notes to process",
    )
    parser.add_argument(
        "--convert", action="store_true",
        help="Convert existing JSONL files to training format (no API calls)",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="Anthropic API key (or set ANTHROPIC_API_KEY env var)",
    )
    args = parser.parse_args()

    if args.convert:
        convert_to_training_format()
    else:
        api_key = args.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("ERROR: Set ANTHROPIC_API_KEY environment variable or "
                  "pass --api-key")
            exit(1)
        generate_pairs(args.source, args.count, api_key)
        print("\nTo convert to training format, run:")
        print("  python generate_synthetic.py --convert")
