"""
Bulk clinical note simplification using Claude Code session.
Reads notes from synthetic_batch_large.json, processes in chunks,
and saves simplified pairs to the synthetic data JSONL.

Usage:
    python bulk_simplify.py --chunk 0    # Process notes 0-49
    python bulk_simplify.py --chunk 1    # Process notes 50-99
    python bulk_simplify.py --status     # Show progress
"""

import argparse
import json
import os
import sys

BATCH_FILE = "./medclear_results/synthetic_batch_large.json"
OUTPUT_FILE = "./medclear_results/synthetic_data/claude_generated_pairs.jsonl"
CHUNK_SIZE = 50


def show_status():
    """Show how many pairs have been generated."""
    if not os.path.exists(OUTPUT_FILE):
        print("No pairs generated yet.")
        return

    count = 0
    datasets = {}
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            pair = json.loads(line)
            ds = pair.get("dataset", "unknown")
            datasets[ds] = datasets.get(ds, 0) + 1
            count += 1

    print(f"Total pairs: {count}")
    for ds, c in sorted(datasets.items()):
        print(f"  {ds}: {c}")

    with open(BATCH_FILE, "r", encoding="utf-8") as f:
        total = len(json.load(f))
    print(f"\nBatch size: {total}")
    print(f"Remaining: ~{total - count + 20}")  # approximate


def get_chunk(chunk_id):
    """Get a chunk of notes to process."""
    with open(BATCH_FILE, "r", encoding="utf-8") as f:
        batch = json.load(f)

    start = chunk_id * CHUNK_SIZE
    end = min(start + CHUNK_SIZE, len(batch))
    return batch[start:end], start, end, len(batch)


def print_chunk(chunk_id):
    """Print notes in a chunk for simplification."""
    notes, start, end, total = get_chunk(chunk_id)
    if not notes:
        print(f"Chunk {chunk_id} is empty (only {total} notes total)")
        return

    print(f"=== CHUNK {chunk_id}: notes {start}-{end-1} of {total} ===\n")
    for i, note in enumerate(notes):
        idx = start + i
        print(f"--- NOTE {idx} ({note['spec']}) ---")
        if note['desc']:
            print(f"DESC: {note['desc']}")
        print(note['text'][:1500])
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", type=int, default=None)
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--print", dest="print_chunk", type=int, default=None)
    args = parser.parse_args()

    if args.status:
        show_status()
    elif args.print_chunk is not None:
        print_chunk(args.print_chunk)
    elif args.chunk is not None:
        print_chunk(args.chunk)
    else:
        show_status()
