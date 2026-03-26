"""
=============================================================================
MedClear: Fine-Tuning FLAN-T5-base for Medical Text Simplification
=============================================================================
Model: google/flan-t5-base (250M params, encoder-decoder)
Datasets:
  - GEM/cochrane-simplification (~3,600 train pairs)
  - PLABA - Plain Language Adaptation of Biomedical Abstracts (~635 train pairs)
  - Med-EASi - Medical Expert-to-Simple (~1,400 train pairs)

Hardware: RTX 4070 Ti Super (16GB VRAM) / 64GB system RAM
Deployment: HuggingFace Spaces (free CPU tier, ~1GB model)

Usage:
    # Variant A: academic data only (Cochrane + PLABA + Med-EASi)
    python medclear_finetune.py --mode all --variant base

    # Variant B: all data including synthetic clinical notes
    python medclear_finetune.py --mode all --variant full

    # Steps can also be run individually:
    python medclear_finetune.py --mode prepare --variant base
    python medclear_finetune.py --mode train --variant base
    python medclear_finetune.py --mode evaluate --variant base
=============================================================================
"""

import argparse
import csv
import json
import os
import textwrap
from pathlib import Path

import torch

# -- Configuration ----------------------------------------------------------
CONFIG = {
    # Model
    "base_model": "google/flan-t5-base",
    "new_model_name": "medclear-t5-base",

    # Training hyperparameters
    "num_train_epochs": 5,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 2,  # effective batch = 8*2 = 16
    "learning_rate": 3e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.05,
    "lr_scheduler_type": "cosine",
    "max_source_length": 512,
    "max_target_length": 512,

    # Output
    "output_dir": "./medclear_results",
    "logging_steps": 50,
    "save_steps": 200,

    # Dataset paths (populated by prepare step)
    "datasets_dir": "./medclear_results/datasets",
    "seed": 42,
}

# -- Task prefix for T5 ----------------------------------------------------
TASK_PREFIX = "simplify medical text: "


# ==========================================================================
# STEP 1: DATA PREPARATION - Combine all datasets
# ==========================================================================
def prepare_data():
    """Download, normalize, and combine all training datasets."""
    from datasets import load_dataset, load_from_disk, Dataset, DatasetDict

    print("\n" + "=" * 70)
    print("STEP 1: PREPARING COMBINED DATASET")
    print("=" * 70)

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    os.makedirs(CONFIG["datasets_dir"], exist_ok=True)

    all_train = []
    all_val = []
    all_test = []

    # -- 1. Cochrane (GEM) -------------------------------------------------
    print("\n[1/4] Loading GEM/cochrane-simplification...")
    cochrane_dir = os.path.join(CONFIG["output_dir"], "raw_data")
    if os.path.exists(os.path.join(cochrane_dir, "train.json")):
        print("  Using cached files...")
    else:
        from huggingface_hub import hf_hub_download
        os.makedirs(cochrane_dir, exist_ok=True)
        for f in ["train.json", "validation.json", "test.json"]:
            hf_hub_download(
                repo_id="GEM/cochrane-simplification",
                filename=f, repo_type="dataset",
                local_dir=cochrane_dir,
            )
    cochrane = load_dataset("json", data_files={
        "train": os.path.join(cochrane_dir, "train.json"),
        "validation": os.path.join(cochrane_dir, "validation.json"),
        "test": os.path.join(cochrane_dir, "test.json"),
    })
    for ex in cochrane["train"]:
        all_train.append({"source": ex["source"], "target": ex["target"],
                          "dataset": "cochrane"})
    for ex in cochrane["validation"]:
        all_val.append({"source": ex["source"], "target": ex["target"],
                        "dataset": "cochrane"})
    for ex in cochrane["test"]:
        all_test.append({"source": ex["source"], "target": ex["target"],
                         "dataset": "cochrane"})
    print(f"  Cochrane: {len(cochrane['train'])} train, "
          f"{len(cochrane['validation'])} val, {len(cochrane['test'])} test")

    # -- 2. PLABA ----------------------------------------------------------
    print("\n[2/4] Loading PLABA...")
    plaba_dir = os.path.join(CONFIG["datasets_dir"], "plaba")
    plaba_files = {
        "train": os.path.join(plaba_dir, "train.csv"),
        "val": os.path.join(plaba_dir, "val.csv"),
        "test": os.path.join(plaba_dir, "test.csv"),
    }
    if all(os.path.exists(p) for p in plaba_files.values()):
        for split_name, path in plaba_files.items():
            with open(path, "r", encoding="utf-8-sig") as f:
                reader = csv.DictReader(f)
                count = 0
                for row in reader:
                    entry = {
                        "source": row["input_text"],
                        "target": row["target_text"],
                        "dataset": "plaba",
                    }
                    if split_name == "train":
                        all_train.append(entry)
                    elif split_name == "val":
                        all_val.append(entry)
                    else:
                        all_test.append(entry)
                    count += 1
                print(f"  PLABA {split_name}: {count} examples")
    else:
        print("  PLABA not found locally. Downloading from OSF...")
        import urllib.request
        os.makedirs(plaba_dir, exist_ok=True)
        osf_url = "https://api.osf.io/v2/nodes/rnpmf/files/osfstorage/"
        req = urllib.request.Request(osf_url,
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            osf_data = json.loads(resp.read().decode())
        for item in osf_data.get("data", []):
            name = item["attributes"]["name"]
            if name in ["train.csv", "val.csv", "test.csv"]:
                dl_url = item["links"]["download"]
                out = os.path.join(plaba_dir, name)
                urllib.request.urlretrieve(dl_url, out)
                print(f"  Downloaded {name}")
        # Recurse to load the downloaded files
        return prepare_data()

    # -- 3. Med-EASi -------------------------------------------------------
    print("\n[3/4] Loading Med-EASi...")
    medeasi_dir = os.path.join(CONFIG["datasets_dir"], "medeasi")
    if os.path.exists(os.path.join(medeasi_dir, "train")):
        medeasi_train = load_from_disk(os.path.join(medeasi_dir, "train"))
        medeasi_val = load_from_disk(os.path.join(medeasi_dir, "validation"))
        medeasi_test = load_from_disk(os.path.join(medeasi_dir, "test"))
    else:
        print("  Downloading from HuggingFace...")
        medeasi = load_dataset("cbasu/Med-EASi")
        os.makedirs(medeasi_dir, exist_ok=True)
        for split in medeasi:
            medeasi[split].save_to_disk(os.path.join(medeasi_dir, split))
        medeasi_train = medeasi["train"]
        medeasi_val = medeasi["validation"]
        medeasi_test = medeasi["test"]

    for ex in medeasi_train:
        all_train.append({"source": ex["Expert"], "target": ex["Simple"],
                          "dataset": "medeasi"})
    for ex in medeasi_val:
        all_val.append({"source": ex["Expert"], "target": ex["Simple"],
                        "dataset": "medeasi"})
    for ex in medeasi_test:
        all_test.append({"source": ex["Expert"], "target": ex["Simple"],
                         "dataset": "medeasi"})
    print(f"  Med-EASi: {len(medeasi_train)} train, "
          f"{len(medeasi_val)} val, {len(medeasi_test)} test")

    # -- 4. Synthetic (Claude-generated) ----------------------------------
    # Choose CoT or regular synthetic file
    if CONFIG.get("use_cot", False):
        synthetic_file = os.path.join(CONFIG["output_dir"],
                                       "synthetic_data",
                                       "claude_generated_pairs_cot.jsonl")
    else:
        synthetic_file = os.path.join(CONFIG["output_dir"],
                                       "synthetic_data",
                                       "claude_generated_pairs.jsonl")
    if CONFIG.get("include_synthetic", True) and os.path.exists(synthetic_file):
        print("\n[4/4] Loading synthetic Claude-generated pairs...")
        import random
        random.seed(CONFIG["seed"])
        synthetic = []
        with open(synthetic_file, "r", encoding="utf-8") as f:
            for line in f:
                pair = json.loads(line)
                synthetic.append({
                    "source": pair["source"],
                    "target": pair["target"],
                    "dataset": "synthetic_" + pair.get("source_dataset",
                                                        pair.get("dataset",
                                                                  "unknown")),
                })
        random.shuffle(synthetic)
        n = len(synthetic)
        train_end = int(n * 0.8)
        val_end = int(n * 0.9)
        all_train.extend(synthetic[:train_end])
        all_val.extend(synthetic[train_end:val_end])
        all_test.extend(synthetic[val_end:])
        print(f"  Synthetic: {train_end} train, "
              f"{val_end - train_end} val, {n - val_end} test")
    elif not CONFIG.get("include_synthetic", True):
        print("\n[4/4] Synthetic data excluded (variant=base)")
    else:
        print("\n[4/4] No synthetic data found (skipping)")

    # -- Combine and save --------------------------------------------------
    print(f"\n--- Combined Dataset ---")
    print(f"  Train:      {len(all_train)}")
    print(f"  Validation: {len(all_val)}")
    print(f"  Test:       {len(all_test)}")

    # Dataset breakdown
    from collections import Counter
    train_counts = Counter(e["dataset"] for e in all_train)
    print(f"\n  Train breakdown: {dict(train_counts)}")

    # Add T5 task prefix to source
    # Use different prefix for CoT clinical notes vs academic text
    COT_PREFIX = "extract facts and simplify clinical note: "
    for split in [all_train, all_val, all_test]:
        for ex in split:
            if CONFIG.get("use_cot") and "synthetic" in ex.get("dataset", ""):
                ex["input_text"] = COT_PREFIX + ex["source"]
            else:
                ex["input_text"] = TASK_PREFIX + ex["source"]

    # Convert to HuggingFace Dataset
    combined = DatasetDict({
        "train": Dataset.from_list(all_train),
        "validation": Dataset.from_list(all_val),
        "test": Dataset.from_list(all_test),
    })

    # Save to variant-specific directory
    variant_dir = CONFIG.get("variant_dir", os.path.join(CONFIG["output_dir"],
                                                          "default"))
    os.makedirs(variant_dir, exist_ok=True)
    combined_dir = os.path.join(variant_dir, "combined_data")
    # Remove existing data to avoid overwrite errors
    import shutil
    if os.path.exists(combined_dir):
        shutil.rmtree(combined_dir)
    combined.save_to_disk(combined_dir)

    # Save raw test examples for evaluation
    test_examples = [{"source": t["source"], "target": t["target"],
                      "dataset": t["dataset"]}
                     for t in all_test]
    with open(os.path.join(variant_dir, "test_examples.json"), "w") as f:
        json.dump(test_examples, f, indent=2)

    # Show a sample
    print(f"\n-- Sample (Cochrane) --")
    sample = all_train[0]
    print(f"INPUT:  {textwrap.shorten(sample['input_text'], 120)}")
    print(f"TARGET: {textwrap.shorten(sample['target'], 120)}")

    print(f"\n-- Sample (PLABA) --")
    plaba_samples = [e for e in all_train if e["dataset"] == "plaba"]
    if plaba_samples:
        sample = plaba_samples[0]
        print(f"INPUT:  {textwrap.shorten(sample['input_text'], 120)}")
        print(f"TARGET: {textwrap.shorten(sample['target'], 120)}")

    # Check token lengths with T5 tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    src_lengths = [len(tokenizer.encode(t["input_text"])) for t in all_train]
    tgt_lengths = [len(tokenizer.encode(t["target"])) for t in all_train]
    print(f"\nSource token lengths (train):")
    print(f"  Mean: {sum(src_lengths)/len(src_lengths):.0f}, "
          f"Max: {max(src_lengths)}, "
          f">512: {sum(1 for l in src_lengths if l > 512)}")
    print(f"Target token lengths (train):")
    print(f"  Mean: {sum(tgt_lengths)/len(tgt_lengths):.0f}, "
          f"Max: {max(tgt_lengths)}, "
          f">512: {sum(1 for l in tgt_lengths if l > 512)}")

    print(f"\nData saved to {combined_dir}/")
    print("[OK] Data preparation complete!")
    return combined


# ==========================================================================
# STEP 2: FINE-TUNING FLAN-T5-base
# ==========================================================================
def train_model():
    """Fine-tune FLAN-T5-base on the combined medical simplification dataset."""
    from datasets import load_from_disk
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        Seq2SeqTrainingArguments,
        Seq2SeqTrainer,
        DataCollatorForSeq2Seq,
    )

    print("\n" + "=" * 70)
    print("STEP 2: FINE-TUNING FLAN-T5-base")
    print("=" * 70)

    # Load combined data
    variant_dir = CONFIG.get("variant_dir", os.path.join(CONFIG["output_dir"],
                                                          "default"))
    combined_dir = os.path.join(variant_dir, "combined_data")
    dataset = load_from_disk(combined_dir)
    print(f"\nLoaded {len(dataset['train'])} train, "
          f"{len(dataset['validation'])} val examples")

    # Load model and tokenizer
    print(f"\nLoading {CONFIG['base_model']}...")
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    model = AutoModelForSeq2SeqLM.from_pretrained(CONFIG["base_model"])

    # Enable gradient checkpointing for large models to save VRAM
    if "large" in CONFIG.get("base_model", ""):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled (saves ~40% VRAM)")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters()
                           if p.requires_grad)
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")

    # Tokenize
    max_src = CONFIG["max_source_length"]
    max_tgt = CONFIG["max_target_length"]

    def preprocess(examples):
        inputs = tokenizer(
            examples["input_text"],
            max_length=max_src,
            truncation=True,
            padding=False,
        )
        targets = tokenizer(
            examples["target"],
            max_length=max_tgt,
            truncation=True,
            padding=False,
        )
        inputs["labels"] = targets["input_ids"]
        return inputs

    print("Tokenizing dataset...")
    tokenized = dataset.map(
        preprocess,
        batched=True,
        remove_columns=dataset["train"].column_names,
    )

    # Data collator (handles dynamic padding)
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
    )

    # Training arguments
    output_model_dir = os.path.join(variant_dir, CONFIG["new_model_name"])
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_model_dir,
        num_train_epochs=CONFIG["num_train_epochs"],
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        warmup_steps=100,
        lr_scheduler_type=CONFIG["lr_scheduler_type"],
        logging_steps=CONFIG["logging_steps"],
        save_steps=CONFIG["save_steps"],
        save_total_limit=2,
        eval_strategy="steps",
        eval_steps=CONFIG["save_steps"],
        predict_with_generate=True,
        generation_max_length=max_tgt,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        report_to="none",
        seed=CONFIG["seed"],
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
    )

    # Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
    )

    print(f"\nStarting fine-tuning...")
    print(f"  Epochs: {CONFIG['num_train_epochs']}")
    print(f"  Effective batch size: "
          f"{CONFIG['per_device_train_batch_size'] * CONFIG['gradient_accumulation_steps']}")
    print(f"  Max source tokens: {max_src}")
    print(f"  Max target tokens: {max_tgt}")

    trainer.train()

    # Save the fine-tuned model
    final_dir = os.path.join(variant_dir, f"{CONFIG['new_model_name']}-final")
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)
    print(f"\n[OK] Model saved to {final_dir}")

    # Save training log
    with open(os.path.join(variant_dir, "training_log.json"), "w") as f:
        json.dump(trainer.state.log_history, f, indent=2)

    print("[OK] Fine-tuning complete!")


# ==========================================================================
# STEP 3: EVALUATION
# ==========================================================================
def evaluate_model():
    """Compare base FLAN-T5 vs fine-tuned MedClear on test examples."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    from rouge_score import rouge_scorer
    import textstat

    print("\n" + "=" * 70)
    print("STEP 3: BEFORE/AFTER EVALUATION")
    print("=" * 70)

    # Load test examples
    variant_dir = CONFIG.get("variant_dir", os.path.join(CONFIG["output_dir"],
                                                          "default"))
    with open(os.path.join(variant_dir, "test_examples.json")) as f:
        test_examples = json.load(f)

    N_EVAL = min(50, len(test_examples))
    test_subset = test_examples[:N_EVAL]
    print(f"\nEvaluating on {N_EVAL} test examples...")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def generate_batch(model, tokenizer, texts, batch_size=8):
        outputs = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            inputs = tokenizer(
                batch, max_length=CONFIG["max_source_length"],
                truncation=True, padding=True, return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=CONFIG["max_target_length"],
                    num_beams=4,
                    early_stopping=True,
                )
            decoded = tokenizer.batch_decode(generated, skip_special_tokens=True)
            outputs.extend(decoded)
            if (i + batch_size) % 16 == 0:
                print(f"  {min(i + batch_size, len(texts))}/{len(texts)} done")
        return outputs

    # Prepare inputs
    input_texts = [TASK_PREFIX + ex["source"] for ex in test_subset]

    # Base model
    print(f"\nLoading base model: {CONFIG['base_model']}...")
    base_tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        CONFIG["base_model"]).to(device)
    base_model.eval()

    print("Generating base model outputs...")
    base_outputs = generate_batch(base_model, base_tokenizer, input_texts)
    del base_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Fine-tuned model
    final_dir = os.path.join(variant_dir,
                              f"{CONFIG['new_model_name']}-final")
    print(f"\nLoading fine-tuned model from {final_dir}...")
    ft_tokenizer = AutoTokenizer.from_pretrained(final_dir)
    ft_model = AutoModelForSeq2SeqLM.from_pretrained(final_dir).to(device)
    ft_model.eval()

    print("Generating fine-tuned model outputs...")
    ft_outputs = generate_batch(ft_model, ft_tokenizer, input_texts)
    del ft_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Compute metrics
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    results = {"base": [], "finetuned": []}
    for i, ex in enumerate(test_subset):
        reference = ex["target"]
        for label, output in [("base", base_outputs[i]),
                               ("finetuned", ft_outputs[i])]:
            rouge = scorer.score(reference, output)
            fk = textstat.flesch_kincaid_grade(output) if len(output) > 20 else None
            smog = textstat.smog_index(output) if len(output) > 20 else None
            ease = textstat.flesch_reading_ease(output) if len(output) > 20 else None
            results[label].append({
                "rouge1_f": rouge["rouge1"].fmeasure,
                "rouge2_f": rouge["rouge2"].fmeasure,
                "rougeL_f": rouge["rougeL"].fmeasure,
                "flesch_kincaid_grade": fk,
                "smog_index": smog,
                "flesch_reading_ease": ease,
                "word_count": len(output.split()),
            })

    # Display results
    def avg(lst, key):
        vals = [d[key] for d in lst if d[key] is not None]
        return sum(vals) / len(vals) if vals else 0

    print("\n" + "=" * 70)
    print("              BEFORE vs AFTER COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<30} {'Base FLAN-T5':>15} {'MedClear (FT)':>15}")
    print("-" * 60)

    metrics = [
        ("ROUGE-1 F1", "rouge1_f"),
        ("ROUGE-2 F1", "rouge2_f"),
        ("ROUGE-L F1", "rougeL_f"),
        ("Flesch-Kincaid Grade", "flesch_kincaid_grade"),
        ("SMOG Index", "smog_index"),
        ("Flesch Reading Ease", "flesch_reading_ease"),
        ("Avg Word Count", "word_count"),
    ]
    comparison = {}
    for label, key in metrics:
        base_val = avg(results["base"], key)
        ft_val = avg(results["finetuned"], key)
        comparison[key] = {"base": base_val, "finetuned": ft_val}
        print(f"{label:<30} {base_val:>15.2f} {ft_val:>15.2f}")

    # Qualitative examples
    print("\n" + "=" * 70)
    print("QUALITATIVE EXAMPLES")
    print("=" * 70)
    for i in range(min(3, N_EVAL)):
        print(f"\n{'-' * 70}")
        print(f"Example {i + 1} (dataset: {test_subset[i].get('dataset', '?')})")
        print(f"{'-' * 70}")
        print(f"\nSOURCE (technical):")
        print(textwrap.fill(test_subset[i]["source"][:400], 80))
        print(f"\nREFERENCE (plain language):")
        print(textwrap.fill(test_subset[i]["target"][:400], 80))
        print(f"\nBASE FLAN-T5 output:")
        print(textwrap.fill(base_outputs[i][:400], 80))
        print(f"\nMEDCLEAR output:")
        print(textwrap.fill(ft_outputs[i][:400], 80))

    # Save results
    full_results = {
        "metrics_summary": comparison,
        "examples": [
            {
                "source": test_subset[i]["source"],
                "reference": test_subset[i]["target"],
                "dataset": test_subset[i].get("dataset", "unknown"),
                "base_output": base_outputs[i],
                "finetuned_output": ft_outputs[i],
                "base_metrics": results["base"][i],
                "finetuned_metrics": results["finetuned"][i],
            }
            for i in range(N_EVAL)
        ],
    }
    eval_path = os.path.join(variant_dir, "evaluation_results.json")
    with open(eval_path, "w") as f:
        json.dump(full_results, f, indent=2)

    print(f"\n[OK] Results saved to {eval_path}")
    print("[OK] Evaluation complete!")
    return full_results


# ==========================================================================
# STEP 4: INTERACTIVE DEMO
# ==========================================================================
def run_demo():
    """Interactive demo for the fine-tuned model."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    print("\n" + "=" * 70)
    print("MedClear -- Interactive Demo")
    print("=" * 70)
    print("Enter medical text to simplify. Type 'quit' to exit.\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    variant_dir = CONFIG.get("variant_dir", os.path.join(CONFIG["output_dir"],
                                                          "default"))
    final_dir = os.path.join(variant_dir,
                              f"{CONFIG['new_model_name']}-final")
    tokenizer = AutoTokenizer.from_pretrained(final_dir)
    model = AutoModelForSeq2SeqLM.from_pretrained(final_dir).to(device)
    model.eval()

    examples = [
        "A meta-analysis of 12 randomized controlled trials (n=3,847) demonstrated "
        "that prophylactic low-molecular-weight heparin significantly reduced the "
        "incidence of venous thromboembolism (RR 0.51, 95% CI 0.41-0.63, p<0.001) "
        "in post-surgical patients compared to unfractionated heparin, with no "
        "significant increase in major bleeding events (RR 1.02, 95% CI 0.73-1.43).",

        "Patient underwent laparoscopic cholecystectomy for acute cholecystitis. "
        "Intraoperative findings revealed a distended, edematous gallbladder with "
        "adhesions to the omentum. Critical view of safety was achieved. "
        "Estimated blood loss was minimal. Patient tolerated the procedure well "
        "and was transferred to PACU in stable condition.",
    ]

    print("-- Type 'ex1' or 'ex2' to try pre-loaded examples --\n")

    while True:
        user_input = input("Enter medical text (or 'quit'): ").strip()
        if user_input.lower() == "quit":
            break
        elif user_input.lower() == "ex1":
            user_input = examples[0]
            print(f"\nUsing example 1:\n{textwrap.fill(user_input, 80)}\n")
        elif user_input.lower() == "ex2":
            user_input = examples[1]
            print(f"\nUsing example 2:\n{textwrap.fill(user_input, 80)}\n")

        if not user_input:
            continue

        input_text = TASK_PREFIX + user_input
        inputs = tokenizer(input_text, return_tensors="pt",
                           max_length=CONFIG["max_source_length"],
                           truncation=True).to(device)

        print("\nGenerating plain language version...")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=CONFIG["max_target_length"],
                num_beams=4,
                early_stopping=True,
            )
        result = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        print(f"\nMedClear output:")
        print(textwrap.fill(result, 80))
        print()


# ==========================================================================
# MAIN
# ==========================================================================
def apply_variant(variant: str):
    """Configure paths based on the training variant."""
    if variant == "large":
        # FLAN-T5-large (783M) with all data + CoT + gradient checkpointing
        CONFIG["base_model"] = "google/flan-t5-large"
        CONFIG["include_synthetic"] = True
        CONFIG["use_cot"] = True
        CONFIG["new_model_name"] = "medclear-t5-large"
        CONFIG["variant_dir"] = os.path.join(CONFIG["output_dir"], "large")
        CONFIG["max_target_length"] = 768
        CONFIG["max_source_length"] = 512
        CONFIG["per_device_train_batch_size"] = 2
        CONFIG["per_device_eval_batch_size"] = 2
        CONFIG["gradient_accumulation_steps"] = 8  # effective batch = 16
        CONFIG["learning_rate"] = 1e-5  # much lower LR for larger model (5e-5 caused divergence)
        CONFIG["num_train_epochs"] = 3  # fewer epochs (more data)
        CONFIG["save_steps"] = 500
        CONFIG["logging_steps"] = 100
    elif variant == "cot":
        CONFIG["include_synthetic"] = True
        CONFIG["use_cot"] = True
        CONFIG["new_model_name"] = "medclear-t5-cot"
        CONFIG["variant_dir"] = os.path.join(CONFIG["output_dir"], "cot")
        CONFIG["max_target_length"] = 768
        CONFIG["per_device_train_batch_size"] = 4
        CONFIG["gradient_accumulation_steps"] = 4
    elif variant == "full":
        CONFIG["include_synthetic"] = True
        CONFIG["use_cot"] = False
        CONFIG["new_model_name"] = "medclear-t5-full"
        CONFIG["variant_dir"] = os.path.join(CONFIG["output_dir"], "full")
    else:  # "base"
        CONFIG["include_synthetic"] = False
        CONFIG["use_cot"] = False
        CONFIG["new_model_name"] = "medclear-t5-base-only"
        CONFIG["variant_dir"] = os.path.join(CONFIG["output_dir"], "base")

    # Each variant gets its own combined_data, model, and eval results
    os.makedirs(CONFIG["variant_dir"], exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MedClear: Medical Text Simplification with FLAN-T5")
    parser.add_argument(
        "--mode",
        choices=["prepare", "train", "evaluate", "demo", "all"],
        default="all",
        help="Which step(s) to run",
    )
    parser.add_argument(
        "--variant",
        choices=["base", "full", "cot", "large"],
        default="base",
        help="'base'=academic; 'full'=+synthetic; 'cot'=+CoT; 'large'=T5-large+all",
    )
    args = parser.parse_args()
    apply_variant(args.variant)

    print(f"\n>>> Variant: {args.variant} "
          f"(synthetic={'YES' if CONFIG['include_synthetic'] else 'NO'})")

    if args.mode in ("prepare", "all"):
        prepare_data()
    if args.mode in ("train", "all"):
        train_model()
    if args.mode in ("evaluate", "all"):
        evaluate_model()
    if args.mode == "demo":
        run_demo()

    if args.mode == "all":
        print("\n" + "=" * 70)
        print("FULL PIPELINE COMPLETE!")
        print("=" * 70)
        print(f"Results in: {CONFIG['output_dir']}/")
        print("Next steps:")
        print("  1. Run --mode demo for interactive testing")
        print("  2. Check evaluation_results.json for metrics")
        print("  3. Deploy to HuggingFace Spaces!")
