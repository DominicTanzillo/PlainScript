"""
=============================================================================
MedClear: Fine-Tuning an LLM for Medical Text Simplification
=============================================================================
Using QLoRA (4-bit quantized LoRA) on Mistral-7B-Instruct-v0.3
Dataset: Cochrane Systematic Review Abstracts → Plain Language Summaries
~4,500 paired examples from GEM/cochrane-simplification (Devaraj et al., 2021)

Hardware: RTX 4070 Ti Super (16GB VRAM) / 64GB system RAM
Estimated training time: ~1-2 hours for 3 epochs on 4,500 examples

Setup:
    pip install torch transformers datasets peft bitsandbytes accelerate trl
    pip install rouge-score nltk textstat sentencepiece protobuf
    # Optional: pip install wandb  (for tracking)

Usage:
    python cochrane_plainlang_finetune.py --mode prepare    # Download & prep data
    python cochrane_plainlang_finetune.py --mode train       # Fine-tune with QLoRA
    python cochrane_plainlang_finetune.py --mode evaluate    # Before/after comparison
    python cochrane_plainlang_finetune.py --mode demo        # Interactive demo
    python cochrane_plainlang_finetune.py --mode all         # Run full pipeline
=============================================================================
"""

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

import torch

# ── Configuration ──────────────────────────────────────────────────────────
CONFIG = {
    # Model
    "base_model": "mistralai/Mistral-7B-Instruct-v0.3",
    "new_model_name": "medclear-mistral-7b-cochrane",

    # QLoRA parameters
    "lora_r": 16,                # Rank of the low-rank matrices
    "lora_alpha": 32,            # Scaling factor (alpha/r = 2)
    "lora_dropout": 0.05,
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],

    # BitsAndBytes 4-bit quantization
    "use_4bit": True,
    "bnb_4bit_compute_dtype": "float16",
    "bnb_4bit_quant_type": "nf4",
    "use_nested_quant": False,

    # Training hyperparameters
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "per_device_eval_batch_size": 2,
    "gradient_accumulation_steps": 4,  # effective batch = 2*4 = 8
    "learning_rate": 2e-4,
    "weight_decay": 0.001,
    "max_grad_norm": 0.3,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "max_seq_length": 1024,

    # Output
    "output_dir": "./medclear_results",
    "logging_steps": 25,
    "save_steps": 100,

    # Dataset
    "dataset_name": "GEM/cochrane-simplification",
    "test_split_ratio": 0.1,
    "seed": 42,
}

# ── Prompt Template ────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are MedClear, a medical communication assistant. "
    "Your task is to rewrite technical medical abstracts into plain language "
    "that a patient without medical training can understand. "
    "Preserve all key findings and conclusions. Use short sentences, "
    "common words, and explain any necessary medical terms."
)

def format_instruction(source_text: str, target_text: str = None) -> str:
    """Format a single example into the Mistral instruct template."""
    user_msg = (
        f"Rewrite the following medical text in plain language that a "
        f"patient can easily understand:\n\n{source_text}"
    )
    if target_text:
        # Training format: include the target
        return (
            f"<s>[INST] {SYSTEM_PROMPT}\n\n{user_msg} [/INST] "
            f"{target_text}</s>"
        )
    else:
        # Inference format: no target
        return f"<s>[INST] {SYSTEM_PROMPT}\n\n{user_msg} [/INST] "


# ==========================================================================
# STEP 1: DATA PREPARATION
# ==========================================================================
def prepare_data():
    """Download and prepare the Cochrane simplification dataset."""
    from datasets import load_dataset

    print("\n" + "="*70)
    print("STEP 1: PREPARING COCHRANE DATASET")
    print("="*70)

    # Load the GEM/cochrane-simplification dataset
    # Download JSON files directly (the repo has a legacy loading script
    # that newer datasets versions refuse to run)
    print("\nDownloading GEM/cochrane-simplification from HuggingFace...")
    from huggingface_hub import hf_hub_download
    data_dir = os.path.join(CONFIG["output_dir"], "raw_data")
    os.makedirs(data_dir, exist_ok=True)
    for split_file in ["train.json", "validation.json", "test.json"]:
        hf_hub_download(
            repo_id=CONFIG["dataset_name"],
            filename=split_file,
            repo_type="dataset",
            local_dir=data_dir,
        )
        print(f"  Downloaded {split_file}")
    dataset = load_dataset("json", data_files={
        "train": os.path.join(data_dir, "train.json"),
        "validation": os.path.join(data_dir, "validation.json"),
        "test": os.path.join(data_dir, "test.json"),
    })

    print(f"\nDataset splits:")
    for split_name, split_data in dataset.items():
        print(f"  {split_name}: {len(split_data)} examples")

    # Inspect a sample
    print("\n-- Sample (before formatting) -----------------------------")
    sample = dataset["train"][0]
    print(f"DOI: {sample['doi']}")
    print(f"\nSOURCE (technical):\n{textwrap.fill(sample['source'][:500], 80)}")
    print(f"\nTARGET (plain language):\n{textwrap.fill(sample['target'][:500], 80)}")

    # Format into instruction-tuning format
    def format_example(example):
        example["text"] = format_instruction(example["source"], example["target"])
        return example

    print("\nFormatting examples into Mistral instruct template...")
    formatted = dataset.map(format_example, remove_columns=["gem_id", "doi"])

    # Use the built-in train/validation/test splits
    train_data = formatted["train"]
    val_data = formatted["validation"]
    test_data = formatted["test"]

    print(f"\nFinal split sizes:")
    print(f"  Train:      {len(train_data)}")
    print(f"  Validation: {len(val_data)}")
    print(f"  Test:       {len(test_data)}")

    # Check token lengths
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lengths = [len(tokenizer.encode(t["text"])) for t in train_data]
    print(f"\nToken length stats (train):")
    print(f"  Mean: {sum(lengths)/len(lengths):.0f}")
    print(f"  Max:  {max(lengths)}")
    print(f"  Min:  {min(lengths)}")
    print(f"  >1024: {sum(1 for l in lengths if l > 1024)} examples")

    # Save formatted data
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    train_data.save_to_disk(f"{CONFIG['output_dir']}/train_data")
    val_data.save_to_disk(f"{CONFIG['output_dir']}/val_data")
    test_data.save_to_disk(f"{CONFIG['output_dir']}/test_data")

    # Also save raw test examples for evaluation
    test_examples = [{"source": t["source"], "target": t["target"]} for t in test_data]
    with open(f"{CONFIG['output_dir']}/test_examples.json", "w") as f:
        json.dump(test_examples, f, indent=2)

    print(f"\nData saved to {CONFIG['output_dir']}/")
    print("[OK] Data preparation complete!")
    return train_data, val_data, test_data


# ==========================================================================
# STEP 2: FINE-TUNING WITH QLoRA
# ==========================================================================
def train_model():
    """Fine-tune Mistral-7B with QLoRA on the Cochrane dataset."""
    from datasets import load_from_disk
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model
    from trl import SFTTrainer

    print("\n" + "="*70)
    print("STEP 2: FINE-TUNING WITH QLoRA")
    print("="*70)

    # Load prepared data
    train_data = load_from_disk(f"{CONFIG['output_dir']}/train_data")
    val_data = load_from_disk(f"{CONFIG['output_dir']}/val_data")
    print(f"\nLoaded {len(train_data)} train, {len(val_data)} val examples")

    # ── Quantization config ───────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=CONFIG["use_4bit"],
        bnb_4bit_quant_type=CONFIG["bnb_4bit_quant_type"],
        bnb_4bit_compute_dtype=getattr(torch, CONFIG["bnb_4bit_compute_dtype"]),
        bnb_4bit_use_double_quant=CONFIG["use_nested_quant"],
    )

    # ── Load base model ───────────────────────────────────────────────────
    print(f"\nLoading {CONFIG['base_model']} in 4-bit precision...")
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="eager",  # safe fallback for consumer GPUs
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1

    # ── Tokenizer ─────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ── LoRA config ───────────────────────────────────────────────────────
    peft_config = LoraConfig(
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=CONFIG["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    # Prepare model
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)

    trainable, total = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"\nTrainable params: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.2f}%)")

    # ── Training arguments ────────────────────────────────────────────────
    training_args = TrainingArguments(
        output_dir=CONFIG["output_dir"],
        num_train_epochs=CONFIG["num_train_epochs"],
        per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
        per_device_eval_batch_size=CONFIG["per_device_eval_batch_size"],
        gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
        learning_rate=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        max_grad_norm=CONFIG["max_grad_norm"],
        warmup_ratio=CONFIG["warmup_ratio"],
        lr_scheduler_type=CONFIG["lr_scheduler_type"],
        logging_steps=CONFIG["logging_steps"],
        save_steps=CONFIG["save_steps"],
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=CONFIG["save_steps"],
        fp16=True,
        optim="paged_adamw_32bit",
        group_by_length=True,
        report_to="none",  # Change to "wandb" if using W&B
        seed=CONFIG["seed"],
    )

    # ── Trainer ───────────────────────────────────────────────────────────
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        eval_dataset=val_data,
        peft_config=peft_config,
        max_seq_length=CONFIG["max_seq_length"],
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    print("\nStarting fine-tuning...")
    print(f"   Epochs: {CONFIG['num_train_epochs']}")
    print(f"   Effective batch size: "
          f"{CONFIG['per_device_train_batch_size'] * CONFIG['gradient_accumulation_steps']}")
    print(f"   LoRA rank: {CONFIG['lora_r']}, alpha: {CONFIG['lora_alpha']}")
    print(f"   Max seq length: {CONFIG['max_seq_length']}")

    trainer.train()

    # Save the adapter
    adapter_path = f"{CONFIG['output_dir']}/{CONFIG['new_model_name']}-adapter"
    trainer.model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)
    print(f"\n[OK] Adapter saved to {adapter_path}")

    # Save training loss history
    log_history = trainer.state.log_history
    with open(f"{CONFIG['output_dir']}/training_log.json", "w") as f:
        json.dump(log_history, f, indent=2)

    print("[OK] Fine-tuning complete!")


# ==========================================================================
# STEP 3: EVALUATION — Before/After Comparison
# ==========================================================================
def evaluate_model():
    """
    Run the same test examples through:
      1. Base Mistral-7B (no fine-tuning)
      2. MedClear (fine-tuned with QLoRA)
    Compute ROUGE, readability (Flesch-Kincaid), and SMOG scores.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    from rouge_score import rouge_scorer
    import textstat
    import nltk
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)

    print("\n" + "="*70)
    print("STEP 3: BEFORE/AFTER EVALUATION")
    print("="*70)

    # Load test examples
    with open(f"{CONFIG['output_dir']}/test_examples.json") as f:
        test_examples = json.load(f)

    # Use a subset for speed (adjust as needed)
    N_EVAL = min(20, len(test_examples))
    test_subset = test_examples[:N_EVAL]
    print(f"\nEvaluating on {N_EVAL} test examples...")

    # ── Load base model ───────────────────────────────────────────────────
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    print(f"\nLoading base model: {CONFIG['base_model']}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    tokenizer.pad_token = tokenizer.eos_token

    def generate_output(model, source_text, max_new_tokens=512):
        prompt = format_instruction(source_text)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
            )
        # Decode only the generated part
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return tokenizer.decode(generated, skip_special_tokens=True)

    # ── Generate BASE model outputs ───────────────────────────────────────
    print("\nGenerating base model outputs...")
    base_outputs = []
    for i, ex in enumerate(test_subset):
        out = generate_output(base_model, ex["source"])
        base_outputs.append(out)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{N_EVAL} done")

    # ── Load fine-tuned model ─────────────────────────────────────────────
    adapter_path = f"{CONFIG['output_dir']}/{CONFIG['new_model_name']}-adapter"
    print(f"\nLoading fine-tuned adapter from {adapter_path}...")
    ft_model = PeftModel.from_pretrained(base_model, adapter_path)
    ft_model.eval()

    # ── Generate FINE-TUNED model outputs ─────────────────────────────────
    print("Generating fine-tuned model outputs...")
    ft_outputs = []
    for i, ex in enumerate(test_subset):
        out = generate_output(ft_model, ex["source"])
        ft_outputs.append(out)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{N_EVAL} done")

    # ── Compute metrics ───────────────────────────────────────────────────
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    results = {"base": [], "finetuned": []}
    for i, ex in enumerate(test_subset):
        reference = ex["target"]

        for label, output in [("base", base_outputs[i]), ("finetuned", ft_outputs[i])]:
            rouge = scorer.score(reference, output)
            fk_grade = textstat.flesch_kincaid_grade(output) if len(output) > 20 else None
            smog = textstat.smog_index(output) if len(output) > 20 else None
            flesch_ease = textstat.flesch_reading_ease(output) if len(output) > 20 else None

            results[label].append({
                "rouge1_f": rouge["rouge1"].fmeasure,
                "rouge2_f": rouge["rouge2"].fmeasure,
                "rougeL_f": rouge["rougeL"].fmeasure,
                "flesch_kincaid_grade": fk_grade,
                "smog_index": smog,
                "flesch_reading_ease": flesch_ease,
                "word_count": len(output.split()),
            })

    # ── Aggregate and display ─────────────────────────────────────────────
    def avg(lst, key):
        vals = [d[key] for d in lst if d[key] is not None]
        return sum(vals) / len(vals) if vals else 0

    print("\n" + "="*70)
    print("              BEFORE vs AFTER COMPARISON")
    print("="*70)
    print(f"{'Metric':<30} {'Base Model':>15} {'MedClear (FT)':>15}")
    print("-"*60)

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

    # ── Show qualitative examples ─────────────────────────────────────────
    print("\n" + "="*70)
    print("QUALITATIVE EXAMPLES")
    print("="*70)

    for i in range(min(3, N_EVAL)):
        print(f"\n{'-'*70}")
        print(f"Example {i+1}")
        print(f"{'-'*70}")
        print(f"\nSOURCE (technical abstract):")
        print(textwrap.fill(test_subset[i]["source"][:400], 80))
        print(f"\nREFERENCE (Cochrane PLS):")
        print(textwrap.fill(test_subset[i]["target"][:400], 80))
        print(f"\nBASE MODEL output:")
        print(textwrap.fill(base_outputs[i][:400], 80))
        print(f"\nMEDCLEAR (fine-tuned) output:")
        print(textwrap.fill(ft_outputs[i][:400], 80))

    # Save full results
    full_results = {
        "metrics_summary": comparison,
        "examples": [
            {
                "source": test_subset[i]["source"],
                "reference": test_subset[i]["target"],
                "base_output": base_outputs[i],
                "finetuned_output": ft_outputs[i],
                "base_metrics": results["base"][i],
                "finetuned_metrics": results["finetuned"][i],
            }
            for i in range(N_EVAL)
        ],
    }
    with open(f"{CONFIG['output_dir']}/evaluation_results.json", "w") as f:
        json.dump(full_results, f, indent=2)

    print(f"\n[OK] Full results saved to {CONFIG['output_dir']}/evaluation_results.json")
    print("[OK] Evaluation complete!")
    return full_results


# ==========================================================================
# STEP 4: INTERACTIVE DEMO
# ==========================================================================
def run_demo():
    """Interactive demo for testing the fine-tuned model."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print("\n" + "="*70)
    print("MedClear — Interactive Demo")
    print("="*70)
    print("Enter a medical abstract to simplify. Type 'quit' to exit.\n")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    tokenizer.pad_token = tokenizer.eos_token

    adapter_path = f"{CONFIG['output_dir']}/{CONFIG['new_model_name']}-adapter"
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()

    # Provide some example inputs
    examples = [
        "A meta-analysis of 12 randomized controlled trials (n=3,847) demonstrated "
        "that prophylactic low-molecular-weight heparin significantly reduced the "
        "incidence of venous thromboembolism (RR 0.51, 95% CI 0.41-0.63, p<0.001) "
        "in post-surgical patients compared to unfractionated heparin, with no "
        "significant increase in major bleeding events (RR 1.02, 95% CI 0.73-1.43).",

        "Systematic review of 8 RCTs examining the efficacy of corticosteroid "
        "injections versus placebo for lateral epicondylitis found short-term pain "
        "reduction (SMD -1.44, 95% CI -2.07 to -0.81) at 4 weeks but no significant "
        "difference at 6 months (SMD -0.20, 95% CI -0.56 to 0.16), with a higher "
        "recurrence rate in the corticosteroid group (OR 2.87, 95% CI 1.51-5.45).",
    ]

    print("-- Pre-loaded examples available. Type 'ex1' or 'ex2' to try them. --\n")

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

        prompt = format_instruction(user_input)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        print("\nGenerating plain language version...")
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.1,
            )
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        result = tokenizer.decode(generated, skip_special_tokens=True)

        print(f"\nMedClear output:")
        print(textwrap.fill(result, 80))
        print()


# ==========================================================================
# MAIN
# ==========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MedClear: Medical Text Simplification")
    parser.add_argument(
        "--mode",
        choices=["prepare", "train", "evaluate", "demo", "all"],
        default="all",
        help="Which step(s) to run",
    )
    args = parser.parse_args()

    if args.mode in ("prepare", "all"):
        prepare_data()
    if args.mode in ("train", "all"):
        train_model()
    if args.mode in ("evaluate", "all"):
        evaluate_model()
    if args.mode == "demo":
        run_demo()

    if args.mode == "all":
        print("\n" + "="*70)
        print("FULL PIPELINE COMPLETE!")
        print("="*70)
        print(f"Results in: {CONFIG['output_dir']}/")
        print("Next steps:")
        print("  1. Run --mode demo for interactive testing")
        print("  2. Check evaluation_results.json for metrics")
        print("  3. Build your pitch slides!")
