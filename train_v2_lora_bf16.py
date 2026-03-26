"""
MedClear V2: Fast LoRA Fine-Tuning with BF16
=============================================
- FLAN-T5-large (783M params)
- LoRA adapters (only 0.6% params trained)
- BF16 precision (3-5x faster than FP32)
- Multi-granularity training data (terms → phrases → sentences → paragraphs)
- Gradient checkpointing for VRAM efficiency

Expected training time: ~2-3 hours on RTX 4070 Ti Super
"""

import json
import os
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

# Config
MODEL_NAME = "google/flan-t5-large"
DATA_DIR = "./medclear_results/training_final_v2"
OUTPUT_DIR = "./medclear_results/v2-lora-bf16"
MAX_SOURCE_LEN = 512
MAX_TARGET_LEN = 256  # Shorter targets for term/phrase level
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
print("Loading V2 training data...")
splits = {}
for split in ["train", "validation", "test"]:
    path = os.path.join(DATA_DIR, f"{split}.jsonl")
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    splits[split] = Dataset.from_list(data)

dataset = DatasetDict(splits)
print(f"Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}")

# Load model
print(f"\nLoading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,  # BF16 for speed
)

# Enable gradient checkpointing
model.gradient_checkpointing_enable()
model.config.use_cache = False  # Required for gradient checkpointing

# Apply LoRA
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q", "v"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Tokenize
def preprocess(examples):
    inputs = tokenizer(
        examples["input_text"],
        max_length=MAX_SOURCE_LEN,
        truncation=True,
        padding=False,
    )
    targets = tokenizer(
        examples["target"],
        max_length=MAX_TARGET_LEN,
        truncation=True,
        padding=False,
    )
    inputs["labels"] = targets["input_ids"]
    return inputs

print("Tokenizing...")
tokenized = dataset.map(
    preprocess, batched=True,
    remove_columns=["input_text", "target", "level"],
)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, model=model, padding=True,
)

# Training args — optimized for speed
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,   # Larger batch with BF16
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,    # Effective batch = 16
    learning_rate=3e-4,
    weight_decay=0.01,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    logging_steps=50,
    save_steps=200,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=200,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LEN,
    bf16=True,                        # BF16 for speed!
    report_to="none",
    seed=SEED,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    dataloader_num_workers=0,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
)

print(f"\nStarting V2 LoRA BF16 training...")
print(f"  Model: {MODEL_NAME} (LoRA)")
print(f"  Data: {len(dataset['train'])} train examples")
print(f"  Batch: {8 * 2} effective")
print(f"  BF16: Yes")
print(f"  Gradient checkpointing: Yes")

trainer.train()

# Save
final_dir = os.path.join(OUTPUT_DIR, "lora-adapter")
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)

# Merge for deployment
print("\nMerging LoRA weights...")
merged = model.merge_and_unload()
merged_dir = os.path.join(OUTPUT_DIR, "merged-model")
merged.save_pretrained(merged_dir)
tokenizer.save_pretrained(merged_dir)

with open(os.path.join(OUTPUT_DIR, "training_log.json"), "w") as f:
    json.dump(trainer.state.log_history, f, indent=2)

print(f"\n[OK] LoRA adapter: {final_dir}")
print(f"[OK] Merged model: {merged_dir}")
print("[OK] V2 training complete!")
