"""
Train FLAN-T5-large with LoRA (Low-Rank Adaptation).
Freezes most weights, trains only small adapters (~1-2% of params).
Much more stable and memory-efficient than full fine-tuning.
"""

import json
import os
import torch
from datasets import load_from_disk
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
DATA_DIR = "./medclear_results/large/combined_data"  # Already prepared
OUTPUT_DIR = "./medclear_results/large-lora"
MAX_SOURCE_LEN = 512
MAX_TARGET_LEN = 512
SEED = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
print("Loading data...")
dataset = load_from_disk(DATA_DIR)
print(f"Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}")

# Load model and tokenizer
print(f"\nLoading {MODEL_NAME}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)

# Apply LoRA
lora_config = LoraConfig(
    r=16,                           # Rank of adaptation matrices
    lora_alpha=32,                  # Scaling factor
    target_modules=["q", "v"],      # Apply to attention Q and V projections
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
tokenized = dataset.map(preprocess, batched=True,
                         remove_columns=dataset["train"].column_names)

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, model=model, padding=True)

# Training args
training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,  # effective batch = 16
    learning_rate=3e-4,             # Higher LR is fine for LoRA
    weight_decay=0.01,
    warmup_steps=100,
    lr_scheduler_type="cosine",
    logging_steps=50,
    save_steps=500,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=500,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LEN,
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    report_to="none",
    seed=SEED,
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

print(f"\nStarting LoRA fine-tuning...")
print(f"  Base model: {MODEL_NAME}")
print(f"  LoRA rank: {lora_config.r}")
print(f"  Effective batch: {4 * 4}")
print(f"  LR: 3e-4")
print(f"  Epochs: 5")

trainer.train()

# Save LoRA adapter + merge for inference
final_dir = os.path.join(OUTPUT_DIR, "medclear-t5-large-lora-final")

# Save adapter
model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)

# Also merge and save full model for easier deployment
print("\nMerging LoRA weights into base model...")
merged_model = model.merge_and_unload()
merged_dir = os.path.join(OUTPUT_DIR, "medclear-t5-large-merged")
merged_model.save_pretrained(merged_dir)
tokenizer.save_pretrained(merged_dir)

# Save training log
with open(os.path.join(OUTPUT_DIR, "training_log.json"), "w") as f:
    json.dump(trainer.state.log_history, f, indent=2)

print(f"\n[OK] LoRA adapter saved to {final_dir}")
print(f"[OK] Merged model saved to {merged_dir}")
print("[OK] Training complete!")
