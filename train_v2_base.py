"""
V2 T5-base: Full fine-tune with multi-granularity term-heavy data.
T5-base produces coherent output. V2 data teaches better vocabulary.
Best of both worlds.
"""
import json, os, torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer, DataCollatorForSeq2Seq,
)

MODEL = "google/flan-t5-base"
DATA_DIR = "./medclear_results/training_final_v2"
OUTPUT_DIR = "./medclear_results/v2-base"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load data
print("Loading V2 data...")
splits = {}
for split in ["train", "validation", "test"]:
    data = []
    with open(os.path.join(DATA_DIR, f"{split}.jsonl"), "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    splits[split] = Dataset.from_list(data)
dataset = DatasetDict(splits)
print(f"Train: {len(dataset['train'])}, Val: {len(dataset['validation'])}")

# Load model
print(f"Loading {MODEL}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL)

def preprocess(examples):
    inputs = tokenizer(examples["input_text"], max_length=512, truncation=True, padding=False)
    targets = tokenizer(examples["target"], max_length=256, truncation=True, padding=False)
    inputs["labels"] = targets["input_ids"]
    return inputs

print("Tokenizing...")
tokenized = dataset.map(preprocess, batched=True, remove_columns=["input_text", "target", "level"])
collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=3e-4,
    warmup_steps=200,
    lr_scheduler_type="cosine",
    logging_steps=100,
    save_steps=500,
    save_total_limit=2,
    eval_strategy="steps",
    eval_steps=500,
    predict_with_generate=True,
    generation_max_length=256,
    bf16=True,
    report_to="none",
    seed=42,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

trainer = Seq2SeqTrainer(
    model=model, args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    processing_class=tokenizer,
    data_collator=collator,
)

print(f"\nTraining T5-base V2...")
print(f"  Data: {len(dataset['train'])} examples (50% terms+phrases)")
print(f"  Batch: 16 effective")
print(f"  BF16: Yes")
trainer.train()

final = os.path.join(OUTPUT_DIR, "final")
trainer.save_model(final)
tokenizer.save_pretrained(final)

with open(os.path.join(OUTPUT_DIR, "training_log.json"), "w") as f:
    json.dump(trainer.state.log_history, f, indent=2)

print(f"\n[OK] Model saved to {final}")
print("[OK] V2 T5-base training complete!")
