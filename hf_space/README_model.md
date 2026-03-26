---
license: apache-2.0
language:
  - en
pipeline_tag: summarization
tags:
  - medical
  - simplification
  - health-literacy
  - flan-t5
  - plain-language
datasets:
  - GEM/cochrane-simplification
  - tttamayo/Med-EASi
base_model: google/flan-t5-base
---

# MedClear V2: Medical Text Simplification

**MedClear** translates doctor-speak into human-speak. Fine-tuned FLAN-T5-base (248M params) that simplifies clinical notes, medical terms, and discharge summaries into plain language patients can understand.

## Usage

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("DTanzillo/medclear-v2-base")
model = AutoModelForSeq2SeqLM.from_pretrained("DTanzillo/medclear-v2-base")

text = "simplify: Patient underwent laparoscopic cholecystectomy for acute cholecystitis. EBL minimal. POD1: afebrile, tolerating PO diet."
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(**inputs, max_new_tokens=256, num_beams=4)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Training Data

23,157 examples across multiple granularity levels:

| Level | Examples | % |
|-------|----------|---|
| Terms | 4,989 | 21.5% |
| Phrases | 6,660 | 28.8% |
| Sentences | 8,000 | 34.5% |
| Flashcards | 2,689 | 11.6% |
| Paragraphs | 574 | 2.5% |
| RAG-augmented | 245 | 1.1% |

Key insight: 50% of training is term/phrase level. The model learns vocabulary mappings first, then composes them into simplified text.

## Results

| Metric | Raw FLAN-T5 | MedClear |
|--------|-------------|----------|
| ROUGE-1 F1 | 0.13 | **0.36** |
| ROUGE-2 F1 | 0.05 | **0.13** |
| ROUGE-L F1 | 0.10 | **0.22** |
| Eval Loss | -- | **1.712** |

## Limitations

- Can hallucinate on complex multi-fact clinical notes
- Best used with RAG pipeline (MedlinePlus) for verification
- Not a substitute for professional medical advice

## Demo

Try the live demo: [MedClear on HuggingFace Spaces](https://huggingface.co/spaces/DTanzillo/medclear)

**Duke University Hackathon 2026**
