# MedClear (PlainScript) - Project Plan

## Overview

**Goal**: Build a web application that takes a doctor's medical summary (discharge
summaries, post-op notes, visit summaries) and returns a patient-friendly plain
language version, powered by a fine-tuned LLM.

**Model**: FLAN-T5-base (250M params, encoder-decoder, runs on CPU)
**Datasets**: 5,616 training pairs from 4 sources (see below)
**Deployment target**: Hugging Face Spaces (free CPU tier, Gradio app)

---

## Phase 1: Environment & Data Setup [COMPLETE]

- [x] Verify hardware: RTX 4070 Ti Super (16GB VRAM), 64GB RAM
- [x] Install Python dependencies
- [x] Download FLAN-T5-base model (google/flan-t5-base, 250M params)
- [x] Download and combine 4 training datasets:

### Training Data Summary

| Dataset | Train | Val | Test | Type |
|---|---|---|---|---|
| GEM/cochrane-simplification | 3,568 | 411 | 480 | Cochrane abstracts -> PLS |
| PLABA (OSF) | 635 | 138 | 148 | Biomedical abstracts -> lay language |
| Med-EASi (HuggingFace) | 1,397 | 196 | 300 | Expert -> simple medical text |
| Synthetic (Claude-generated) | 16 | 2 | 2 | MTSamples clinical notes -> plain English |
| **Total** | **5,616** | **747** | **930** | |

### Additional datasets downloaded (for synthetic generation)
- MTSamples: 4,999 medical transcriptions (108 discharge summaries, 1,103 surgery)
- Asclepius Synthetic Clinical Notes: 158,114 synthetic discharge summaries

### Token length stats (T5 tokenizer, train split)
- Source: mean 398 tokens, max 1,191, 1,781 exceed 512
- Target: mean 231 tokens, max 1,000, 374 exceed 512

---

## Phase 2: Fine-Tuning FLAN-T5-base [NEXT STEP - awaiting go-ahead]

### Configuration (in `medclear_finetune.py`)
| Parameter | Value |
|---|---|
| Model | google/flan-t5-base (250M params) |
| Architecture | Encoder-decoder (Seq2Seq) |
| Epochs | 5 |
| Batch size (effective) | 16 (8 x 2 gradient accum) |
| Learning rate | 3e-4 (cosine schedule) |
| Max source tokens | 512 |
| Max target tokens | 512 |
| Estimated time | ~30-45 min on RTX 4070 Ti |

### Run command
```bash
python medclear_finetune.py --mode train
```

---

## Phase 2b: Synthetic Data Expansion (Parallel Track)

### Pipeline (`generate_synthetic.py`)
1. Read clinical notes from MTSamples (5K) and Asclepius (158K)
2. Generate plain-language versions using Claude (in this Claude Code session
   or via Anthropic API)
3. Quality filter: discard pairs where FK grade > 12
4. Save as JSONL -> auto-included in training via `medclear_finetune.py`

### Current status
- 20 synthetic pairs generated (avg FK grade: source 10.5 -> target 8.9)
- Pipeline ready for scaling (can generate hundreds more)

### To scale up
```bash
# Via API (requires ANTHROPIC_API_KEY):
python generate_synthetic.py --source mtsamples --count 500

# Or generate in Claude Code session (free, no API key needed)
# Then convert to training format:
python generate_synthetic.py --convert
```

---

## Phase 3: Evaluation

### Automatic metrics
- **ROUGE-1/2/L F1**: Lexical overlap with gold-standard references
- **Flesch-Kincaid Grade Level**: Target 6-8th grade
- **SMOG Index**: Health literacy standard
- **Flesch Reading Ease**: Higher = easier

### Evaluation tracks
1. **Academic track**: Test on Cochrane/PLABA/Med-EASi test sets
2. **Clinical track**: Test on synthetic discharge summary pairs
3. **Qualitative**: Side-by-side base vs fine-tuned examples

### Run command
```bash
python medclear_finetune.py --mode evaluate
```

---

## Phase 4: Web Application & Deployment

### Architecture
```
User (patient/provider) --> [Gradio Web UI] --> [FLAN-T5-base fine-tuned] --> Plain Language Output
```

### Hugging Face Spaces Deployment
1. Push fine-tuned model to HF Hub (~1GB)
2. Create Gradio app with:
   - Text input for medical summary / discharge note
   - "Simplify" button
   - Plain-language output
   - Readability scores (FK grade, SMOG)
3. Deploy on free CPU tier (no GPU needed!)
4. Optional: ONNX quantization for faster CPU inference (~250MB)

---

## File Structure

```
PlainScript/
  medclear_finetune.py            # Main training/eval pipeline (FLAN-T5)
  generate_synthetic.py           # Synthetic data generation with Claude
  append_synthetic.py             # Helper to append synthetic pairs
  cochrane_plainlang_finetune.py  # Original Mistral pipeline (deprecated)
  setup.sh                        # Environment setup script
  PITCH_PLAN.md                   # Pitch presentation plan
  PROJECT_PLAN.md                 # This file
  medclear_results/
    raw_data/                     # Cochrane JSON files
    combined_data/                # Combined training dataset (all sources)
    test_examples.json            # Test examples for evaluation
    datasets/
      plaba/                      # PLABA CSV files
      medeasi/                    # Med-EASi Arrow datasets
      mtsamples/                  # MTSamples transcriptions
      asclepius/                  # Asclepius synthetic clinical notes
    synthetic_data/
      claude_generated_pairs.jsonl  # Claude-generated training pairs
    medclear-t5-base-final/       # [after training] Fine-tuned model
    training_log.json             # [after training] Loss history
    evaluation_results.json       # [after evaluation] Metrics
```

---

## Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Factual hallucination | "Assistant" framing; human review required |
| T5-base capacity limits | Can upgrade to FLAN-T5-large (770M) if needed |
| Truncation (1,781 examples > 512 tokens) | Most clinical value in first 512 tokens |
| Synthetic data quality | Quality filtering by readability scores |
| Training data bias (English, Western) | Acknowledge in disclaimers |
