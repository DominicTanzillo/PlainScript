# MedClear: Fine-Tuning Mistral-7B for Medical Text Simplification
## 3-Minute Hackathon Pitch — Planning Document

---

## 🎯 PITCH STRUCTURE (3 minutes)

### Opening Hook (20 seconds)
"When a patient reads their Cochrane review summary and sees 'heterogeneity was 
assessed using the I² statistic' — they've already lost the message. MedClear 
fixes that by teaching an LLM to speak patient, not physician."

### What We Built (40 seconds)
- **Model**: Mistral-7B-Instruct-v0.3, fine-tuned with QLoRA (4-bit quantized LoRA)
- **Dataset**: ~4,500 paired examples from the Cochrane Database of Systematic 
  Reviews — each pair is a technical abstract matched with its professionally 
  written Plain Language Summary (Devaraj et al., NAACL 2021)
- **Method**: Parameter-efficient fine-tuning — only ~1-2% of weights trained 
  via low-rank adapters, making this feasible on a single consumer GPU
- **Framing**: This is a text simplification/adaptation task, not summarization — 
  the model learns to preserve clinical meaning while transforming the register

### Before/After Demo (60 seconds)
Show 2 side-by-side examples:
1. **Technical input** → **Base Mistral output** vs **MedClear output**
2. Point out: tone shift, jargon removal, sentence structure, reading level

**Key metrics to highlight:**
- ROUGE scores (alignment with Cochrane gold-standard PLS)
- Flesch-Kincaid Grade Level (target: 6-8th grade for patient materials)
- SMOG Index (Simple Measure of Gobbledygook — standard for health literacy)

### Why This Matters (30 seconds)
- ~36% of US adults have limited health literacy (NCES/IOM)
- Cochrane PLSs themselves often exceed recommended readability (SMOG ~12 vs 
  target of ~6-8) — even the "simple" versions are too hard
- Real application: clinical decision support alerts (like Duke's Scout system) 
  could include auto-generated patient-facing explanations
- Every discharge summary, radiology report, and clinical trial result could 
  have a patient-readable companion

### Risks, Ethics & Evaluation Challenges (40 seconds)
1. **Factual fidelity**: Simplification ≠ changing the conclusion. A model that 
   says "the treatment works great!" when the evidence is uncertain is dangerous. 
   We evaluate this qualitatively and flag it as the critical unsolved challenge.
2. **Hallucination**: LLMs may introduce medical claims not present in the source. 
   For clinical deployment, a factuality verification layer is essential.
3. **Cultural/linguistic bias**: Cochrane PLSs are written by English-speaking 
   medical professionals — "plain language" as defined by this dataset reflects 
   their assumptions about what patients understand.
4. **Evaluation gap**: ROUGE measures lexical overlap, not whether a patient 
   actually understood the text. Human evaluation and patient comprehension 
   testing are the gold standard, but weren't feasible in hackathon scope.
5. **Not a replacement**: This assists writers — it shouldn't auto-publish to 
   patients without human review.

### Close (10 seconds)
"MedClear shows that with 4,500 expert-curated examples and a few hours of 
LoRA training, we can meaningfully shift an LLM's communication register 
from physician to patient — a capability that could improve health equity 
one simplified paragraph at a time."

---

## 📊 SLIDE STRUCTURE (suggested 4-5 slides)

### Slide 1: Title + Hook
- "MedClear: Teaching an LLM to Speak Patient"
- One-line problem statement
- Team name

### Slide 2: Method
- Diagram: Cochrane Abstract → [Mistral-7B + QLoRA] → Plain Language
- Key numbers: 4,500 pairs, 16M trainable params out of 7B, ~1.5 hrs training
- LoRA visualization (frozen weights + small adapters)

### Slide 3: Before/After
- Side-by-side comparison table
- Metrics bar chart (ROUGE, readability)
- Highlight the reading level drop

### Slide 4: Risks & Ethics
- Bullet the top 3 risks with one-line mitigations
- "What ROUGE doesn't capture" callout

### Slide 5: Impact & Next Steps
- Scout notifications hook
- Health literacy statistics
- "Every clinical document could have a patient companion"

---

## 🔧 TECHNICAL QUICK REFERENCE

### Requirements
```
pip install torch transformers datasets peft bitsandbytes accelerate trl
pip install rouge-score nltk textstat sentencepiece protobuf
```

### Run Commands
```bash
# Full pipeline
python cochrane_plainlang_finetune.py --mode all

# Or step by step
python cochrane_plainlang_finetune.py --mode prepare
python cochrane_plainlang_finetune.py --mode train
python cochrane_plainlang_finetune.py --mode evaluate
python cochrane_plainlang_finetune.py --mode demo
```

### Hardware
- GPU: RTX 4070 Ti Super (16GB VRAM)
- RAM: 64GB system
- Disk: ~20GB for model + checkpoints
- Time: ~1-2 hours for training

### Key Citations
- Devaraj et al. (2021). "Paragraph-level Simplification of Medical Texts." NAACL.
- Bakker & Kamps (2024). "Cochrane-auto: An Aligned Dataset." TSAR Workshop.
- Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." ICLR.
- HuggingFace dataset: GEM/cochrane-simplification (~4,500 pairs)

---

## ⚠️ TROUBLESHOOTING

**OOM (Out of Memory)**:
- Reduce `per_device_train_batch_size` to 1
- Reduce `max_seq_length` to 768 or 512
- Increase `gradient_accumulation_steps` to compensate

**Slow training**:
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check GPU utilization: `nvidia-smi`

**Poor outputs**:
- Try more epochs (4-5)
- Increase `lora_r` to 32 (more capacity, more VRAM)
- Check that the data formatting matches Mistral's instruct template exactly

**Model won't stop generating**:
- The pad_token = eos_token setup can cause this
- Use `tokenizer.add_special_tokens({"pad_token": "[PAD]"})` and resize embeddings
- Or add `eos_token_id=tokenizer.eos_token_id` to generate() call explicitly
