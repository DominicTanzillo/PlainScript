# MedClear: Teaching an LLM to Speak Patient, Not Physician

> *Your doctor hands you this:*
>
> **"Pt: 72y/o M. CC: SOB, DOE, R/O Acute MI. PMHx: HTN, DMII, CAD, HFpEF. Presented to ED via EMS with progressive SOB and 3-pillow orthopnea x24h. Noncompliant with PO meds (ASA, Lisinopril) d/t financial constraints. Tachycardic HR 115, hypotensive BP 90/50. CXR: pulmonary edema. ECG: sinus tach with PVCs, no STEMI. Labs: Cr 2.1 from 0.9 baseline, K+ 5.5, BNP 2000. Pre-renal AKI. Troponin mildly elevated, likely demand ischemia."**
>
> *You nod politely. You understand exactly zero of that. You just want to know if Grandpa is going to be okay.*
>
> **MedClear translates:**
>
> "Your 72-year-old grandfather came to the emergency room by ambulance because he was having serious trouble breathing. He has several ongoing conditions -- high blood pressure, type 2 diabetes, coronary artery disease, and heart failure. He had stopped taking his blood pressure and aspirin medications because he couldn't afford them. When he arrived, his heart was beating too fast (115, normal is 60-100) and his blood pressure was dangerously low (90/50). A chest X-ray showed fluid in his lungs from his heart not pumping well enough. Blood tests showed his kidneys were struggling (likely from low blood flow), his potassium was high, and a heart strain marker (BNP) was very elevated at 2000. The good news: he is NOT having a heart attack (no STEMI on ECG)."
>
> *Now you know what's happening. Now you can ask the right questions. Now you can help.*

---

## What Is This?

MedClear is an AI tool that translates doctor-speak into human-speak. Paste in a discharge summary, post-op note, or visit summary, and MedClear will:

1. **Extract key facts** from the clinical note
2. **Look up definitions** on [MedlinePlus](https://medlineplus.gov) (NIH's plain-language health resource)
3. **Generate a patient-friendly version** of the text
4. **Hyperlink every medical term** so patients can click to learn more

Think of it as Google Translate, but instead of English to Spanish, it's *Physician to Patient*.

---

## The Problem (It's Bigger Than You Think)

- **~36% of US adults** have limited health literacy
- The average discharge summary reads at a **college level** (Flesch-Kincaid grade 15+)
- Patient materials should be at **6th-8th grade level**
- That 72-year-old man? He couldn't afford his meds. He *definitely* can't afford to misunderstand his diagnosis.

Every patient deserves to understand what happened to them. Every parent deserves to understand their kid's surgery. Every person deserves to know what "noncompliance with PO meds due to financial constraints" actually means -- *he couldn't pay for his pills.*

---

## How It Works

```
Doctor's Note (the alphabet soup)
        |
        v
[Term Extraction]  ───>  [MedlinePlus API]  ───>  Definitions + Links
   920+ terms              (NIH/NLM)                SOB = shortness of breath
        |                                           (not what you're thinking)
        v
[FLAN-T5-large + LoRA]  ───>  Plain Language Translation
   783M params                 Trained on 23K examples
   0.6% fine-tuned             50% term/phrase vocabulary
        |
        v
    React Web App
    Clickable terms | Hover tooltips | MedlinePlus glossary
```

### The Three-Layer Approach

**Layer 1 -- Vocabulary (the foundation):** 920+ medical terms mapped to plain English definitions. The model learns that `"cholecystectomy"` = `"surgery to remove the gallbladder"` before it tries to simplify an entire surgical report.

**Layer 2 -- RAG with MedlinePlus:** Every medical term is looked up on [MedlinePlus](https://medlineplus.gov) (NIH's authoritative patient health resource). Definitions are injected as context at inference time, grounding the output in verified medical information.

**Layer 3 -- Generation:** FLAN-T5-large fine-tuned with LoRA adapters generates the simplified text, having learned from 23,157 training examples spanning terms, phrases, sentences, and full clinical documents.

---

## Quick Start

```bash
cd PlainScript

# Start the API server (loads model + MedlinePlus RAG)
python api_server.py

# In another terminal, start the React frontend
cd frontend && npm start

# Open http://localhost:3000
```

Or run the pipeline directly:
```bash
# Train the model (V2 LoRA BF16)
python train_v2_lora_bf16.py

# Run the RAG pipeline on a clinical note
python rag_pipeline.py --demo
```

---

## Data Augmentation Strategy

The key insight: **teach the vocabulary first, composition second.**

Rather than training on full paragraph pairs (where the model must simultaneously learn jargon AND generate coherent output), we built a multi-granularity dataset weighted toward short, precise translations:

### Training Data (23,157 examples)

| Level | Examples | % | What It Teaches |
|-------|----------|---|-----------------|
| **Terms** | 4,989 | 21.5% | `"DVT"` -> `"a blood clot in a deep vein, usually in the leg"` |
| **Phrases** | 6,660 | 28.8% | `"afebrile, tolerating PO"` -> `"no fever, eating and drinking normally"` |
| Sentences | 8,000 | 34.5% | Sentence-level simplification from aligned academic pairs |
| Flashcards | 2,689 | 11.6% | Medical knowledge Q&A |
| Paragraphs | 574 | 2.5% | Full clinical note simplification |
| RAG-augmented | 245 | 1.1% | Using MedlinePlus context |

### Data Sources

| Source | Raw Pairs | How We Used It |
|--------|-----------|---------------|
| Cochrane (GEM) | 3,568 | Paragraph pairs + 20,678 sentence chunks + 467 extracted terms |
| PLABA (OSF) | 635 | Paragraph pairs + 8,028 sentence chunks + 141 terms |
| Med-EASi (HuggingFace) | 1,397 | Already sentence-level, used directly |
| Synthetic Clinical (Claude) | 626 | 155 specialties, discharge/surgery/ER notes |
| Medical Flashcards | 33,955 | 3,000 selected "What is X?" definitions |
| Term Dictionary | 920 | Hand-written + agent-generated + extracted |
| MedlinePlus RAG | 270 | Context-injected training pairs |

### Why Multi-Granularity Works

Previous approaches trained on full paragraphs. The model had to learn vocabulary, abbreviations, AND coherent generation simultaneously. Results: hallucination, Cochrane-style "We found..." leaking into clinical output.

V2 dedicates 50.3% of training to terms + phrases. The model learns the **mapping** first (`"EBL minimal"` = `"very little blood loss"`), then learns to compose these mappings into sentences and paragraphs.

---

## Model Architecture

| Component | Choice | Why |
|-----------|--------|-----|
| Base model | FLAN-T5-large (783M) | Encoder-decoder for translation, instruction-tuned, fits on CPU |
| Fine-tuning | LoRA (rank 16) | 0.6% params trained, preserves pre-trained knowledge, stable |
| Precision | BF16 | 20x faster than FP32, native RTX 40-series support |
| Retrieval | MedlinePlus API | Free, authoritative, plain-language, hyperlinked |
| Frontend | React + Flask | Clickable term annotations, hover tooltips |

### Why Not Just Use GPT-4 / Claude?

1. **Cost**: API calls for every patient document at scale = expensive
2. **Privacy**: Clinical notes contain PHI; can't send to external APIs without HIPAA compliance
3. **Latency**: Local model = instant; API = network round-trip
4. **Deployment**: Runs on HuggingFace Spaces free tier (CPU, ~3.5GB RAM)
5. **Reproducibility**: Open weights, open training data, open methodology

---

## Results (T5-base, Academic Evaluation)

| Metric | Raw FLAN-T5 | MedClear (best) |
|--------|-------------|-----------------|
| ROUGE-1 F1 | 0.13 | **0.36** |
| ROUGE-2 F1 | 0.05 | **0.13** |
| ROUGE-L F1 | 0.10 | **0.22** |
| Flesch-Kincaid Grade | 15.7 | **13.9** |
| Flesch Reading Ease | 9.9 | **34.4** |

T5-large V2 LoRA results pending (training in progress).

---

## Training Iterations & What We Learned

| # | Model | Data | Result | Lesson |
|---|-------|------|--------|--------|
| 1 | T5-base full FT | Academic only | ROUGE 0.36, Cochrane style leaks | Data distribution > model size |
| 2 | T5-base full FT | + 541 synthetic | Slight improvement | Small domain data helps |
| 3 | T5-base full FT | + CoT format | Learned structure, still hallucinated | Small models lack capacity for reasoning |
| 4 | T5-large full FT | Same data | Gibberish output | Full FT of large models is unstable |
| 5 | T5-large LoRA FP32 | Paragraph-heavy | 19 hours, poor output | FP32 too slow, data too paragraph-heavy |
| **6** | **T5-large LoRA BF16** | **23K V2 (50% terms)** | **Training...** | **Vocabulary first, 20x faster** |

---

## Ethics & Honest Limitations

**This tool is an assistant, not an authority.**

- The model can hallucinate medical facts. Every output should be verified.
- ROUGE measures word overlap, not patient comprehension.
- Training data reflects English-speaking medical professionals' assumptions about "plain language."
- **This is not a replacement for talking to your doctor.** It's a starting point for understanding.
- All medical terms link to MedlinePlus (NIH) for authoritative verification.

---

## The Pitch

*"When that 72-year-old man's family gets handed a discharge summary full of acronyms they've never seen, they shouldn't need a medical degree to understand it. MedClear shows that with 23,000 training examples, a medical vocabulary of 920 terms, and a connection to the National Library of Medicine, we can meaningfully shift clinical communication from physician to patient -- one simplified paragraph at a time."*

---

## Built With

FLAN-T5 | LoRA/PEFT | MedlinePlus (NIH) | PyTorch | Transformers | React | Flask

See [METHODS.md](METHODS.md) for full technical documentation.

**Duke University Hackathon 2026**