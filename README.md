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

## Try It

**Live Demo (Gradio):** [huggingface.co/spaces/DTanzillo/medclear](https://huggingface.co/spaces/DTanzillo/medclear)

**Web App (GitHub Pages):** [dominictanzillo.github.io/PlainScript](https://dominictanzillo.github.io/PlainScript)

**Model on HuggingFace:** [huggingface.co/DTanzillo/medclear-v2-base](https://huggingface.co/DTanzillo/medclear-v2-base)

---

## What Is This?

MedClear translates doctor-speak into human-speak. Paste in a discharge summary, post-op note, or visit summary, and MedClear will:

1. **Generate a patient-friendly version** using a fine-tuned FLAN-T5 model
2. **Identify every medical term** from a dictionary of 920+ terms
3. **Link each term to MedlinePlus** (NIH) for authoritative definitions
4. **Build an interactive glossary** with hover tooltips and click-through links

Think of it as Google Translate, but instead of English to Spanish, it's *Physician to Patient*.

---

## Architecture

```
                           +-------------------+
                           |   React Frontend  |
                           |  (GitHub Pages)   |
                           +--------+----------+
                                    |
                              POST /api/simplify
                                    |
                           +--------v----------+
                           |   Flask API Server |
                           |   (api_server.py)  |
                           +--------+----------+
                                    |
                     +--------------+--------------+
                     |              |              |
              +------v------+ +----v-----+ +------v-------+
              |  FLAN-T5    | |  Term    | |  MedlinePlus |
              |  base       | |  Dict    | |  API (NIH)   |
              |  (248M)     | |  920+    | |              |
              +------+------+ +----+-----+ +------+-------+
                     |              |              |
                     v              v              v
              Plain language   Term matches   Definitions
              simplification   with positions  + URLs
                     |              |              |
                     +--------------+--------------+
                                    |
                           +--------v----------+
                           |   JSON Response   |
                           | plain_language     |
                           | source_annotations |
                           | output_annotations |
                           +-------------------+
```

### The Three-Layer Approach

**Layer 1 -- Vocabulary:** 920+ medical terms mapped to plain English with curated MedlinePlus URLs. Abbreviations like PO, PRN, DVT, NSTEMI all link to the correct NIH page.

**Layer 2 -- RAG with MedlinePlus:** The [MedlinePlus API](https://medlineplus.gov) (NIH/NLM) provides authoritative definitions at inference time. Every term the model encounters gets a verified definition and link.

**Layer 3 -- Generation:** FLAN-T5-base fine-tuned on 23,157 training examples generates the simplified text. 50% of training is term/phrase level -- the model learns vocabulary first, then composition.

---

## Running Locally

```bash
git clone https://github.com/DominicTanzillo/PlainScript.git
cd PlainScript

# Install dependencies
pip install flask flask-cors torch transformers

# Start the server (loads model from HuggingFace, serves React app)
python api_server.py

# Open http://localhost:5000
```

The server automatically downloads the model from HuggingFace on first run (~950MB).

### Using the Model Directly

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("DTanzillo/medclear-v2-base")
model = AutoModelForSeq2SeqLM.from_pretrained("DTanzillo/medclear-v2-base")

text = "simplify: Patient underwent laparoscopic cholecystectomy. EBL minimal. Afebrile, tolerating PO diet."
inputs = tokenizer(text, return_tensors="pt", max_length=512, truncation=True)
outputs = model.generate(**inputs, max_new_tokens=256, num_beams=4)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

---

## Training Data (23,157 examples)

The key insight: **teach the vocabulary first, composition second.**

| Level | Examples | % | What It Teaches |
|-------|----------|---|-----------------|
| **Terms** | 4,989 | 21.5% | `"DVT"` -> `"a blood clot in a deep vein"` |
| **Phrases** | 6,660 | 28.8% | `"afebrile, tolerating PO"` -> `"no fever, eating normally"` |
| Sentences | 8,000 | 34.5% | Sentence-level simplification |
| Flashcards | 2,689 | 11.6% | Medical knowledge Q&A |
| Paragraphs | 574 | 2.5% | Full clinical note simplification |
| RAG-augmented | 245 | 1.1% | MedlinePlus context-injected |

**Sources:** Cochrane (GEM), PLABA, Med-EASi, 626 synthetic clinical pairs (155 specialties), 920-term medical dictionary, MedlinePlus RAG pairs.

---

## Results

| Metric | Raw FLAN-T5 | MedClear V2 |
|--------|-------------|-------------|
| ROUGE-1 F1 | 0.13 | **0.36** |
| ROUGE-2 F1 | 0.05 | **0.13** |
| ROUGE-L F1 | 0.10 | **0.22** |
| Flesch-Kincaid Grade | 15.7 | **13.9** |
| Flesch Reading Ease | 9.9 | **34.4** |
| Eval Loss | -- | **1.712** |

Trained in **18 minutes** on RTX 4070 Ti Super (3 epochs, BF16).

---

## HuggingFace Deployment

The model and demo are hosted on HuggingFace:

- **Model:** [`DTanzillo/medclear-v2-base`](https://huggingface.co/DTanzillo/medclear-v2-base) -- FLAN-T5-base (248M params), ~950MB
- **Space:** [`DTanzillo/medclear`](https://huggingface.co/spaces/DTanzillo/medclear) -- Gradio demo with MedlinePlus RAG

The Space runs on HuggingFace's free CPU tier (~16GB RAM). The Gradio app (`hf_space/app.py`) loads the model, runs simplification, and builds a MedlinePlus glossary for every input.

To update the Space:
```bash
# Files are in hf_space/
# Push via HuggingFace Hub or git
cd hf_space
# Edit app.py or requirements.txt
# Upload with: huggingface-cli upload DTanzillo/medclear . --repo-type space
```

---

## Project Structure

```
PlainScript/
  api_server.py              # Flask API + React static server
  rag_pipeline.py            # MedlinePlus RAG pipeline
  train_v2_base.py           # V2 training script (the winning config)
  build_training_data_v2.py  # Multi-granularity data builder
  build_final_training_data.py  # Final dataset assembly
  frontend/
    src/App.js               # React app (term highlighting, tooltips)
    src/App.css              # Styles
    public/index.html        # HTML shell
  hf_space/
    app.py                   # Gradio app for HuggingFace Spaces
    requirements.txt         # Space dependencies
```

---

## Why Not Just Use GPT-4 / Claude?

1. **Cost**: API calls for every patient document at scale = expensive
2. **Privacy**: Clinical notes contain PHI; can't send to external APIs
3. **Latency**: Local model = instant; API = network round-trip
4. **Deployment**: Runs on HuggingFace free tier (CPU, ~3.5GB RAM)
5. **Reproducibility**: Open weights, open data, open methodology

---

## Ethics & Limitations

**This tool is an assistant, not an authority.**

- The model can hallucinate medical facts. Every output should be verified.
- Excels at surgical/procedural notes; struggles with complex multi-system cases.
- Training data reflects English-speaking assumptions about "plain language."
- **Not a replacement for talking to your doctor.** A starting point for understanding.
- All terms link to MedlinePlus (NIH) for authoritative verification.

---

## The Pitch

*"When that 72-year-old man's family gets handed a discharge summary full of acronyms they've never seen, they shouldn't need a medical degree to understand it. MedClear shows that with 23,000 training examples, a medical vocabulary of 920 terms, and a connection to the National Library of Medicine, we can meaningfully shift clinical communication from physician to patient -- one simplified paragraph at a time."*

---

## Built With

FLAN-T5 | MedlinePlus (NIH) | PyTorch | Transformers | React | Flask

**Duke University Hackathon 2026**
