"""
MedClear RAG Pipeline
=====================
Retrieval-Augmented Generation for medical text simplification.

Pipeline:
1. Extract medical terms from clinical note (NLP-based)
2. Retrieve plain-language content from MedlinePlus (NIH)
3. Generate grounded simplification using retrieved context + model

MedlinePlus is a free, authoritative source from the NIH/National Library
of Medicine, providing patient-facing health information.

Usage:
    python rag_pipeline.py --text "Patient underwent laparoscopic cholecystectomy..."
    python rag_pipeline.py --demo
"""

import argparse
import json
import os
import re
import textwrap
import time
import urllib.request
import xml.etree.ElementTree as ET
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# -- Configuration ----------------------------------------------------------
MODEL_DIR = "./medclear_results/cot/medclear-t5-cot-final"
MEDLINEPLUS_API = "https://wsearch.nlm.nih.gov/ws/query"
TASK_PREFIX = "extract facts and simplify clinical note: "
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================================================
# STEP 1: Medical Term Extraction
# ==========================================================================

# Common medical terms to search for in MedlinePlus
MEDICAL_TERM_PATTERNS = [
    # Procedures
    r'\b(cholecystectomy|appendectomy|hysterectomy|mastectomy|'
    r'colectomy|nephrectomy|thyroidectomy|prostatectomy|'
    r'arthroplasty|arthroscopy|laminectomy|discectomy|'
    r'craniotomy|thoracotomy|laparotomy|cesarean section|'
    r'CABG|coronary artery bypass|PCI|stent placement|'
    r'endoscopy|colonoscopy|bronchoscopy|catheterization|'
    r'dialysis|transplant|amputation|biopsy|tracheostomy|'
    r'pacemaker|defibrillator|ablation|fasciotomy)\b',

    # Conditions / diagnoses
    r'\b(cholecystitis|appendicitis|pneumonia|sepsis|'
    r'myocardial infarction|heart attack|STEMI|NSTEMI|'
    r'heart failure|atrial fibrillation|hypertension|'
    r'diabetes|diabetic ketoacidosis|DKA|'
    r'stroke|CVA|hemorrhage|aneurysm|embolism|DVT|'
    r'fracture|dislocation|tear|rupture|'
    r'cancer|carcinoma|lymphoma|melanoma|sarcoma|'
    r'cirrhosis|hepatitis|pancreatitis|'
    r'asthma|COPD|pneumothorax|'
    r'seizure|epilepsy|meningitis|encephalitis|'
    r'osteoarthritis|rheumatoid arthritis|'
    r'anemia|thrombocytopenia|leukemia|'
    r'preeclampsia|placental abruption|'
    r'endometriosis|fibroids?|'
    r'hernia|obstruction|stenosis|'
    r'infection|abscess|cellulitis|osteomyelitis)\b',

    # Medications (common classes)
    r'\b(aspirin|warfarin|heparin|enoxaparin|'
    r'metformin|insulin|'
    r'lisinopril|metoprolol|atorvastatin|'
    r'oxycodone|morphine|fentanyl|'
    r'amoxicillin|ceftriaxone|vancomycin|'
    r'prednisone|dexamethasone|'
    r'chemotherapy|immunotherapy|radiation therapy)\b',
]


def extract_medical_terms(text: str) -> list[str]:
    """Extract medical terms from clinical text for MedlinePlus lookup."""
    terms = set()
    for pattern in MEDICAL_TERM_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for m in matches:
            terms.add(m.lower())

    # Map abbreviations to searchable terms
    abbrev_map = {
        "cabg": "coronary artery bypass",
        "pci": "angioplasty",
        "stemi": "heart attack",
        "nstemi": "heart attack",
        "cva": "stroke",
        "dvt": "deep vein thrombosis",
        "copd": "chronic obstructive pulmonary disease",
        "dka": "diabetic ketoacidosis",
    }
    expanded = set()
    for t in terms:
        expanded.add(abbrev_map.get(t, t))

    return list(expanded)[:5]  # Top 5 terms to avoid too many API calls


# ==========================================================================
# STEP 2: MedlinePlus Retrieval
# ==========================================================================

def search_medlineplus(term: str, retmax: int = 1) -> list[dict]:
    """Search MedlinePlus health topics for a term."""
    try:
        encoded = urllib.parse.quote(term)
        url = f"{MEDLINEPLUS_API}?db=healthTopics&term={encoded}&retmax={retmax}"
        req = urllib.request.Request(url, headers={"User-Agent": "MedClear/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = resp.read().decode()

        root = ET.fromstring(data)
        results = []
        for doc in root.findall(".//document"):
            title_elem = doc.find('.//content[@name="title"]')
            summary_elem = doc.find('.//content[@name="FullSummary"]')
            url_attr = doc.get("url", "")

            if title_elem is not None and title_elem.text:
                title = re.sub(r"<[^>]+>", "", title_elem.text).strip()
            else:
                title = ""

            if summary_elem is not None and summary_elem.text:
                summary = re.sub(r"<[^>]+>", " ", summary_elem.text)
                summary = re.sub(r"\s+", " ", summary).strip()
            else:
                summary = ""

            if title and summary:
                results.append({
                    "term": term,
                    "title": title,
                    "url": url_attr,
                    "summary": summary[:500],  # Cap length
                })
        return results
    except Exception as e:
        print(f"  [MedlinePlus] Error searching '{term}': {e}")
        return []


def retrieve_context(clinical_text: str) -> str:
    """Extract terms and retrieve MedlinePlus context."""
    terms = extract_medical_terms(clinical_text)
    if not terms:
        return ""

    print(f"  Extracted terms: {terms}")
    all_context = []
    seen_titles = set()

    for term in terms:
        results = search_medlineplus(term)
        for r in results:
            if r["title"] not in seen_titles:
                seen_titles.add(r["title"])
                # Take first 2-3 sentences of the summary
                sentences = r["summary"].split(". ")
                brief = ". ".join(sentences[:3]) + "."
                all_context.append(
                    f"[{r['title']}]: {brief}"
                )
        time.sleep(0.3)  # Rate limit

    if not all_context:
        return ""

    # Cap total context to ~300 words
    context = "\n".join(all_context)
    words = context.split()
    if len(words) > 300:
        context = " ".join(words[:300]) + "..."

    return context


# ==========================================================================
# STEP 3: RAG-Augmented Generation
# ==========================================================================

def load_model():
    """Load the fine-tuned model."""
    print(f"Loading model from {MODEL_DIR}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to(DEVICE)
    model.eval()
    return tokenizer, model


def generate_with_rag(
    clinical_text: str,
    tokenizer,
    model,
    use_rag: bool = True,
) -> dict:
    """Generate a plain-language simplification with RAG context."""

    # Step 1 & 2: Retrieve context
    context = ""
    if use_rag:
        print("  Retrieving MedlinePlus context...")
        context = retrieve_context(clinical_text)
        if context:
            print(f"  Retrieved {len(context.split())} words of context")

    # Step 3: Build augmented input
    if context:
        augmented_input = (
            f"{TASK_PREFIX}"
            f"REFERENCE INFORMATION:\n{context}\n\n"
            f"CLINICAL NOTE:\n{clinical_text}"
        )
    else:
        augmented_input = f"{TASK_PREFIX}{clinical_text}"

    # Generate
    inputs = tokenizer(
        augmented_input,
        return_tensors="pt",
        max_length=512,
        truncation=True,
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=768,
            num_beams=4,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    raw_output = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Parse output
    result = {
        "raw_output": raw_output,
        "context_used": context,
        "terms_searched": extract_medical_terms(clinical_text),
    }

    if "PLAIN LANGUAGE:" in raw_output:
        parts = raw_output.split("PLAIN LANGUAGE:", 1)
        result["facts"] = parts[0].replace("FACTS:", "").strip()
        result["plain_language"] = parts[1].strip()
    else:
        result["facts"] = None
        result["plain_language"] = raw_output

    return result


# ==========================================================================
# DEMO
# ==========================================================================

def run_demo():
    """Run interactive demo with RAG pipeline."""
    tokenizer, model = load_model()

    examples = [
        {
            "name": "Post-Op Cholecystectomy",
            "text": (
                "Patient underwent laparoscopic cholecystectomy for acute "
                "cholecystitis. Intraoperative findings revealed a distended, "
                "edematous gallbladder with adhesions to the omentum. Critical "
                "view of safety was achieved. EBL minimal. Patient tolerated "
                "the procedure well. POD1: afebrile, tolerating PO diet, "
                "ambulating independently. Discharged on ibuprofen and "
                "oxycodone PRN. Follow-up in 2 weeks."
            ),
        },
        {
            "name": "Cardiac Catheterization",
            "text": (
                "68-year-old male with NSTEMI. Left heart catheterization "
                "with PCI to LAD. Angiography revealed 95% stenosis of "
                "proximal LAD. Successful DES placement with TIMI 3 flow. "
                "Echo showed EF 45% with anterior wall hypokinesis. "
                "Discharge medications: Aspirin 81mg daily, Ticagrelor 90mg "
                "BID x12 months, Metoprolol 50mg daily, Atorvastatin 80mg."
            ),
        },
        {
            "name": "Total Knee Replacement",
            "text": (
                "71-year-old female with severe tricompartmental "
                "osteoarthritis right knee. Right total knee arthroplasty "
                "under spinal anesthesia. Cemented posterior-stabilized "
                "implant. EBL 250mL. DVT prophylaxis enoxaparin 40mg SQ "
                "daily. PT initiated POD0, ambulating 150 feet with walker. "
                "ROM 0-90 degrees. Discharge medications: Oxycodone 5mg "
                "q4-6h PRN, Acetaminophen 1000mg q8h, Enoxaparin 40mg "
                "SQ daily x14 days."
            ),
        },
    ]

    print("\n" + "=" * 70)
    print("MedClear RAG Pipeline Demo")
    print("=" * 70)
    print("Using MedlinePlus (NIH) as reference source\n")

    for ex in examples:
        print("=" * 70)
        print(f"CLINICAL NOTE: {ex['name']}")
        print("=" * 70)
        print(f"\nINPUT: {textwrap.fill(ex['text'], 80)}\n")

        # With RAG
        print("--- WITH RAG (MedlinePlus context) ---")
        result_rag = generate_with_rag(ex["text"], tokenizer, model,
                                        use_rag=True)
        if result_rag["facts"]:
            print(f"\nEXTRACTED FACTS:\n{result_rag['facts']}")
        print(f"\nPLAIN LANGUAGE:\n{textwrap.fill(result_rag['plain_language'], 80)}")

        # Without RAG for comparison
        print("\n--- WITHOUT RAG ---")
        result_no_rag = generate_with_rag(ex["text"], tokenizer, model,
                                           use_rag=False)
        print(f"\n{textwrap.fill(result_no_rag['plain_language'], 80)}")

        # Show MedlinePlus sources
        if result_rag["context_used"]:
            print(f"\n--- MEDLINEPLUS SOURCES ---")
            for line in result_rag["context_used"].split("\n"):
                if line.startswith("["):
                    title = line.split("]:")[0] + "]"
                    print(f"  {title}")
        print()

    # Interactive mode
    print("\n" + "=" * 70)
    print("Interactive Mode - Enter clinical text (or 'quit')")
    print("=" * 70)
    while True:
        text = input("\nClinical text: ").strip()
        if text.lower() in ("quit", "exit", "q"):
            break
        if not text:
            continue
        if text == "ex1":
            text = examples[0]["text"]
        elif text == "ex2":
            text = examples[1]["text"]
        elif text == "ex3":
            text = examples[2]["text"]

        result = generate_with_rag(text, tokenizer, model, use_rag=True)
        if result["facts"]:
            print(f"\nFACTS: {result['facts']}")
        print(f"\nPLAIN LANGUAGE:\n{textwrap.fill(result['plain_language'], 80)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MedClear RAG Pipeline with MedlinePlus")
    parser.add_argument("--text", type=str, default=None,
                        help="Clinical text to simplify")
    parser.add_argument("--demo", action="store_true",
                        help="Run demo with example clinical notes")
    parser.add_argument("--no-rag", action="store_true",
                        help="Disable RAG (model only)")
    args = parser.parse_args()

    if args.text:
        tokenizer, model = load_model()
        result = generate_with_rag(args.text, tokenizer, model,
                                    use_rag=not args.no_rag)
        if result["facts"]:
            print(f"FACTS:\n{result['facts']}\n")
        print(f"PLAIN LANGUAGE:\n{result['plain_language']}")
    elif args.demo:
        run_demo()
    else:
        run_demo()
