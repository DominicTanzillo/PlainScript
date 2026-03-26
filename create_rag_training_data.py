"""
Create RAG-augmented training data.
Takes existing synthetic pairs and creates versions where MedlinePlus
reference information is prepended to the source, teaching the model
to use retrieved context in its simplification.

This creates pairs that match how the RAG pipeline works at inference time:
  Input: "extract facts and simplify clinical note: REFERENCE INFORMATION:
          [MedlinePlus content] CLINICAL NOTE: [clinical text]"
  Target: "FACTS: ... PLAIN LANGUAGE: ..."
"""

import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

INPUT_FILE = "./medclear_results/synthetic_data/claude_generated_pairs.jsonl"
OUTPUT_FILE = "./medclear_results/synthetic_data/claude_generated_pairs_rag.jsonl"
MEDLINEPLUS_API = "https://wsearch.nlm.nih.gov/ws/query"

# Medical terms to search for
TERM_PATTERNS = [
    r'\b(cholecystectomy|appendectomy|hysterectomy|mastectomy|arthroplasty|'
    r'laminectomy|craniotomy|thoracotomy|catheterization|dialysis|transplant|'
    r'tracheostomy|colonoscopy|bronchoscopy|biopsy|amputation)\b',
    r'\b(cholecystitis|appendicitis|pneumonia|sepsis|heart attack|'
    r'heart failure|atrial fibrillation|hypertension|diabetes|stroke|'
    r'fracture|cancer|carcinoma|cirrhosis|pancreatitis|asthma|COPD|'
    r'seizure|epilepsy|anemia|hernia|stenosis|embolism)\b',
]

ABBREV_MAP = {
    "cabg": "coronary artery bypass",
    "pci": "angioplasty",
    "stemi": "heart attack",
    "nstemi": "heart attack",
    "cva": "stroke",
    "dvt": "deep vein thrombosis",
    "copd": "chronic obstructive pulmonary disease",
    "dka": "diabetic ketoacidosis",
    "ards": "acute respiratory distress syndrome",
    "aki": "acute kidney injury",
}


def extract_terms(text):
    terms = set()
    for pat in TERM_PATTERNS:
        for m in re.findall(pat, text, re.IGNORECASE):
            terms.add(ABBREV_MAP.get(m.lower(), m.lower()))
    return list(terms)[:3]  # Top 3 to keep API calls manageable


def fetch_medlineplus(term):
    try:
        encoded = urllib.parse.quote(term)
        url = f"{MEDLINEPLUS_API}?db=healthTopics&term={encoded}&retmax=1"
        req = urllib.request.Request(url, headers={"User-Agent": "MedClear/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = resp.read().decode()
        root = ET.fromstring(data)
        doc = root.find(".//document")
        if doc is not None:
            summary_elem = doc.find('.//content[@name="FullSummary"]')
            if summary_elem is not None and summary_elem.text:
                summary = re.sub(r"<[^>]+>", " ", summary_elem.text)
                summary = re.sub(r"\s+", " ", summary).strip()
                sentences = summary.split(". ")
                return ". ".join(sentences[:2]) + "."
    except Exception:
        pass
    return None


def main():
    print("Creating RAG-augmented training data...")

    pairs = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))

    print(f"Input pairs: {len(pairs)}")

    rag_pairs = []
    no_context_count = 0

    for i, pair in enumerate(pairs):
        if i % 50 == 0:
            print(f"  Processing {i}/{len(pairs)}...")

        source = pair["source"]
        target = pair["target"]

        # Extract terms and fetch MedlinePlus context
        terms = extract_terms(source)
        context_parts = []

        for term in terms:
            summary = fetch_medlineplus(term)
            if summary:
                context_parts.append(f"[{term.title()}]: {summary}")
            time.sleep(0.25)

        if context_parts:
            context = "\n".join(context_parts)
            # Create RAG-augmented source
            rag_source = (
                f"REFERENCE INFORMATION:\n{context}\n\n"
                f"CLINICAL NOTE:\n{source}"
            )
        else:
            rag_source = source
            no_context_count += 1

        rag_pairs.append({
            "source": rag_source,
            "target": target,
            "dataset": pair.get("dataset", "unknown"),
            "specialty": pair.get("specialty", "unknown"),
        })

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for p in rag_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nOutput pairs: {len(rag_pairs)}")
    print(f"Pairs with MedlinePlus context: {len(rag_pairs) - no_context_count}")
    print(f"Pairs without context: {no_context_count}")

    # Show a sample
    for p in rag_pairs[:1]:
        print(f"\n--- SAMPLE ---")
        print(f"SOURCE (first 400 chars):\n{p['source'][:400]}...")
        print(f"\nTARGET (first 200 chars):\n{p['target'][:200]}...")

    # Token lengths
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    src_lengths = [len(tokenizer.encode(p["source"])) for p in rag_pairs]
    print(f"\nRAG source token lengths:")
    print(f"  Mean: {sum(src_lengths)/len(src_lengths):.0f}")
    print(f"  Max: {max(src_lengths)}")
    print(f"  >512: {sum(1 for l in src_lengths if l > 512)}")


if __name__ == "__main__":
    main()
