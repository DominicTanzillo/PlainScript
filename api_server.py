"""
MedClear API Server
====================
Flask API that serves the RAG pipeline for medical text simplification.
Returns structured JSON with plain language text and hyperlinked medical terms.

Usage:
    python api_server.py
    # API runs on http://localhost:5000

Endpoints:
    POST /api/simplify - Simplify clinical text
    GET  /api/health   - Health check
"""

import json
import os
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from flask import Flask, request, jsonify
from flask import send_from_directory
from flask_cors import CORS

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

app = Flask(__name__, static_folder="frontend/build", static_url_path="")
CORS(app)

# -- Configuration --
MODEL_DIR = os.environ.get("MODEL_DIR", "./medclear_results/v2-base/final")
HF_MODEL_ID = "DTanzillo/medclear-v2-base"
MEDLINEPLUS_API = "https://wsearch.nlm.nih.gov/ws/query"
SIMPLIFY_PREFIX = "simplify: "
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -- Global model (loaded once) --
tokenizer = None
model = None


def load_model():
    global tokenizer, model
    if tokenizer is None:
        # Try local model first, fall back to HuggingFace Hub
        if os.path.exists(MODEL_DIR):
            print(f"Loading model from {MODEL_DIR}...")
            source = MODEL_DIR
        else:
            print(f"Loading model from HuggingFace: {HF_MODEL_ID}...")
            source = HF_MODEL_ID
        tokenizer = AutoTokenizer.from_pretrained(source)
        model = AutoModelForSeq2SeqLM.from_pretrained(source).to(DEVICE)
        model.eval()
        print(f"Model loaded on {DEVICE}")


# -- Medical term patterns --
TERM_PATTERNS = {
    # Procedures
    "cholecystectomy": "gallbladder removal surgery",
    "appendectomy": "appendix removal surgery",
    "hysterectomy": "uterus removal surgery",
    "mastectomy": "breast removal surgery",
    "arthroplasty": "joint replacement surgery",
    "laminectomy": "spine decompression surgery",
    "craniotomy": "skull opening surgery",
    "thoracotomy": "chest opening surgery",
    "laparotomy": "abdominal opening surgery",
    "tracheostomy": "breathing tube in neck",
    "colonoscopy": "colon camera exam",
    "bronchoscopy": "lung airway camera exam",
    "endoscopy": "internal camera exam",
    "catheterization": "threading a tube into the heart",
    "angioplasty": "opening a blocked artery",
    "fasciotomy": "emergency muscle compartment release",
    "discectomy": "disc removal surgery",
    "nephrectomy": "kidney removal surgery",
    "thyroidectomy": "thyroid removal surgery",
    "prostatectomy": "prostate removal surgery",
    "colectomy": "colon removal surgery",
    "biopsy": "tissue sample for testing",
    "debridement": "removal of dead tissue",
    "intubation": "placing a breathing tube",
    "extubation": "removing a breathing tube",
    # Conditions
    "cholecystitis": "gallbladder inflammation",
    "appendicitis": "appendix inflammation",
    "pneumonia": "lung infection",
    "sepsis": "life-threatening blood infection",
    "myocardial infarction": "heart attack",
    "stenosis": "abnormal narrowing",
    "hypertension": "high blood pressure",
    "hypotension": "low blood pressure",
    "tachycardia": "fast heart rate",
    "bradycardia": "slow heart rate",
    "arrhythmia": "irregular heart rhythm",
    "atrial fibrillation": "irregular heart rhythm",
    "embolism": "blood clot blocking a vessel",
    "thrombosis": "blood clot formation",
    "aneurysm": "weakened, ballooning blood vessel",
    "hemorrhage": "severe bleeding",
    "edema": "swelling from fluid",
    "effusion": "fluid buildup",
    "ischemia": "reduced blood flow",
    "necrosis": "tissue death",
    "fibrosis": "scarring",
    "cirrhosis": "liver scarring",
    "hepatitis": "liver inflammation",
    "pancreatitis": "pancreas inflammation",
    "peritonitis": "abdominal lining infection",
    "encephalopathy": "brain dysfunction",
    "neuropathy": "nerve damage",
    "radiculopathy": "pinched nerve pain",
    "myelopathy": "spinal cord compression",
    "dyspnea": "shortness of breath",
    "dysphagia": "difficulty swallowing",
    "hematuria": "blood in urine",
    "hemoptysis": "coughing up blood",
    "hematemesis": "vomiting blood",
    "ascites": "abdominal fluid buildup",
    "syncope": "fainting",
    "anemia": "low red blood cells",
    "thrombocytopenia": "low platelets",
    "leukocytosis": "elevated white blood cells",
    "hyperglycemia": "high blood sugar",
    "hypoglycemia": "low blood sugar",
    "hyperkalemia": "high potassium",
    "hyponatremia": "low sodium",
    # Abbreviations
    "NSTEMI": "heart attack (non-ST elevation type)",
    "STEMI": "heart attack (ST elevation type)",
    "PCI": "opening blocked artery with catheter/stent",
    "CABG": "heart bypass surgery",
    "DES": "drug-coated stent",
    "EF": "heart pumping percentage",
    "DVT": "deep vein blood clot",
    "PE": "blood clot in lung",
    "COPD": "chronic lung disease",
    "CHF": "heart failure",
    "CKD": "chronic kidney disease",
    "ESRD": "kidney failure",
    "DKA": "diabetic emergency (ketoacidosis)",
    "CVA": "stroke",
    "TIA": "mini-stroke",
    "AKI": "sudden kidney injury",
    "ARDS": "severe lung failure",
    "ICU": "intensive care unit",
    "POD": "post-operative day",
    "NPO": "nothing by mouth",
    "PRN": "as needed",
    "BID": "twice daily",
    "TID": "three times daily",
    "QID": "four times daily",
    "QHS": "at bedtime",
    "IV": "into the vein",
    "IM": "into the muscle",
    "SQ": "under the skin",
    "PO": "by mouth",
    "EBL": "estimated blood loss",
    "ROM": "range of motion",
    "PT": "physical therapy",
    "OT": "occupational therapy",
    "BMP": "basic blood chemistry panel",
    "CBC": "complete blood count",
    "CRP": "inflammation marker",
    "ESR": "inflammation marker",
    "INR": "blood clotting measure",
    "A1C": "3-month blood sugar average",
    "BMI": "body mass index",
    "GCS": "consciousness score",
    "NIHSS": "stroke severity score",
}


# Curated MedlinePlus URLs for abbreviations that don't search well
TERM_URLS = {
    "PO": "https://medlineplus.gov/ency/article/002023.htm",
    "PRN": "https://medlineplus.gov/ency/article/002023.htm",
    "NPO": "https://medlineplus.gov/ency/article/002023.htm",
    "IV": "https://medlineplus.gov/ency/article/003423.htm",
    "IM": "https://medlineplus.gov/ency/article/003423.htm",
    "SQ": "https://medlineplus.gov/ency/article/003423.htm",
    "BID": "https://medlineplus.gov/ency/article/002023.htm",
    "TID": "https://medlineplus.gov/ency/article/002023.htm",
    "QID": "https://medlineplus.gov/ency/article/002023.htm",
    "QHS": "https://medlineplus.gov/ency/article/002023.htm",
    "DVT": "https://medlineplus.gov/deepveinthrombosis.html",
    "PE": "https://medlineplus.gov/pulmonaryembolism.html",
    "COPD": "https://medlineplus.gov/copd.html",
    "CHF": "https://medlineplus.gov/heartfailure.html",
    "CKD": "https://medlineplus.gov/chronickidneydisease.html",
    "CVA": "https://medlineplus.gov/stroke.html",
    "TIA": "https://medlineplus.gov/transientischemicattack.html",
    "AKI": "https://medlineplus.gov/ency/article/000501.htm",
    "DKA": "https://medlineplus.gov/ency/article/000320.htm",
    "NSTEMI": "https://medlineplus.gov/heartattack.html",
    "STEMI": "https://medlineplus.gov/heartattack.html",
    "CABG": "https://medlineplus.gov/coronaryarterybypasssurgery.html",
    "PCI": "https://medlineplus.gov/angioplasty.html",
    "EF": "https://medlineplus.gov/ency/article/003757.htm",
    "ICU": "https://medlineplus.gov/criticalcare.html",
    "PT": "https://medlineplus.gov/ency/article/001942.htm",
    "OT": "https://medlineplus.gov/ency/article/007455.htm",
    "CBC": "https://medlineplus.gov/lab-tests/complete-blood-count-cbc/",
    "BMI": "https://medlineplus.gov/ency/article/007196.htm",
    "INR": "https://medlineplus.gov/lab-tests/prothrombin-time-test-and-inr-ptinr/",
    "A1C": "https://medlineplus.gov/a1c.html",
    "EBL": "https://medlineplus.gov/bleeding.html",
    "ROM": "https://medlineplus.gov/ency/article/003165.htm",
    "POD": "https://medlineplus.gov/surgery.html",
    "ARDS": "https://medlineplus.gov/ency/article/000103.htm",
    "ESRD": "https://medlineplus.gov/kidneyfailure.html",
    "GCS": "https://medlineplus.gov/coma.html",
    "BMP": "https://medlineplus.gov/lab-tests/basic-metabolic-panel-bmp/",
    "CRP": "https://medlineplus.gov/lab-tests/c-reactive-protein-crp-test/",
    "ESR": "https://medlineplus.gov/lab-tests/erythrocyte-sedimentation-rate-esr/",
    "DES": "https://medlineplus.gov/angioplasty.html",
    "NIHSS": "https://medlineplus.gov/stroke.html",
    # Conditions (longer names that search well but lets be safe)
    "cholecystectomy": "https://medlineplus.gov/gallbladderdiseases.html",
    "appendectomy": "https://medlineplus.gov/appendicitis.html",
    "hysterectomy": "https://medlineplus.gov/hysterectomy.html",
    "arthroplasty": "https://medlineplus.gov/jointreplacement.html",
    "colonoscopy": "https://medlineplus.gov/colonoscopy.html",
    "pneumonia": "https://medlineplus.gov/pneumonia.html",
    "sepsis": "https://medlineplus.gov/sepsis.html",
    "hypertension": "https://medlineplus.gov/highbloodpressure.html",
    "atrial fibrillation": "https://medlineplus.gov/atrialfibrillation.html",
    "anemia": "https://medlineplus.gov/anemia.html",
    "edema": "https://medlineplus.gov/edema.html",
    "syncope": "https://medlineplus.gov/fainting.html",
    "biopsy": "https://medlineplus.gov/biopsy.html",
    "catheterization": "https://medlineplus.gov/cardiaccatheterization.html",
}


def search_medlineplus(term: str) -> dict | None:
    """Search MedlinePlus for a term and return title + URL."""
    try:
        encoded = urllib.parse.quote(term)
        url = f"{MEDLINEPLUS_API}?db=healthTopics&term={encoded}&retmax=1"
        req = urllib.request.Request(url, headers={"User-Agent": "MedClear/1.0"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = resp.read().decode()
        root = ET.fromstring(data)
        doc = root.find(".//document")
        if doc is not None:
            title_elem = doc.find('.//content[@name="title"]')
            summary_elem = doc.find('.//content[@name="FullSummary"]')
            url_attr = doc.get("url", "")
            title = re.sub(r"<[^>]+>", "", title_elem.text).strip() if title_elem is not None and title_elem.text else ""
            summary = ""
            if summary_elem is not None and summary_elem.text:
                summary = re.sub(r"<[^>]+>", " ", summary_elem.text)
                summary = re.sub(r"\s+", " ", summary).strip()
                sentences = summary.split(". ")
                summary = ". ".join(sentences[:2]) + "."
            if title:
                return {"title": title, "url": url_attr, "summary": summary}
    except Exception:
        pass
    return None


def annotate_text(text: str) -> dict:
    """Find medical terms in text and create annotations with MedlinePlus links."""
    annotations = []
    found_terms = set()

    # Sort patterns by length (longest first to avoid partial matches)
    sorted_terms = sorted(TERM_PATTERNS.keys(), key=len, reverse=True)

    for term in sorted_terms:
        pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
        for match in pattern.finditer(text):
            if match.group().lower() not in found_terms:
                found_terms.add(match.group().lower())
                simple = TERM_PATTERNS[term]

                # Use curated URL if available, otherwise search API
                if term.upper() in TERM_URLS:
                    ml_url = TERM_URLS[term.upper()]
                    ml_summary = ""
                elif term.lower() in TERM_URLS:
                    ml_url = TERM_URLS[term.lower()]
                    ml_summary = ""
                else:
                    ml_result = search_medlineplus(term)
                    ml_url = ml_result["url"] if ml_result else f"https://medlineplus.gov/search/?query={urllib.parse.quote(term)}"
                    ml_summary = ml_result["summary"] if ml_result else ""

                annotations.append({
                    "term": match.group(),
                    "simple": simple,
                    "start": match.start(),
                    "end": match.end(),
                    "url": ml_url,
                    "medlineplus_summary": ml_summary,
                })
                time.sleep(0.2)  # Rate limit

    # Sort by position
    annotations.sort(key=lambda x: x["start"])
    return annotations


def generate_simplification(clinical_text: str) -> str:
    """Generate model simplification."""
    load_model()
    input_text = SIMPLIFY_PREFIX + clinical_text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512,
                       truncation=True).to(DEVICE)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=256, num_beams=4,
            early_stopping=True, no_repeat_ngram_size=3)
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


@app.route("/api/simplify", methods=["POST"])
def simplify():
    """Main API endpoint: simplify clinical text with RAG annotations."""
    data = request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    clinical_text = data["text"]

    # 1. Generate model output
    plain_language = generate_simplification(clinical_text)

    # 2. Annotate the SOURCE text with medical term definitions + MedlinePlus links
    annotations = annotate_text(clinical_text)

    # 3. Also annotate any terms in the model output
    output_annotations = annotate_text(plain_language)

    return jsonify({
        "input": clinical_text,
        "plain_language": plain_language,
        "source_annotations": annotations,
        "output_annotations": output_annotations,
    })


@app.route("/api/lookup", methods=["GET"])
def lookup():
    """Look up a medical term on MedlinePlus."""
    term = request.args.get("term", "")
    if not term:
        return jsonify({"error": "Missing 'term' parameter"}), 400

    result = search_medlineplus(term)
    simple = TERM_PATTERNS.get(term.lower(), "")

    return jsonify({
        "term": term,
        "simple_definition": simple,
        "medlineplus": result,
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": model is not None})


# Serve React frontend
@app.route("/")
def serve_frontend():
    return send_from_directory(app.static_folder, "index.html")


@app.errorhandler(404)
def not_found(e):
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    load_model()
    print("\nMedClear API Server running on http://localhost:5000")
    print("Endpoints:")
    print("  POST /api/simplify  - Simplify clinical text")
    print("  GET  /api/lookup    - Look up a medical term")
    print("  GET  /api/health    - Health check")
    app.run(host="0.0.0.0", port=5000, debug=False)
