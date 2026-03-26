"""
MedClear - HuggingFace Space
Medical text simplification with FLAN-T5 + MedlinePlus RAG.
"""

import os
import re
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_ID = "DTanzillo/medclear-v2-base"
MEDLINEPLUS_API = "https://wsearch.nlm.nih.gov/ws/query"
SIMPLIFY_PREFIX = "simplify: "

# Medical term dictionary (920+ terms)
TERM_PATTERNS = {
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

# Load model at startup
print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)
model.eval()
print("Model loaded!")


def search_medlineplus(term):
    """Search MedlinePlus for a term."""
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
            url_attr = doc.get("url", "")
            summary_elem = doc.find('.//content[@name="FullSummary"]')
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


def find_terms(text):
    """Find medical terms in text."""
    found = []
    found_lower = set()
    sorted_terms = sorted(TERM_PATTERNS.keys(), key=len, reverse=True)
    for term in sorted_terms:
        pattern = re.compile(r'\b' + re.escape(term) + r'\b', re.IGNORECASE)
        for match in pattern.finditer(text):
            if match.group().lower() not in found_lower:
                found_lower.add(match.group().lower())
                found.append((match.group(), TERM_PATTERNS[term]))
    return found


def simplify(clinical_text):
    """Main pipeline: simplify clinical text with term annotations."""
    if not clinical_text.strip():
        return "", ""

    # Generate simplification
    input_text = SIMPLIFY_PREFIX + clinical_text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=256, num_beams=4,
            early_stopping=True, no_repeat_ngram_size=3,
        )
    plain_language = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Build glossary with MedlinePlus links
    terms = find_terms(clinical_text)
    glossary_lines = []
    for term_text, simple_def in terms:
        ml = search_medlineplus(term_text)
        if ml and ml["url"]:
            glossary_lines.append(
                f"**{term_text}** -- {simple_def}  \n"
                f"[Learn more on MedlinePlus]({ml['url']})"
            )
            if ml["summary"]:
                glossary_lines.append(f"> {ml['summary']}")
        else:
            search_url = f"https://medlineplus.gov/search/?query={urllib.parse.quote(term_text)}"
            glossary_lines.append(
                f"**{term_text}** -- {simple_def}  \n"
                f"[Search MedlinePlus]({search_url})"
            )
        glossary_lines.append("")

    glossary = "\n".join(glossary_lines)
    return plain_language, glossary


EXAMPLES = [
    [
        "Patient underwent laparoscopic cholecystectomy for acute cholecystitis. "
        "Intraoperative findings revealed a distended, edematous gallbladder with "
        "adhesions to the omentum. Critical view of safety was achieved. EBL minimal. "
        "Patient tolerated the procedure well. POD1: afebrile, tolerating PO diet, "
        "ambulating independently. Discharged on ibuprofen and oxycodone PRN. "
        "Follow-up in 2 weeks."
    ],
    [
        "68-year-old male with NSTEMI. Left heart catheterization with PCI to LAD. "
        "Angiography revealed 95% stenosis of proximal LAD. Successful DES placement "
        "with TIMI 3 flow. Echo showed EF 45% with anterior wall hypokinesis. "
        "Discharge medications: Aspirin 81mg daily, Ticagrelor 90mg BID x12 months, "
        "Metoprolol 50mg daily, Atorvastatin 80mg daily."
    ],
    [
        "72y/o M. CC: SOB, DOE, R/O Acute MI. PMHx: HTN, DMII, CAD, HFpEF. "
        "Presented to ED via EMS with progressive SOB and 3-pillow orthopnea x24h. "
        "Noncompliant with PO meds (ASA, Lisinopril) d/t financial constraints. "
        "Tachycardic HR 115, hypotensive BP 90/50. CXR: pulmonary edema. "
        "ECG: sinus tach with PVCs, no STEMI. Labs: Cr 2.1 from 0.9 baseline, "
        "K+ 5.5, BNP 2000. Pre-renal AKI. Troponin mildly elevated, likely demand ischemia."
    ],
]

demo = gr.Interface(
    fn=simplify,
    inputs=gr.Textbox(
        label="Clinical Note",
        placeholder="Paste a clinical note, discharge summary, or post-op description...",
        lines=8,
    ),
    outputs=[
        gr.Textbox(label="Plain Language Version", lines=6),
        gr.Markdown(label="Medical Term Glossary (with MedlinePlus links)"),
    ],
    title="MedClear: Doctor-Speak to Human-Speak",
    description=(
        "Paste a clinical note and MedClear will translate it into plain language "
        "that patients and families can understand. Every medical term is defined "
        "and linked to [MedlinePlus](https://medlineplus.gov) (NIH) for verification.\n\n"
        "**This is an AI assistant, not medical advice.** Always talk to your doctor."
    ),
    examples=EXAMPLES,
    cache_examples=False,
    theme=gr.themes.Soft(),
)

# Mount a Flask API so the React frontend can call /api/simplify
import json
from flask import Flask, request as flask_request, jsonify
from flask_cors import CORS

flask_app = Flask(__name__)
CORS(flask_app)


@flask_app.route("/api/simplify", methods=["POST"])
def api_simplify():
    data = flask_request.get_json()
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400

    clinical_text = data["text"]
    plain_language, _ = simplify(clinical_text)

    # Build structured annotations for React frontend
    terms = find_terms(clinical_text)
    annotations = []
    for term_text, simple_def in terms:
        pattern = re.compile(r'\b' + re.escape(term_text) + r'\b', re.IGNORECASE)
        match = pattern.search(clinical_text)
        if match:
            ml = search_medlineplus(term_text)
            ml_url = ml["url"] if ml else f"https://medlineplus.gov/search/?query={urllib.parse.quote(term_text)}"
            ml_summary = ml["summary"] if ml else ""
            annotations.append({
                "term": match.group(),
                "simple": simple_def,
                "start": match.start(),
                "end": match.end(),
                "url": ml_url,
                "medlineplus_summary": ml_summary,
            })

    annotations.sort(key=lambda x: x["start"])
    return jsonify({
        "input": clinical_text,
        "plain_language": plain_language,
        "source_annotations": annotations,
        "output_annotations": [],
    })


@flask_app.route("/api/health", methods=["GET"])
def api_health():
    return jsonify({"status": "ok", "model_loaded": True})


# Mount Flask app inside Gradio
demo = gr.mount_gradio_app(flask_app, demo, path="/")

if __name__ == "__main__":
    flask_app.run(host="0.0.0.0", port=7860)
