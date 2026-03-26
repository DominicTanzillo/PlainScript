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
    "A1C": "3-month blood sugar average",
    "ABG": "arterial blood gas test",
    "ACL": "anterior cruciate ligament (knee)",
    "ADHD": "attention deficit hyperactivity disorder",
    "adhesions": "internal scars connecting body structures",
    "ADL": "activities of daily living",
    "AIDS": "acquired immunodeficiency syndrome",
    "AKA": "above-knee amputation",
    "AKI": "sudden kidney injury",
    "ALS": "amyotrophic lateral sclerosis (Lou Gehrig's disease)",
    "AMA": "against medical advice",
    "AMI": "heart attack",
    "amniocentesis": "test of fluid around baby in the womb",
    "anaemia": "low red blood cell count causing tiredness",
    "anaesthesia": "medication to prevent pain during procedures",
    "anemia": "low red blood cells",
    "aneurysm": "weakened, ballooning blood vessel",
    "angioplasty": "opening a blocked artery",
    "antibiotics": "medicines to fight bacterial infections",
    "anticoagulant": "blood-thinning medication",
    "appendectomy": "appendix removal surgery",
    "appendicitis": "appendix inflammation",
    "ARDS": "severe lung failure",
    "arrhythmia": "irregular heart rhythm",
    "arthroplasty": "joint replacement surgery",
    "ASAP": "as soon as possible",
    "ascites": "abdominal fluid buildup",
    "ASD": "autism spectrum disorder",
    "atrial fibrillation": "irregular heart rhythm",
    "BID": "twice daily",
    "biopsy": "tissue sample for testing",
    "BKA": "below-knee amputation",
    "BMI": "body mass index",
    "BMP": "basic blood chemistry panel",
    "BP": "blood pressure",
    "BPH": "enlarged prostate",
    "BR": "bed rest",
    "bradycardia": "slow heart rate",
    "breech": "baby positioned bottom-first in the womb",
    "bronchoscopy": "lung airway camera exam",
    "BUN": "blood urea nitrogen (kidney test)",
    "CABG": "heart bypass surgery",
    "CAD": "coronary artery disease",
    "caesarean": "surgical delivery of a baby through the abdomen",
    "CAT": "CT scan (a type of X-ray)",
    "catheter": "a small tube passed through the body",
    "catheterization": "threading a tube into the heart",
    "CBC": "complete blood count",
    "CC": "chief complaint",
    "CCU": "coronary care unit",
    "cervix": "the entrance or neck of the womb",
    "CHD": "congenital heart disease",
    "CHF": "heart failure",
    "CHI": "closed head injury",
    "cholecystectomy": "gallbladder removal surgery",
    "cholecystitis": "gallbladder inflammation",
    "cirrhosis": "liver scarring",
    "CKD": "chronic kidney disease",
    "CMV": "cytomegalovirus (a common virus)",
    "CNS": "central nervous system (brain and spinal cord)",
    "colectomy": "colon removal surgery",
    "colonoscopy": "colon camera exam",
    "colposcopy": "examination of the cervix using a microscope",
    "conception": "when an egg is fertilized by sperm",
    "contraception": "birth control",
    "COPD": "chronic lung disease",
    "corticosteroids": "anti-inflammatory hormonal medications",
    "CP": "cerebral palsy",
    "CPAP": "continuous positive airway pressure",
    "CPR": "cardiopulmonary resuscitation",
    "craniotomy": "skull opening surgery",
    "CRF": "chronic kidney failure",
    "CRP": "inflammation marker",
    "CSF": "cerebrospinal fluid (fluid around brain and spine)",
    "CT": "computerized tomography scan",
    "CVA": "stroke",
    "CXR": "chest X-ray",
    "D&C": "surgical procedure on the uterus",
    "DC": "discharge",
    "debridement": "removal of dead tissue",
    "DES": "drug-coated stent",
    "diabetes": "condition causing high blood sugar",
    "dilatation": "the cervix opening during labor",
    "discectomy": "disc removal surgery",
    "DKA": "diabetic emergency (ketoacidosis)",
    "DM": "diabetes",
    "DNR": "do not resuscitate",
    "DOA": "dead on arrival",
    "DOE": "shortness of breath with exertion",
    "DVT": "deep vein blood clot",
    "dysphagia": "difficulty swallowing",
    "dyspnea": "shortness of breath",
    "EBL": "estimated blood loss",
    "ECG": "heart tracing test",
    "ECHO": "heart ultrasound",
    "eclampsia": "seizures as a complication of pre-eclampsia",
    "ectopic pregnancy": "pregnancy growing outside the womb",
    "ED": "emergency department",
    "edema": "swelling from fluid",
    "EEG": "brain wave test",
    "EF": "heart pumping percentage",
    "effusion": "fluid buildup",
    "EKG": "heart tracing test",
    "embolism": "blood clot blocking a vessel",
    "EMG": "muscle electrical test",
    "encephalopathy": "brain dysfunction",
    "endometriosis": "womb lining tissue growing in other places",
    "endoscopy": "internal camera exam",
    "ENT": "ear, nose and throat",
    "epidural": "pain relief injection into the lower back",
    "episiotomy": "a cut to widen the birth opening during delivery",
    "ER": "emergency room",
    "ERCP": "procedure to check liver, gallbladder, bile ducts and pancreas",
    "ESR": "inflammation marker",
    "ESRD": "kidney failure",
    "ETOH": "alcohol",
    "extubation": "removing a breathing tube",
    "fasciotomy": "emergency muscle compartment release",
    "fibroids": "non-cancerous growths in the womb wall",
    "fibrosis": "scarring",
    "forceps": "instruments used to help deliver a baby",
    "FWB": "full weight bearing",
    "Fx": "fracture",
    "GCS": "consciousness score",
    "GERD": "acid reflux disease",
    "gestational diabetes": "diabetes triggered during pregnancy",
    "GFR": "kidney function test",
    "GI": "gastrointestinal (digestive system)",
    "GSW": "gunshot wound",
    "H/A": "headache",
    "HAV": "hepatitis A virus",
    "HBV": "hepatitis B virus",
    "HCV": "hepatitis C virus",
    "HDL": "good cholesterol",
    "HEENT": "head, eyes, ears, nose, throat",
    "hematemesis": "vomiting blood",
    "hematuria": "blood in urine",
    "hemoptysis": "coughing up blood",
    "hemorrhage": "severe bleeding",
    "HEP": "home exercise program",
    "heparin": "blood-thinning injection",
    "hepatitis": "liver inflammation",
    "HIV": "human immunodeficiency virus",
    "HPV": "human papillomavirus",
    "HR": "heart rate",
    "HRT": "hormone replacement therapy",
    "HTN": "high blood pressure",
    "Hx": "history",
    "hyperglycemia": "high blood sugar",
    "hyperkalemia": "high potassium",
    "hypertension": "high blood pressure",
    "hypoglycemia": "low blood sugar",
    "hyponatremia": "low sodium",
    "hypotension": "low blood pressure",
    "hysterectomy": "uterus removal surgery",
    "IBD": "inflammatory bowel disease",
    "IBS": "irritable bowel syndrome",
    "ICD": "implantable heart defibrillator",
    "ICU": "intensive care unit",
    "IM": "into the muscle",
    "incontinence": "loss of bladder or bowel control",
    "induction": "starting labor artificially",
    "infusion": "medication given slowly through an IV",
    "INR": "blood clotting measure",
    "intubation": "placing a breathing tube",
    "ischemia": "reduced blood flow",
    "IUD": "intrauterine device (birth control)",
    "IV": "into the vein",
    "IVF": "in vitro fertilization (test tube baby procedure)",
    "jaundice": "yellowing of the skin and eyes",
    "laminectomy": "spine decompression surgery",
    "laparoscopy": "keyhole surgery using small cuts",
    "laparotomy": "abdominal opening surgery",
    "LDL": "bad cholesterol",
    "leukocytosis": "elevated white blood cells",
    "LFT": "liver function tests",
    "LOS": "length of stay",
    "LP": "lumbar puncture (spinal tap)",
    "LUE": "left upper extremity",
    "mastectomy": "breast removal surgery",
    "meconium": "baby's first bowel movement (black and sticky)",
    "meningitis": "inflammation of the brain lining",
    "menopause": "when periods stop, usually around age 50",
    "MI": "heart attack",
    "miscarriage": "loss of pregnancy before 23 weeks",
    "MRI": "magnetic resonance imaging",
    "MRSA": "antibiotic-resistant staph infection",
    "MS": "multiple sclerosis",
    "MVA": "motor vehicle accident",
    "myelopathy": "spinal cord compression",
    "myocardial infarction": "heart attack",
    "necrosis": "tissue death",
    "nephrectomy": "kidney removal surgery",
    "neuropathy": "nerve damage",
    "NG": "nasogastric (tube through nose to stomach)",
    "NIHSS": "stroke severity score",
    "NKA": "no known allergies",
    "NPO": "nothing by mouth",
    "NSAID": "anti-inflammatory drug (like ibuprofen)",
    "NSTEMI": "heart attack (non-ST elevation type)",
    "NWB": "non-weight bearing",
    "OA": "osteoarthritis",
    "OCD": "obsessive-compulsive disorder",
    "oedema": "swelling from fluid buildup",
    "OR": "operating room",
    "OT": "occupational therapy",
    "ovulation": "when an egg is released from the ovary",
    "PACU": "post-anesthesia care unit (recovery room)",
    "PAD": "peripheral artery disease",
    "pancreatitis": "pancreas inflammation",
    "PCI": "opening blocked artery with catheter/stent",
    "PD": "Parkinson's disease",
    "PE": "blood clot in lung",
    "PEG": "feeding tube through the stomach wall",
    "peritonitis": "abdominal lining infection",
    "PET": "PET scan (imaging test)",
    "PFT": "lung function test",
    "PID": "pelvic inflammatory disease",
    "placenta": "organ connecting mother and baby during pregnancy",
    "placenta praevia": "placenta covering the cervix",
    "platelets": "blood cells needed for clotting",
    "PMH": "past medical history",
    "PNA": "pneumonia",
    "pneumonia": "lung infection",
    "PO": "by mouth",
    "POD": "post-operative day",
    "polyp": "a growth of tissue on the lining of an organ",
    "pre-eclampsia": "high blood pressure and protein in urine during pregnancy",
    "premature birth": "baby born before 37 weeks",
    "PRN": "as needed",
    "prolapse": "organ pushing through the vaginal wall",
    "prostaglandin": "hormone that causes womb contractions",
    "prostatectomy": "prostate removal surgery",
    "PSA": "prostate specific antigen (prostate test)",
    "PT": "physical therapy",
    "PTSD": "post-traumatic stress disorder",
    "PWB": "partial weight bearing",
    "QHS": "at bedtime",
    "QID": "four times daily",
    "RA": "rheumatoid arthritis",
    "radiculopathy": "pinched nerve pain",
    "RBC": "red blood cell",
    "rehab": "rehabilitation",
    "RLE": "right lower extremity",
    "ROM": "range of motion",
    "RSV": "respiratory syncytial virus",
    "RUE": "right upper extremity",
    "Rx": "prescription",
    "SCI": "spinal cord injury",
    "sepsis": "life-threatening blood infection",
    "SIDS": "sudden infant death syndrome",
    "SLE": "lupus",
    "SNF": "skilled nursing facility",
    "SOB": "shortness of breath",
    "speculum": "instrument to open the vagina for examination",
    "SQ": "under the skin",
    "STAT": "immediately",
    "STD": "sexually transmitted disease",
    "STEMI": "heart attack (ST elevation type)",
    "stenosis": "abnormal narrowing",
    "stillbirth": "baby born dead after 23 weeks of pregnancy",
    "sutures": "stitches",
    "syncope": "fainting",
    "tachycardia": "fast heart rate",
    "TB": "tuberculosis",
    "TBI": "traumatic brain injury",
    "thoracotomy": "chest opening surgery",
    "THR": "total hip replacement",
    "thrombocytopenia": "low platelets",
    "thrombosis": "blood clot formation",
    "thyroidectomy": "thyroid removal surgery",
    "TIA": "mini-stroke",
    "TID": "three times daily",
    "TKR": "total knee replacement",
    "tocolysis": "treatment to delay or prevent early labor",
    "TPN": "total parenteral nutrition (IV feeding)",
    "trach": "tracheostomy",
    "tracheostomy": "breathing tube in neck",
    "TSH": "thyroid stimulating hormone",
    "Tx": "treatment",
    "UA": "urinalysis (urine test)",
    "ultrasound": "imaging using sound waves",
    "urethra": "tube for passing urine",
    "URI": "upper respiratory infection (common cold)",
    "UTI": "urinary tract infection",
    "ventouse": "suction cup used to help deliver a baby",
    "VS": "vital signs",
    "WBC": "white blood cell",
    "WNL": "within normal limits",
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
