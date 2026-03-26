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

# Add REST API routes via FastAPI (Gradio's underlying server)
import json
from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

app = gr.routes.App.create_app(demo)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.post("/api/simplify")
async def api_simplify(request: Request):
    data = await request.json()
    if not data or "text" not in data:
        return JSONResponse({"error": "Missing 'text' field"}, status_code=400)

    clinical_text = data["text"]
    plain_language, _ = simplify(clinical_text)

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
    return JSONResponse({
        "input": clinical_text,
        "plain_language": plain_language,
        "source_annotations": annotations,
        "output_annotations": [],
    })


@app.get("/api/health")
async def api_health():
    return JSONResponse({"status": "ok", "model_loaded": True})


if __name__ == "__main__":
    demo.launch()
