"""
MedClear Training Data V2 - Phrase-Level Medical Jargon Translation
====================================================================
Instead of training on long paragraph pairs, we build a multi-granularity
dataset weighted toward SHORT, PRECISE translations:

Level 1: TERM (60% of training)
    "cholecystectomy" → "surgery to remove the gallbladder"
    "DVT prophylaxis" → "medication to prevent blood clots"

Level 2: PHRASE (25% of training)
    "afebrile, tolerating PO diet, ambulating independently" →
    "no fever, eating and drinking normally, walking on their own"

Level 3: SENTENCE (10% of training)
    Full sentence simplification with context

Level 4: PARAGRAPH (5% of training)
    Full document for coherent output generation

This teaches the model the VOCABULARY first, then composition.
"""

import json
import os
import re
import random

OUTPUT_DIR = "./medclear_results/training_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# LEVEL 1: TERM-LEVEL TRANSLATIONS (the foundation)
# ============================================================
MEDICAL_TERMS = {
    # --- PROCEDURES ---
    "cholecystectomy": "surgery to remove the gallbladder",
    "laparoscopic cholecystectomy": "gallbladder removal through small incisions using a camera",
    "appendectomy": "surgery to remove the appendix",
    "hysterectomy": "surgery to remove the uterus",
    "total abdominal hysterectomy": "surgery to remove the uterus through a cut in the belly",
    "mastectomy": "surgery to remove the breast",
    "lumpectomy": "surgery to remove a breast lump and small amount of surrounding tissue",
    "colectomy": "surgery to remove part or all of the large intestine",
    "right hemicolectomy": "surgery to remove the right half of the large intestine",
    "nephrectomy": "surgery to remove a kidney",
    "partial nephrectomy": "surgery to remove part of a kidney while saving the rest",
    "thyroidectomy": "surgery to remove the thyroid gland",
    "prostatectomy": "surgery to remove the prostate gland",
    "craniotomy": "surgery where the skull is opened to access the brain",
    "thoracotomy": "surgery where the chest is opened",
    "laparotomy": "surgery where the abdomen is opened",
    "laminectomy": "surgery to remove bone from the spine to relieve pressure on nerves",
    "discectomy": "surgery to remove a herniated disc pressing on a nerve",
    "arthroplasty": "joint replacement surgery",
    "total knee arthroplasty": "total knee replacement surgery",
    "total hip arthroplasty": "total hip replacement surgery",
    "arthroscopy": "surgery using a tiny camera inside a joint through small incisions",
    "tracheostomy": "a surgical opening in the front of the neck for a breathing tube",
    "fasciotomy": "emergency surgery to cut open muscle compartments to relieve dangerous pressure",
    "coronary artery bypass grafting": "open-heart surgery to create new blood flow routes around blocked heart arteries",
    "CABG": "heart bypass surgery",
    "PCI": "a procedure to open a blocked heart artery using a catheter and stent",
    "stent placement": "putting a tiny mesh tube inside a blood vessel to hold it open",
    "DES placement": "placing a drug-coated stent inside a heart artery to keep it open",
    "catheterization": "threading a thin tube through blood vessels to the heart",
    "left heart catheterization": "threading a tube through a leg artery up to the heart to check for blockages",
    "angiography": "injecting dye and taking X-rays to see inside blood vessels",
    "angioplasty": "using a balloon to widen a narrowed blood vessel",
    "endarterectomy": "surgery to remove plaque buildup from inside an artery",
    "carotid endarterectomy": "surgery to clean out blockage from the neck artery that supplies blood to the brain",
    "ERCP": "a procedure using a scope through the mouth to examine and treat bile duct problems",
    "EGD": "a procedure where a camera is passed through the mouth to look at the esophagus, stomach, and upper intestine",
    "colonoscopy": "a procedure where a camera examines the inside of the large intestine",
    "bronchoscopy": "a procedure where a camera is passed into the airways of the lungs",
    "cystoscopy": "a procedure where a camera looks inside the bladder",
    "biopsy": "removing a small tissue sample for testing under a microscope",
    "excision": "surgical removal",
    "resection": "surgical removal of part of an organ or tissue",
    "debridement": "removing dead, damaged, or infected tissue from a wound",
    "I&D": "cutting open and draining an infection or abscess",
    "incision and drainage": "cutting open and draining a pocket of infection",
    "intubation": "placing a breathing tube through the mouth into the windpipe",
    "extubation": "removing a breathing tube",
    "thoracentesis": "draining fluid from around the lung using a needle",
    "paracentesis": "draining fluid from the belly using a needle",
    "lumbar puncture": "inserting a needle into the lower back to collect spinal fluid",
    "cardioversion": "using an electrical shock to reset the heart to a normal rhythm",
    "ablation": "using heat or cold energy to destroy small areas of tissue causing problems",
    "dialysis": "a machine that filters the blood when the kidneys cannot",
    "hemodialysis": "cleaning the blood through a machine three times per week",
    "CRRT": "continuous kidney dialysis for critically ill patients",
    "exchange transfusion": "replacing a patient's blood with donor blood",
    "ECMO": "a machine that takes over the work of the heart and lungs",
    "mechanical ventilation": "a machine that breathes for the patient",
    "CPR": "chest compressions and rescue breathing to restart the heart",
    "defibrillation": "delivering an electrical shock to restart a stopped heart",
    "pacemaker implantation": "placing a small device under the skin to control the heartbeat",
    "ICD implantation": "placing a device that can shock the heart back to normal rhythm if it stops",
    "TAVR": "replacing the aortic heart valve through a catheter without open-heart surgery",
    "AVR": "open-heart surgery to replace the aortic valve",

    # --- CONDITIONS / DIAGNOSES ---
    "cholecystitis": "inflammation of the gallbladder",
    "acute cholecystitis": "sudden, severe inflammation of the gallbladder",
    "cholelithiasis": "gallstones",
    "choledocholithiasis": "a gallstone stuck in the bile duct",
    "appendicitis": "inflammation of the appendix",
    "pneumonia": "a lung infection",
    "aspiration pneumonia": "a lung infection caused by food or liquid going into the lungs",
    "sepsis": "a life-threatening condition where infection causes organ damage throughout the body",
    "septic shock": "the most severe form of sepsis where blood pressure drops dangerously low",
    "myocardial infarction": "a heart attack",
    "STEMI": "a severe type of heart attack where a heart artery is completely blocked",
    "NSTEMI": "a type of heart attack where a heart artery is partially blocked",
    "acute coronary syndrome": "a sudden reduction in blood flow to the heart",
    "heart failure": "a condition where the heart cannot pump blood effectively",
    "congestive heart failure": "heart failure causing fluid to build up in the lungs and body",
    "CHF": "heart failure",
    "atrial fibrillation": "an irregular and often rapid heart rhythm",
    "ventricular tachycardia": "a dangerous fast heart rhythm from the lower chambers of the heart",
    "ventricular fibrillation": "a life-threatening chaotic heart rhythm",
    "cardiac arrest": "the heart has stopped beating",
    "hypertension": "high blood pressure",
    "hypotension": "dangerously low blood pressure",
    "tachycardia": "an abnormally fast heart rate over 100 beats per minute",
    "bradycardia": "an abnormally slow heart rate under 60 beats per minute",
    "aortic stenosis": "narrowing of the heart's aortic valve",
    "mitral regurgitation": "leaking of the heart's mitral valve",
    "cardiomyopathy": "a disease of the heart muscle that makes it harder to pump blood",
    "pericardial effusion": "fluid buildup in the sac around the heart",
    "endocarditis": "an infection of the heart valves",
    "DVT": "a blood clot in a deep vein, usually in the leg",
    "deep vein thrombosis": "a blood clot forming in a deep vein of the leg",
    "PE": "a blood clot that has traveled to the lungs",
    "pulmonary embolism": "a blood clot blocking an artery in the lungs",
    "stroke": "brain damage from blocked or burst blood vessels",
    "CVA": "a stroke",
    "TIA": "a mini-stroke where symptoms resolve completely within 24 hours",
    "subarachnoid hemorrhage": "bleeding on the surface of the brain",
    "subdural hematoma": "a collection of blood between the brain and skull",
    "aneurysm": "a weak, balloon-like bulge in a blood vessel that can burst",
    "stenosis": "abnormal narrowing",
    "occlusion": "a complete blockage",
    "embolism": "a blood clot that traveled and blocked a vessel",
    "thrombosis": "formation of a blood clot inside a blood vessel",
    "ischemia": "inadequate blood flow to part of the body",
    "hemorrhage": "severe or uncontrolled bleeding",
    "hematoma": "a collection of blood outside blood vessels",
    "edema": "swelling from excess fluid in body tissues",
    "effusion": "abnormal fluid buildup in a body space",
    "ascites": "fluid buildup in the belly, usually from liver disease",
    "pleural effusion": "fluid buildup around the lungs",
    "necrosis": "death of body tissue",
    "fibrosis": "scarring of tissue",
    "cirrhosis": "severe scarring of the liver",
    "hepatitis": "inflammation of the liver",
    "pancreatitis": "inflammation of the pancreas",
    "peritonitis": "infection of the abdominal lining",
    "cellulitis": "a spreading skin infection",
    "osteomyelitis": "a bone infection",
    "abscess": "a pocket of pus from infection",
    "COPD": "chronic obstructive pulmonary disease, a long-term lung condition making breathing difficult",
    "asthma": "a condition where airways narrow and swell, making breathing difficult",
    "ARDS": "severe lung failure where the lungs fill with fluid",
    "pneumothorax": "a collapsed lung from air leaking into the chest cavity",
    "atelectasis": "collapse of part of the lung",
    "DKA": "diabetic ketoacidosis, a dangerous complication of diabetes with very high blood sugar",
    "hyperglycemia": "high blood sugar",
    "hypoglycemia": "dangerously low blood sugar",
    "hyperkalemia": "dangerously high potassium in the blood",
    "hyponatremia": "low sodium in the blood",
    "AKI": "sudden kidney injury where the kidneys stop working properly",
    "acute kidney injury": "sudden kidney failure",
    "CKD": "chronic kidney disease, long-term reduced kidney function",
    "ESRD": "end-stage kidney failure requiring dialysis or transplant",
    "encephalopathy": "brain dysfunction causing confusion",
    "hepatic encephalopathy": "confusion caused by a failing liver unable to filter toxins",
    "neuropathy": "nerve damage causing numbness, tingling, or weakness",
    "peripheral neuropathy": "nerve damage in the hands and feet",
    "radiculopathy": "pain from a pinched nerve in the spine",
    "myelopathy": "spinal cord damage affecting walking and hand function",
    "seizure": "sudden uncontrolled electrical activity in the brain",
    "status epilepticus": "a seizure that will not stop and is a medical emergency",
    "syncope": "fainting",
    "anemia": "low red blood cells, reducing the blood's ability to carry oxygen",
    "thrombocytopenia": "low platelet count, increasing bleeding risk",
    "pancytopenia": "low counts of all blood cell types",
    "leukocytosis": "elevated white blood cells, often indicating infection or inflammation",
    "coagulopathy": "a problem with blood clotting",
    "DIC": "a dangerous condition where the blood both clots and bleeds uncontrollably",
    "osteoarthritis": "wear-and-tear arthritis where joint cartilage breaks down",
    "fracture": "a broken bone",
    "dislocation": "a joint where the bones have come apart",
    "herniated disc": "a spinal disc that has bulged out and is pressing on a nerve",
    "spinal stenosis": "narrowing of the spinal canal putting pressure on the spinal cord",
    "hernia": "tissue pushing through a weak spot in the muscle wall",
    "bowel obstruction": "a blockage in the intestine preventing food from passing through",
    "ileus": "the intestines have temporarily stopped moving",
    "diverticulitis": "infection of a small pouch in the colon wall",
    "intussusception": "when one part of the intestine telescopes inside another, blocking it",
    "volvulus": "when the intestine twists on itself, cutting off blood supply",
    "preeclampsia": "high blood pressure during pregnancy that can damage organs",
    "eclampsia": "seizures caused by preeclampsia",
    "placenta previa": "the placenta covering the cervix, blocking the baby's exit",
    "placental abruption": "the placenta separating from the uterus before delivery",

    # --- SYMPTOMS ---
    "dyspnea": "shortness of breath",
    "orthopnea": "difficulty breathing when lying flat",
    "paroxysmal nocturnal dyspnea": "waking up at night gasping for air",
    "PND": "waking up at night gasping for air",
    "dysphagia": "difficulty swallowing",
    "odynophagia": "pain when swallowing",
    "dysuria": "pain or burning when urinating",
    "hematuria": "blood in the urine",
    "hemoptysis": "coughing up blood",
    "hematemesis": "vomiting blood",
    "melena": "black, tarry stools from bleeding in the upper digestive tract",
    "hematochezia": "bright red blood in the stool",
    "stridor": "a high-pitched sound when breathing in, indicating airway narrowing",
    "wheezing": "a whistling sound when breathing out",
    "cyanosis": "blue coloring of the skin from low oxygen",
    "jaundice": "yellowing of the skin and eyes from liver problems",
    "pruritus": "itching",
    "diaphoresis": "excessive sweating",
    "rigors": "severe shaking chills",
    "malaise": "a general feeling of being unwell",
    "lethargy": "extreme tiredness and drowsiness",
    "altered mental status": "confusion or change in level of consciousness",
    "obtunded": "significantly decreased level of consciousness",
    "comatose": "completely unconscious and unresponsive",

    # --- CLINICAL PHRASES ---
    "afebrile": "no fever",
    "febrile": "having a fever",
    "tolerating PO": "able to eat and drink by mouth",
    "tolerating PO diet": "eating and drinking normally",
    "NPO": "not allowed to eat or drink",
    "ambulating independently": "walking on their own without help",
    "ambulating with assistance": "walking with help",
    "hemodynamically stable": "blood pressure and heart rate are stable",
    "hemodynamically unstable": "dangerously unstable blood pressure",
    "neurovascularly intact": "normal blood flow and nerve function",
    "oriented x3": "knows who they are, where they are, and what day it is",
    "GCS 15": "fully conscious and alert (best possible score)",
    "acute on chronic": "a sudden worsening of a long-standing condition",
    "status post": "after having",
    "s/p": "after having",
    "EBL minimal": "very little blood loss during surgery",
    "wound CDI": "surgical wound is clean, dry, and intact",
    "incision CDI": "surgical cut is clean, dry, and intact",
    "POD0": "the day of surgery",
    "POD1": "the first day after surgery",
    "POD2": "the second day after surgery",
    "DNR/DNI": "do not resuscitate / do not intubate (no CPR, no breathing machine)",
    "code status full": "wants all life-saving measures if the heart stops",
    "comfort measures only": "treatment focused on reducing suffering, not curing",
    "palliative care": "care focused on comfort and quality of life for serious illness",
    "hospice": "end-of-life care focused on comfort at home or in a facility",
    "prognosis": "the expected course and outcome of a disease",
    "guarded prognosis": "uncertain outlook, could go either way",
    "poor prognosis": "the condition is expected to worsen or not improve",

    # --- MEDICATIONS (common abbreviations) ---
    "PRN": "as needed",
    "BID": "twice a day",
    "TID": "three times a day",
    "QID": "four times a day",
    "QHS": "at bedtime",
    "QAM": "every morning",
    "q4h": "every 4 hours",
    "q6h": "every 6 hours",
    "q8h": "every 8 hours",
    "PO": "by mouth",
    "IV": "into the vein",
    "IM": "into the muscle",
    "SQ": "under the skin (subcutaneous injection)",
    "SL": "under the tongue",
    "PR": "rectally",
    "DVT prophylaxis": "medication to prevent blood clots in the legs",
    "anticoagulation": "blood-thinning treatment to prevent clots",
    "antibiotic prophylaxis": "antibiotics given before surgery to prevent infection",
    "empiric antibiotics": "antibiotics started before knowing the exact bacteria, based on best guess",
    "broad-spectrum antibiotics": "antibiotics that work against many types of bacteria",
    "vasopressors": "medications that raise dangerously low blood pressure",
    "inotropes": "medications that help the heart pump more strongly",
    "diuretics": "medications that help the body remove excess fluid through urination",
    "analgesics": "pain medications",
    "antipyretics": "fever-reducing medications",
    "antiemetics": "medications to prevent or treat nausea and vomiting",
    "bronchodilators": "medications that open the airways to make breathing easier",
    "immunosuppressants": "medications that calm down the immune system",
    "chemotherapy": "medications that kill cancer cells",
    "immunotherapy": "treatment that helps the immune system fight cancer",
    "neoadjuvant": "treatment given before surgery to shrink a tumor",
    "adjuvant": "additional treatment given after surgery to prevent cancer from coming back",

    # --- LAB VALUES / TESTS ---
    "CBC": "complete blood count, a test measuring red cells, white cells, and platelets",
    "BMP": "basic metabolic panel, a blood test checking kidney function and electrolytes",
    "CMP": "comprehensive metabolic panel, a blood test checking kidney, liver, and electrolytes",
    "LFTs": "liver function tests",
    "troponin": "a protein released when heart muscle is damaged, used to diagnose heart attacks",
    "BNP": "a blood test measuring heart strain (higher means worse heart failure)",
    "lactate": "a blood marker that rises when the body is not getting enough oxygen",
    "INR": "a measure of how fast the blood clots (higher means thinner blood)",
    "PT/INR": "a blood clotting test, often used to monitor blood thinners",
    "A1C": "a blood test showing average blood sugar over the past 3 months",
    "HbA1c": "hemoglobin A1C, average blood sugar over 3 months",
    "eGFR": "estimated kidney filtration rate (higher is better, normal is over 90)",
    "creatinine": "a waste product that rises in the blood when the kidneys are not working well",
    "BUN": "blood urea nitrogen, another marker of kidney function",
    "AST": "a liver enzyme that rises when the liver is damaged",
    "ALT": "a liver enzyme that rises when the liver is damaged",
    "bilirubin": "a substance that builds up and causes yellowing when the liver is not working",
    "albumin": "a protein made by the liver; low levels indicate poor nutrition or liver disease",
    "hemoglobin": "the protein in red blood cells that carries oxygen",
    "hematocrit": "the percentage of blood that is red blood cells",
    "WBC": "white blood cell count, a measure of immune system activity",
    "platelets": "blood cells that help with clotting",
    "ESR": "erythrocyte sedimentation rate, an inflammation marker",
    "CRP": "C-reactive protein, an inflammation marker",
    "PSA": "prostate-specific antigen, a blood test related to prostate health",
    "AFP": "alpha-fetoprotein, a tumor marker for liver cancer",
    "CA 19-9": "a tumor marker often elevated in pancreatic cancer",
    "urinalysis": "a urine test checking for infection, blood, protein, and other abnormalities",
    "blood cultures": "blood tests to check for bacteria in the bloodstream",
    "sputum culture": "testing mucus from the lungs to identify which bacteria is causing infection",
    "ABG": "arterial blood gas, a test measuring oxygen, carbon dioxide, and acid levels in the blood",
    "CT scan": "a detailed X-ray that creates cross-sectional images of the body",
    "MRI": "a scan using magnetic fields to create detailed images without radiation",
    "CTA": "a CT scan with dye injected to see blood vessels",
    "MRA": "an MRI scan of blood vessels",
    "PET scan": "a scan that shows how active cells are, used to find cancer",
    "echocardiogram": "an ultrasound of the heart",
    "TTE": "an ultrasound of the heart done from outside the chest",
    "TEE": "an ultrasound of the heart done through a scope in the esophagus for clearer images",
    "EKG": "a recording of the heart's electrical activity",
    "ECG": "a recording of the heart's electrical activity",
    "EEG": "a recording of the brain's electrical activity",
    "EMG": "a test of nerve and muscle electrical activity",
    "NCS": "nerve conduction study, testing how fast electrical signals travel through nerves",
    "CXR": "a chest X-ray",
    "KUB": "an X-ray of the kidneys, ureters, and bladder",

    # --- ANATOMY / LOCATION ---
    "RUQ": "right upper area of the belly",
    "LUQ": "left upper area of the belly",
    "RLQ": "right lower area of the belly",
    "LLQ": "left lower area of the belly",
    "bilateral": "on both sides",
    "unilateral": "on one side only",
    "proximal": "closer to the center of the body",
    "distal": "farther from the center of the body",
    "anterior": "front",
    "posterior": "back",
    "superior": "upper / above",
    "inferior": "lower / below",
    "medial": "toward the middle",
    "lateral": "toward the side",
    "periorbital": "around the eye",
    "suprapubic": "above the pubic bone",
    "substernal": "behind the breastbone",
    "epigastric": "upper middle area of the belly",
    "cervical": "in the neck area",
    "thoracic": "in the chest area",
    "lumbar": "in the lower back area",
    "sacral": "at the base of the spine",
    "femoral": "related to the thigh bone",
    "tibial": "related to the shinbone",
    "radial": "related to the forearm bone on the thumb side",
    "ulnar": "related to the forearm bone on the pinky side",
}


def build_term_pairs():
    """Build Level 1: term → plain English pairs."""
    pairs = []
    for term, definition in MEDICAL_TERMS.items():
        # Standard direction
        pairs.append({
            "input_text": f"define: {term}",
            "target": definition,
            "level": "term",
        })
        # Also add as "what does X mean" format
        pairs.append({
            "input_text": f"what does {term} mean in simple terms?",
            "target": definition,
            "level": "term",
        })
    return pairs


# ============================================================
# LEVEL 2: PHRASE-LEVEL TRANSLATIONS
# ============================================================
PHRASE_PAIRS = [
    # Post-op status phrases
    ("afebrile, tolerating PO diet, ambulating independently",
     "no fever, eating and drinking normally, walking on their own"),
    ("hemodynamically stable, neurologically intact",
     "blood pressure and heart rate are normal, brain and nerve function is normal"),
    ("wound CDI, no erythema or drainage",
     "surgical wound is clean, dry, and intact with no redness or fluid leaking"),
    ("EBL minimal, no complications",
     "very little blood loss during surgery, no problems occurred"),
    ("patient tolerated the procedure well",
     "the patient handled the surgery without problems"),
    ("transferred to PACU in stable condition",
     "moved to the recovery room in stable condition"),
    ("discharged home in good condition",
     "sent home feeling well"),

    # Vital signs
    ("tachycardic to 120, hypotensive to 80/50",
     "heart rate is fast at 120 (normal is 60-100), blood pressure is dangerously low at 80/50"),
    ("febrile to 102.4, tachycardic",
     "has a fever of 102.4 and a fast heart rate"),
    ("SpO2 88% on room air",
     "oxygen level is low at 88% without supplemental oxygen (normal is above 95%)"),
    ("BP 142/92 on current regimen",
     "blood pressure is 142/92 on current medications (slightly high, target is under 130/80)"),

    # Lab descriptions
    ("WBC 18.4, left shift",
     "white blood cell count is elevated at 18.4 with immature cells, suggesting active infection"),
    ("Hgb 7.2, transfusion threshold",
     "hemoglobin is very low at 7.2, low enough that a blood transfusion may be needed"),
    ("Cr 2.8, up from baseline 1.0",
     "creatinine is elevated at 2.8 (was normally 1.0), meaning the kidneys are not working well"),
    ("troponin peaked at 4.2",
     "the heart damage marker peaked at 4.2, confirming heart muscle injury"),
    ("INR 3.8, supratherapeutic",
     "blood clotting time is 3.8, which is too thin (above the target range)"),
    ("A1C 9.2%, poorly controlled",
     "3-month blood sugar average is 9.2%, much higher than the goal of under 7%"),
    ("K+ 6.2, EKG changes",
     "potassium is dangerously high at 6.2 and is affecting the heart rhythm"),
    ("lactate 4.8, trending down",
     "lactic acid is elevated at 4.8 (indicating the body is stressed) but is improving"),

    # Clinical assessment phrases
    ("acute on chronic systolic heart failure",
     "a sudden worsening of long-standing heart failure where the heart cannot pump strongly enough"),
    ("community-acquired pneumonia failing outpatient treatment",
     "a lung infection caught outside the hospital that is not getting better with prescribed medications"),
    ("sepsis secondary to UTI",
     "a life-threatening blood infection caused by a urinary tract infection"),
    ("NSTEMI with 2-vessel CAD",
     "a heart attack with blockages in two of the heart's main arteries"),
    ("decompensated cirrhosis with new-onset ascites",
     "liver disease that has gotten worse, now causing fluid buildup in the belly"),
    ("acute limb ischemia, Rutherford IIa",
     "sudden loss of blood flow to the leg that threatens the limb but can be saved with prompt treatment"),
    ("status post CABG x4, LIMA to LAD",
     "after quadruple heart bypass surgery, with a chest wall artery connected to the main heart artery"),

    # Discharge instruction phrases
    ("weight bearing as tolerated with walker",
     "you can put weight on the leg as much as is comfortable, using a walker for support"),
    ("NWB right LE x6 weeks",
     "do not put any weight on the right leg for 6 weeks"),
    ("advance diet as tolerated",
     "gradually return to eating normal food as your stomach allows"),
    ("follow up in 2 weeks for wound check and staple removal",
     "come back in 2 weeks so the doctor can check how the wound is healing and remove the staples"),
    ("call 911 if chest pain, shortness of breath, or arm/jaw pain",
     "call 911 immediately if you have chest pain, trouble breathing, or pain in your arm or jaw"),
    ("do not take aspirin or ibuprofen for 7 days",
     "avoid aspirin and ibuprofen for one week because they can increase bleeding"),
    ("incentive spirometry q1h while awake",
     "use the breathing exercise device every hour while you are awake to keep your lungs expanded"),
    ("DVT prophylaxis with enoxaparin 40mg SQ daily x14 days",
     "take a daily blood-thinning injection (enoxaparin) under the skin for 14 days to prevent blood clots"),
]


def build_phrase_pairs():
    """Build Level 2: phrase → plain English pairs."""
    pairs = []
    for medical, plain in PHRASE_PAIRS:
        pairs.append({
            "input_text": f"simplify: {medical}",
            "target": plain,
            "level": "phrase",
        })
    return pairs


def build_all_data():
    """Build complete multi-granularity training dataset with proper weighting."""

    # Level 1: Terms (target: 60% of data)
    term_pairs = build_term_pairs()
    print(f"Level 1 (terms): {len(term_pairs)} pairs")

    # Level 2: Phrases (target: 25% of data)
    phrase_pairs = build_phrase_pairs()
    # Oversample phrases to reach target proportion
    phrase_pairs_oversampled = phrase_pairs * 5
    print(f"Level 2 (phrases): {len(phrase_pairs)} base x5 = {len(phrase_pairs_oversampled)} pairs")

    # Level 3: Sentences from aligned data
    sentence_pairs = []
    aligned_file = "./medclear_results/synthetic_data/aligned_training_data.jsonl"
    if os.path.exists(aligned_file):
        with open(aligned_file, "r", encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                if p.get("granularity") == "sentence":
                    sentence_pairs.append({
                        "input_text": f"simplify: {p['source']}",
                        "target": p["target"],
                        "level": "sentence",
                    })
    print(f"Level 3 (sentences): {len(sentence_pairs)} pairs")

    # Level 4: Paragraphs from synthetic data
    paragraph_pairs = []
    synth_file = "./medclear_results/synthetic_data/claude_generated_pairs.jsonl"
    if os.path.exists(synth_file):
        with open(synth_file, "r", encoding="utf-8") as f:
            for line in f:
                p = json.loads(line)
                paragraph_pairs.append({
                    "input_text": f"simplify clinical note: {p['source']}",
                    "target": p["target"],
                    "level": "paragraph",
                })
    print(f"Level 4 (paragraphs): {len(paragraph_pairs)} pairs")

    # Level 5: Medical flashcards (term-level knowledge)
    flashcard_pairs = []
    fc_dir = "./medclear_results/datasets/medical_flashcards/train"
    if os.path.exists(fc_dir):
        from datasets import load_from_disk
        ds = load_from_disk(fc_dir)
        for i in range(len(ds)):
            inp = ds[i]["input"]
            out = ds[i]["output"]
            if inp.startswith("What is ") or inp.startswith("What are ") or inp.startswith("Define "):
                if len(out) < 200:
                    flashcard_pairs.append({
                        "input_text": inp,
                        "target": out,
                        "level": "flashcard",
                    })
            if len(flashcard_pairs) >= 5000:  # Cap at 5000
                break
    print(f"Level 5 (flashcards): {len(flashcard_pairs)} pairs")

    # Level 6: Academic data (Cochrane/PLABA/Med-EASi)
    # These are loaded separately by the training script

    # Combine with weighting
    all_pairs = (
        term_pairs +           # ~840 (60% weight through volume)
        phrase_pairs_oversampled +  # ~350 (25% weight)
        sentence_pairs[:2000] +    # Cap sentences
        paragraph_pairs +          # ~626
        flashcard_pairs            # ~5000
    )

    random.seed(42)
    random.shuffle(all_pairs)

    # Split 90/5/5
    n = len(all_pairs)
    train_end = int(n * 0.9)
    val_end = int(n * 0.95)

    splits = {
        "train": all_pairs[:train_end],
        "validation": all_pairs[train_end:val_end],
        "test": all_pairs[val_end:],
    }

    # Save
    for split_name, data in splits.items():
        path = os.path.join(OUTPUT_DIR, f"{split_name}.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for p in data:
                f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\n=== TRAINING DATA V2 ===")
    print(f"Total: {n}")
    print(f"  Train: {len(splits['train'])}")
    print(f"  Val:   {len(splits['validation'])}")
    print(f"  Test:  {len(splits['test'])}")

    # Stats by level
    from collections import Counter
    level_counts = Counter(p["level"] for p in all_pairs)
    print(f"\nBy level:")
    for level, count in level_counts.most_common():
        pct = count / n * 100
        print(f"  {level}: {count} ({pct:.1f}%)")

    return splits


if __name__ == "__main__":
    build_all_data()
