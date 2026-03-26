"""
Create sentence-aligned and term-aligned training data.

Three granularity levels:
1. PARAGRAPH: Full clinical note -> full plain language (existing data)
2. SENTENCE: Aligned source/target sentences
3. TERM: Medical term -> plain definition + MedlinePlus context

This teaches the model to simplify at every level of detail.
"""

import json
import re
import os

INPUT_FILE = "./medclear_results/synthetic_data/claude_generated_pairs.jsonl"
OUTPUT_FILE = "./medclear_results/synthetic_data/aligned_training_data.jsonl"


def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences, handling medical abbreviations."""
    # Protect common abbreviations from sentence splitting
    protected = text
    abbrevs = ['Dr.', 'Mr.', 'Mrs.', 'Ms.', 'vs.', 'etc.', 'i.e.', 'e.g.',
               'approx.', 'mg.', 'mL.', 'cm.', 'mm.', 'kg.', 'lb.',
               'hr.', 'min.', 'sec.', 'dept.']
    for a in abbrevs:
        protected = protected.replace(a, a.replace('.', '<DOT>'))

    # Split on sentence boundaries
    sentences = re.split(r'(?<=[.!?])\s+', protected)

    # Restore dots
    sentences = [s.replace('<DOT>', '.').strip() for s in sentences if s.strip()]
    return sentences


def align_sentences(source_sents: list[str], target_sents: list[str]) -> list[tuple]:
    """Simple alignment: map source sentences to target sentences by position.
    Uses a sliding window approach for approximate alignment."""
    pairs = []

    if len(source_sents) == 0 or len(target_sents) == 0:
        return pairs

    # Ratio-based alignment
    ratio = len(target_sents) / max(len(source_sents), 1)

    for i, src in enumerate(source_sents):
        if len(src) < 20:
            continue

        # Find corresponding target sentence(s)
        tgt_idx = int(i * ratio)
        tgt_idx = min(tgt_idx, len(target_sents) - 1)

        # Take 1-2 target sentences around the aligned position
        tgt_start = max(0, tgt_idx)
        tgt_end = min(len(target_sents), tgt_idx + 2)
        tgt = " ".join(target_sents[tgt_start:tgt_end])

        if len(tgt) > 20 and len(src) > 20:
            pairs.append((src, tgt))

    return pairs


# Term-level definitions (medical term -> plain English)
TERM_DEFINITIONS = {
    "cholecystectomy": "surgery to remove the gallbladder",
    "appendectomy": "surgery to remove the appendix",
    "hysterectomy": "surgery to remove the uterus",
    "mastectomy": "surgery to remove the breast",
    "colectomy": "surgery to remove part or all of the colon",
    "nephrectomy": "surgery to remove a kidney",
    "thyroidectomy": "surgery to remove the thyroid gland",
    "prostatectomy": "surgery to remove the prostate",
    "craniotomy": "surgery that opens the skull to access the brain",
    "thoracotomy": "surgery that opens the chest",
    "laparotomy": "surgery that opens the abdomen",
    "laminectomy": "surgery to remove bone from the spine to relieve pressure on nerves",
    "discectomy": "surgery to remove a herniated disc from the spine",
    "arthroplasty": "surgery to replace a joint with an artificial one",
    "arthroscopy": "surgery using a tiny camera inside a joint",
    "tracheostomy": "a surgical opening in the neck for a breathing tube",
    "fasciotomy": "emergency surgery to cut open muscle compartments to relieve dangerous pressure",
    "laparoscopic": "done through small incisions using a camera, minimally invasive",
    "cholecystitis": "inflammation of the gallbladder",
    "appendicitis": "inflammation of the appendix",
    "pneumonia": "infection in the lungs",
    "sepsis": "a life-threatening condition where the body's response to infection damages its own organs",
    "myocardial infarction": "heart attack, where blood flow to part of the heart is blocked",
    "heart failure": "a condition where the heart cannot pump blood well enough to meet the body's needs",
    "atrial fibrillation": "an irregular, often rapid heart rhythm in the upper chambers of the heart",
    "hypertension": "high blood pressure",
    "hypotension": "dangerously low blood pressure",
    "tachycardia": "abnormally fast heart rate, over 100 beats per minute",
    "bradycardia": "abnormally slow heart rate, under 60 beats per minute",
    "stenosis": "abnormal narrowing of a blood vessel or other tube-like structure in the body",
    "embolism": "a blood clot that has traveled and is blocking a blood vessel",
    "thrombosis": "formation of a blood clot inside a blood vessel",
    "aneurysm": "a weak, balloon-like bulge in a blood vessel wall that can burst",
    "hemorrhage": "severe or uncontrolled bleeding",
    "edema": "swelling caused by excess fluid trapped in body tissues",
    "effusion": "abnormal buildup of fluid in a body space, like around the lungs or heart",
    "ischemia": "inadequate blood flow to a part of the body",
    "necrosis": "death of body tissue due to lack of blood supply or infection",
    "fibrosis": "scarring of tissue, often from chronic inflammation or injury",
    "cirrhosis": "severe scarring of the liver, usually from long-term damage",
    "pancreatitis": "inflammation of the pancreas",
    "peritonitis": "infection or inflammation of the lining of the abdominal cavity",
    "encephalopathy": "brain dysfunction causing confusion, often from liver failure or other causes",
    "neuropathy": "damage to nerves, often causing numbness, tingling, or weakness",
    "radiculopathy": "pain, numbness, or weakness caused by a pinched nerve root in the spine",
    "myelopathy": "damage to the spinal cord, affecting walking, hand function, and bladder control",
    "dyspnea": "shortness of breath or difficulty breathing",
    "dysphagia": "difficulty swallowing food or liquids",
    "hematuria": "blood in the urine",
    "hemoptysis": "coughing up blood",
    "ascites": "buildup of fluid in the belly, usually from liver disease",
    "syncope": "fainting or temporary loss of consciousness",
    "anemia": "a condition where you don't have enough healthy red blood cells to carry oxygen",
    "thrombocytopenia": "abnormally low platelet count, which increases bleeding risk",
    "hyperglycemia": "abnormally high blood sugar level",
    "hypoglycemia": "dangerously low blood sugar level",
    "hyperkalemia": "dangerously high potassium level in the blood",
    "hyponatremia": "abnormally low sodium level in the blood",
    "osteoarthritis": "wear-and-tear arthritis where joint cartilage breaks down over time",
    "DVT": "deep vein thrombosis, a blood clot in a deep vein, usually in the leg",
    "PE": "pulmonary embolism, a blood clot that travels to the lungs",
    "COPD": "chronic obstructive pulmonary disease, a long-term lung condition that makes breathing difficult",
    "DKA": "diabetic ketoacidosis, a dangerous complication of diabetes with very high blood sugar and acid buildup",
    "STEMI": "ST-elevation myocardial infarction, a type of heart attack where a coronary artery is completely blocked",
    "NSTEMI": "non-ST-elevation myocardial infarction, a type of heart attack where a coronary artery is partially blocked",
    "PCI": "percutaneous coronary intervention, a procedure to open blocked heart arteries using a catheter and stent",
    "CABG": "coronary artery bypass grafting, open-heart surgery to reroute blood around blocked heart arteries",
    "EF": "ejection fraction, the percentage of blood pumped out of the heart with each beat (normal is 55-70%)",
    "BNP": "brain natriuretic peptide, a blood test that measures heart strain (higher means worse heart failure)",
    "INR": "international normalized ratio, a measure of how quickly blood clots (higher means thinner blood)",
    "A1C": "hemoglobin A1C, a blood test that shows average blood sugar over the past 3 months",
    "eGFR": "estimated glomerular filtration rate, a measure of how well the kidneys filter blood (normal is above 90)",
    "creatinine": "a waste product in the blood that rises when kidneys are not working well",
    "BUN": "blood urea nitrogen, another measure of kidney function",
    "troponin": "a protein released when heart muscle is damaged, used to diagnose heart attacks",
    "lactate": "lactic acid in the blood, which rises when tissues are not getting enough oxygen",
    "prognosis": "the likely course and outcome of a disease",
    "palliative": "focused on relieving symptoms and improving quality of life, not curing the disease",
    "benign": "not cancer, not harmful",
    "malignant": "cancerous, able to spread to other parts of the body",
    "metastatic": "cancer that has spread from where it started to other parts of the body",
    "bilateral": "on both sides",
    "unilateral": "on one side only",
    "acute": "sudden onset, short duration",
    "chronic": "long-lasting, ongoing",
    "refractory": "not responding to treatment",
    "prophylaxis": "preventive treatment",
    "empiric": "treatment started based on best guess before test results are available",
    "idiopathic": "of unknown cause",
}


def create_term_pairs() -> list[dict]:
    """Create term-level training pairs."""
    pairs = []
    for term, definition in TERM_DEFINITIONS.items():
        pairs.append({
            "source": term,
            "target": definition,
            "dataset": "term_definition",
            "specialty": "Glossary",
            "granularity": "term",
        })
    return pairs


def main():
    print("Creating multi-granularity aligned training data...")

    # Load existing paragraph-level pairs
    paragraph_pairs = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            p["granularity"] = "paragraph"
            paragraph_pairs.append(p)

    print(f"Paragraph-level pairs: {len(paragraph_pairs)}")

    # Create sentence-level pairs
    sentence_pairs = []
    for pp in paragraph_pairs:
        src_sents = split_into_sentences(pp["source"])
        tgt_sents = split_into_sentences(pp["target"])
        aligned = align_sentences(src_sents, tgt_sents)
        for src, tgt in aligned:
            sentence_pairs.append({
                "source": src,
                "target": tgt,
                "dataset": pp.get("dataset", "unknown") + "_sentence",
                "specialty": pp.get("specialty", "unknown"),
                "granularity": "sentence",
            })

    print(f"Sentence-level pairs: {len(sentence_pairs)}")

    # Create term-level pairs
    term_pairs = create_term_pairs()
    print(f"Term-level pairs: {len(term_pairs)}")

    # Combine all
    all_pairs = paragraph_pairs + sentence_pairs + term_pairs

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for p in all_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"\nTotal aligned pairs: {len(all_pairs)}")
    print(f"  Paragraph: {len(paragraph_pairs)}")
    print(f"  Sentence: {len(sentence_pairs)}")
    print(f"  Term: {len(term_pairs)}")
    print(f"\nSaved to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
