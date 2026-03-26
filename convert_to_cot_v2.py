"""
Convert synthetic pairs to CoT format V2.
Uses the TARGET text (which is already well-written by Claude) to extract facts,
since the target already contains the key information in clear language.

Strategy: Parse the plain-language target to extract structured facts,
then prepend them to the target.
"""

import json
import re
import os

INPUT_FILE = "./medclear_results/synthetic_data/claude_generated_pairs.jsonl"
OUTPUT_FILE = "./medclear_results/synthetic_data/claude_generated_pairs_cot.jsonl"


def extract_facts_from_target(source: str, target: str) -> list[str]:
    """Extract key facts by analyzing both source and target text."""
    facts = []
    combined = source + " " + target

    # 1. Patient demographics (from target, which is cleaner)
    age_patterns = [
        r'(?:This|The)\s+(\d{1,3})[-\s]?year[-\s]?old\s+(man|woman|male|female|boy|girl|baby\s+\w+|child|patient|mother|infant)',
        r'(\d{1,3})[-\s]?year[-\s]?old\s+(man|woman|male|female|boy|girl)',
    ]
    for pat in age_patterns:
        m = re.search(pat, target, re.IGNORECASE)
        if m:
            facts.append(f"Patient: {m.group(1)}-year-old {m.group(2)}")
            break

    # 2. Diagnosis/Condition - look for key medical conditions in target
    condition_keywords = [
        (r'(?:diagnosed with|has|have|had)\s+([\w\s\-]+(?:cancer|carcinoma|lymphoma|sarcoma|melanoma|tumor))', 'Cancer'),
        (r'(?:diagnosed with|has|had)\s+([\w\s\-]+(?:disease|syndrome|disorder|deficiency|failure))', 'Condition'),
        (r'(?:diagnosed with|confirmed|found to have)\s+([^,.]+)', 'Diagnosis'),
        (r'(heart attack|stroke|pneumonia|appendicitis|cholecystitis|pancreatitis|sepsis|DKA|diabetic ketoacidosis)', 'Condition'),
        (r'(type [12] diabetes|hypertension|COPD|asthma|heart failure|atrial fibrillation|epilepsy)', 'Condition'),
        (r'(fracture|hernia|tear|rupture|obstruction|stenosis|occlusion)\s+(?:of\s+)?(?:the\s+)?([^,.]+)', 'Injury'),
    ]
    for pat, label in condition_keywords:
        m = re.search(pat, target, re.IGNORECASE)
        if m:
            condition = m.group(0).strip()[:100]
            # Clean up
            condition = re.sub(r'^(?:diagnosed with|has|had|confirmed|found to have)\s+', '', condition, flags=re.IGNORECASE)
            facts.append(f"Condition: {condition}")
            break

    # 3. Procedure/Surgery
    proc_patterns = [
        r'(?:had|underwent|received|performed)\s+(?:a\s+|an\s+)?((?:surgery|procedure|operation)(?:\s+\w+){0,5})',
        r'(?:had|underwent|received)\s+(?:a\s+|an\s+)?([\w\s\-]+(?:ectomy|otomy|plasty|oscopy|scopy|orrhaphy|tion|graft|replacement|repair|fusion|biopsy|drainage|ablation|transplant|implant))',
        r'(?:had|underwent)\s+(?:a\s+|an\s+)?([^,.]{10,60})',
    ]
    for pat in proc_patterns:
        m = re.search(pat, target, re.IGNORECASE)
        if m:
            proc = m.group(1).strip()[:100]
            if len(proc) > 5:
                facts.append(f"Procedure: {proc}")
                break

    # 4. Key findings from target
    finding_patterns = [
        r'(?:found|showed|revealed|confirmed|discovered)\s+(?:that\s+)?([^,.]{15,100})',
        r'(?:scan|X-ray|MRI|CT|ultrasound|testing|biopsy|blood test)\s+(?:showed|revealed|confirmed|found)\s+([^,.]+)',
    ]
    for pat in finding_patterns:
        m = re.search(pat, target, re.IGNORECASE)
        if m:
            finding = m.group(1).strip()[:100]
            if len(finding) > 10 and finding not in str(facts):
                facts.append(f"Findings: {finding}")
                break

    # 5. Medications mentioned in target
    med_names = re.findall(
        r'\b(aspirin|metformin|lisinopril|atorvastatin|metoprolol|oxycodone|'
        r'ibuprofen|acetaminophen|warfarin|heparin|ceftriaxone|vancomycin|'
        r'prednisone|prednisolone|insulin|gabapentin|furosemide|amoxicillin|'
        r'ticagrelor|clopidogrel|enoxaparin|levothyroxine|duloxetine|'
        r'amlodipine|carvedilol|spironolactone|omeprazole|morphine|'
        r'hydromorphone|fentanyl|propranolol|methimazole|allopurinol|'
        r'colchicine|rituximab|pembrolizumab|methotrexate|cyclosporine|'
        r'tacrolimus|mycophenolate|buprenorphine|naloxone|epinephrine|'
        r'dexamethasone|methylprednisolone|nimodipine|levetiracetam|'
        r'risperidone|fluoxetine|sertraline|ketamine|adalimumab|'
        r'sacubitril|valsartan|dapagliflozin|empagliflozin|ocrelizumab|'
        r'nitrofurantoin|ciprofloxacin|azithromycin|piperacillin|meropenem)\b',
        target, re.IGNORECASE)
    if med_names:
        unique_meds = list(dict.fromkeys(m.lower() for m in med_names))[:5]
        facts.append(f"Medications: {', '.join(unique_meds)}")

    # 6. Outcome
    outcome_patterns = [
        r'(?:went home|discharged|sent home|recovery|recovered|improving|improved|stable|doing well|tolerated .+ well)',
        r'(?:no complications|uncomplicated|successful)',
        r'(?:died|passed away|comfort care|hospice|palliative)',
    ]
    for pat in outcome_patterns:
        m = re.search(pat, target, re.IGNORECASE)
        if m:
            facts.append(f"Outcome: {m.group(0).strip()[:60]}")
            break

    # 7. Follow-up
    fu_patterns = [
        r'(?:follow[- ]?up|return|come back|appointment)\s+(?:in\s+|at\s+)?(\d+\s+(?:days?|weeks?|months?))',
        r'(?:follow[- ]?up|return|see)\s+(?:the\s+|her\s+|his\s+)?(?:doctor|surgeon|specialist|cardiologist|neurologist)',
    ]
    for pat in fu_patterns:
        m = re.search(pat, target, re.IGNORECASE)
        if m:
            facts.append(f"Follow-up: {m.group(0).strip()[:60]}")
            break

    # 8. Key numbers (blood pressure, lab values, etc.)
    number_patterns = [
        r'(?:blood pressure|BP)\s+(?:was\s+|of\s+)?(\d+/\d+)',
        r'(?:A1C|HbA1c)\s+(?:of\s+|was\s+|is\s+)?(\d+\.?\d*%?)',
        r'(?:ejection fraction|EF)\s+(?:of\s+|was\s+|is\s+|at\s+)?(\d+%?)',
        r'(?:hemoglobin|Hgb)\s+(?:of\s+|was\s+|is\s+)?(\d+\.?\d*)',
        r'(?:creatinine|Cr)\s+(?:of\s+|was\s+|is\s+)?(\d+\.?\d*)',
    ]
    for pat in number_patterns:
        m = re.search(pat, target, re.IGNORECASE)
        if m:
            facts.append(f"Key value: {m.group(0).strip()[:60]}")
            break

    # Deduplicate
    seen = set()
    unique_facts = []
    for f in facts:
        key = f.split(":")[0]
        if key not in seen:
            seen.add(key)
            unique_facts.append(f)

    return unique_facts[:8]


def convert_pair(pair: dict) -> dict:
    """Convert a pair to CoT format."""
    source = pair["source"]
    target = pair["target"]

    facts = extract_facts_from_target(source, target)

    if len(facts) < 2:
        # Not enough facts extracted; keep original format
        return pair

    facts_section = "FACTS:\n" + "\n".join(f"- {f}" for f in facts)
    new_target = f"{facts_section}\n\nPLAIN LANGUAGE:\n{target}"

    return {
        "source": source,
        "target": new_target,
        "dataset": pair.get("dataset", "unknown"),
        "specialty": pair.get("specialty", "unknown"),
    }


def main():
    print("Converting to CoT format V2 (target-based extraction)...")

    pairs = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))

    print(f"Input pairs: {len(pairs)}")

    converted = []
    cot_count = 0
    fact_counts = []

    for p in pairs:
        c = convert_pair(p)
        converted.append(c)
        if "FACTS:" in c["target"]:
            cot_count += 1
            n_facts = c["target"].split("PLAIN LANGUAGE:")[0].count("- ")
            fact_counts.append(n_facts)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for c in converted:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Output pairs: {len(converted)}")
    print(f"Pairs with CoT format: {cot_count}/{len(converted)}")
    print(f"Pairs without (kept original): {len(converted) - cot_count}")
    if fact_counts:
        print(f"Avg facts per CoT pair: {sum(fact_counts)/len(fact_counts):.1f}")
        print(f"Min/Max facts: {min(fact_counts)}/{max(fact_counts)}")

    # Show samples
    for i in [0, 100, 300, 500]:
        if i < len(converted) and "FACTS:" in converted[i]["target"]:
            print(f"\n--- Sample {i} ({converted[i].get('specialty', '?')}) ---")
            facts_part = converted[i]["target"].split("PLAIN LANGUAGE:")[0]
            print(facts_part.strip()[:400])

    # Token lengths
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
    tgt_lengths = [len(tokenizer.encode(c["target"])) for c in converted]
    print(f"\nTarget token lengths:")
    print(f"  Mean: {sum(tgt_lengths)/len(tgt_lengths):.0f}")
    print(f"  Max: {max(tgt_lengths)}")
    print(f"  >768: {sum(1 for l in tgt_lengths if l > 768)}")
    print(f"  >512: {sum(1 for l in tgt_lengths if l > 512)}")


if __name__ == "__main__":
    main()
