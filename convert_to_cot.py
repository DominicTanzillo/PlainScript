"""
Convert existing synthetic pairs to Chain-of-Thought (CoT) format.
Extracts key facts from source text, then appends plain language.

New target format:
    FACTS:
    - Patient: 67-year-old male
    - Condition: acute cholecystitis
    - Procedure: laparoscopic cholecystectomy
    ...

    PLAIN LANGUAGE:
    This 67-year-old man had surgery to remove his gallbladder...
"""

import json
import re
import os

INPUT_FILE = "./medclear_results/synthetic_data/claude_generated_pairs.jsonl"
OUTPUT_FILE = "./medclear_results/synthetic_data/claude_generated_pairs_cot.jsonl"


def extract_facts_from_source(source: str) -> list[str]:
    """Rule-based extraction of key facts from clinical text."""
    facts = []

    # Patient demographics
    age_match = re.search(
        r'(\d{1,3})[-\s]?(?:year[-\s]?old|y/?o|yo)\s*'
        r'(male|female|man|woman|boy|girl|gentleman|lady)?',
        source, re.IGNORECASE)
    if age_match:
        age = age_match.group(1)
        sex = age_match.group(2) or ""
        facts.append(f"Patient: {age}-year-old {sex}".strip())

    # Diagnoses / conditions
    dx_patterns = [
        r'(?:DIAGNOS[EI]S?|DIAGNOSIS|ASSESSMENT|IMPRESSION)[:\s,]+(.+?)(?:\.|$)',
        r'(?:PREOPERATIVE DIAGNOS[EI]S?|ADMITTING DIAGNOS[EI]S?)[:\s,]+(.+?)(?:\.|$)',
        r'(?:diagnosed with|diagnosis of|presenting with)\s+(.+?)(?:\.|,|$)',
    ]
    for pat in dx_patterns:
        m = re.search(pat, source, re.IGNORECASE)
        if m:
            dx_text = m.group(1).strip()[:120]
            if len(dx_text) > 10:
                facts.append(f"Condition: {dx_text}")
                break

    # Procedure
    proc_patterns = [
        r'(?:PROCEDURE[S]?\s*(?:PERFORMED)?|OPERATION[S]?\s*(?:PERFORMED)?)[:\s,]+(.+?)(?:\.|$)',
        r'(?:underwent|s/p|status post)\s+(.+?)(?:\.|,|$)',
    ]
    for pat in proc_patterns:
        m = re.search(pat, source, re.IGNORECASE)
        if m:
            proc = m.group(1).strip()[:120]
            if len(proc) > 5:
                facts.append(f"Procedure: {proc}")
                break

    # Key findings
    finding_patterns = [
        r'(?:FINDINGS?|INTRAOPERATIVE FINDINGS?)[:\s,]+(.+?)(?:\.|$)',
        r'(?:revealed|showed|demonstrated|confirmed)\s+(.+?)(?:\.|,|$)',
    ]
    for pat in finding_patterns:
        m = re.search(pat, source, re.IGNORECASE)
        if m:
            finding = m.group(1).strip()[:120]
            if len(finding) > 10:
                facts.append(f"Findings: {finding}")
                break

    # Medications
    med_patterns = [
        r'(?:MEDICATIONS?|DISCHARGE MED|PLAN)[:\s,].*?'
        r'((?:aspirin|metformin|lisinopril|atorvastatin|metoprolol|oxycodone|'
        r'ibuprofen|acetaminophen|warfarin|heparin|ceftriaxone|vancomycin|'
        r'prednisone|insulin|gabapentin|omeprazole|furosemide|amoxicillin|'
        r'ticagrelor|clopidogrel|enoxaparin|levothyroxine|duloxetine|'
        r'hydrochlorothiazide|amlodipine|carvedilol|spironolactone)'
        r'[^.]*)',
    ]
    for pat in med_patterns:
        m = re.search(pat, source, re.IGNORECASE)
        if m:
            meds = m.group(1).strip()[:150]
            facts.append(f"Medications: {meds}")
            break

    # Outcome / status
    outcome_patterns = [
        r'(?:CONDITION|STATUS|OUTCOME)[:\s]+(.+?)(?:\.|$)',
        r'(?:tolerated .+? well|stable condition|discharged|improved)',
    ]
    for pat in outcome_patterns:
        m = re.search(pat, source, re.IGNORECASE)
        if m:
            outcome = m.group(0).strip()[:80] if m.lastindex is None else m.group(1).strip()[:80]
            facts.append(f"Outcome: {outcome}")
            break

    # Follow-up
    fu_match = re.search(
        r'(?:follow[- ]?up|f/u|return)[:\s]+(.+?)(?:\.|$)',
        source, re.IGNORECASE)
    if fu_match:
        fu = fu_match.group(1).strip()[:80]
        facts.append(f"Follow-up: {fu}")

    # Blood loss
    ebl_match = re.search(r'(?:EBL|blood loss)[:\s]*(?:approximately?\s*)?(\d+\s*m?L)',
                           source, re.IGNORECASE)
    if ebl_match:
        facts.append(f"Blood loss: {ebl_match.group(1)}")

    # Complications
    comp_match = re.search(r'COMPLICATIONS?[:\s]+(.+?)(?:\.|$)',
                            source, re.IGNORECASE)
    if comp_match:
        comp = comp_match.group(1).strip()[:60]
        facts.append(f"Complications: {comp}")

    # If we got very few facts, add a generic summary from the source
    if len(facts) < 3:
        # Take the first meaningful sentence
        sentences = source.split('.')
        for s in sentences[:3]:
            s = s.strip()
            if len(s) > 20 and s not in str(facts):
                facts.append(f"Summary: {s[:120]}")
                break

    return facts[:8]  # Cap at 8 facts


def convert_pair_to_cot(pair: dict) -> dict:
    """Convert a source/target pair to CoT format."""
    source = pair["source"]
    target = pair["target"]

    # Extract facts from source
    facts = extract_facts_from_source(source)

    if not facts:
        # Fallback: keep original format
        return pair

    # Build new target
    facts_section = "FACTS:\n" + "\n".join(f"- {f}" for f in facts)
    new_target = f"{facts_section}\n\nPLAIN LANGUAGE:\n{target}"

    return {
        "source": source,
        "target": new_target,
        "dataset": pair.get("dataset", "unknown"),
        "specialty": pair.get("specialty", "unknown"),
    }


def main():
    print("Converting synthetic pairs to CoT format...")

    pairs = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            pairs.append(json.loads(line))

    print(f"Input pairs: {len(pairs)}")

    converted = []
    fact_counts = []
    for p in pairs:
        c = convert_pair_to_cot(p)
        converted.append(c)
        # Count facts
        if "FACTS:" in c["target"]:
            n_facts = c["target"].split("PLAIN LANGUAGE:")[0].count("- ")
            fact_counts.append(n_facts)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for c in converted:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Output pairs: {len(converted)}")
    print(f"Pairs with facts: {len(fact_counts)}")
    if fact_counts:
        print(f"Avg facts per pair: {sum(fact_counts)/len(fact_counts):.1f}")
        print(f"Min/Max facts: {min(fact_counts)}/{max(fact_counts)}")

    # Show a sample
    print("\n--- SAMPLE ---")
    sample = converted[0]
    print(f"SOURCE: {sample['source'][:200]}...")
    print(f"\nTARGET:\n{sample['target'][:500]}...")

    # Check token lengths
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
