"""Extra phrase-level pairs for V2 training. These are the MOST valuable
training examples — they teach the exact clinical shorthand → plain English
mappings that patients encounter on discharge papers."""

import json

OUTPUT = "./medclear_results/training_v2/extra_phrases.jsonl"

PHRASE_PAIRS = [
    # === POST-OP STATUS ===
    ("s/p laparoscopic cholecystectomy", "after gallbladder removal surgery through small incisions"),
    ("s/p CABG x4", "after quadruple heart bypass surgery"),
    ("s/p TKA", "after total knee replacement surgery"),
    ("s/p THA", "after total hip replacement surgery"),
    ("s/p appendectomy", "after appendix removal surgery"),
    ("s/p C-section", "after a cesarean delivery"),
    ("s/p ORIF", "after surgery to realign and fix a broken bone with metal hardware"),
    ("s/p AVR with bioprosthetic valve", "after aortic valve replacement with a tissue valve"),
    ("s/p PCI with DES to LAD", "after placing a drug-coated stent in the main heart artery"),
    ("s/p exploratory laparotomy", "after emergency surgery to open and examine the abdomen"),

    # === VITAL SIGN DESCRIPTIONS ===
    ("VS stable, afebrile", "vital signs are normal, no fever"),
    ("T 101.4, HR 110, BP 90/60, RR 24, SpO2 92% on RA",
     "temperature 101.4 (fever), heart rate 110 (fast), blood pressure 90/60 (low), breathing rate 24 (fast), oxygen 92% on room air (low)"),
    ("hemodynamically stable on no vasopressors",
     "blood pressure is stable without needing any blood pressure support medications"),
    ("HR 45, symptomatic bradycardia",
     "heart rate is dangerously slow at 45 and causing symptoms"),
    ("BP 188/110, hypertensive urgency",
     "blood pressure is dangerously high at 188/110 and needs urgent treatment"),
    ("SpO2 85% on 6L NC, escalated to high-flow",
     "oxygen level was only 85% despite 6 liters of oxygen through a nasal tube, so a stronger oxygen delivery system was started"),
    ("GCS 7, intubated for airway protection",
     "consciousness level was very low (7 out of 15), so a breathing tube was placed to protect the airway"),

    # === LAB RESULT INTERPRETATIONS ===
    ("Hgb 6.8, transfused 2U PRBCs",
     "hemoglobin was critically low at 6.8, so 2 units of donated red blood cells were given"),
    ("WBC 22K with left shift",
     "white blood cells are very elevated at 22,000 with immature cells, indicating active infection"),
    ("plt 42K, transfusion threshold",
     "platelet count is dangerously low at 42,000 (bleeding risk), approaching the level where a transfusion is needed"),
    ("Cr 3.4, up from baseline 1.0, consistent with AKI",
     "creatinine jumped from normal (1.0) to 3.4, meaning the kidneys have suddenly stopped working properly (acute kidney injury)"),
    ("troponin trending 0.8 -> 2.4 -> 4.1",
     "the heart damage marker is rising (0.8 to 2.4 to 4.1), confirming ongoing heart muscle injury"),
    ("BNP 3200, consistent with decompensated CHF",
     "heart strain marker is very high at 3200, confirming the heart failure has gotten much worse"),
    ("INR 4.8, supratherapeutic, hold warfarin",
     "blood is too thin (INR 4.8, above target), stop the blood thinner warfarin until it comes down"),
    ("lactate 5.2, suggesting tissue hypoperfusion",
     "lactic acid is high at 5.2, meaning the body's tissues are not getting enough blood flow"),
    ("pH 7.18, bicarb 8, AG 28, consistent with severe metabolic acidosis",
     "the blood is dangerously acidic (pH 7.18), with very low bicarbonate and a high anion gap, indicating a severe acid imbalance"),
    ("LFTs grossly elevated: AST 1200, ALT 980",
     "liver enzymes are extremely elevated (AST 1200, ALT 980), indicating severe liver damage"),
    ("TSH <0.01, free T4 5.8, consistent with hyperthyroidism",
     "thyroid is extremely overactive (TSH almost zero, thyroid hormone very high)"),
    ("A1C 11.4%, significantly above goal",
     "3-month blood sugar average is 11.4%, much higher than the goal of under 7%, meaning diabetes is poorly controlled"),
    ("urine tox screen positive for opioids and benzodiazepines",
     "drug test showed opioid pain medications and anti-anxiety medications in the system"),
    ("blood cultures 2/2 positive for MSSA",
     "both blood culture samples grew staph bacteria, confirming a bloodstream infection"),

    # === IMAGING FINDINGS ===
    ("CXR: bilateral pulmonary edema with cephalization",
     "chest X-ray shows fluid in both lungs with blood being redirected upward, signs of heart failure"),
    ("CXR: RLL consolidation with air bronchograms",
     "chest X-ray shows a solid-appearing area in the right lower lung with visible airways, consistent with pneumonia"),
    ("CT head: no acute intracranial hemorrhage or mass effect",
     "brain CT scan shows no bleeding or swelling in the brain"),
    ("CT abdomen: free air under diaphragm",
     "belly CT shows air outside the intestines, meaning something has perforated (a hole in the bowel)"),
    ("MRI brain: acute left MCA territory infarct",
     "brain MRI shows a new stroke on the left side of the brain"),
    ("CT PE study: bilateral pulmonary emboli",
     "CT scan shows blood clots in the arteries of both lungs"),
    ("US abdomen: gallstones with wall thickening and pericholecystic fluid",
     "ultrasound shows gallstones with an inflamed, swollen gallbladder and surrounding fluid"),
    ("echo: EF 25%, severe global hypokinesis, moderate MR",
     "heart ultrasound shows the heart is pumping at only 25% (normal is over 55%), with all walls moving weakly, and the mitral valve is leaking moderately"),
    ("DEXA: T-score -2.8 lumbar spine",
     "bone density scan shows osteoporosis (weakened bones) in the lower spine"),

    # === MEDICATION INSTRUCTIONS ===
    ("continue home medications",
     "keep taking all your regular medications as before"),
    ("resume home medications except hold warfarin pending INR",
     "restart all your regular medications, but do NOT take warfarin until your blood clotting level is checked"),
    ("taper prednisone: 40mg x5d, 30mg x5d, 20mg x5d, 10mg x5d, then stop",
     "gradually reduce prednisone over 20 days: 40mg for 5 days, then 30mg for 5 days, then 20mg for 5 days, then 10mg for 5 days, then stop"),
    ("Tylenol 650mg PO q6h PRN pain, max 3g/day",
     "take 650mg of Tylenol by mouth every 6 hours as needed for pain, do not exceed 3 grams (about 4-5 doses) per day"),
    ("oxycodone 5mg PO q4-6h PRN breakthrough pain",
     "take oxycodone 5mg by mouth every 4-6 hours as needed for pain that is not controlled by other medications"),
    ("enoxaparin 40mg SQ daily x14 days for DVT prophylaxis",
     "inject enoxaparin 40mg under the skin once daily for 14 days to prevent blood clots"),
    ("ASA 81mg daily indefinitely, DO NOT STOP without consulting cardiology",
     "take baby aspirin (81mg) every day for the rest of your life, NEVER stop taking it without talking to your heart doctor first"),
    ("dual antiplatelet therapy: ASA + ticagrelor x12 months",
     "take two blood-thinning medications together (aspirin plus ticagrelor) for 12 months to prevent stent clots"),
    ("sliding scale insulin with meals",
     "insulin doses adjusted before each meal based on your blood sugar reading at that time"),
    ("NPH insulin 20 units QHS",
     "take 20 units of NPH insulin at bedtime"),

    # === DISCHARGE INSTRUCTIONS ===
    ("activity as tolerated",
     "be as active as you feel comfortable being"),
    ("no heavy lifting >10 lbs x6 weeks",
     "do not lift anything heavier than 10 pounds for 6 weeks"),
    ("sternal precautions x8 weeks: no pushing, pulling, or lifting >5 lbs",
     "protect your breastbone for 8 weeks: do not push, pull, or lift anything over 5 pounds"),
    ("posterior hip precautions x6 weeks",
     "for 6 weeks: do not cross your legs, do not bend your hip past 90 degrees, do not twist your body"),
    ("NWB LLE x6 weeks",
     "do not put any weight on the left leg for 6 weeks"),
    ("WBAT with FWW",
     "put weight on the leg as much as comfortable, using a front-wheeled walker"),
    ("toe-touch weight bearing",
     "only touch your toes to the ground for balance while using crutches, do not put full weight on the leg"),
    ("advance diet from clears to regular as tolerated",
     "start with clear liquids, then work up to regular food as your stomach allows"),
    ("soft diet x2 weeks",
     "eat only soft foods for 2 weeks (mashed potatoes, soup, yogurt, etc.)"),
    ("fluid restriction 1.5L/day",
     "limit total fluid intake (including water, coffee, soup) to 1.5 liters (about 6 cups) per day"),
    ("low sodium diet <2g/day",
     "limit salt intake to less than 2 grams per day (read food labels, avoid processed foods)"),
    ("daily weights, call if gain >3 lbs in 1 day or >5 lbs in 1 week",
     "weigh yourself every morning and call the doctor if you gain more than 3 pounds in one day or 5 pounds in one week (this means fluid is building up)"),
    ("incentive spirometry 10 reps q1h while awake",
     "use the plastic breathing exercise device 10 times every hour while you are awake to keep your lungs expanded"),
    ("wound care: keep incision clean and dry x48h, then may shower",
     "keep the surgical cut clean and dry for 48 hours, after that you may shower but do not soak in a bath"),
    ("call 911 for chest pain, sudden SOB, weakness on one side, or severe bleeding",
     "call 911 immediately if you have: chest pain, sudden trouble breathing, weakness or numbness on one side of your body, or uncontrolled bleeding"),

    # === FOLLOW-UP INSTRUCTIONS ===
    ("f/u with PCP in 1 week",
     "follow up with your primary care doctor in 1 week"),
    ("f/u with surgery in 2 weeks for staple removal",
     "come back to the surgeon's office in 2 weeks to have staples removed"),
    ("f/u with cardiology post-discharge for echo",
     "see your heart doctor after leaving the hospital for a follow-up heart ultrasound"),
    ("cardiac rehab referral placed",
     "a referral was made for a supervised exercise program to help your heart recover"),
    ("VNA/home health for wound care and vitals",
     "a visiting nurse will come to your home to check on your wound and vital signs"),
    ("outpatient PT/OT 3x/week",
     "physical therapy and occupational therapy 3 times per week at an outpatient clinic"),
    ("return to ED if: fever >101.5, wound redness/drainage, uncontrolled pain, or inability to eat/drink",
     "go to the emergency room if you develop: fever above 101.5, redness or fluid from the wound, pain that medications cannot control, or you cannot eat or drink"),
]

pairs = []
for medical, plain in PHRASE_PAIRS:
    pairs.append({
        "input_text": f"simplify: {medical}",
        "target": plain,
        "level": "phrase",
    })
    # Also reverse format for variety
    pairs.append({
        "input_text": f"what does this mean: {medical}",
        "target": plain,
        "level": "phrase",
    })

with open(OUTPUT, "w", encoding="utf-8") as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")

print(f"Generated {len(pairs)} phrase pairs from {len(PHRASE_PAIRS)} base pairs")
print(f"Saved to {OUTPUT}")
