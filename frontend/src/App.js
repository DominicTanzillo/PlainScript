import React, { useState } from 'react';
import './App.css';

const API_BASE = process.env.REACT_APP_API_URL || '';

const EXAMPLE_NOTES = [
  {
    name: "Appendectomy",
    text: "22-year-old male presenting with RLQ pain, nausea, and low-grade fever x12 hours. CT abdomen showed acute appendicitis with no perforation. Underwent uncomplicated laparoscopic appendectomy. EBL less than 10mL. POD0: tolerating clear liquids, pain controlled with IV Tylenol. Discharged POD1 on PO antibiotics x5 days and ibuprofen PRN."
  },
  {
    name: "Gallbladder Surgery",
    text: "Patient underwent laparoscopic cholecystectomy for acute cholecystitis. Intraoperative findings revealed a distended, edematous gallbladder with adhesions to the omentum. Critical view of safety was achieved. EBL minimal. Patient tolerated the procedure well. POD1: afebrile, tolerating PO diet, ambulating independently. Discharged on ibuprofen and oxycodone PRN. Follow-up in 2 weeks."
  },
  {
    name: "Cataract Surgery",
    text: "72-year-old female with visually significant bilateral cataracts. Underwent right phacoemulsification with posterior chamber IOL implant under topical anesthesia. No complications. Visual acuity improved from 20/200 to 20/40 on POD1. Prescribed prednisolone drops QID x4 weeks and moxifloxacin drops QID x1 week. Left eye scheduled in 2 weeks."
  },
  {
    name: "Dog Bite",
    text: "35-year-old male bitten by neighbor's dog on right hand. Two puncture wounds over dorsal hand, no tendon involvement, full ROM intact. X-ray negative for fracture or foreign body. Wound irrigated copiously. Not sutured due to bite wound infection risk. Started on augmentin 875mg BID x7 days. Tetanus booster given. Wound check in 48 hours. Report filed with animal control."
  },
  {
    name: "Abscess I&D",
    text: "35-year-old male with 3cm fluctuant abscess on right buttock x5 days. Incision and drainage performed under local anesthesia. 15mL of purulent material expressed. Wound packed with iodoform gauze. Culture sent. Prescribed TMP-SMX 160/800mg BID x7 days. Packing removal in 48 hours. Daily wound care with repacking. Follow-up in 1 week."
  },
  {
    name: "Chalazion Drainage",
    text: "45-year-old male with persistent right upper eyelid chalazion x3 months, failed warm compresses. Incision and curettage performed under local anesthesia from the inner eyelid approach. Granulomatous material removed. Antibiotic-steroid ointment applied. Warm compresses QID x2 weeks. Follow-up in 2 weeks."
  },
  {
    name: "Dupuytren Release",
    text: "62-year-old male with Dupuytren contracture of right ring finger, unable to fully extend. Needle aponeurotomy performed in office under local anesthesia. Cord disrupted. Finger achieved full extension. Bandaid applied. ROM exercises immediately. May recur. Hand therapy referral. Follow-up in 4 weeks."
  },
  {
    name: "Ear Piercing Infection",
    text: "16-year-old female with infected right ear piercing x5 days. Erythema, swelling, and purulent discharge around earring. No abscess. Earring removed. Wound cleaned with saline. Mupirocin ointment TID x7 days. Warm compresses QID. May re-pierce in 6 weeks after full healing. Return if worsening or fever."
  },
  {
    name: "Tongue Laceration",
    text: "8-year-old male with 1.5cm tongue laceration after biting tongue during fall. Actively bleeding. Repaired with 3 absorbable sutures under local anesthesia. Hemostasis achieved. Soft diet x5 days. Saltwater rinses after meals. Sutures dissolve in 7-10 days. Acetaminophen for pain. Return if excessive bleeding or signs of infection."
  }
];

function App() {
  const [inputText, setInputText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [tooltip, setTooltip] = useState(null);

  const handleSimplify = async () => {
    if (!inputText.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      if (API_BASE && API_BASE.includes('hf.space')) {
        // HuggingFace Space: use Gradio API (SSE call pattern)
        const callResp = await fetch(`${API_BASE}/gradio_api/call/simplify`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ data: [inputText] }),
        });
        if (!callResp.ok) throw new Error('API error: ' + callResp.status);
        const { event_id } = await callResp.json();

        // Read the SSE stream
        const resultResp = await fetch(`${API_BASE}/gradio_api/call/simplify/${event_id}`);
        const reader = resultResp.body.getReader();
        const decoder = new TextDecoder();
        let fullText = '';
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          fullText += decoder.decode(value, { stream: true });
        }

        // Parse SSE: find the "data:" line after "event: complete"
        const sseLines = fullText.split('\n');
        let plainLanguage = '';
        let glossaryMd = '';
        for (let i = 0; i < sseLines.length; i++) {
          if (sseLines[i].startsWith('data:')) {
            try {
              const jsonStr = sseLines[i].replace(/^data:\s*/, '');
              const parsed = JSON.parse(jsonStr);
              if (Array.isArray(parsed)) {
                plainLanguage = parsed[0] || '';
                glossaryMd = parsed[1] || '';
              }
            } catch (e) {
              // skip malformed lines
            }
          }
        }

        // Parse glossary markdown into annotations
        const annotations = [];
        const covered = new Set();
        const blocks = glossaryMd.split('\n\n');
        for (const block of blocks) {
          if (!block.trim()) continue;
          const termMatch = /\*\*(.+?)\*\*\s*--\s*(.+?)(?:\s{2}\n|\s{2}$|\n|$)/.exec(block);
          if (termMatch) {
            const term = termMatch[1];
            const simple = termMatch[2].trim();
            const urlMatch = /\[.*?\]\((.+?)\)/.exec(block);
            const url = urlMatch ? urlMatch[1] : '';
            const summaryMatch = /^>\s*(.+)/m.exec(block);
            const summary = summaryMatch ? summaryMatch[1].trim() : '';
            // Find term in input text with boundary checks
            const inputLower = inputText.toLowerCase();
            const termLower = term.toLowerCase();
            const isShortAbbrev = term.length <= 3 && term === term.toUpperCase();
            let searchFrom = 0;
            let placed = false;
            while (!placed && searchFrom < inputLower.length) {
              const termIdx = inputLower.indexOf(termLower, searchFrom);
              if (termIdx < 0) break;
              const endIdx = termIdx + term.length;
              // Check letter boundaries for short abbreviations (PO, IV, etc.)
              if (isShortAbbrev) {
                const charBefore = termIdx > 0 ? inputText[termIdx - 1] : ' ';
                const charAfter = endIdx < inputText.length ? inputText[endIdx] : ' ';
                if (/[A-Za-z]/.test(charBefore) || /[A-Za-z]/.test(charAfter)) {
                  searchFrom = termIdx + 1;
                  continue;
                }
              }
              // Skip if overlaps with covered positions
              let overlaps = false;
              for (let p = termIdx; p < endIdx; p++) {
                if (covered.has(p)) { overlaps = true; break; }
              }
              if (overlaps) {
                searchFrom = termIdx + 1;
                continue;
              }
              for (let p = termIdx; p < endIdx; p++) covered.add(p);
              annotations.push({
                term: inputText.slice(termIdx, endIdx),
                simple, start: termIdx, end: endIdx,
                url, medlineplus_summary: summary,
              });
              placed = true;
            }
          }
        }
        annotations.sort((a, b) => a.start - b.start);

        setResult({
          input: inputText,
          plain_language: plainLanguage || 'Model is loading, please try again in a moment...',
          source_annotations: annotations,
          output_annotations: [],
        });
      } else {
        // Local Flask server
        const response = await fetch(`${API_BASE}/api/simplify`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: inputText }),
        });
        if (!response.ok) throw new Error('API error: ' + response.status);
        const data = await response.json();
        setResult(data);
      }
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const renderAnnotatedText = (text, annotations) => {
    if (!annotations || annotations.length === 0) return <p className="result-text">{text}</p>;
    const elements = [];
    let lastEnd = 0;

    annotations.forEach((ann, idx) => {
      if (ann.start > lastEnd) {
        elements.push(<span key={'t' + idx}>{text.slice(lastEnd, ann.start)}</span>);
      }
      elements.push(
        <span key={'a' + idx} className="medical-term"
          onMouseEnter={(e) => {
            const rect = e.target.getBoundingClientRect();
            setTooltip({ ...ann, x: rect.left, y: rect.bottom + window.scrollY });
          }}
          onMouseLeave={() => setTooltip(null)}>
          {ann.url ? (
            <a href={ann.url} target="_blank" rel="noopener noreferrer" className="term-link">
              {text.slice(ann.start, ann.end)}
            </a>
          ) : (
            <span className="term-link term-no-link">{text.slice(ann.start, ann.end)}</span>
          )}
        </span>
      );
      lastEnd = ann.end;
    });
    if (lastEnd < text.length) elements.push(<span key="end">{text.slice(lastEnd)}</span>);
    return <p className="annotated-text">{elements}</p>;
  };

  return (
    <div className="app">
      <header className="header">
        <h1 className="logo">MedClear</h1>
        <p className="tagline">Doctor-Speak to Human-Speak</p>
        <p className="subtitle">
          Paste a clinical note below and get a plain-language version your family can understand.
          Medical terms are defined and linked to <a href="https://medlineplus.gov" target="_blank" rel="noopener noreferrer">MedlinePlus</a> (NIH) where available.
        </p>
      </header>

      <section className="input-section">
        <label htmlFor="clinical-input" className="input-label">Clinical Note</label>
        <textarea
          id="clinical-input"
          value={inputText}
          onChange={e => setInputText(e.target.value)}
          placeholder="Paste a clinical note, discharge summary, or post-op description here..."
          className="input-textarea"
        />
        <div className="controls">
          <button
            onClick={handleSimplify}
            disabled={loading || !inputText.trim()}
            className="simplify-btn"
          >
            {loading ? (
              <><span className="spinner" /> Simplifying...</>
            ) : (
              'Simplify for Patient'
            )}
          </button>
          <div className="examples">
            <span className="examples-label">Try an example:</span>
            {EXAMPLE_NOTES.map((ex, i) => (
              <button key={i} onClick={() => setInputText(ex.text)} className="example-btn">
                {ex.name}
              </button>
            ))}
          </div>
        </div>
      </section>

      {error && <div className="error-banner">{error}</div>}

      {result && (
        <section className="results">
          <div className="result-card original">
            <h3>Original Note</h3>
            <p className="card-hint">Hover underlined terms for definitions. Click to open MedlinePlus.</p>
            {renderAnnotatedText(result.input, result.source_annotations)}
          </div>

          <div className="result-card simplified">
            <h3>Plain Language Version</h3>
            <p className="result-text">{result.plain_language}</p>
          </div>

          {result.source_annotations && result.source_annotations.length > 0 && (
            <div className="result-card glossary">
              <h3>Medical Term Glossary</h3>
              <div className="glossary-grid">
                {result.source_annotations.map((ann, i) => (
                  <div key={i} className="glossary-item">
                    {ann.url ? (
                      <a href={ann.url} target="_blank" rel="noopener noreferrer" className="glossary-term">
                        {ann.term}
                      </a>
                    ) : (
                      <span className="glossary-term">{ann.term}</span>
                    )}
                    <span className="glossary-def">{ann.simple}</span>
                    {ann.medlineplus_summary && (
                      <p className="glossary-summary">{ann.medlineplus_summary}</p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>
      )}

      {tooltip && (
        <div className="tooltip" style={{ left: tooltip.x, top: tooltip.y + 8 }}>
          <strong>{tooltip.term}</strong>: {tooltip.simple}
          {tooltip.medlineplus_summary && (
            <p className="tooltip-detail">{tooltip.medlineplus_summary}</p>
          )}
        </div>
      )}

      <footer className="footer">
        <p>Powered by FLAN-T5 + MedlinePlus (NIH/NLM)</p>
        <p className="disclaimer">This is an AI assistant, not medical advice. Always consult your doctor.</p>
      </footer>
    </div>
  );
}

export default App;
