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
        // HuggingFace Space: use Gradio API
        const callResp = await fetch(`${API_BASE}/gradio_api/call/simplify`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ data: [inputText] }),
        });
        if (!callResp.ok) throw new Error('API error: ' + callResp.status);
        const { event_id } = await callResp.json();

        const resultResp = await fetch(`${API_BASE}/gradio_api/call/simplify/${event_id}`);
        const text = await resultResp.text();
        const dataLine = text.split('\n').find(l => l.startsWith('data:'));
        if (!dataLine) throw new Error('No result from API');
        const jsonStr = dataLine.replace(/^data:\s*/, '');
        const parsed = JSON.parse(jsonStr);
        const plainLanguage = Array.isArray(parsed) ? parsed[0] : parsed;

        setResult({
          input: inputText,
          plain_language: plainLanguage || 'No output generated',
          source_annotations: [],
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
          <a href={ann.url} target="_blank" rel="noopener noreferrer" className="term-link">
            {text.slice(ann.start, ann.end)}
          </a>
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
          Every medical term is defined and linked to <a href="https://medlineplus.gov" target="_blank" rel="noopener noreferrer">MedlinePlus</a> (NIH).
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
                    <a href={ann.url} target="_blank" rel="noopener noreferrer" className="glossary-term">
                      {ann.term}
                    </a>
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
