import React, { useState } from 'react';

function App() {
  const [inputText, setInputText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [tooltip, setTooltip] = useState(null);

  const exampleNotes = [
    {
      name: "Post-Op Cholecystectomy",
      text: "Patient underwent laparoscopic cholecystectomy for acute cholecystitis. Intraoperative findings revealed a distended, edematous gallbladder with adhesions to the omentum. Critical view of safety was achieved. EBL minimal. Patient tolerated the procedure well. POD1: afebrile, tolerating PO diet, ambulating independently. Discharged on ibuprofen and oxycodone PRN. Follow-up in 2 weeks."
    },
    {
      name: "Cardiac Catheterization",
      text: "68-year-old male with NSTEMI. Left heart catheterization with PCI to LAD. Angiography revealed 95% stenosis of proximal LAD. Successful DES placement with TIMI 3 flow. Echo showed EF 45% with anterior wall hypokinesis. Discharge medications: Aspirin 81mg daily, Ticagrelor 90mg BID x12 months, Metoprolol 50mg daily, Atorvastatin 80mg daily."
    },
    {
      name: "Total Knee Replacement",
      text: "71-year-old female with severe tricompartmental osteoarthritis right knee. Right total knee arthroplasty under spinal anesthesia. Cemented posterior-stabilized implant. EBL 250mL. DVT prophylaxis enoxaparin 40mg SQ daily. PT initiated POD0, ambulating 150 feet with walker. ROM 0-90 degrees."
    }
  ];

  const handleSimplify = async () => {
    if (!inputText.trim()) return;
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('/api/simplify', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: inputText }),
      });
      if (!response.ok) throw new Error('API error: ' + response.status);
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const renderAnnotatedText = (text, annotations) => {
    if (!annotations || annotations.length === 0) return <p>{text}</p>;
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
          <span className="term-badge">{ann.simple}</span>
        </span>
      );
      lastEnd = ann.end;
    });
    if (lastEnd < text.length) elements.push(<span key="end">{text.slice(lastEnd)}</span>);
    return <p className="annotated-text">{elements}</p>;
  };

  return (
    <div style={{ maxWidth: 900, margin: '0 auto', padding: 20, fontFamily: 'system-ui, sans-serif' }}>
      <header style={{ textAlign: 'center', marginBottom: 30 }}>
        <h1 style={{ color: '#1a73e8', fontSize: 36 }}>MedClear</h1>
        <p style={{ color: '#666' }}>Medical Text Simplification with AI + MedlinePlus</p>
      </header>

      <section style={{ marginBottom: 20 }}>
        <h2>Clinical Note</h2>
        <textarea value={inputText} onChange={e => setInputText(e.target.value)}
          placeholder="Paste a clinical note, discharge summary, or post-op description here..."
          style={{ width: '100%', minHeight: 150, padding: 12, fontSize: 14, border: '2px solid #ddd', borderRadius: 8 }} />
        <div style={{ marginTop: 10, display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
          <button onClick={handleSimplify} disabled={loading || !inputText.trim()}
            style={{ padding: '10px 24px', fontSize: 16, background: '#1a73e8', color: 'white', border: 'none', borderRadius: 6, cursor: 'pointer' }}>
            {loading ? 'Simplifying...' : 'Simplify for Patient'}
          </button>
          <span style={{ color: '#888' }}>Examples: </span>
          {exampleNotes.map((ex, i) => (
            <button key={i} onClick={() => setInputText(ex.text)}
              style={{ padding: '6px 12px', fontSize: 12, background: '#f0f0f0', border: '1px solid #ddd', borderRadius: 4, cursor: 'pointer' }}>
              {ex.name}
            </button>
          ))}
        </div>
      </section>

      {error && <div style={{ padding: 12, background: '#fee', border: '1px solid #f88', borderRadius: 6, marginBottom: 20 }}>{error}</div>}

      {result && (
        <section>
          <div style={{ background: '#fff3e0', padding: 20, borderRadius: 8, marginBottom: 16, border: '1px solid #ffcc80' }}>
            <h3 style={{ marginTop: 0 }}>Original (hover terms for definitions, click for MedlinePlus)</h3>
            {renderAnnotatedText(result.input, result.source_annotations)}
          </div>

          <div style={{ background: '#e8f5e9', padding: 20, borderRadius: 8, marginBottom: 16, border: '1px solid #a5d6a7' }}>
            <h3 style={{ marginTop: 0 }}>Plain Language Version</h3>
            <p style={{ fontSize: 16, lineHeight: 1.6 }}>{result.plain_language}</p>
          </div>

          {result.facts && (
            <div style={{ background: '#e3f2fd', padding: 20, borderRadius: 8, marginBottom: 16, border: '1px solid #90caf9' }}>
              <h3 style={{ marginTop: 0 }}>Key Facts Extracted</h3>
              <pre style={{ whiteSpace: 'pre-wrap', fontFamily: 'inherit' }}>{result.facts}</pre>
            </div>
          )}

          {result.source_annotations && result.source_annotations.length > 0 && (
            <div style={{ background: '#f3e5f5', padding: 20, borderRadius: 8, border: '1px solid #ce93d8' }}>
              <h3 style={{ marginTop: 0 }}>Medical Term Glossary</h3>
              {result.source_annotations.map((ann, i) => (
                <div key={i} style={{ marginBottom: 12, paddingBottom: 12, borderBottom: '1px solid #e0e0e0' }}>
                  <a href={ann.url} target="_blank" rel="noopener noreferrer"
                    style={{ fontWeight: 'bold', color: '#7b1fa2', fontSize: 16 }}>{ann.term}</a>
                  <span style={{ marginLeft: 10, color: '#333' }}>{ann.simple}</span>
                  {ann.medlineplus_summary && (
                    <p style={{ margin: '4px 0 0', color: '#666', fontSize: 13 }}>{ann.medlineplus_summary}</p>
                  )}
                </div>
              ))}
            </div>
          )}
        </section>
      )}

      {tooltip && (
        <div style={{
          position: 'absolute', left: tooltip.x, top: tooltip.y + 5,
          background: '#333', color: 'white', padding: '8px 12px', borderRadius: 6,
          maxWidth: 350, fontSize: 13, zIndex: 1000, boxShadow: '0 2px 8px rgba(0,0,0,0.3)'
        }}>
          <strong>{tooltip.term}</strong>: {tooltip.simple}
          {tooltip.medlineplus_summary && <p style={{ margin: '6px 0 0', opacity: 0.9, fontSize: 12 }}>{tooltip.medlineplus_summary}</p>}
        </div>
      )}

      <footer style={{ textAlign: 'center', marginTop: 40, color: '#999', fontSize: 12 }}>
        Powered by FLAN-T5 + MedlinePlus (NIH/NLM). Not a substitute for professional medical advice.
      </footer>

      <style>{`
        .medical-term { position: relative; }
        .term-link { color: #1a73e8; text-decoration: underline; text-decoration-style: dotted; cursor: pointer; }
        .term-badge { display: none; position: absolute; bottom: 100%; left: 0; background: #1a73e8; color: white; padding: 2px 6px; border-radius: 3px; font-size: 11px; white-space: nowrap; }
        .medical-term:hover .term-badge { display: block; }
        .annotated-text { line-height: 2; font-size: 15px; }
      `}</style>
    </div>
  );
}

export default App;
