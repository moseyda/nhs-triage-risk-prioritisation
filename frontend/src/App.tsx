import { useState } from 'react'
import './App.css'

interface TriageResponse {
  risk_score: number;
  priority_band: 'High' | 'Medium' | 'Low';
  prioritisation_score: number;
}

function App() {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<TriageResponse | null>(null)
  const [error, setError] = useState<string | null>(null)

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!text.trim()) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://localhost:8000/api/v1/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error('Failed to fetch prediction from AI model.');
      }

      const data = await response.json();
      setResult(data);
    } catch (err: any) {
      setError(err.message || 'An error occurred during prediction.');
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="dashboard">
      <div className="header-span">
        <h1>MH Triage Prototype</h1>
        <p>AI-Assisted Clinical Decision Support</p>
      </div>

      {/* Left Panel: Input Form */}
      <div className="panel">
        <h2>📝 New Referral Intake</h2>
        <form onSubmit={handleSubmit}>
          <div className="form-group">
            <label htmlFor="referral">Clinical Notes / Patient Referral</label>
            <textarea 
              id="referral"
              placeholder="Enter patient details, symptoms, and presenting complaints..."
              value={text}
              onChange={(e) => setText(e.target.value)}
              disabled={loading}
            />
          </div>
          
          <button type="submit" className="btn-submit" disabled={loading || !text.trim()}>
            {loading ? <div className="spinner"></div> : 'Analyze Risk & Priority'}
          </button>
        </form>

        {error && (
          <div style={{ marginTop: '1rem', color: 'var(--priority-high)', fontWeight: 500 }}>
            ⚠️ {error}
          </div>
        )}
      </div>

      {/* Right Panel: AI Results Display */}
      <div className="panel">
        <h2>🧠 AI Triage Analysis</h2>
        
        {!result && !loading && (
          <div className="alert-info">
            <p><strong>Awaiting Input</strong></p>
            <p style={{ marginTop: '0.5rem' }}>Submit a clinical referral on the left to view the AI's real-time risk classification and priority queue routing.</p>
          </div>
        )}

        {loading && (
          <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '200px', flexDirection: 'column', gap: '1rem' }}>
            <div className="spinner" style={{ borderColor: 'var(--nhs-blue)', borderTopColor: 'transparent', width: '40px', height: '40px' }}></div>
            <p style={{ color: 'var(--nhs-dark-blue)', fontWeight: 600 }}>Analyzing semantic intent...</p>
          </div>
        )}

        {result && !loading && (
          <div className="result-card">
            <div className="metric-group">
              <span className="metric-label">Recommended Triage Priority</span>
              <span className={`metric-value band-${result.priority_band}`}>
                {result.priority_band} Risk
              </span>
            </div>

            <div className="metric-group">
              <span className="metric-label">Model Confidence (Risk Probability)</span>
              <span className="metric-value">
                {(result.risk_score * 100).toFixed(1)}%
              </span>
            </div>

            <div className="metric-group">
              <span className="metric-label">Urgency Routing Score</span>
              <span className="metric-value">
                {result.prioritisation_score.toFixed(2)} / 100
              </span>
            </div>

            <div className="alert-info" style={{ marginTop: '0.5rem' }}>
              ℹ️ High precision NLP triage prioritisation. Always verify with human clinical judgement.
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
