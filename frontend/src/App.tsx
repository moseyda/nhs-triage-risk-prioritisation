import { useState, useEffect } from 'react'
import { RefreshCw, Inbox, Stethoscope, AlertTriangle, CheckCircle } from 'lucide-react'
import './App.css'

interface TriageResponse {
  risk_score: number;
  priority_band: 'High' | 'Medium' | 'Low';
  prioritisation_score: number;
}

interface PatientCase {
  id: string;
  mrn: string;
  age: number;
  gender: string;
  referral_text: string;
  ai_triage: TriageResponse;
}

function App() {
  const [queue, setQueue] = useState<PatientCase[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null)

  const fetchQueue = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await fetch('http://localhost:8000/api/v1/queue');
      if (!response.ok) throw new Error('Failed to fetch EHR triage queue.');
      const data = await response.json();
      setQueue(data);
      if (data.length > 0) {
        setSelectedCaseId(data[0].id);
      }
    } catch (err: any) {
      setError(err.message || 'An error occurred.');
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchQueue();
  }, [])

  const handleApprove = (id: string) => {
    // In a real app, this would send an API request to mark as Triaged
    setQueue(prev => prev.filter(c => c.id !== id));
    if (queue.length > 1) {
      const remaining = queue.filter(c => c.id !== id);
      setSelectedCaseId(remaining[0].id);
    } else {
      setSelectedCaseId(null);
    }
  }

  const selectedCase = queue.find(c => c.id === selectedCaseId);

  return (
    <div className="dashboard">
      <div className="header-span">
        <div>
          <h1>CDSS Triage Workspace</h1>
          <p>AI-Assisted Mental Health Referral Prioritisation</p>
        </div>
        <button onClick={fetchQueue} style={{ padding: '0.5rem 1rem', borderRadius: '8px', border: '1px solid #ccc', cursor: 'pointer', background: 'white', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <RefreshCw size={16} /> Refresh EHR Queue
        </button>
      </div>

      {/* Left Panel: The Triage Inbox Queue */}
      <div className="panel sidebar">
        <div className="panel-header">
          <h2><Inbox size={20} /> Waiting List ({queue.length})</h2>
        </div>
        
        {loading ? (
          <div className="spinner-container" style={{ padding: '2rem' }}>
            <div className="spinner"></div>
            <p>Evaluating clinical queue...</p>
          </div>
        ) : error ? (
          <div style={{ padding: '1.5rem', color: 'red' }}>⚠️ {error}</div>
        ) : queue.length === 0 ? (
          <div style={{ padding: '2rem', textAlign: 'center', color: '#666' }}>
            All clear. No pending referrals.
          </div>
        ) : (
          <div className="queue-list">
            {queue.map((pat) => (
              <div 
                key={pat.id} 
                className={`queue-item ${selectedCaseId === pat.id ? 'active' : ''}`}
                onClick={() => setSelectedCaseId(pat.id)}
              >
                <div className="queue-item-info">
                  <h3>{pat.mrn}</h3>
                  <p>{pat.age}{pat.gender} • {pat.referral_text}</p>
                </div>
                <div className={`priority-badge band-${pat.ai_triage.priority_band}`}>
                  {pat.ai_triage.priority_band}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Right Panel: Human Review Window */}
      <div className="panel">
        {!selectedCase ? (
          <div className="empty-state">
            <Stethoscope size={48} color="#888" />
            <p>Select a patient referral from the queue to review the AI's triage recommendation.</p>
          </div>
        ) : (
          <div className="detail-view">
            <div className="detail-header">
              <div>
                <div className="patient-id">Patient {selectedCase.mrn}</div>
                <div className="patient-demographics">{selectedCase.age} Years Old • {selectedCase.gender}</div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <span className="metric-label" style={{ marginBottom: '0.2rem' }}>AI Router Assignment</span>
                <span className={`priority-badge band-${selectedCase.ai_triage.priority_band}`} style={{ fontSize: '1.1rem' }}>
                  {selectedCase.ai_triage.priority_band} Priority
                </span>
              </div>
            </div>

            <div>
              <span className="metric-label">Raw Clinical Referral Text</span>
              <div className="referral-box">
                "{selectedCase.referral_text}"
              </div>
            </div>

            <div>
              <span className="metric-label" style={{ marginBottom: '1rem' }}>AI Clinical Decision Support Metrics</span>
              <div className="ai-reasoning">
                <div className="metric-card">
                  <span className="metric-label">Confidence (High Risk Probability)</span>
                  <span className="metric-value">
                    {(selectedCase.ai_triage.risk_score * 100).toFixed(1)}%
                  </span>
                </div>
                <div className="metric-card" style={{ borderColor: selectedCase.ai_triage.prioritisation_score > 70 ? 'var(--priority-high)' : '#f0f0f0' }}>
                  <span className="metric-label">Queue Ranking Score (0-100)</span>
                  <span className="metric-value">
                    {selectedCase.ai_triage.prioritisation_score.toFixed(1)}
                  </span>
                </div>
              </div>
            </div>

            <div className="action-buttons">
              <button className="btn btn-override" onClick={() => handleApprove(selectedCase.id)}>
                <AlertTriangle size={20} /> Override & Escalate
              </button>
              <button className="btn btn-approve" onClick={() => handleApprove(selectedCase.id)}>
                <CheckCircle size={20} /> Approve Triage
              </button>
            </div>
            <p style={{ textAlign: 'center', color: '#888', fontSize: '0.85rem', marginTop: '-0.5rem' }}>
              Final clinical responsibility remains with the attending human clinician.
            </p>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
