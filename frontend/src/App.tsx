import { useState, useEffect } from 'react'
import { RefreshCw, Inbox, Stethoscope, AlertTriangle, CheckCircle, Info, Activity } from 'lucide-react'
import './App.css'

interface WordAttribution {
  word: string;
  impact_score: number;
}

interface TriageResponse {
  risk_score: number;
  priority_band: 'High' | 'Medium' | 'Low';
  prioritisation_score: number;
  word_attributions: WordAttribution[];
  recommended_protocol?: string;
}

interface PatientCase {
  id: string;
  mrn: string;
  age: number;
  gender: string;
  referral_text: string;
  ai_triage: TriageResponse;
}

interface OverrideHistoryItem {
  timestamp: string;
  patient_id: string;
  ai_risk_score: number;
  human_corrected_band: string;
  referral_text: string;
  reasoning: string;
}

const getAttributionColor = (score: number, maxAbsScore: number) => {
  if (maxAbsScore === 0) return 'transparent';
  const normalised = score / maxAbsScore; // Scale relative to strongest word
  if (normalised > 0.6) return 'rgba(218, 41, 28, 0.6)';  // Strong risk driver (NHS Red)
  if (normalised > 0.3) return 'rgba(218, 41, 28, 0.35)';
  if (normalised > 0.1) return 'rgba(218, 41, 28, 0.15)';
  if (normalised < -0.3) return 'rgba(0, 150, 57, 0.25)'; // Safety indicator (NHS Green)
  if (normalised < -0.1) return 'rgba(0, 150, 57, 0.12)';
  return 'transparent';
};

function App() {
  const [queue, setQueue] = useState<PatientCase[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [selectedCaseId, setSelectedCaseId] = useState<string | null>(null)
  const [overrideTarget, setOverrideTarget] = useState<PatientCase | null>(null)
  const [isRetraining, setIsRetraining] = useState(false)
  const [toast, setToast] = useState<{ message: string, type: 'success' | 'error' | 'info' } | null>(null)
  const [confirmRetrainOpen, setConfirmRetrainOpen] = useState(false)
  const [activeTab, setActiveTab] = useState<'queue' | 'history'>('queue')
  const [history, setHistory] = useState<OverrideHistoryItem[]>([])
  const [reasoningText, setReasoningText] = useState("")

  const showToast = (message: string, type: 'success' | 'error' | 'info' = 'info') => {
    setToast({ message, type });
    setTimeout(() => setToast(null), 6000);
  }

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

  const loadHistory = async () => {
    try {
      const response = await fetch('http://localhost:8000/api/v1/feedback-history');
      if (response.ok) setHistory(await response.json());
    } catch (err) { }
  }

  useEffect(() => {
    fetchQueue();
    loadHistory();
  }, [])

  const handleRetrainClick = () => {
    setConfirmRetrainOpen(true);
  }

  const executeRetrain = async () => {
    setConfirmRetrainOpen(false);
    setIsRetraining(true);
    try {
      const response = await fetch('http://localhost:8000/api/v1/trigger-retrain', { method: 'POST' });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to trigger retraining.");
      }
      showToast("Retraining successfully triggered! BERT model weights will mathematically adjust to the recent Ground Truth overrides.", "success");
    } catch (err: any) {
      showToast("Retraining failed: " + err.message, "error");
    } finally {
      setIsRetraining(false);
    }
  }

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

  const handleOverrideClick = (id: string) => {
    const targetCase = queue.find(c => c.id === id);
    if (targetCase) {
      setOverrideTarget(targetCase);
    }
  }

  const submitOverride = async (band: string) => {
    if (!overrideTarget) return;

    try {
      await fetch('http://localhost:8000/api/v1/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          patient_id: overrideTarget.mrn,
          referral_text: overrideTarget.referral_text,
          ai_risk_score: overrideTarget.ai_triage.risk_score,
          human_corrected_band: band,
          age: overrideTarget.age,
          gender: overrideTarget.gender,
          reasoning: reasoningText || "No clinical justification documented."
        })
      });

      showToast(`Override Captured! Patient ${overrideTarget.mrn} escalated to ${band} Priority. Trace logged to MLOps database.`, 'success');

      const removedId = overrideTarget.id;
      setOverrideTarget(null);
      setReasoningText("");
      loadHistory();
      handleApprove(removedId);
    } catch (err) {
      showToast("Failed to submit MLOps feedback override.", 'error');
    }
  }

  const selectedCase = queue.find(c => c.id === selectedCaseId);

  return (
    <div className="dashboard">
      <div className="header-span">
        <div>
          <h1>CDSS Triage Workspace</h1>
          <p>BERT-Based Mental Health Referral Prioritisation</p>
        </div>
        <div style={{ display: 'flex', gap: '1rem' }}>
          <button onClick={handleRetrainClick} disabled={isRetraining} style={{ padding: '0.5rem 1rem', borderRadius: '8px', border: '1px solid #da291c', cursor: isRetraining ? 'not-allowed' : 'pointer', background: 'rgba(218, 41, 28, 0.05)', color: '#da291c', display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 600 }}>
            {isRetraining ? <div className="spinner" style={{ width: '16px', height: '16px', margin: 0, borderWidth: '2px', borderColor: 'rgba(218, 41, 28, 0.2)', borderTopColor: '#da291c' }}></div> : <><Activity size={16} /> Force MLOps Retrain</>}
          </button>
          <button onClick={fetchQueue} style={{ padding: '0.5rem 1rem', borderRadius: '8px', border: '1px solid #ccc', cursor: 'pointer', background: 'white', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <RefreshCw size={16} /> Refresh EHR Queue
          </button>
        </div>
      </div>

      {/* Left Panel: The Triage Inbox Queue */}
      <div className="panel sidebar">
        <div className="panel-header" style={{ padding: 0 }}>
          <div style={{ display: 'flex', borderBottom: '1px solid #eee' }}>
            <button
              className={`tab-button ${activeTab === 'queue' ? 'active' : ''}`}
              onClick={() => setActiveTab('queue')}
            >
              <Inbox size={18} /> Queue ({queue.length})
            </button>
            <button
              className={`tab-button ${activeTab === 'history' ? 'active' : ''}`}
              onClick={() => { setActiveTab('history'); loadHistory(); }}
            >
              <Activity size={18} /> Audit Log
            </button>
          </div>
        </div>

        {activeTab === 'queue' && (
          loading ? (
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
          )
        )}

        {activeTab === 'history' && (
          <div className="history-list">
            {history.length === 0 ? (
              <div style={{ padding: '2rem', textAlign: 'center', color: '#666' }}>No clinical overrides logged yet.</div>
            ) : (
              history.map((item, idx) => (
                <div key={idx} className="history-item">
                  <div style={{ fontSize: '0.8rem', color: '#888', marginBottom: '6px' }}>
                    {new Date(item.timestamp).toLocaleString()} • Patient {item.patient_id}
                  </div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', marginBottom: '10px' }}>
                    <span style={{ fontSize: '0.9rem', color: '#666' }}>Risk Score: {(item.ai_risk_score * 100).toFixed(1)}%</span>
                    <span>→</span>
                    <span className={`priority-badge band-${item.human_corrected_band}`} style={{ fontSize: '0.7rem' }}>
                      {item.human_corrected_band}
                    </span>
                  </div>
                  <div style={{ fontSize: '0.9rem', color: '#333', background: '#f5f5f5', padding: '10px', borderRadius: '6px', fontStyle: 'italic', borderLeft: '3px solid #ccc' }}>
                    "{item.reasoning}"
                  </div>
                </div>
              ))
            )}
          </div>
        )}
      </div>

      {/* Right Panel: Human Review Window */}
      <div className="panel">
        {!selectedCase ? (
          <div className="empty-state">
            <Stethoscope size={48} color="#888" />
            <p>Select a patient referral from the queue to review the model’s triage recommendation.</p>
          </div>
        ) : (
          <div className="detail-view">
            <div className="detail-header">
              <div>
                <div className="patient-id">Patient {selectedCase.mrn}</div>
                <div className="patient-demographics">{selectedCase.age} Years Old • {selectedCase.gender}</div>
              </div>
              <div style={{ textAlign: 'right' }}>
                <span className="metric-label" style={{ marginBottom: '0.2rem' }}>Assigned Priority Band</span>
                <span className={`priority-badge band-${selectedCase.ai_triage.priority_band}`} style={{ fontSize: '1.1rem' }}>
                  {selectedCase.ai_triage.priority_band} Priority
                </span>
              </div>
            </div>

            <div>
              <span className="metric-label" style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Explainable AI (XAI) Feature Attribution</span>
                <span style={{ fontSize: '0.8rem', color: '#888', fontWeight: 'normal' }}>Hover words for impact score</span>
              </span>
              <div className="referral-box" style={{ lineHeight: '1.8' }}>
                {selectedCase.ai_triage.word_attributions && selectedCase.ai_triage.word_attributions.length > 0 ? (() => {
                  const maxAbsScore = Math.max(...selectedCase.ai_triage.word_attributions.map(a => Math.abs(a.impact_score)));
                  return selectedCase.ai_triage.word_attributions.map((attr, idx) => (
                    <span
                      key={idx}
                      className="xai-word"
                      style={{ backgroundColor: getAttributionColor(attr.impact_score, maxAbsScore) }}
                    >
                      {attr.word}
                      <span className="tooltip">
                        Impact: {attr.impact_score > 0 ? '+' : ''}{(attr.impact_score * 100).toFixed(1)}%
                      </span>
                    </span>
                  ));
                })() : (
                  `"${selectedCase.referral_text}"`
                )}
              </div>
            </div>

            <div>
              <span className="metric-label" style={{ marginBottom: '1rem' }}>Clinical Decision Support Metrics</span>
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

            {selectedCase.ai_triage.recommended_protocol && (
              <div style={{ marginTop: '1.5rem', background: 'rgba(238, 245, 255, 0.7)', padding: '1rem', borderRadius: '8px', borderLeft: '4px solid var(--nhs-blue)' }}>
                <span className="metric-label" style={{ marginBottom: '0.4rem', color: 'var(--nhs-dark-blue)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                  <Info size={16} /> RAG Protocol Assistant (NICE Guidelines)
                </span>
                <p style={{ margin: 0, fontSize: '0.95rem', lineHeight: '1.5', color: '#333' }}>
                  {selectedCase.ai_triage.recommended_protocol}
                </p>
              </div>
            )}

            <div className="action-buttons">
              <button className="btn btn-override" onClick={() => handleOverrideClick(selectedCase.id)}>
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

      {overrideTarget && (
        <div className="modal-overlay">
          <div className="modal-content">
            <h2 style={{ color: 'var(--nhs-dark-blue)', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
              <AlertTriangle color="var(--priority-high)" size={28} />
              Active Learning Override
            </h2>
            <p className="modal-desc">
              The model originally evaluated Patient <strong>{overrideTarget.mrn}</strong> as a <strong>{overrideTarget.ai_triage.priority_band} Risk</strong>.
              <br /><br />
              Select the correct clinical Ground Truth below. Your correction will be logged directly to the MLOps retraining pipeline to perpetually improve the model's accuracy.
            </p>

            <div style={{ textAlign: 'left', marginBottom: '1.5rem' }}>
              <label style={{ display: 'block', marginBottom: '0.5rem', fontWeight: 600, color: 'var(--nhs-dark-blue)' }}>Clinical Justification (Mandatory for NHS Audit)</label>
              <textarea
                value={reasoningText}
                onChange={(e) => setReasoningText(e.target.value)}
                placeholder="Briefly explain the physiological or psychiatric reasoning for contradicting the model's assessment..."
                style={{ width: '100%', padding: '12px', borderRadius: '8px', border: '1px solid #ccc', minHeight: '90px', fontFamily: 'inherit', resize: 'vertical' }}
              />
            </div>

            <div className="modal-buttons">
              <button
                className="band-button high"
                disabled={!reasoningText.trim()}
                style={{ opacity: !reasoningText.trim() ? 0.5 : 1, cursor: !reasoningText.trim() ? 'not-allowed' : 'pointer' }}
                onClick={() => submitOverride('High')}>Escalate to High Priority</button>
              <button
                className="band-button medium"
                disabled={!reasoningText.trim()}
                style={{ opacity: !reasoningText.trim() ? 0.5 : 1, cursor: !reasoningText.trim() ? 'not-allowed' : 'pointer' }}
                onClick={() => submitOverride('Medium')}>Re-classify as Medium Priority</button>
              <button
                className="band-button low"
                disabled={!reasoningText.trim()}
                style={{ opacity: !reasoningText.trim() ? 0.5 : 1, cursor: !reasoningText.trim() ? 'not-allowed' : 'pointer' }}
                onClick={() => submitOverride('Low')}>Downgrade to Low Priority</button>
            </div>

            <button className="modal-close" onClick={() => { setOverrideTarget(null); setReasoningText(""); }}>Cancel Override</button>
          </div>
        </div>
      )}

      {confirmRetrainOpen && (
        <div className="modal-overlay">
          <div className="modal-content">
            <h2 style={{ color: 'var(--priority-high)', marginBottom: '0.5rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem' }}>
              <Activity size={28} />
              Force MLOps Retrain
            </h2>
            <p className="modal-desc">
              Are you sure you want to force massive GPU retraining calculations? <br /><br />
              This process will dynamically update live system weights. Model inference will be paused for approximately 15 seconds.
            </p>
            <div className="modal-buttons" style={{ marginTop: '2rem' }}>
              <button className="band-button high" style={{ background: 'var(--priority-high)', color: 'white' }} onClick={executeRetrain}>Yes, Execute Retraining Run</button>
            </div>
            <button className="modal-close" onClick={() => setConfirmRetrainOpen(false)}>Cancel</button>
          </div>
        </div>
      )}

      {toast && (
        <div className={`toast-container`}>
          <div className={`toast ${toast.type}`}>
            {toast.type === 'success' && <CheckCircle size={28} color="var(--priority-low)" />}
            {toast.type === 'error' && <AlertTriangle size={28} color="var(--priority-high)" />}
            {toast.type === 'info' && <Info size={28} color="var(--nhs-blue)" />}
            <div>
              <strong style={{ fontSize: '1.05rem' }}>{toast.type === 'success' ? 'Success' : toast.type === 'error' ? 'Action Failed' : 'System Notice'}</strong><br />
              <span style={{ fontSize: '0.9rem', color: '#666' }}>{toast.message}</span>
            </div>
          </div>
        </div>
      )}

    </div>
  )
}

export default App
