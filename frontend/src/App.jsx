import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  CheckCircle2,
  ChevronDown,
  ChevronRight,
  Cpu,
  FileImage,
  FileVideo,
  Layers,
  RefreshCw,
  RotateCcw,
  ShieldAlert,
  Sparkles,
  Upload,
  UserCheck,
  X,
  Zap,
} from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

function formatPercent(value) {
  if (typeof value !== 'number' || isNaN(value)) return '0.0%'
  return `${(value * 100).toFixed(1)}%`
}

const DEFAULT_MODELS = [
  {
    id: 'cross_ensemble',
    display_name: 'Cross-Architecture Hybrid Ensemble',
    dataset: 'Celeb-DF v2 & FaceForensics++',
    architecture: 'Hybrid Ensemble (CNN + ViT)',
    description:
      'Experimental ensemble. Only eligible models contribute; some registered checkpoints are excluded from fusion.',
    available: false,
    is_ensemble: true,
  },
  {
    id: 'celebdf_b0',
    display_name: 'EfficientNet-B0',
    dataset: 'Celeb-DF v2',
    architecture: 'CNN',
    description: 'EfficientNet-B0 trained on Celeb-DF v2 with frame-level aggregation.',
    available: false,
  },
  {
    id: 'ff_b0',
    display_name: 'EfficientNet-B0',
    dataset: 'FaceForensics++',
    architecture: 'CNN',
    description: 'EfficientNet-B0 trained on FaceForensics++ (c23 compressed).',
    available: false,
  },
  {
    id: 'celebdf_vit',
    display_name: 'ViT-Small',
    dataset: 'Celeb-DF v2',
    architecture: 'Vision Transformer',
    description: 'Vision Transformer (ViT-Small/16) trained on Celeb-DF v2.',
    available: false,
  },
  {
    id: 'ff_vit',
    display_name: 'ViT-Small',
    dataset: 'FaceForensics++',
    architecture: 'Vision Transformer',
    description: 'Vision Transformer (ViT-Small/16) trained on FaceForensics++.',
    available: false,
  },
  {
    id: 'dfdc_b0',
    display_name: 'EfficientNet-B0 (Pilot)',
    dataset: 'DFDC (8-video pilot)',
    architecture: 'CNN',
    description:
      'Experimental DFDC pilot checkpoint trained for one epoch on a tiny subset.',
    available: false,
  },
  {
    id: 'combined_multidomain_b0',
    display_name: 'EfficientNet-B0 (Universal Multi-Domain)',
    dataset: 'All 4 Combined (FF++, Celeb-DF, DFDC, Diffusion)',
    architecture: 'CNN',
    description:
      'Universal detector trained on all 4 datasets combined (FaceForensics++, Celeb-DF v2, DFDC, and Diffusion synthetic faces).',
    available: true,
  },
]

function App() {
  const [file, setFile] = useState(null)
  const [fileUrl, setFileUrl] = useState(null)
  const [models, setModels] = useState(DEFAULT_MODELS)
  const [model, setModel] = useState('cross_ensemble')
  const [result, setResult] = useState(null)
  const [ensembleResult, setEnsembleResult] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [backendStatus, setBackendStatus] = useState('checking')
  const [isDragging, setIsDragging] = useState(false)
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [mode, setMode] = useState('ensemble') // 'ensemble' (default) or 'single'
  const [loadingSample, setLoadingSample] = useState(false)
  const [sampleNotice, setSampleNotice] = useState('')
  const inputRef = useRef(null)

  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/api/health`)
      if (res.ok) {
        setBackendStatus('online')
        const modelsRes = await fetch(`${API_BASE}/api/models`)
        if (modelsRes.ok) {
          const data = await modelsRes.json()
          if (data.models && data.models.length > 0) {
            setModels(data.models)
            setModel((currentModel) => {
              if (data.models.some((m) => m.id === currentModel && m.available)) {
                return currentModel
              }
              return (data.models.find((m) => m.available) || data.models[0]).id
            })
          }
        }
      } else {
        setBackendStatus('offline')
      }
    } catch {
      setBackendStatus('offline')
    }
  }

  useEffect(() => {
    checkHealth()
    const interval = setInterval(checkHealth, 10000)
    return () => clearInterval(interval)
  }, [])

  useEffect(() => {
    if (!file) {
      setFileUrl(null)
      return
    }
    const url = URL.createObjectURL(file)
    setFileUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  const fileKind = useMemo(() => {
    if (!file) return null
    return file.type.startsWith('image') ? 'image' : 'video'
  }, [file])

  const handleFileChange = (newFile) => {
    if (!newFile) return
    setFile(newFile)
    setResult(null)
    setEnsembleResult(null)
    setError('')
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setIsDragging(false)
    const dropped = e.dataTransfer.files?.[0]
    if (dropped) handleFileChange(dropped)
  }

  const handleDragOver = (e) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e) => {
    e.preventDefault()
    setIsDragging(false)
  }

  const handleLoadDiffusionSample = async () => {
    setLoadingSample(true)
    setError('')
    setSampleNotice('')
    try {
      const res = await fetch(`${API_BASE}/api/diffusion-sample/random`)
      if (!res.ok) throw new Error('Could not fetch diffusion face sample.')
      const blob = await res.blob()
      const sampleName = res.headers.get('X-Sample-Name') || 'sdxl_diffusion_face.png'
      const sampleFile = new File([blob], sampleName, { type: 'image/png' })
      handleFileChange(sampleFile)
      setSampleNotice(`Loaded: ${sampleName} (SDXL Diffusion Face)`)
    } catch (err) {
      setError(err.message || 'Failed to load diffusion sample.')
    } finally {
      setLoadingSample(false)
    }
  }

  const handleLoadRealSample = async () => {
    setLoadingSample(true)
    setError('')
    setSampleNotice('')
    try {
      const res = await fetch(`${API_BASE}/api/diffusion-sample/random-real`)
      if (!res.ok) throw new Error('Could not fetch real face sample.')
      const blob = await res.blob()
      const sampleName = res.headers.get('X-Sample-Name') || 'real_celeba_face.png'
      const sampleFile = new File([blob], sampleName, { type: 'image/png' })
      handleFileChange(sampleFile)
      setSampleNotice(`Loaded: ${sampleName} (CelebA-HQ Real Face)`)
    } catch (err) {
      setError(err.message || 'Failed to load real face sample.')
    } finally {
      setLoadingSample(false)
    }
  }

  const analyzeEnsemble = async () => {
    if (!file) return
    setBusy(true)
    setError('')
    setEnsembleResult(null)
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)

    // 3-minute timeout — videos take ~15-30s on CPU, give generous buffer
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 180000)

    try {
      const response = await fetch(`${API_BASE}/api/ensemble-predict`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      })
      clearTimeout(timeoutId)
      const rawText = await response.text()
      let payload = {}
      try {
        payload = JSON.parse(rawText)
      } catch {
        throw new Error(
          `Inference service returned invalid response (HTTP ${response.status}): ${rawText.slice(0, 120) || 'Empty payload'}`
        )
      }
      if (!response.ok) {
        throw new Error(payload.detail || `Inference error: HTTP ${response.status}`)
      }
      setEnsembleResult(payload)
    } catch (err) {
      clearTimeout(timeoutId)
      if (err.name === 'AbortError') {
        setError('Analysis timed out after 3 minutes. Try a shorter video clip.')
      } else {
        setError(err.message || 'The inference service is temporarily unreachable.')
      }
    } finally {
      setBusy(false)
    }
  }

  const analyzeSingle = async () => {
    if (!file) return
    setBusy(true)
    setError('')
    setResult(null)
    setEnsembleResult(null)

    const formData = new FormData()
    formData.append('file', file)
    formData.append('model', model)

    try {
      const response = await fetch(`${API_BASE}/api/predict?model_id=${encodeURIComponent(model)}`, {
        method: 'POST',
        body: formData,
      })
      const rawText = await response.text()
      let payload = {}
      try {
        payload = JSON.parse(rawText)
      } catch {
        throw new Error(
          `Inference service returned invalid response (HTTP ${response.status}): ${rawText.slice(0, 120) || 'Empty payload'}`
        )
      }
      if (!response.ok) {
        throw new Error(payload.detail || `Inference error: HTTP ${response.status}`)
      }
      setResult(payload)
    } catch (err) {
      setError(err.message || 'The inference service is temporarily unreachable.')
    } finally {
      setBusy(false)
    }
  }

  const analyze = () => {
    if (mode === 'ensemble') {
      analyzeEnsemble()
    } else {
      analyzeSingle()
    }
  }

  const resetAll = () => {
    setFile(null)
    setResult(null)
    setEnsembleResult(null)
    setError('')
  }

  const selectedModelMeta = models.find((m) => m.id === model)

  return (
    <main className="app-shell">
      {/* Top Navbar */}
      <nav className="topbar">
        <div className="brand">
          <span className="brand-mark">R</span>
          <span>
            ROBUST<span className="brand-accent">DEEP</span>
          </span>
          <span className="brand-badge">RESEARCH CONSOLE</span>
        </div>

        <div className="nav-actions">
          <div className={`nav-status status-${backendStatus}`}>
            <span className="status-dot" />
            <span>
              {backendStatus === 'online'
                ? 'API ACTIVE'
                : backendStatus === 'checking'
                ? 'CONNECTING...'
                : 'API OFFLINE'}
            </span>
          </div>
          {backendStatus === 'offline' && (
            <button className="btn-icon" onClick={checkHealth} title="Retry API connection">
              <RefreshCw size={14} />
            </button>
          )}
        </div>
      </nav>

      {/* Hero Section */}
      <section className="hero-grid">
        <div className="hero-copy">
          <div className="eyebrow-container">
            <span className="eyebrow">RESEARCH SYSTEM / LIVE FORENSICS</span>
            <span className="badge-pill">All-Model Ensemble Analysis</span>
          </div>
          <h1>
            One verdict.<br />
            <em>Every model. Every face.</em>
          </h1>
          <p className="hero-text">
            Upload face media and get a single <strong>REAL / FAKE / UNCERTAIN</strong> verdict
            from the registered checkpoints. Only models eligible for fusion contribute to the
            verdict; low-confidence or unverified models are shown but excluded. Use the advanced
            panel to inspect an individual model.
          </p>
          <div className="hero-tags">
            <span className="tag-item tag-highlight">
              <Sparkles size={14} /> Validation-Weighted Ensemble
            </span>
            <span className="tag-item">
              <Cpu size={14} /> EfficientNet-B0 ×4
            </span>
            <span className="tag-item">
              <Layers size={14} /> ViT-Small ×2
            </span>
          </div>
        </div>

        <div className="hero-orbit" aria-hidden="true">
          <div className="orbit-ring ring-one" />
          <div className="orbit-ring ring-two" />
          <div className="orbit-core">
            <Sparkles size={28} />
          </div>
          <span className="orbit-label label-a">Ensemble Fusion</span>
          <span className="orbit-label label-b">6 Models</span>
          <span className="orbit-label label-c">One Verdict</span>
        </div>
      </section>

      {/* Main Workspace */}
      <section className="workspace">
        {/* Left Column: Control Rail */}
        <aside className="control-rail panel">
          <div className="section-kicker">01 / INPUT MEDIA</div>
          <h2>Upload Sample</h2>
          <p className="muted">Upload a video or face image to run all models and get one verdict.</p>

          <input
            ref={inputRef}
            type="file"
            accept="image/*,video/*"
            onChange={(e) => handleFileChange(e.target.files?.[0])}
            hidden
          />

          <div
            className={`dropzone ${file ? 'has-file' : ''} ${isDragging ? 'dragging' : ''}`}
            onClick={() => inputRef.current?.click()}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
          >
            {file ? (
              <div className="file-info-preview">
                <CheckCircle2 size={30} className="file-check-icon" />
                <span className="file-name">{file.name}</span>
                <span className="file-meta">
                  {(file.size / 1024 / 1024).toFixed(2)} MB · {fileKind?.toUpperCase()}
                </span>
                <span className="replace-text">Click or drop to replace</span>
              </div>
            ) : (
              <>
                <Upload size={32} className="upload-icon" />
                <span>Drag & drop media here</span>
                <small>MP4, MOV, AVI, WEBP, PNG, JPG</small>
              </>
            )}
          </div>

          <div className="sample-quick-actions-panel">
            <div className="sample-quick-header">
              <span className="sample-badge">BENCHMARK DATASET</span>
              <span className="sample-count">100 Crops (50 Real · 50 Fake)</span>
            </div>
            <div className="sample-buttons-row">
              <button
                type="button"
                className="btn-text-sample btn-diffusion-sample"
                onClick={handleLoadDiffusionSample}
                disabled={loadingSample || busy}
                title="Load a synthetic face generated by Stable Diffusion XL"
              >
                {loadingSample ? (
                  <>
                    <Activity className="spin" size={13} />
                    <span>Loading...</span>
                  </>
                ) : (
                  <>
                    <Sparkles size={13} />
                    <span>Fake (SDXL)</span>
                  </>
                )}
              </button>
              <button
                type="button"
                className="btn-text-sample btn-real-sample"
                onClick={handleLoadRealSample}
                disabled={loadingSample || busy}
                title="Load an authentic photograph from CelebA-HQ"
              >
                {loadingSample ? (
                  <>
                    <Activity className="spin" size={13} />
                    <span>Loading...</span>
                  </>
                ) : (
                  <>
                    <UserCheck size={13} />
                    <span>Real (CelebA)</span>
                  </>
                )}
              </button>
              {file && (
                <button type="button" className="btn-text-clear" onClick={resetAll}>
                  <RotateCcw size={13} /> Clear
                </button>
              )}
            </div>
            {sampleNotice && (
              <div className="sample-notice-pill">
                <CheckCircle2 size={12} className="sample-notice-icon" />
                <span>{sampleNotice}</span>
              </div>
            )}
          </div>

          {/* Primary action: Analyze (All Models) */}
          <button
            className="analyze-button"
            disabled={!file || busy}
            onClick={() => { setMode('ensemble'); analyzeEnsemble() }}
          >
            {busy && mode === 'ensemble' ? (
              <>
                <Activity className="spin" size={18} />
                <span>Running All Models...</span>
              </>
            ) : (
              <>
                <Sparkles size={18} />
                <span>Analyze (All Models)</span>
              </>
            )}
          </button>

          {/* Advanced: single model inspection */}
          <div className="advanced-toggle" onClick={() => setShowAdvanced(!showAdvanced)}>
            {showAdvanced ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            <span className="advanced-toggle-text">Advanced: Inspect Single Model</span>
          </div>

          {showAdvanced && (
            <div className="advanced-panel">
              <div className="field-group">
                <label className="field-label" htmlFor="model-select">
                  SINGLE MODEL CHECKPOINT
                </label>
                <select
                  id="model-select"
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  className="select-input"
                >
                  {models.map((m) => (
                    <option key={m.id} value={m.id} disabled={!m.available}>
                      {m.id === 'cross_ensemble'
                        ? `★ ${m.display_name} (${m.members?.filter((member) => member.available).length ?? 0}/4 available)`
                        : `${m.architecture} · ${m.display_name} (${m.dataset})${m.available ? '' : ' (weights missing)'}`}
                    </option>
                  ))}
                </select>
                {selectedModelMeta && (
                  <p className="field-hint">
                    <strong>{selectedModelMeta.architecture}:</strong> {selectedModelMeta.description}
                  </p>
                )}
              </div>

              <button
                className="analyze-button analyze-button-secondary"
                disabled={!file || busy}
                onClick={() => { setMode('single'); analyzeSingle() }}
              >
                {busy && mode === 'single' ? (
                  <>
                    <Activity className="spin" size={18} />
                    <span>Evaluating...</span>
                  </>
                ) : (
                  <>
                    <Zap size={18} />
                    <span>Run Single Model</span>
                  </>
                )}
              </button>
            </div>
          )}

          {error && (
            <div className="error-box">
              <AlertTriangle size={18} className="error-icon" />
              <div>
                <strong>Inference Notice</strong>
                <p>{error}</p>
              </div>
            </div>
          )}

          <div className="protocol-card">
            <span>RESEARCH PROTOCOL</span>
            <p>
              Primary mode evaluates available checkpoints with per-model preprocessing. Only
              eligible models contribute to fusion; excluded models do not affect the verdict or
              disagreement score. Face detection is required — no face means no analysis.
            </p>
          </div>
        </aside>

        {/* Right Column: Result Stage */}
        <div className="result-stage">
          <div className="section-kicker">02 / FORENSIC EVIDENCE</div>

          {!result && !ensembleResult && !busy && (
            <div className="empty-state">
              <div className="empty-icon">
                <FileVideo size={36} />
              </div>
              <h2>Awaiting Media Input</h2>
              <p>
                Upload a face image or video, then click <strong>Analyze (All Models)</strong> to
                get a single ensemble verdict with per-model breakdown.
              </p>
            </div>
          )}

          {busy && (
            <div className="empty-state loading-state">
              <div className="loading-bars">
                <i />
                <i />
                <i />
                <i />
              </div>
              <h2>{mode === 'ensemble' ? 'Running All Models' : 'Analyzing Neural Signatures'}</h2>
              <p>
                {mode === 'ensemble'
                  ? fileKind === 'video'
                    ? '🎬 Video detected — sampling frames, detecting faces, running all models... this takes ~15–30s on CPU. Please wait.'
                    : 'Detecting faces, running EfficientNet-B0 ×4 and ViT-Small ×2, computing weighted fusion...'
                  : 'Running Haar cascade face localization, extracting frame sequences, computing activations...'}
              </p>
              {fileKind === 'video' && mode === 'ensemble' && (
                <p style={{ fontSize: '0.78rem', marginTop: '0.4rem', opacity: 0.6 }}>
                  Analyzing up to 16 frames across available checkpoints — do not close or refresh.
                </p>
              )}
            </div>
          )}

          {ensembleResult && <EnsembleResult data={ensembleResult} onReset={resetAll} />}

          {result && !ensembleResult && (
            <Result
              result={result}
              fileKind={fileKind}
              clientPreviewUrl={fileUrl}
              onReset={resetAll}
            />
          )}
        </div>
      </section>
    </main>
  )
}


// =============================================================================
// NEW: Ensemble Result Component — ONE verdict, collapsible details
// =============================================================================
function EnsembleResult({ data, onReset }) {
  const [detailsOpen, setDetailsOpen] = useState(false)

  // Handle special cases
  if (data.verdict === 'NO_FACE') {
    return (
      <div className="result-content">
        <div className="verdict verdict-uncertain">
          <div className="verdict-primary">
            <span className="verdict-label">DETECTION RESULT</span>
            <strong>NO FACE DETECTED</strong>
            <p className="verdict-sub">{data.message}</p>
          </div>
        </div>
        <div className="biometric-warning-banner">
          <AlertTriangle size={20} className="warning-icon" />
          <div className="warning-text">
            <strong>No human face identified in the uploaded media.</strong>
            <p>Deepfake detectors require visible facial landmarks. No models were run.</p>
          </div>
        </div>
        <div className="result-footnote">
          <ShieldAlert size={18} />
          <span>{data.disclaimer}</span>
          <button type="button" className="btn-reset-analysis" onClick={onReset}>
            <RotateCcw size={14} /> Analyze Another Sample
          </button>
        </div>
      </div>
    )
  }

  if (data.verdict === 'INSUFFICIENT_FRAMES') {
    return (
      <div className="result-content">
        <div className="verdict verdict-uncertain">
          <div className="verdict-primary">
            <span className="verdict-label">DETECTION RESULT</span>
            <strong>INSUFFICIENT DATA</strong>
            <p className="verdict-sub">{data.message}</p>
          </div>
        </div>
        <div className="result-footnote">
          <ShieldAlert size={18} />
          <span>{data.disclaimer}</span>
          <button type="button" className="btn-reset-analysis" onClick={onReset}>
            <RotateCcw size={14} /> Analyze Another Sample
          </button>
        </div>
      </div>
    )
  }

  const isFake = data.verdict === 'FAKE'
  const isUncertain = data.verdict === 'UNCERTAIN'
  const verdictClass = isFake ? 'verdict-fake' : isUncertain ? 'verdict-uncertain' : 'verdict-real'
  const modelDetails = data.model_details || []
  const includedModels = modelDetails.filter((m) => m.included)
  const excludedModels = modelDetails.filter((m) => !m.included)
  const perFace = data.per_face || []

  return (
    <div className="result-content">
      {/* ── Main Verdict Banner ── */}
      <div className={`verdict ${verdictClass}`}>
        <div className="verdict-primary">
          <span className="verdict-label">ENSEMBLE VERDICT</span>
          <strong>{data.verdict}</strong>
          <p className="verdict-sub">
            {data.models_loaded} checkpoints loaded · {includedModels.length} used in fusion · {data.face_count} face{data.face_count !== 1 ? 's' : ''} detected
            {data.frames_analyzed > 1 ? ` · ${data.frames_analyzed} frames` : ''}
          </p>
        </div>

        <div className="verdict-scores">
          <div className="score-block">
            <span>CONFIDENCE</span>
            <b className={isFake ? 'text-fake' : isUncertain ? 'text-uncertain' : 'text-real'}>
              {data.confidence.toFixed(1)}%
            </b>
          </div>
          <div className="score-block">
            <span>FUSED FAKE PROBABILITY</span>
            <b className={data.fused_score >= 0.5 ? 'text-fake' : 'text-real'}>
              {formatPercent(data.fused_score)}
            </b>
          </div>
          <div className="score-bar-container">
            <div
              className={`score-bar-fill ${data.fused_score >= 0.5 ? 'bar-fake' : 'bar-real'}`}
              style={{ width: `${Math.min(100, Math.max(0, data.fused_score * 100))}%` }}
            />
          </div>
        </div>
      </div>

      {/* ── High disagreement warning ── */}
      {data.disagreement_std > 0.25 && (
        <div className="biometric-warning-banner">
          <AlertTriangle size={20} className="warning-icon" />
          <div className="warning-text">
            <strong>High Model Disagreement (σ = {data.disagreement_std.toFixed(3)})</strong>
            <p>The models strongly disagree on this media. The confidence has been reduced.</p>
          </div>
        </div>
      )}

      {/* ── Video-specific stats ── */}
      {data.is_video && data.face_frame_fraction && (
        <div className="snapshots-stats-bar">
          <div className="stat-pill">
            <span>Frames Analyzed</span>
            <strong>{data.frames_analyzed}</strong>
          </div>
          <div className="stat-pill">
            <span>Face Frames</span>
            <strong>{data.face_frame_fraction}</strong>
          </div>
          {data.frame_variance != null && (
            <div className="stat-pill">
              <span>Frame-to-Frame Variance</span>
              <strong>{data.frame_variance.toFixed(4)}</strong>
            </div>
          )}
        </div>
      )}

      {/* ── Per-face results (images with multiple faces) ── */}
      {perFace.length > 1 && (
        <div className="ensemble-panel panel">
          <div className="panel-heading">
            <span>PER-FACE RESULTS ({perFace.length} FACES)</span>
          </div>
          <div className="ensemble-grid">
            {perFace.map((face) => {
              const fFake = face.verdict === 'FAKE'
              const fUncertain = face.verdict === 'UNCERTAIN'
              return (
                <div key={face.face_index} className="ensemble-card">
                  <div className="ensemble-card-header">
                    <span className="ensemble-arch-badge">Face {face.face_index + 1}</span>
                    <span
                      className={`ensemble-vote-badge ${fFake ? 'vote-fake' : fUncertain ? 'vote-uncertain' : 'vote-real'}`}
                    >
                      {face.verdict} ({face.confidence.toFixed(1)}%)
                    </span>
                  </div>
                  <div className="ensemble-score-row">
                    <span>Fused Score:</span>
                    <strong>{formatPercent(face.fused_score)}</strong>
                  </div>
                  <div className="ensemble-bar">
                    <div
                      className={`ensemble-bar-fill ${face.fused_score >= 0.5 ? 'bar-fake' : 'bar-real'}`}
                      style={{ width: `${Math.min(100, Math.max(0, face.fused_score * 100))}%` }}
                    />
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* ── Collapsible Details Section ── */}
      <div className="details-toggle-bar" onClick={() => setDetailsOpen(!detailsOpen)}>
        {detailsOpen ? <ChevronDown size={16} /> : <ChevronRight size={16} />}
        <span>
          {detailsOpen ? 'Hide' : 'Show'} Details — Per-Model Breakdown ({modelDetails.length} models)
        </span>
      </div>

      {detailsOpen && (
        <div className="ensemble-panel panel">
          <div className="panel-heading">
            <span>PER-MODEL BREAKDOWN</span>
            <span className="tag tag-accent">
              {includedModels.length}/{modelDetails.length} IN FUSION
            </span>
          </div>
          <div className="ensemble-grid">
            {modelDetails.map((item) => {
              const itemFake = item.probability >= 0.5
              return (
                <div
                  key={item.model_id}
                  className={`ensemble-card${!item.included ? ' ensemble-card-excluded' : ''}`}
                >
                  <div className="ensemble-card-header">
                    <span className="ensemble-arch-badge">{item.architecture}</span>
                    <span
                      className={`ensemble-vote-badge ${!item.included ? 'vote-missing' : itemFake ? 'vote-fake' : 'vote-real'}`}
                    >
                      {!item.included
                        ? 'EXCLUDED'
                        : item.probability >= 0.5
                        ? 'FAKE'
                        : 'REAL'}
                    </span>
                  </div>
                  <div className="ensemble-card-title">{item.display_name}</div>
                  <div className="ensemble-card-dataset">{item.dataset}</div>
                  <div className="ensemble-score-row">
                    <span>Fake Prob:</span>
                    <strong>{formatPercent(item.probability)}</strong>
                  </div>
                  <div className="ensemble-score-row">
                    <span>Weight:</span>
                    <strong>{item.weight.toFixed(4)}</strong>
                  </div>
                  <div className="ensemble-score-row">
                    <span>Val AUC:</span>
                    <strong>{item.val_auc.toFixed(4)}</strong>
                  </div>
                  {!item.included && item.excluded_reason && (
                    <div className="ensemble-excluded-reason">
                      <AlertTriangle size={12} /> {item.excluded_reason}
                    </div>
                  )}
                  {item.included && (
                    <div className="ensemble-bar">
                      <div
                        className={`ensemble-bar-fill ${itemFake ? 'bar-fake' : 'bar-real'}`}
                        style={{
                          width: `${Math.min(100, Math.max(0, item.probability * 100))}%`,
                        }}
                      />
                    </div>
                  )}
                  {/* Video frame stats */}
                  {item.total_frames != null && (
                    <div className="ensemble-frame-stats">
                      <span>
                        {item.frames_flagged_fake}/{item.total_frames} frames flagged · var{' '}
                        {item.frame_variance?.toFixed(4)}
                      </span>
                    </div>
                  )}
                </div>
              )
            })}
          </div>

          {/* Excluded models summary */}
          {data.models_excluded && data.models_excluded.length > 0 && (
            <div className="excluded-summary">
              <strong>Excluded from fusion:</strong>
              <ul>
                {data.models_excluded.map((ex, i) => (
                  <li key={i}>
                    {ex.id}: {ex.reason}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Metrics Strip */}
      <div className="metric-strip">
        <div>
          <span>CONFIDENCE</span>
          <b>{data.confidence.toFixed(1)}%</b>
        </div>
        <div>
          <span>ENSEMBLE THRESHOLD</span>
          <b>{data.threshold_used?.toFixed(2) ?? '0.50'}</b>
        </div>
        <div>
          <span>MODEL DISAGREEMENT (σ)</span>
          <b>{data.disagreement_std?.toFixed(4) ?? '0.0000'}</b>
        </div>
        <div>
          <span>INFERENCE LATENCY</span>
          <b>{data.latency_ms ? `${data.latency_ms} ms` : '< 1s'}</b>
        </div>
      </div>

      {/* Footer */}
      <div className="result-footnote">
        <ShieldAlert size={18} />
        <span>{data.disclaimer}</span>
        <button type="button" className="btn-reset-analysis" onClick={onReset}>
          <RotateCcw size={14} /> Analyze Another Sample
        </button>
      </div>
    </div>
  )
}


// =============================================================================
// Original single-model Result Component (kept for advanced single-model mode)
// =============================================================================
function Result({ result, fileKind, clientPreviewUrl, onReset }) {
  const isFake = result.label === 'FAKE'
  const hasSnapshots = Array.isArray(result.snapshots) && result.snapshots.length > 0
  const initialIdx = result.multi_snapshot_analysis?.peak_anomaly_index ?? 0
  const [activeSnapshotIdx, setActiveSnapshotIdx] = useState(initialIdx)

  // Ensure selected index is valid
  const currentIdx = hasSnapshots && activeSnapshotIdx < result.snapshots.length ? activeSnapshotIdx : 0
  const activeSnapshot = hasSnapshots ? result.snapshots[currentIdx] : null

  // Active display items
  const displayPreview = activeSnapshot?.preview || result.input_preview
  const displayExplanation = activeSnapshot?.explanation_image || result.explanation_image
  const displayFaithfulness = activeSnapshot?.faithfulness_confidence_drop ?? result.faithfulness_confidence_drop

  return (
    <div className="result-content">
      {/* Verdict Banner */}
      <div className={`verdict ${isFake ? 'verdict-fake' : 'verdict-real'}`}>
        <div className="verdict-primary">
          <span className="verdict-label">SINGLE MODEL RESULT</span>
          <strong>{result.label}</strong>
          <p className="verdict-sub">
            {result.dataset} · {result.model_name || result.model} ({result.architecture || 'CNN'})
          </p>
          {result.consensus && (
            <span className="consensus-badge">
              <CheckCircle2 size={13} /> {result.consensus}
            </span>
          )}
        </div>

        <div className="verdict-scores">
          <div className="score-block">
            <span>AGGREGATE FAKE PROBABILITY</span>
            <b className={isFake ? 'text-fake' : 'text-real'}>{formatPercent(result.score)}</b>
          </div>
          <div className="score-bar-container">
            <div
              className={`score-bar-fill ${isFake ? 'bar-fake' : 'bar-real'}`}
              style={{ width: `${Math.min(100, Math.max(0, result.score * 100))}%` }}
            />
          </div>
        </div>
      </div>

      {/* Face Biometric Verification Advisory Banner */}
      {!result.face_detected && (
        <div className="biometric-warning-banner">
          <AlertTriangle size={20} className="warning-icon" />
          <div className="warning-text">
            <strong>Biometric Verification Alert: No Human Face Identified</strong>
            <p>
              The detector could not confirm a high-confidence human face with verified facial geometry in this media.
            </p>
          </div>
        </div>
      )}

      {/* Cross-Model Ensemble Breakdown (if available) */}
      {result.ensemble_breakdown && (
        <div className="ensemble-panel panel">
          <div className="panel-heading">
            <span>CROSS-ARCHITECTURE ENSEMBLE BREAKDOWN</span>
            <span className="tag tag-accent">
              {result.ensemble_breakdown.filter((item) => item.status !== 'missing').length}/4 AVAILABLE
            </span>
          </div>
          <div className="ensemble-grid">
            {result.ensemble_breakdown.map((item) => {
              const isMissing = item.status === 'missing'
              const itemFake = item.label === 'FAKE'
              return (
                <div key={item.id} className={`ensemble-card${isMissing ? ' ensemble-card-missing' : ''}`}>
                  <div className="ensemble-card-header">
                    <span className="ensemble-arch-badge">{item.architecture}</span>
                    <span className={`ensemble-vote-badge ${isMissing ? 'vote-missing' : itemFake ? 'vote-fake' : 'vote-real'}`}>
                      {isMissing ? 'WEIGHTS MISSING' : item.label}
                    </span>
                  </div>
                  <div className="ensemble-card-title">{item.name}</div>
                  <div className="ensemble-card-dataset">{item.dataset}</div>
                  <div className="ensemble-score-row">
                    <span>{isMissing ? 'Expected checkpoint:' : 'Fake Score:'}</span>
                    <strong>{isMissing ? item.checkpoint_path : formatPercent(item.score)}</strong>
                  </div>
                  {!isMissing && (
                    <div className="ensemble-bar">
                      <div
                        className={`ensemble-bar-fill ${itemFake ? 'bar-fake' : 'bar-real'}`}
                        style={{ width: `${Math.min(100, Math.max(0, item.score * 100))}%` }}
                      />
                    </div>
                  )}
                </div>
              )
            })}
          </div>
        </div>
      )}

      {/* Multi-Snapshot Evidence & Temporal Timeline */}
      {hasSnapshots && (
        <div className="snapshots-panel panel">
          <div className="panel-heading">
            <div className="heading-with-icon">
              <Layers size={16} className="text-accent" />
              <span>
                MULTI-SNAPSHOT FORENSIC EVIDENCE ({result.snapshots.length} SAMPLES ANALYZED)
              </span>
            </div>
            {result.multi_snapshot_analysis?.agreement_text && (
              <span className="tag tag-accent">
                {result.multi_snapshot_analysis.agreement_text}
              </span>
            )}
          </div>

          <p className="snapshots-desc">
            Click any snapshot below to inspect its local facial geometry, individual fake score, and Grad-CAM attention heatmap:
          </p>

          <div className="snapshots-strip">
            {result.snapshots.map((snap, idx) => {
              const isSelected = idx === currentIdx
              const isSnapFake = snap.label === 'FAKE'
              return (
                <div
                  key={snap.id ?? idx}
                  className={`snapshot-card ${isSelected ? 'snapshot-active' : ''} ${
                    snap.is_peak_anomaly ? 'snapshot-peak' : ''
                  }`}
                  onClick={() => setActiveSnapshotIdx(idx)}
                >
                  <div className="snapshot-card-top">
                    <span className="snapshot-tag">{snap.name}</span>
                    {snap.is_peak_anomaly && (
                      <span className="peak-anomaly-badge" title="Highest deepfake anomaly score">
                        ★ Peak Anomaly
                      </span>
                    )}
                  </div>
                  <div className="snapshot-biometric-meta">
                    {snap.face_detected ? (
                      <span className="bio-tag bio-verified">
                        <CheckCircle2 size={10} /> Verified Face
                      </span>
                    ) : (
                      <span className="bio-tag bio-unverified">
                        <AlertTriangle size={10} /> No Face
                      </span>
                    )}
                    {snap.face_clarity && (
                      <span className="bio-tag bio-clarity">{snap.face_clarity}</span>
                    )}
                  </div>
                  <div className="snapshot-thumb-wrap">
                    <img src={snap.preview} alt={snap.name} className="snapshot-thumb" />
                    <span className={`snapshot-score-badge ${isSnapFake ? 'vote-fake' : 'vote-real'}`}>
                      {formatPercent(snap.score)} {snap.label}
                    </span>
                  </div>
                  <div className="snapshot-card-footer">
                    <span className="snapshot-region">{snap.region || 'Crop'}</span>
                    <span className="snapshot-select-hint">
                      {isSelected ? '● Active View' : 'Click to inspect'}
                    </span>
                  </div>
                </div>
              )
            })}
          </div>

          {/* Multi-Snapshot Summary Statistics Bar */}
          {result.multi_snapshot_analysis && (
            <div className="snapshots-stats-bar">
              <div className="stat-pill">
                <span>Samples Analyzed</span>
                <strong>{result.multi_snapshot_analysis.total_snapshots} snapshots</strong>
              </div>
              <div className="stat-pill">
                <span>Mean Anomaly</span>
                <strong>{formatPercent(result.multi_snapshot_analysis.mean_score)}</strong>
              </div>
              <div className="stat-pill">
                <span>Peak Anomaly Score</span>
                <strong className={result.multi_snapshot_analysis.max_score >= 0.5 ? 'text-fake' : 'text-real'}>
                  {formatPercent(result.multi_snapshot_analysis.max_score)}
                </strong>
              </div>
              <div className="stat-pill">
                <span>Sequence Variance</span>
                <strong>{result.multi_snapshot_analysis.variance.toFixed(4)}</strong>
              </div>
              <button
                type="button"
                className="btn-select-peak"
                onClick={() => setActiveSnapshotIdx(result.multi_snapshot_analysis.peak_anomaly_index)}
              >
                Focus Peak Anomaly
              </button>
            </div>
          )}
        </div>
      )}

      {/* Media Inspection Panels */}
      <div className="result-grid">
        <div className="preview-panel panel">
          <div className="panel-heading">
            <span>
              {activeSnapshot ? `SNAPSHOT: ${activeSnapshot.name.toUpperCase()}` : 'INPUT CROPPED FACE'}
            </span>
            <span className="tag">
              {activeSnapshot?.face_detected ? 'FACE DETECTED' : result.face_detected ? 'FACE DETECTED' : 'FULL FRAME'}
            </span>
          </div>
          <div className="panel-body">
            <img src={displayPreview} alt="Inspected face snapshot" className="preview-image" />
          </div>
          <div className="panel-footer">
            <span>
              {activeSnapshot
                ? `Local Score: ${formatPercent(activeSnapshot.score)} (${activeSnapshot.label}) · Region: ${activeSnapshot.region}`
                : 'Normalized 224x224 RGB crop'}
            </span>
          </div>
        </div>

        <div className="explain-panel panel">
          <div className="panel-heading">
            <span>
              {activeSnapshot ? `ATTRIBUTION: ${activeSnapshot.name.toUpperCase()}` : 'EXPLAINABILITY ATTRIBUTION'}
            </span>
            <span className="tag tag-accent">{result.explanation_method || 'Grad-CAM'}</span>
          </div>
          <div className="panel-body">
            {displayExplanation ? (
              <img
                src={displayExplanation}
                alt="Model attribution heatmap"
                className="preview-image"
              />
            ) : (
              <div className="no-explanation">Attribution heatmap unavailable</div>
            )}
          </div>
          <div className="panel-footer">
            <span className="image-disclaimer">
              {activeSnapshot
                ? `Faithfulness Drop: ${(displayFaithfulness * 100).toFixed(2)} pts upon masking salient facial features.`
                : 'Warmer regions highlight spatial features influencing the decision.'}
            </span>
          </div>
        </div>
      </div>

      {/* Uploaded Source Media */}
      {clientPreviewUrl && (
        <div className="source-media-panel panel">
          <div className="panel-heading">
            <span>ORIGINAL INPUT MEDIA</span>
            <span className="tag">{fileKind?.toUpperCase()} SOURCE</span>
          </div>
          <div className="source-media-body">
            {fileKind === 'video' ? (
              <video
                src={clientPreviewUrl}
                controls
                className="source-media-player"
              />
            ) : (
              <img
                src={clientPreviewUrl}
                alt="Original uploaded media"
                className="source-media-image"
              />
            )}
          </div>
        </div>
      )}

      {/* Metrics Strip */}
      <div className="metric-strip">
        <div>
          <span>MODEL CONFIDENCE</span>
          <b>{formatPercent(result.confidence)}</b>
        </div>
        <div>
          <span>DECISION THRESHOLD</span>
          <b>0.50</b>
        </div>
        <div>
          <span>FAITHFULNESS DROP</span>
          <b>{(result.faithfulness_confidence_drop * 100).toFixed(2)} pts</b>
        </div>
        <div>
          <span>INFERENCE LATENCY</span>
          <b>{result.latency_ms ? `${result.latency_ms} ms` : '< 1s'}</b>
        </div>
      </div>

      {/* Bottom Footer Notice */}
      <div className="result-footnote">
        <ShieldAlert size={18} />
        <span>{result.disclaimer}</span>
        <button type="button" className="btn-reset-analysis" onClick={onReset}>
          <RotateCcw size={14} /> Analyze Another Sample
        </button>
      </div>
    </div>
  )
}

export default App
