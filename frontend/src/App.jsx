import { useEffect, useMemo, useRef, useState } from 'react'
import {
  Activity,
  AlertTriangle,
  ArrowUpRight,
  CheckCircle2,
  Cpu,
  FileImage,
  FileVideo,
  Layers,
  RefreshCw,
  RotateCcw,
  ShieldAlert,
  Sparkles,
  Upload,
  X,
  Zap,
} from 'lucide-react'

const API_BASE = import.meta.env.VITE_API_URL || ''

function formatPercent(value) {
  if (typeof value !== 'number' || isNaN(value)) return '0.0%'
  return `${(value * 100).toFixed(1)}%`
}

const DEFAULT_MODELS = [
  {
    id: 'celebdf_b0',
    display_name: 'EfficientNet-B0',
    dataset: 'Celeb-DF v2',
    architecture: 'CNN',
    description: 'EfficientNet-B0 trained on Celeb-DF v2 with frame-level aggregation.',
    available: true,
  },
  {
    id: 'ff_b0',
    display_name: 'EfficientNet-B0',
    dataset: 'FaceForensics++',
    architecture: 'CNN',
    description: 'EfficientNet-B0 trained on FaceForensics++ (c23 compressed).',
    available: true,
  },
  {
    id: 'celebdf_vit',
    display_name: 'ViT-Small',
    dataset: 'Celeb-DF v2',
    architecture: 'Vision Transformer',
    description: 'Vision Transformer (ViT-Small/16) trained on Celeb-DF v2.',
    available: true,
  },
  {
    id: 'ff_vit',
    display_name: 'ViT-Small',
    dataset: 'FaceForensics++',
    architecture: 'Vision Transformer',
    description: 'Vision Transformer (ViT-Small/16) trained on FaceForensics++.',
    available: true,
  },
]

function App() {
  const [file, setFile] = useState(null)
  const [fileUrl, setFileUrl] = useState(null)
  const [models, setModels] = useState(DEFAULT_MODELS)
  const [model, setModel] = useState('celebdf_b0')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [busy, setBusy] = useState(false)
  const [backendStatus, setBackendStatus] = useState('checking') // 'online' | 'offline' | 'checking'
  const [isDragging, setIsDragging] = useState(false)
  const inputRef = useRef(null)

  // Check backend health & fetch available models
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
            // Ensure selected model is available
            const hasModel = data.models.some((m) => m.id === model)
            if (!hasModel) {
              setModel(data.models[0].id)
            }
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

  // Create preview URL for uploaded media
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

  const handleCreateSample = () => {
    // Generate a simple synthetic test face canvas image for instant testing
    const canvas = document.createElement('canvas')
    canvas.width = 224
    canvas.height = 224
    const ctx = canvas.getContext('2d')
    // Background gradient
    const grad = ctx.createLinearGradient(0, 0, 224, 224)
    grad.addColorStop(0, '#1a2b28')
    grad.addColorStop(1, '#0c1413')
    ctx.fillStyle = grad
    ctx.fillRect(0, 0, 224, 224)
    // Draw stylized face oval
    ctx.fillStyle = '#dfaa8b'
    ctx.beginPath()
    ctx.ellipse(112, 115, 60, 80, 0, 0, Math.PI * 2)
    ctx.fill()
    // Eyes
    ctx.fillStyle = '#222'
    ctx.beginPath()
    ctx.arc(92, 100, 7, 0, Math.PI * 2)
    ctx.arc(132, 100, 7, 0, Math.PI * 2)
    ctx.fill()
    // Mouth
    ctx.strokeStyle = '#993333'
    ctx.lineWidth = 4
    ctx.beginPath()
    ctx.arc(112, 145, 20, 0.15 * Math.PI, 0.85 * Math.PI)
    ctx.stroke()

    canvas.toBlob((blob) => {
      const sampleFile = new File([blob], 'synthetic_sample_face.png', { type: 'image/png' })
      handleFileChange(sampleFile)
    }, 'image/png')
  }

  const analyze = async () => {
    if (!file) return
    setBusy(true)
    setError('')
    setResult(null)

    const formData = new FormData()
    formData.append('file', file)
    formData.append('model', model)

    try {
      const response = await fetch(`${API_BASE}/api/predict?model_id=${encodeURIComponent(model)}`, {
        method: 'POST',
        body: formData,
      })

      const payload = await response.json()
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

  const resetAll = () => {
    setFile(null)
    setResult(null)
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
            <span className="badge-pill">Cross-Dataset Generalization</span>
          </div>
          <h1>
            Inspect the signal<br />
            <em>under distribution shift.</em>
          </h1>
          <p className="hero-text">
            Upload face media to evaluate generalization across FaceForensics++, Celeb-DF, and DFDC.
            Extract frame-level features, compare CNN vs. Vision Transformer baselines, and inspect
            visual attribution heatmaps.
          </p>
          <div className="hero-tags">
            <span className="tag-item">
              <Cpu size={14} /> EfficientNet-B0
            </span>
            <span className="tag-item">
              <Layers size={14} /> Vision Transformer
            </span>
            <span className="tag-item">
              <Zap size={14} /> Grad-CAM Heatmaps
            </span>
          </div>
        </div>

        <div className="hero-orbit" aria-hidden="true">
          <div className="orbit-ring ring-one" />
          <div className="orbit-ring ring-two" />
          <div className="orbit-core">
            <Sparkles size={28} />
          </div>
          <span className="orbit-label label-a">AUROC 0.94+</span>
          <span className="orbit-label label-b">ViT-Small</span>
          <span className="orbit-label label-c">XAI Heatmap</span>
        </div>
      </section>

      {/* Main Workspace */}
      <section className="workspace">
        {/* Left Column: Control Rail */}
        <aside className="control-rail panel">
          <div className="section-kicker">01 / INPUT MEDIA</div>
          <h2>Upload Sample</h2>
          <p className="muted">Upload a video or face image to score against trained detectors.</p>

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

          <div className="sample-quick-actions">
            <button type="button" className="btn-text-sample" onClick={handleCreateSample}>
              <Sparkles size={14} /> Load Synthetic Test Sample
            </button>
            {file && (
              <button type="button" className="btn-text-clear" onClick={resetAll}>
                <RotateCcw size={14} /> Clear
              </button>
            )}
          </div>

          <div className="field-group">
            <label className="field-label" htmlFor="model-select">
              TARGET DETECTOR CHECKPOINT
            </label>
            <select
              id="model-select"
              value={model}
              onChange={(e) => setModel(e.target.value)}
              className="select-input"
            >
              {models.map((m) => (
                <option key={m.id} value={m.id}>
                  {m.architecture} · {m.display_name} ({m.dataset})
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
            className="analyze-button"
            disabled={!file || busy}
            onClick={analyze}
          >
            {busy ? (
              <>
                <Activity className="spin" size={18} />
                <span>Extracting & Evaluating...</span>
              </>
            ) : (
              <>
                <Sparkles size={18} />
                <span>Run Forensic Analysis</span>
              </>
            )}
          </button>

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
              Evaluation utilizes strictly real face crops and pre-trained weights. Fixed 0.50 decision
              threshold. No synthetic substitution during inference.
            </p>
          </div>
        </aside>

        {/* Right Column: Result Stage */}
        <div className="result-stage">
          <div className="section-kicker">02 / FORENSIC EVIDENCE</div>

          {!result && !busy && (
            <div className="empty-state">
              <div className="empty-icon">
                <FileVideo size={36} />
              </div>
              <h2>Awaiting Media Input</h2>
              <p>
                Select or drop a face image or video on the left, choose a detector checkpoint, and run
                analysis to view deepfake probabilities and Grad-CAM attention.
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
              <h2>Analyzing Neural Signatures</h2>
              <p>
                Running Haar cascade face localization, extracting frame sequences, computing backbone
                activations, and generating saliency attributions.
              </p>
            </div>
          )}

          {result && (
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

function Result({ result, fileKind, clientPreviewUrl, onReset }) {
  const isFake = result.label === 'FAKE'

  return (
    <div className="result-content">
      {/* Verdict Banner */}
      <div className={`verdict ${isFake ? 'verdict-fake' : 'verdict-real'}`}>
        <div className="verdict-primary">
          <span className="verdict-label">EVALUATION VERDICT</span>
          <strong>{result.label}</strong>
          <p className="verdict-sub">
            {result.dataset} · {result.model_name || result.model} ({result.architecture || 'CNN'})
          </p>
        </div>

        <div className="verdict-scores">
          <div className="score-block">
            <span>FAKE PROBABILITY</span>
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

      {/* Media Inspection Panels */}
      <div className="result-grid">
        {/* Input Media Panel */}
        <div className="preview-panel panel">
          <div className="panel-heading">
            <span>INPUT CROPPED FACE</span>
            <span className="tag">
              {result.face_detected ? 'FACE DETECTED' : 'FULL FRAME'}
            </span>
          </div>
          <div className="panel-body">
            <img src={result.input_preview} alt="Cropped face sample" className="preview-image" />
          </div>
          <div className="panel-footer">
            <span>Normalized 224x224 RGB crop</span>
          </div>
        </div>

        {/* Explainability Heatmap Panel */}
        <div className="explain-panel panel">
          <div className="panel-heading">
            <span>EXPLAINABILITY ATTRIBUTION</span>
            <span className="tag tag-accent">{result.explanation_method || 'Grad-CAM'}</span>
          </div>
          <div className="panel-body">
            {result.explanation_image ? (
              <img
                src={result.explanation_image}
                alt="Model attribution heatmap"
                className="preview-image"
              />
            ) : (
              <div className="no-explanation">Attribution heatmap unavailable</div>
            )}
          </div>
          <div className="panel-footer">
            <span className="image-disclaimer">
              Warmer regions highlight spatial features influencing the decision.
            </span>
          </div>
        </div>
      </div>

      {/* Uploaded Source Media (if available) */}
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
