import { useEffect, useState, useRef, useMemo, type CSSProperties } from 'react'
import { useParams, Link, useNavigate, useSearchParams, useLocation } from 'react-router-dom'
import { client } from '../api/client'
import { useDialog } from '../components/Dialog'
import type { Video, MediaMetadata, Segment, Frame, Detection, Track, Event, SegmentDescription } from '../api/types'

function highlightMatched(text: string, matchedText: string) {
  if (!text || !matchedText.trim()) return text
  let idx = text.indexOf(matchedText)
  let len = matchedText.length
  if (idx === -1) {
    const ci = text.toLowerCase().indexOf(matchedText.toLowerCase())
    if (ci === -1) return text
    idx = ci
    len = matchedText.length
  }
  return (
    <>
      {text.slice(0, idx)}
      <mark style={{ background: '#fef08a', color: '#422006', padding: '0 2px', borderRadius: 3, fontWeight: 600 }}>
        {text.slice(idx, idx + len)}
      </mark>
      {text.slice(idx + len)}
    </>
  )
}

function highlightQuery(text: string, query: string) {
  if (!text || !query.trim()) return text
  const tokens = query.toLowerCase().split(/\s+/).map(t => t.replace(/[^a-z0-9]/g, '')).filter(t => t.length >= 3)
  if (tokens.length === 0) return text
  // Find earliest token hit and highlight that span.
  const lower = text.toLowerCase()
  let best = -1, bestTok = ''
  for (const tok of tokens) {
    const i = lower.indexOf(tok)
    if (i !== -1 && (best === -1 || i < best)) { best = i; bestTok = tok }
  }
  if (best === -1) return text
  return (
    <>
      {text.slice(0, best)}
      <mark style={{ background: '#fef08a', color: '#422006', padding: '0 2px', borderRadius: 3, fontWeight: 600 }}>
        {text.slice(best, best + bestTok.length)}
      </mark>
      {text.slice(best + bestTok.length)}
    </>
  )
}

function formatDur(sec: number) {
  if (!isFinite(sec) || sec < 0) sec = 0
  const h = Math.floor(sec / 3600), m = Math.floor((sec % 3600) / 60), s = Math.floor(sec % 60)
  return `${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

// One sampled observation of a tracked object. Bounding boxes are normalized
// (0–1) relative to the video frame, matching the video_frame_detections table.
type OverlayPoint = { t: number; x: number; y: number; w: number; h: number; label: string; conf: number; held?: boolean }
type OverlayTrack = { trackIndex: number | null; color: string; pts: OverlayPoint[] }

// Color per object label, shared between the timeline markers and the player overlay.
function trackColor(label: string): string {
  const l = (label || '').toLowerCase()
  if (l === 'person') return '#10a37f'
  if (['vehicle', 'car', 'truck', 'bus', 'motorcycle', 'bicycle'].includes(l)) return '#3b82f6'
  if (l) return '#f59e0b'
  return '#ef4444'
}

// Detections are only sampled at select timestamps. Within a track's observed
// range the box is linearly interpolated between consecutive observations; when
// consecutive observations are farther apart than this, the box is HELD at the
// last known position (drawn dashed) instead of interpolating guessed motion.
const MAX_INTERP_GAP = 5
// Single observations (one-off detections with no track path) stay visible this
// many seconds around their timestamp instead of flashing for a split second.
const SINGLETON_WINDOW = 1.5

function interpolateBox(pts: OverlayPoint[], t: number): OverlayPoint | null {
  if (pts.length === 0) return null
  const eps = 1e-3
  const first = pts[0], last = pts[pts.length - 1]
  // Just outside the observed range: keep the edge box visible briefly (dashed)
  // so objects don't blink in/out at track boundaries.
  if (t < first.t - eps) {
    return t >= first.t - SINGLETON_WINDOW ? { ...first, held: true } : null
  }
  if (t > last.t + eps) {
    return t <= last.t + SINGLETON_WINDOW ? { ...last, held: true } : null
  }
  let lo = 0, hi = pts.length - 1
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1
    if (pts[mid].t <= t) lo = mid
    else hi = mid
  }
  const a = pts[lo]
  const b = pts[Math.min(lo + 1, pts.length - 1)]
  if (a === b || b.t - a.t > MAX_INTERP_GAP) {
    // Single observation (e.g. an untracked detection) or a dead gap: hold the
    // observed box in place (dashed) instead of guessing motion across the gap.
    return { ...a, held: true }
  }
  const alpha = Math.min(1, Math.max(0, (t - a.t) / (b.t - a.t)))
  const lerp = (av: number, bv: number) => av + (bv - av) * alpha
  const nearest = alpha < 0.5 ? a : b
  return { t, x: lerp(a.x, b.x), y: lerp(a.y, b.y), w: lerp(a.w, b.w), h: lerp(a.h, b.h), label: nearest.label, conf: nearest.conf }
}

// Styles a normalized bbox for a container that displays the frame with
// object-fit: contain. When the frame's aspect ratio differs from the
// container's, the image is letterboxed, so the box must be mapped into the
// image's content rect, not placed with raw percentages of the container.
function detectionBoxStyle(d: { bbox_x: number; bbox_y: number; bbox_width: number; bbox_height: number }, frameW: number, frameH: number, containerAspect: number): CSSProperties {
  const fa = frameW > 0 && frameH > 0 ? frameW / frameH : containerAspect
  const cw = Math.min(1, fa / containerAspect)
  const ch = Math.min(1, containerAspect / fa)
  const pct = (v: number) => `${(v * 100).toFixed(3)}%`
  return {
    left: pct((1 - cw) / 2 + d.bbox_x * cw),
    top: pct((1 - ch) / 2 + d.bbox_y * ch),
    width: pct(d.bbox_width * cw),
    height: pct(d.bbox_height * ch),
  }
}

export default function VideoDetail() {
  const { id } = useParams()
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const location = useLocation() as any
  const deepSegment = searchParams.get('segment')
  const deepQuery = searchParams.get('q') || ''
  const deepSim = searchParams.get('sim') || ''
  const deepRel = searchParams.get('rel') || ''
  const deepFrom = searchParams.get('from') || ''
  const deepMatch = searchParams.get('match') || ''
  const navResult = location?.state?.searchResult as any | undefined
  const navMatch: string = navResult?.matched_text || ''
  // Exact snippet from search — URL param survives reload, nav state is fast path.
  const matchedSnippet = deepMatch || navMatch || ''
  const isDeepLink = !!deepSegment
  const videoRef = useRef<HTMLVideoElement>(null)
  const [video, setVideo] = useState<Video | null>(null)
  const [media, setMedia] = useState<MediaMetadata | null>(null)
  const [segments, setSegments] = useState<Segment[]>([])
  const [frames, setFrames] = useState<Frame[]>([])
  const [detections, setDetections] = useState<Detection[]>([])
  const [tracks, setTracks] = useState<Track[]>([])
  const [detToTrack, setDetToTrack] = useState<Record<string, number>>({})
  const [selectedTrack, setSelectedTrack] = useState<number | 'all'>('all')
  const overlayRef = useRef<HTMLCanvasElement>(null)
  const [overlayOn, setOverlayOn] = useState(true)
  const [events, setEvents] = useState<Event[]>([])
  const [eventFilter, setEventFilter] = useState<string>('all')
  const [descriptions, setDescriptions] = useState<SegmentDescription[]>([])
  const [err, setErr] = useState<string | null>(null)
  const [selectedFrame, setSelectedFrame] = useState<Frame | null>(null)
  const [tracksCollapsed, setTracksCollapsed] = useState(false)
  const [eventsCollapsed, setEventsCollapsed] = useState(false)
  const [segmentsCollapsed, setSegmentsCollapsed] = useState(false)
  const [descriptionsCollapsed, setDescriptionsCollapsed] = useState(false)
  const [mediaCollapsed, setMediaCollapsed] = useState(false)
  const [analysisCollapsed, setAnalysisCollapsed] = useState(false)
  const [activeTab, setActiveTab] = useState<'segments' | 'tracks' | 'events' | 'descriptions'>('segments')
  const [pvQuery, setPvQuery] = useState('')
  const [pvResults, setPvResults] = useState<any[] | null>(null)
  const [pvLoading, setPvLoading] = useState(false)
  const [pvError, setPvError] = useState<string | null>(null)
  const { confirm, notify } = useDialog()
  const [filterSegmentId, setFilterSegmentId] = useState<string | null>(null)

  const fetchAll = async () => {
    if (!id) return
    try {
      const v = await client.get<Video>(`/api/v1/videos/${id}`)
      setVideo(v)
      try { setMedia(await client.get<MediaMetadata>(`/api/v1/videos/${id}/media`)) } catch { setMedia(null) }
      try { setSegments(await client.get<Segment[]>(`/api/v1/videos/${id}/segments`)) } catch { setSegments([]) }
      try { setFrames(await client.get<Frame[]>(`/api/v1/videos/${id}/frames`)) } catch { setFrames([]) }
      try { setDetections(await client.get<Detection[]>(`/api/v1/videos/${id}/detections`)) } catch { setDetections([]) }
      try {
        const t = await client.get<Track[]>(`/api/v1/videos/${id}/tracks`)
        setTracks(t)
        const map: Record<string, number> = {}
        await Promise.all(t.map(async (tr) => {
          try {
            const links: any[] = await client.get<any[]>(`/api/v1/tracks/${tr.id}/detections`)
            links.forEach((l: any) => { map[l.detection_id] = tr.track_index })
          } catch {}
        }))
        setDetToTrack(map)
      } catch { setTracks([]); setDetToTrack({}) }
      try { setEvents(await client.get<Event[]>(`/api/v1/videos/${id}/events`)) } catch { setEvents([]) }
      try { setDescriptions(await client.get<SegmentDescription[]>(`/api/v1/videos/${id}/descriptions`)) } catch { setDescriptions([]) }
      setErr(null)
    } catch (e: any) { setErr(e.message) }
  }

  const fetchEvidence = async (segmentId: string) => {
    if (!id) return
    try {
      const v = await client.get<Video>(`/api/v1/videos/${id}`)
      setVideo(v)
      try { setMedia(await client.get<MediaMetadata>(`/api/v1/videos/${id}/media`)) } catch { setMedia(null) }
      try { setSegments(await client.get<Segment[]>(`/api/v1/videos/${id}/segments`)) } catch { setSegments([]) }
      // Scoped evidence: single bounded request instead of full-video fan-out.
      const ev: any = await client.get(`/api/v1/videos/${id}/segments/${segmentId}/evidence`)
      setFrames(ev.frames || [])
      setDetections(ev.detections || [])
      const scopedTracks: Track[] = ev.tracks || []
      setTracks(scopedTracks)
      setEvents(ev.events || [])
      if (ev.description) setDescriptions([ev.description])
      else { try { setDescriptions(await client.get<SegmentDescription[]>(`/api/v1/videos/${id}/descriptions`)) } catch { setDescriptions([]) } }
      // Track→detection links only for scoped tracks (bounded, not whole video).
      const map: Record<string, number> = {}
      await Promise.all(scopedTracks.slice(0, 20).map(async (tr) => {
        try {
          const links: any[] = await client.get<any[]>(`/api/v1/tracks/${tr.id}/detections`)
          links.forEach((l: any) => { map[l.detection_id] = tr.track_index })
        } catch {}
      }))
      setDetToTrack(map)
      setFilterSegmentId(segmentId)
      setActiveTab('descriptions')
      setAnalysisCollapsed(false)
      setErr(null)
    } catch (e: any) {
      // Fall back to full fetch if evidence endpoint fails.
      await fetchAll()
      setFilterSegmentId(segmentId)
    }
  }

  useEffect(() => {
    if (deepSegment) fetchEvidence(deepSegment)
    else fetchAll()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id])

  const clearDeepLink = () => {
    setFilterSegmentId(null)
    setSearchParams({}, { replace: true })
    navigate(`/videos/${id}`, { replace: true })
    fetchAll()
  }
  useEffect(() => {
    const tStr = searchParams.get('t')
    if (tStr == null) return
    const tVal = parseFloat(tStr)
    if (isNaN(tVal) || !isFinite(tVal)) return
    let cancelled = false
    let timeoutId: any
    const trySeek = () => {
      if (cancelled) return
      const el = videoRef.current
      if (!el) {
        timeoutId = setTimeout(trySeek, 100)
        return
      }
      const doSeek = () => {
        if (cancelled) return
        try {
          el.currentTime = Math.max(0, tVal)
          el.play().catch(() => {})
        } catch {}
      }
      if (el.readyState >= 1) doSeek()
      else el.addEventListener('loadedmetadata', doSeek, { once: true })
    }
    trySeek()
    return () => { cancelled = true; if (timeoutId) clearTimeout(timeoutId) }
  }, [searchParams, video])
  useEffect(() => {
    if (!video || (video.status !== 'UPLOADED' && video.status !== 'PROCESSING')) return
    const t = setInterval(fetchAll, 3000)
    return () => clearInterval(t)
  }, [video?.status])

  // -- Bounding-box playback overlay -----------------------------------------
  // frame_id -> video-timeline timestamp, needed to place detections on playback.
  // Frame timestamps are recorded on the source file's PTS timeline; the browser
  // playback timeline starts at 0, so container start_time offsets must be
  // removed for the boxes to line up with what is on screen.
  const frameTimesById = useMemo(() => {
    const m: Record<string, number> = {}
    const st = media?.start_time
    const offset = st != null && isFinite(st) && st > 0 ? st : 0
    frames.forEach(f => { m[f.id] = f.timestamp_seconds - offset })
    return m
  }, [frames, media?.start_time])

  // Group detections per track into sorted (timestamp, bbox) keypoints.
  // Detections without a track become singleton keys: their box is shown for a
  // short window around the timestamp (dashed "held") instead of interpolated.
  const overlayTracks = useMemo<OverlayTrack[]>(() => {
    const byKey = new Map<string, OverlayPoint[]>()
    detections.forEach(d => {
      const t = frameTimesById[d.frame_id]
      if (t == null || !isFinite(t)) return
      const ti = detToTrack[d.id]
      const key = ti !== undefined ? `t${ti}` : `d${d.id}`
      let arr = byKey.get(key)
      if (!arr) { arr = []; byKey.set(key, arr) }
      arr.push({ t, x: d.bbox_x, y: d.bbox_y, w: d.bbox_width, h: d.bbox_height, label: d.label, conf: d.confidence })
    })
    const out: OverlayTrack[] = []
    byKey.forEach((pts, key) => {
      pts.sort((p, q) => p.t - q.t)
      const trackIndex = key.startsWith('t') ? Number(key.slice(1)) : null
      out.push({ trackIndex, color: trackColor(pts[pts.length - 1].label), pts })
    })
    return out
  }, [detections, frameTimesById, detToTrack])

  // Redraw every animation frame while the overlay is enabled: boxes are
  // interpolated at the current playback time, not only at sampled timestamps.
  useEffect(() => {
    const canvas = overlayRef.current
    if (!overlayOn || !canvas) return
    let raf = 0
    const draw = () => {
      const el = videoRef.current
      const cw = canvas.clientWidth, ch = canvas.clientHeight
      const dpr = window.devicePixelRatio || 1
      const pw = Math.max(1, Math.round(cw * dpr)), ph = Math.max(1, Math.round(ch * dpr))
      if (canvas.width !== pw || canvas.height !== ph) { canvas.width = pw; canvas.height = ph }
      const ctx = canvas.getContext('2d')
      if (el && ctx && cw > 0 && ch > 0) {
        ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
        ctx.clearRect(0, 0, cw, ch)
        const vw = el.videoWidth, vh = el.videoHeight
        if (vw > 0 && vh > 0) {
          // object-fit: contain — the video may be letterboxed inside the
          // player; compute the content rect so normalized bboxes land on the
          // right pixels for any video aspect ratio.
          const scale = Math.min(cw / vw, ch / vh)
          const dw = vw * scale, dh = vh * scale
          const ox = (cw - dw) / 2, oy = (ch - dh) / 2
          const now = el.currentTime
          overlayTracks.forEach(tr => {
            const box = interpolateBox(tr.pts, now)
            if (!box) return
            const isDim = selectedTrack !== 'all' && tr.trackIndex !== selectedTrack
            const x = ox + box.x * dw, y = oy + box.y * dh, bw = box.w * dw, bh = box.h * dh
            if (bw < 2 || bh < 2) return
            ctx.globalAlpha = isDim ? 0.25 : box.held ? 0.85 : 1
            ctx.lineWidth = 2
            // Interpolated boxes are solid; held boxes (between far-apart
            // observations) are dashed to mark them as approximate positions.
            ctx.setLineDash(box.held ? [5, 4] : [])
            ctx.strokeStyle = tr.color
            ctx.strokeRect(x, y, bw, bh)
            ctx.setLineDash([])
            const text = `${box.label} ${(box.conf * 100).toFixed(0)}%${tr.trackIndex !== null ? ` T${tr.trackIndex}` : ''}`
            ctx.font = '700 11px ui-sans-serif, system-ui, -apple-system, sans-serif'
            const tw = ctx.measureText(text).width
            const ly = Math.max(0, y - 16)
            ctx.fillStyle = tr.color
            ctx.fillRect(x, ly, tw + 8, 15)
            ctx.fillStyle = tr.color === '#f59e0b' ? '#1f2937' : '#ffffff'
            ctx.fillText(text, x + 4, ly + 11)
            ctx.globalAlpha = 1
          })
        }
      }
      raf = requestAnimationFrame(draw)
    }
    raf = requestAnimationFrame(draw)
    return () => cancelAnimationFrame(raf)
  }, [overlayOn, overlayTracks, selectedTrack])

  const onDelete = async () => {
    if (!video) return
    const ok = await confirm({
      title: 'Delete video',
      message: `Delete "${video.filename}"? This will permanently delete the video and all its frames, tracks and metadata. This cannot be undone.`,
      confirmLabel: 'Delete',
      danger: true,
    })
    if (!ok) return
    try { await client.del(`/api/v1/videos/${video.id}`); navigate('/videos') } catch (e: any) { await notify({ title: 'Delete failed', message: e.message }) }
  }

  const seekTo = (sec: number) => {
    const el = videoRef.current
    if (!el) return
    try {
      el.currentTime = Math.max(0, sec)
      el.play().catch(() => {})
      el.scrollIntoView({ behavior: 'smooth', block: 'center' })
    } catch {}
  }

  const doPerVideoSearch = async () => {
    const q = pvQuery.trim(); if (!q) { setPvError('Query is required'); return }
    if (!video) return
    setPvLoading(true); setPvError(null)
    try {
      const res: any = await client.post('/api/v1/search', { query: q, video_id: video.id, limit: 8 })
      setPvResults(res.results || [])
    } catch (e: any) { setPvError(e.message || 'Search failed'); setPvResults([]) } finally { setPvLoading(false) }
  }

  const formatTs = (sec: number) => {
    const s = Math.floor(sec), h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec2 = s % 60
    if (h >= 1) return `${String(h).padStart(2,'0')}:${String(m).padStart(2,'0')}:${String(sec2).padStart(2,'0')}`
    return `${String(m).padStart(2,'0')}:${String(sec2).padStart(2,'0')}`
  }

  if (err) return <div className="card" style={{ padding: 16 }}><div style={{ color: '#b91c1c', marginBottom: 8 }}>Error: {err}</div><button className="btn" onClick={fetchAll}>Retry</button></div>
  if (!video) return <div className="card" style={{ padding: 32, textAlign: 'center', color: 'var(--muted)' }}>Loading…</div>

  // filtered view when a per-video search result is selected
  const activeSegment = filterSegmentId ? segments.find(s => s.id === filterSegmentId) : null
  const displaySegments = activeSegment ? [activeSegment] : segments
  const displayTracks = activeSegment ? tracks.filter(t => t.start_timestamp < activeSegment.end_time && t.end_timestamp > activeSegment.start_time) : tracks
  const displayEvents = activeSegment ? events.filter(e => {
    const segStart = activeSegment.start_time
    const segEnd = activeSegment.end_time
    const evEnd = e.end_timestamp ?? e.start_timestamp
    return evEnd > segStart && e.start_timestamp < segEnd
  }) : events
  const displayDescriptions = activeSegment ? descriptions.filter(d => d.segment_id === filterSegmentId) : descriptions
  const displayFrames = activeSegment ? frames.filter(f => f.segment_id === filterSegmentId) : frames
  const displayDetections = activeSegment ? detections.filter(d => d.segment_id === filterSegmentId) : detections
  const detectionsByFrame = displayDetections.reduce<Record<string, Detection[]>>((acc, d) => { (acc[d.frame_id] = acc[d.frame_id] || []).push(d); return acc }, {})
  const summary = displayDetections.reduce<Record<string, number>>((acc, d) => { acc[d.label] = (acc[d.label] || 0) + 1; return acc }, {})
  const frameCountBySeg = displayFrames.reduce<Record<string, number>>((a, f) => { a[f.segment_id] = (a[f.segment_id] || 0) + 1; return a }, {})
  const detCountBySeg = displayDetections.reduce<Record<string, number>>((a, d) => { a[d.segment_id] = (a[d.segment_id] || 0) + 1; return a }, {})
  const imgSrc = (f: Frame) => client.imageUrl(`/api/v1/videos/${video.id}/frames/${f.id}/image`)
  const isReady = video.status === 'READY'
  const total = media?.duration_seconds ?? (segments.length ? Math.max(...segments.map(s => s.end_time)) : 0) ?? 0
  const effectiveTotal = total > 0 ? total : 1
  const tickLabels = (() => {
    if (total <= 0) return ['0:00']
    const steps = 4
    const labels: string[] = []
    for (let i = 0; i <= steps; i++) labels.push(formatDur((total * i) / steps))
    return labels
  })()

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
      <div>
        <Link to="/videos" style={{ fontSize: 11, color: 'var(--muted)', display: 'inline-flex', alignItems: 'center', gap: 4 }}>← Back to videos</Link>
        {(isDeepLink || filterSegmentId) && activeSegment && (
          <div style={{ marginTop: 8, display: 'flex', alignItems: 'center', gap: 8, padding: '8px 10px', background: '#fefce8', border: '1px solid #fde68a', borderRadius: 8, fontSize: 12, flexWrap: 'wrap' }}>
            <span style={{ fontWeight: 700 }}>Matched {deepQuery ? `"${deepQuery}"` : 'segment'}{deepRel ? ` (relevance ${deepRel}%)` : deepSim ? ` (${deepSim}%)` : navResult?.rerank_score_normalized != null ? ` (relevance ${Math.round(navResult.rerank_score_normalized * 100)}%)` : navResult?.similarity ? ` (${Math.round(navResult.similarity * 100)}%)` : ''}</span>
            <span style={{ color: 'var(--muted)' }}>• Segment #{activeSegment.segment_index} {formatDur(activeSegment.start_time)}–{formatDur(activeSegment.end_time)}{deepFrom ? ` • from ${deepFrom} search` : ''} • showing only this segment</span>
            <button className="btn" onClick={clearDeepLink} style={{ marginLeft: 'auto', padding: '4px 8px', fontSize: 11, background: 'white' }}>Show full video ✕</button>
          </div>
        )}
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginTop: 4, gap: 12 }}>
          <div>
            <h1 style={{ fontSize: 20, fontWeight: 700, margin: '0 0 4px 0', letterSpacing: -0.02 }}>{video.filename}</h1>
            <div style={{ fontSize: 11, color: 'var(--muted)', display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
              <span>{new Date(video.created_at).toLocaleDateString('en-US', { month: 'short', day: '2-digit', year: 'numeric' })}</span><span className="dot" /><span>{media?.duration_seconds ? formatDur(media.duration_seconds) : '-'}</span><span className="dot" /><span>{segments.length} segments</span><span className="dot" /><span className={`badge ${isReady ? 'badge-green' : 'badge-gray'}`} style={{ fontSize: 10 }}>{isReady ? 'Processed' : video.status}</span>
            </div>
          </div>
          <button className="btn" style={{ color: '#b91c1c', borderColor: '#fecaca', background: '#fef2f2' }} onClick={onDelete}>Delete Video</button>
        </div>
      </div>

      <div className="player-layout" style={{ display: 'flex', gap: 14 }}>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div className="player-wrap">
            <div className="player-media">
              <video ref={videoRef} controls preload="metadata" style={{ width: '100%', height: '100%', objectFit: 'contain', background: '#000' }} src={client.imageUrl(`/api/v1/videos/${video.id}/stream`)} poster={frames[0] ? imgSrc(frames[0]) : undefined} />
              <canvas ref={overlayRef} style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none', display: overlayOn ? 'block' : 'none' }} />
              <button onClick={() => setOverlayOn(v => !v)} title="Interpolated bounding-box overlay for tracked objects" style={{ position: 'absolute', top: 8, right: 8, zIndex: 3, padding: '5px 10px', fontSize: 11, fontWeight: 600, borderRadius: 8, border: '1px solid rgba(255,255,255,0.25)', background: overlayOn ? 'rgba(16,163,127,0.92)' : 'rgba(15,17,20,0.65)', color: '#fff', cursor: 'pointer', backdropFilter: 'blur(4px)' }}>
                {overlayOn ? '◉ Track boxes' : '○ Track boxes'}
              </button>
            </div>
          </div>
        </div>
        <div className="info-card" style={{ display: 'flex', flexDirection: 'column', maxHeight: 520, overflowY: 'auto' }}>
          <h4 style={{ flexShrink: 0 }}>Video Information</h4>
          <div className="info-row"><span className="info-label">Name</span><span className="info-value">{video.filename}</span></div>
          <div className="info-row"><span className="info-label">File Path</span><span className="info-value" title={video.source_path || ''}>{video.source_path || '-'}</span></div>
          <div className="info-row"><span className="info-label">Date</span><span className="info-value">{new Date(video.created_at).toLocaleString()}</span></div>
          <div className="info-row"><span className="info-label">Duration</span><span className="info-value">{media?.duration_seconds ? formatDur(media.duration_seconds) : '-'}</span></div>
          <div className="info-row"><span className="info-label">Segments</span><span className="info-value">{segments.length}</span></div>
          <div className="info-row"><span className="info-label">Status</span><span className={`badge ${isReady ? 'badge-green' : 'badge-gray'}`} style={{ marginLeft: 'auto' }}>{isReady ? 'Processed' : video.status}</span></div>

          <div style={{ height: 1, background: 'var(--border)', margin: '12px 0', flexShrink: 0 }} />
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: mediaCollapsed ? 0 : 8, flexShrink: 0 }}>
            <h4 style={{ margin: 0, fontSize: 12, fontWeight: 600 }}>File & Media Details</h4>
            <button className="collapse-btn" onClick={() => setMediaCollapsed(v => !v)} aria-label={mediaCollapsed ? 'Expand' : 'Collapse'} style={{ width: 22, height: 22, fontSize: 10 }}>{mediaCollapsed ? '▶' : '▼'}</button>
          </div>
          {!mediaCollapsed && (
            <div style={{ flexShrink: 0 }}>
              <dl className="kv" style={{ fontSize: 11.5, gap: '4px 8px' }}>
                <dt>Video ID</dt><dd style={{ fontSize: 11 }}>{video.id.slice(0, 8)}…</dd>
                <dt>Source type</dt><dd>{video.source_type}</dd>
                <dt>Storage key</dt><dd style={{ wordBreak: 'break-all', fontSize: 11 }}>{video.storage_key || '-'}</dd>
                <dt>Size</dt><dd>{(video.size_bytes / 1024 / 1024).toFixed(2)} MB</dd>
                <dt>MIME</dt><dd>{video.mime_type}</dd>
                <dt>Content hash</dt><dd style={{ wordBreak: 'break-all', fontSize: 10 }}>{video.content_hash?.slice(0, 16) || '-'}…</dd>
                <dt>Created</dt><dd style={{ fontSize: 11 }}>{new Date(video.created_at).toLocaleString()}</dd>
                <dt>Updated</dt><dd style={{ fontSize: 11 }}>{new Date(video.updated_at).toLocaleString()}</dd>
                {video.processing_error && <><dt>Error</dt><dd style={{ color: '#b91c1c', fontSize: 11 }}>{video.processing_error}</dd></>}
              </dl>
              <div style={{ height: 1, background: 'var(--border)', margin: '10px 0' }} />
              {media ? (
                <dl className="kv" style={{ fontSize: 11.5, gap: '4px 8px' }}>
                  <dt>Format</dt><dd>{media.format_name || '-'} {media.format_long_name ? `(${media.format_long_name})` : ''}</dd>
                  <dt>Duration</dt><dd>{media.duration_seconds?.toFixed(2) + 's' || '-'}</dd>
                  <dt>Bitrate</dt><dd>{media.bit_rate ? `${(Number(media.bit_rate) / 1000).toFixed(0)} kb/s` : '-'}</dd>
                  <dt>Video codec</dt><dd>{media.video_codec || '-'}</dd>
                  <dt>Resolution</dt><dd>{media.video_width && media.video_height ? `${media.video_width}×${media.video_height}` : '-'}</dd>
                  <dt>FPS</dt><dd>{media.video_fps?.toFixed(2) || '-'} {media.video_fps_raw ? `(${media.video_fps_raw})` : ''}</dd>
                  <dt>Audio codec</dt><dd>{media.audio_codec || 'no audio'}</dd>
                  <dt>Sample rate</dt><dd>{media.audio_sample_rate || '-'}</dd>
                  <dt>Channels</dt><dd>{media.audio_channels ?? '-'}</dd>
                  <dt>Channel layout</dt><dd>{media.audio_channel_layout || '-'}</dd>
                </dl>
              ) : <div style={{ padding: 8, color: 'var(--muted)', fontSize: 11, textAlign: 'center' }}>No media metadata yet</div>}
            </div>
          )}
        </div>
      </div>

      <div className="card timeline-card">
        <div className="timeline-head">
          <div className="timeline-title">Timeline</div>
          <div className="legend"><span><span className="legend-dot" style={{ background: '#10a37f' }} />Person</span><span><span className="legend-dot" style={{ background: '#3b82f6' }} />Vehicle</span><span><span className="legend-dot" style={{ background: '#f59e0b' }} />Object</span><span><span className="legend-dot" style={{ background: '#ef4444' }} />Event</span></div>
        </div>
        <div className="timeline-track">
          <div className="timeline-ticks">
            {segments.map(s => (
              <div key={s.id} style={{ position: 'absolute', left: `${(s.start_time / effectiveTotal) * 100}%`, width: `${Math.max(0.5, (s.duration / effectiveTotal) * 100)}%`, top: 4, bottom: 4, background: '#d4ede5', borderRight: '1px solid #b8ddd0', borderRadius: 2 }} title={`S${s.segment_index} ${formatDur(s.start_time)}-${formatDur(s.end_time)}`} />
            ))}
            {tracks.map(tr => {
              const left = (tr.start_timestamp / effectiveTotal) * 100
              const w = ((tr.end_timestamp - tr.start_timestamp) / effectiveTotal) * 100
              if (left > 100 || left < 0) return null
              return <div key={`tr-${tr.id}`} style={{ position: 'absolute', left: `${left}%`, width: `${Math.max(0.8, Math.min(w, 100 - left))}%`, top: 10, height: 4, background: trackColor(tr.label), opacity: 0.55, borderRadius: 1 }} title={`${tr.label} #${tr.track_index} ${formatDur(tr.start_timestamp)}-${formatDur(tr.end_timestamp)}`} />
            })}
            {events.map(ev => {
              const left = (ev.start_timestamp / effectiveTotal) * 100
              if (left > 100 || left < 0) return null
              return <div key={`ev-${ev.id}`} className="marker" style={{ left: `${left}%`, background: '#ef4444', height: 14, width: 2.5, opacity: 0.95, borderRadius: 1, top: 7 }} title={`${ev.event_type} ${ev.label} @${formatDur(ev.start_timestamp)}`} />
            })}
          </div>
        </div>
        <div className="timeline-labels">{tickLabels.map(l => <span key={l}>{l}</span>)}</div>
      </div>

      <div className="card" style={{ padding: 12 }}>
        <h3 style={{ margin: '0 0 8px 0', fontSize: 13, fontWeight: 600 }}>Search in this video</h3>
        <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 8 }}>Find moments in this video using natural language</div>
        <div className="search-hero">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#9aa3b1" strokeWidth="1.8"><circle cx="11" cy="11" r="6" /><path d="M20 20L15.3 15.3" /></svg>
          <input placeholder="person entering, vehicle near entrance, person at counter" value={pvQuery} onChange={e => setPvQuery(e.target.value)} onKeyDown={e => e.key === 'Enter' && doPerVideoSearch()} style={{ flex: 1, border: 'none', outline: 'none', fontSize: 12 }} />
          <button className="btn btn-primary" onClick={doPerVideoSearch} style={{ padding: '6px 12px', borderRadius: 8, fontSize: 12 }}>{pvLoading ? 'Searching…' : 'Search'}</button>
        </div>
        {pvError && <div style={{ marginTop: 8, color: '#b91c1c', fontSize: 12 }}>{pvError}</div>}
        {filterSegmentId && (() => {
          const seg = segments.find(s => s.id === filterSegmentId)
          return (
            <div style={{ marginTop: 10, display: 'flex', alignItems: 'center', gap: 8, padding: '8px 10px', background: '#e6f6ef', border: '1px solid #b8e0d0', borderRadius: 8, fontSize: 12 }}>
              <span style={{ color: '#0e8f6f', fontWeight: 600 }}>Filtered to segment {seg ? `#${seg.segment_index} • ${formatDur(seg.start_time)} – ${formatDur(seg.end_time)}` : filterSegmentId!.slice(0,8)}</span>
              <span style={{ color: 'var(--muted)' }}>• Analysis & Frames below show only this segment</span>
              <button className="btn" onClick={() => setFilterSegmentId(null)} style={{ marginLeft: 'auto', padding: '4px 8px', fontSize: 11, background: 'white' }}>Clear filter ✕</button>
            </div>
          )
        })()}
        {pvResults !== null && (
          <div style={{ marginTop: 10 }}>
            {pvLoading ? <div style={{ padding: 16, textAlign: 'center', color: 'var(--muted)', fontSize: 12 }}>Searching…</div> : pvResults.length === 0 ? <div style={{ padding: 16, textAlign: 'center', color: 'var(--muted)', fontSize: 12 }}>No matching moments in this video</div> : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {pvResults.map((r: any) => {
                  const matched: string = r.matched_text || ''
                  const desc: string = r.description || ''
                  const idx = matched && desc ? desc.indexOf(matched) : -1
                  const ciIdx = idx === -1 && matched ? desc.toLowerCase().indexOf(matched.toLowerCase()) : -1
                  const hasHighlight = matched && (idx !== -1 || ciIdx !== -1)
                  const pos = idx !== -1 ? idx : ciIdx
                  const isSelected = filterSegmentId === r.segment_id
                  return (
                    <div key={r.segment_id} onClick={() => { setFilterSegmentId(r.segment_id); seekTo(r.start_time); setActiveTab('segments'); setAnalysisCollapsed(false) }} style={{ display: 'flex', gap: 10, padding: 10, background: isSelected ? '#e6f6ef' : '#f8f9f8', border: `1px solid ${isSelected ? '#b8e0d0' : 'var(--border)'}`, borderRadius: 8, cursor: 'pointer' }}>
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 3 }}>{formatTs(r.start_time)} – {formatTs(r.end_time)} • Similarity {Math.round((r.similarity||0)*100)}%</div>
                        <div style={{ fontSize: 12, lineHeight: 1.4, color: 'var(--text)' }}>
                          {hasHighlight && matched ? (() => {
                            const before = desc.slice(0, pos)
                            const match = desc.slice(pos, pos + matched.length)
                            const after = desc.slice(pos + matched.length)
                            return <>{before}<mark style={{ background: '#fef08a', color: '#422006', padding: '0 2px', borderRadius: 3, fontWeight: 600 }}>{match}</mark>{after}</>
                          })() : desc}
                        </div>
                      </div>
                      <span style={{ alignSelf: 'center', fontSize: 11, color: 'var(--accent)', fontWeight: 600, whiteSpace: 'nowrap' }}>▶ Play</span>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
        )}
      </div>

      <div className="card" style={{ padding: '10px 14px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 12, marginBottom: analysisCollapsed ? 0 : 12 }}>
          <h3 style={{ margin: 0, fontSize: 13, fontWeight: 600 }}>Analysis</h3>
          <button className="collapse-btn" onClick={() => setAnalysisCollapsed(v => !v)}>{analysisCollapsed ? '▶' : '▼'}</button>
        </div>
        {!analysisCollapsed && (
          <>
            <div className="tabs">
              <div className={`tab ${activeTab === 'segments' ? 'active' : ''}`} onClick={() => setActiveTab('segments')}><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"><rect x="3" y="3" width="18" height="18" rx="2" /><path d="M9 3v18M15 3v18M3 9h18M3 15h18" /></svg> Segments</div>
              <div className={`tab ${activeTab === 'tracks' ? 'active' : ''}`} onClick={() => setActiveTab('tracks')}><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"><circle cx="12" cy="8" r="3" /><path d="M6 20c0-3 4-5 6-5s6 2 6 5" /></svg> Tracks</div>
              <div className={`tab ${activeTab === 'events' ? 'active' : ''}`} onClick={() => setActiveTab('events')}><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M3 9h18" /><path d="M8 3v4M16 3v4" /></svg> Events</div>
              <div className={`tab ${activeTab === 'descriptions' ? 'active' : ''}`} onClick={() => setActiveTab('descriptions')}><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"><path d="M14 2H7a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 2v5h5"/><path d="M10 13H8M16 17H8M13 13h1"/></svg> Descriptions</div>
            </div>

            {activeTab === 'segments' && (
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <button className="collapse-btn" onClick={() => setSegmentsCollapsed(v => !v)} title={segmentsCollapsed ? 'Expand' : 'Collapse'} style={{ width: 24, height: 24, fontSize: 10 }}>{segmentsCollapsed ? '▶' : '▼'}</button>
                  <span style={{ fontSize: 11, color: 'var(--muted)' }}>{displaySegments.length} segments{filterSegmentId ? ' • filtered' : ''}</span>
                </div>
                {!segmentsCollapsed && (
                  <div className="table-wrap" style={{ border: 'none' }}>
                    <table className="table">
                      <thead><tr><th>Segment</th><th>Start</th><th>End</th><th>Duration</th><th>Frames</th><th>Detections</th><th></th></tr></thead>
                      <tbody>
                        {displaySegments.map(s => (
                          <tr key={s.id} id={`seg-row-${s.id}`} onClick={() => seekTo(s.start_time)} style={{ cursor: 'pointer' }}><td>{s.segment_index}</td><td>{s.start_time.toFixed(2)}s</td><td>{s.end_time.toFixed(2)}s</td><td>{s.duration.toFixed(2)}s</td><td>{frameCountBySeg[s.id] || 0}</td><td>{detCountBySeg[s.id] || 0}</td><td><button className="btn" onClick={(e) => { e.stopPropagation(); seekTo(s.start_time) }} style={{ padding: '3px 6px', borderRadius: 20, background: '#e6f6ef', border: '1px solid #cfe9de', color: '#0e8f6f', fontSize: 10 }}>▶</button></td></tr>
                        ))}
                        {displaySegments.length === 0 && <tr><td colSpan={7} style={{ textAlign: 'center', color: 'var(--muted)', padding: 24 }}>No segments</td></tr>}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}

            {activeTab === 'tracks' && (
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <button className="collapse-btn" onClick={() => setTracksCollapsed(v => !v)} title={tracksCollapsed ? 'Expand' : 'Collapse'} style={{ width: 24, height: 24, fontSize: 10 }}>{tracksCollapsed ? '▶' : '▼'}</button>
                  <span style={{ fontSize: 11, color: 'var(--muted)' }}>{displayTracks.length} tracks{filterSegmentId ? ' • filtered' : ''}</span>
                </div>
                {!tracksCollapsed && (
                  displayTracks.length === 0 ? <div className="empty">No tracks{filterSegmentId ? ' in this segment' : ''}</div> : (
                    <>
                      <div style={{ marginBottom: 12 }}>
                        <select value={selectedTrack === 'all' ? 'all' : String(selectedTrack)} onChange={e => setSelectedTrack(e.target.value === 'all' ? 'all' : Number(e.target.value))} style={{ minWidth: 320, padding: '8px 10px', fontSize: 12, borderRadius: 8, border: '1px solid var(--border)' }}>
                          <option value="all">All tracks ({displayTracks.length}){filterSegmentId ? ' in segment' : ''}</option>
                          {displayTracks.map(t => <option key={t.id} value={String(t.track_index)}>Track {t.track_index} — {t.label} — {t.start_timestamp.toFixed(1)}s → {t.end_timestamp.toFixed(1)}s — {t.detection_count} dets</option>)}
                        </select>
                      </div>
                      {selectedTrack === 'all' ? (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 8, maxHeight: 260, overflowY: 'auto', paddingRight: 4 }}>
                          {displayTracks.filter(t => selectedTrack === 'all' || t.track_index === selectedTrack).map(t => (
                            <div key={t.id} onClick={() => seekTo(t.start_timestamp)} title="Click to play from here" style={{ display: 'flex', gap: 12, alignItems: 'center', padding: '8px 10px', background: '#f8f9f8', border: '1px solid var(--border)', borderRadius: 8, cursor: 'pointer' }}>
                              <span style={{ minWidth: 80, fontWeight: 700, textTransform: 'uppercase', fontSize: 12 }}>{t.label}</span>
                              <span style={{ fontSize: 12 }}>Track {t.track_index}</span>
                              <span style={{ fontSize: 12, color: 'var(--muted)' }}>{t.start_timestamp.toFixed(1)}s → {t.end_timestamp.toFixed(1)}s</span>
                              <span style={{ fontSize: 12 }}>{t.detection_count} detections</span>
                              <span style={{ color: 'var(--muted)', fontSize: 11 }}>{t.tracker_name} v{t.tracker_version}</span>
                              <span style={{ marginLeft: 'auto', fontSize: 10, color: 'var(--accent)' }}>▶</span>
                            </div>
                          ))}
                        </div>
                      ) : (
                        (() => {
                          const t = displayTracks.find(x => x.track_index === selectedTrack)
                          if (!t) return <div className="empty">Track not found{filterSegmentId ? ' in this segment' : ''}</div>
                          return <div onClick={() => seekTo(t.start_timestamp)} title="Click to play from here" style={{ display: 'flex', gap: 12, alignItems: 'center', flexWrap: 'wrap', padding: 10, background: '#f8f9f8', border: '1px solid var(--border)', borderRadius: 8, cursor: 'pointer' }}><span style={{ minWidth: 80, fontWeight: 700, textTransform: 'uppercase', fontSize: 12 }}>{t.label}</span><span style={{ fontSize: 12 }}>Track {t.track_index}</span><span style={{ fontSize: 12 }}>{t.start_timestamp.toFixed(1)}s → {t.end_timestamp.toFixed(1)}s</span><span style={{ fontSize: 12 }}>{t.detection_count} detections</span><span style={{ color: 'var(--muted)', fontSize: 11 }}>{t.tracker_name} v{t.tracker_version}</span><span style={{ marginLeft: 'auto', fontSize: 10, color: 'var(--accent)' }}>▶ Play</span></div>
                        })()
                      )}
                    </>
                  )
                )}
              </div>
            )}

            {activeTab === 'events' && (
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <button className="collapse-btn" onClick={() => setEventsCollapsed(v => !v)} title={eventsCollapsed ? 'Expand' : 'Collapse'} style={{ width: 24, height: 24, fontSize: 10 }}>{eventsCollapsed ? '▶' : '▼'}</button>
                  <span style={{ fontSize: 11, color: 'var(--muted)' }}>{displayEvents.length} events{filterSegmentId ? ' • filtered' : ''}</span>
                </div>
                {!eventsCollapsed && (
                  <>
                    <div style={{ display: 'flex', gap: 8, marginBottom: 12, flexWrap: 'wrap' }}>
                      <select value={eventFilter} onChange={e => setEventFilter(e.target.value)} style={{ padding: '6px 8px', borderRadius: 8, border: '1px solid var(--border)', fontSize: 12 }}>
                        <option value="all">All events</option>
                        <option value="OBJECT_APPEARED">Appeared</option>
                        <option value="OBJECT_PRESENT">Present</option>
                        <option value="OBJECT_DISAPPEARED">Last observed</option>
                        <option value="OBJECT_MOVED">Movement</option>
                      </select>
                      {selectedTrack !== 'all' && <span style={{ fontSize: 11, color: 'var(--muted)', alignSelf: 'center' }}>Filtered to Track {selectedTrack} + event filter{filterSegmentId ? ' + segment' : ''}</span>}
                    </div>
                    {media?.duration_seconds && displayTracks.length > 0 && (
                      <div style={{ marginBottom: 12, padding: 8, background: '#f8f9f8', borderRadius: 8, border: '1px solid var(--border)' }}>
                        <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 4 }}>Timeline — Present (blue) & Movement (green)</div>
                        {displayTracks.filter(t => selectedTrack === 'all' || t.track_index === selectedTrack).map(t => {
                          const dur = media.duration_seconds || effectiveTotal
                          const presentEvents = displayEvents.filter(e => e.track_id === t.id && e.event_type === 'OBJECT_PRESENT')
                          const movedEvents = displayEvents.filter(e => e.track_id === t.id && e.event_type === 'OBJECT_MOVED')
                          return (
                            <div key={t.id} style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                              <span style={{ width: 90, fontSize: 11, fontWeight: 600 }}>T{t.track_index} {t.label}</span>
                              <div style={{ flex: 1, height: 18, background: 'white', border: '1px solid var(--border)', borderRadius: 4, position: 'relative', overflow: 'hidden' }}>
                                {presentEvents.map(e => { const s = (e.start_timestamp / dur) * 100; const ee = ((e.end_timestamp ?? e.start_timestamp) / dur) * 100; return <div key={e.id} style={{ position: 'absolute', left: `${s}%`, width: `${Math.max(1, ee - s)}%`, top: 0, bottom: 0, background: '#3b82f6', opacity: 0.7 }} title={`${e.start_timestamp.toFixed(1)}→${(e.end_timestamp ?? e.start_timestamp).toFixed(1)}`} /> })}
                                {movedEvents.map(e => { const s = (e.start_timestamp / dur) * 100; const ee = ((e.end_timestamp ?? e.start_timestamp) / dur) * 100; return <div key={e.id} style={{ position: 'absolute', left: `${s}%`, width: `${Math.max(1, ee - s)}%`, top: 4, bottom: 4, background: '#10a37f', opacity: 0.9, borderRadius: 2 }} title={`moved ${e.start_timestamp.toFixed(1)}→${(e.end_timestamp ?? e.start_timestamp).toFixed(1)}`} /> })}
                              </div>
                              <span style={{ fontSize: 10, color: 'var(--muted)', minWidth: 90 }}>{t.start_timestamp.toFixed(1)}s → {t.end_timestamp.toFixed(1)}s</span>
                            </div>
                          )
                        })}
                      </div>
                    )}
                    <div className="table-wrap" style={{ border: '1px solid var(--border)' }}>
                      <table className="table">
                        <thead><tr><th>Time</th><th>Type</th><th>Label</th><th>Track</th><th>Confidence</th></tr></thead>
                        <tbody>
                          {displayEvents.filter(e => eventFilter === 'all' || e.event_type === eventFilter).filter(e => { if (selectedTrack === 'all') return true; const tr = displayTracks.find(t => t.id === e.track_id); return tr ? tr?.track_index === selectedTrack : false }).map(e => (
                            <tr key={e.id} onClick={() => seekTo(e.start_timestamp)} title="Click to play from here" style={{ cursor: 'pointer', ...(e.event_type === 'OBJECT_MOVED' ? { background: '#f0faf7' } : {}) }}>
                              <td>{e.end_timestamp ? `${e.start_timestamp.toFixed(2)} → ${e.end_timestamp.toFixed(2)}` : e.start_timestamp.toFixed(2)}</td>
                              <td>{e.event_type === 'OBJECT_APPEARED' ? 'Appeared' : e.event_type === 'OBJECT_DISAPPEARED' ? 'Last observed' : e.event_type === 'OBJECT_PRESENT' ? 'Present' : e.event_type === 'OBJECT_MOVED' ? 'Movement' : e.event_type}</td>
                              <td>{e.label}</td>
                              <td>{(() => { const tr = displayTracks.find(t => t.id === e.track_id); return tr ? `Track ${tr.track_index}` : '-' })()}</td>
                              <td>{e.confidence != null ? `${(e.confidence * 100).toFixed(0)}%` : '-'}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                      {displayEvents.filter(e => eventFilter === 'all' || e.event_type === eventFilter).filter(e => { if (selectedTrack === 'all') return true; const tr = displayTracks.find(t => t.id === e.track_id); return tr ? tr?.track_index === selectedTrack : false }).length === 0 && <div className="empty">No events{filterSegmentId ? ' in this segment' : ''}</div>}
                    </div>
                  </>
                )}
              </div>
            )}

            {activeTab === 'descriptions' && (
              <div>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
                  <button className="collapse-btn" onClick={() => setDescriptionsCollapsed(v => !v)} title={descriptionsCollapsed ? 'Expand' : 'Collapse'} style={{ width: 24, height: 24, fontSize: 10 }}>{descriptionsCollapsed ? '▶' : '▼'}</button>
                  <span style={{ fontSize: 11, color: 'var(--muted)' }}>{displayDescriptions.length} descriptions{filterSegmentId ? ' • filtered' : ''}</span>
                </div>
                {!descriptionsCollapsed && (
                  displayDescriptions.length === 0 ? <div className="empty">No descriptions{filterSegmentId ? ' in this segment' : ' — waiting for processing or VLM unavailable'}</div> : (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
                      {displayDescriptions.slice().sort((a, b) => { const sa = segments.find(s => s.id === a.segment_id)?.start_time ?? 0; const sb = segments.find(s => s.id === b.segment_id)?.start_time ?? 0; return sa - sb }).map(d => {
                        const seg = segments.find(s => s.id === d.segment_id)
                        const fmtTime = (s: number) => { const m = Math.floor(s / 60); const ss = Math.floor(s % 60); return `${String(m).padStart(2, '0')}:${String(ss).padStart(2, '0')}` }
                        const timeLabel = seg ? `${fmtTime(seg.start_time)} → ${fmtTime(seg.end_time)}` : d.segment_id.slice(0, 8)
                        return (
                          <div key={d.id} style={{ padding: 12, background: '#f8f9f8', border: '1px solid var(--border)', borderRadius: 8, cursor: seg ? 'pointer' : 'default' }} onClick={() => { if (!seg) return; seekTo(seg.start_time); setActiveTab('segments'); const el = document.getElementById(`seg-row-${seg.id}`); if (el) el.scrollIntoView({ behavior: 'smooth', block: 'center' }) }} title={seg ? `Click to play from ${formatDur(seg.start_time)}` : undefined}>
                            <div style={{ fontSize: 11, color: 'var(--muted)', marginBottom: 4 }}>{timeLabel} — Segment {seg?.segment_index ?? '?'} • {seg ? formatDur(seg.start_time) : ''} ▶{(isDeepLink || filterSegmentId) && <span style={{ marginLeft: 6, background: '#fef08a', padding: '0 4px', borderRadius: 3, fontWeight: 700 }}>MATCHED</span>}</div>
                            <div style={{ fontSize: 13, lineHeight: 1.5, whiteSpace: 'pre-wrap', color: 'var(--text)' }}>
                              {(isDeepLink || filterSegmentId) && matchedSnippet
                                ? highlightMatched(d.description, matchedSnippet)
                                : (isDeepLink && deepQuery) ? highlightQuery(d.description, deepQuery) : d.description}
                            </div>
                            <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 6 }}>Model: {d.model_name} </div>
                          </div>
                        )
                      })}
                    </div>
                  )
                )}
              </div>
            )}
          </>
        )}
      </div>

      <div className="card">
        <h3 style={{ margin: '0 0 8px 0', fontSize: 13, fontWeight: 600 }}>Detection Summary</h3>
        {Object.keys(summary).length === 0 ? <div className="empty">No detections</div> : (
          <div className="summary">
            {Object.entries(summary).map(([label, count]) => <span key={label} className="item">{label} <b>{count}</b></span>)}
          </div>
        )}
      </div>

      <div className="card">
        <h3 style={{ margin: '0 0 12px 0', fontSize: 13, fontWeight: 600 }}>Frames ({displayFrames.length}) {filterSegmentId ? '• filtered to segment' : ''} {selectedTrack !== 'all' ? `• Track ${selectedTrack}` : ''}</h3>
        <div className="frame-grid">
          {displayFrames.filter(f => { if (selectedTrack === 'all') return true; const dets = detectionsByFrame[f.id] || []; return dets.some(d => detToTrack[d.id] === selectedTrack) }).map(f => {
            const dets = detectionsByFrame[f.id] || []
            return (
              <div key={f.id} className="frame-card" onClick={() => setSelectedFrame(f)} style={{ cursor: 'pointer', opacity: selectedTrack !== 'all' && !dets.some(d => detToTrack[d.id] === selectedTrack) ? 0.6 : 1 }}>
                <div className="img-wrap">
                  <img src={imgSrc(f)} alt={`frame ${f.frame_index}`} loading="lazy" />
                  {dets.map(d => {
                    const trackIdx = detToTrack[d.id]
                    const highlighted = selectedTrack !== 'all' && trackIdx === selectedTrack
                    const dimmed = selectedTrack !== 'all' && trackIdx !== selectedTrack
                    return (
                      // 16/9 = the .frame-card .img-wrap aspect ratio; the helper
                      // maps the bbox into the image's letterboxed content rect.
                      <div key={d.id} className="bbox" style={{ ...detectionBoxStyle(d, f.width, f.height, 16 / 9), borderColor: highlighted ? '#10a37f' : undefined, background: highlighted ? 'rgba(16,163,127,0.18)' : dimmed ? 'rgba(250,204,21,0.06)' : undefined, opacity: dimmed ? 0.4 : 1 }}>
                        <span className="bbox-label" style={highlighted ? { background: '#10a37f' } : undefined}>{d.label} {(d.confidence * 100).toFixed(0)}%{trackIdx !== undefined ? ` T${trackIdx}` : ''}</span>
                      </div>
                    )
                  })}
                </div>
                <div className="meta">
                  <div><b>Frame {String(f.frame_index).padStart(6, '0')}</b> {f.timestamp_seconds.toFixed(2)}s</div>
                  <div>Segment {segments.find(s => s.id === f.segment_id)?.segment_index ?? '-'} · {f.width}×{f.height}</div>
                  <div>{dets.length} detections {selectedTrack !== 'all' ? `(${dets.filter(d => detToTrack[d.id] === selectedTrack).length} in T${selectedTrack})` : ''}</div>
                </div>
              </div>
            )
          })}
        </div>
        {displayFrames.length === 0 && <div className="empty">No frames{filterSegmentId ? ' in this segment' : ''}</div>}
      </div>

      {selectedFrame && (
        <div style={{ position: 'fixed', inset: 0, background: 'rgba(15,17,20,0.72)', zIndex: 20, overflow: 'auto', padding: 20, backdropFilter: 'blur(2px)' }} onClick={() => setSelectedFrame(null)}>
          <div style={{ maxWidth: 900, margin: '0 auto', background: 'white', padding: 16, borderRadius: 12, border: '1px solid var(--border)', boxShadow: '0 20px 40px rgba(16,24,40,0.18)' }} onClick={e => e.stopPropagation()}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
              <h3 style={{ margin: 0, fontSize: 14 }}>Frame {selectedFrame.frame_index} — {selectedFrame.timestamp_seconds.toFixed(2)}s</h3>
              <button className="btn" onClick={() => setSelectedFrame(null)}>Close</button>
            </div>
            <div className="img-wrap" style={{ position: 'relative', background: '#000', borderRadius: 8, overflow: 'hidden' }}>
              <img src={imgSrc(selectedFrame)} alt="selected" style={{ width: '100%' }} />
              {(detectionsByFrame[selectedFrame.id] || []).map(d => {
                const trackIdx = detToTrack[d.id]
                const highlighted = selectedTrack !== 'all' && trackIdx === selectedTrack
                const dimmed = selectedTrack !== 'all' && trackIdx !== selectedTrack
                return (
                  // Modal image has no forced aspect ratio, so the content rect
                  // equals the image itself: the helper reduces to raw ~percentages.
                  <div key={d.id} className="bbox" style={{ ...detectionBoxStyle(d, selectedFrame.width, selectedFrame.height, selectedFrame.width > 0 && selectedFrame.height > 0 ? selectedFrame.width / selectedFrame.height : 16 / 9), borderColor: highlighted ? '#10a37f' : undefined, background: highlighted ? 'rgba(16,163,127,0.18)' : dimmed ? 'rgba(250,204,21,0.06)' : undefined, opacity: dimmed ? 0.4 : 1 }}>
                    <span className="bbox-label" style={highlighted ? { background: '#10a37f' } : undefined}>{d.label} {(d.confidence * 100).toFixed(0)}%{trackIdx !== undefined ? ` T${trackIdx}` : ''}</span>
                  </div>
                )
              })}
            </div>
            <div style={{ marginTop: 12 }}>
              <h4 style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>Detections ({(detectionsByFrame[selectedFrame.id] || []).length})</h4>
              <div className="table-wrap">
                <table className="table">
                  <thead><tr><th>Label</th><th>Confidence</th><th>BBox</th><th>Track</th><th>Detector</th></tr></thead>
                  <tbody>
                    {(detectionsByFrame[selectedFrame.id] || []).map(d => {
                      const trackIdx = detToTrack[d.id]
                      const highlighted = selectedTrack !== 'all' && trackIdx === selectedTrack
                      return <tr key={d.id} style={highlighted ? { background: '#e6f6ef', fontWeight: 600 } : undefined}><td>{d.label}</td><td>{(d.confidence * 100).toFixed(1)}%</td><td>{d.bbox_x.toFixed(2)},{d.bbox_y.toFixed(2)},{d.bbox_width.toFixed(2)},{d.bbox_height.toFixed(2)}</td><td>{trackIdx !== undefined ? `Track ${trackIdx}` : '-'}</td><td>{d.detector_name} v{d.detector_version}</td></tr>
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
