import { useEffect, useRef, useState, useMemo } from 'react'
import { Link } from 'react-router-dom'
import { client } from '../api/client'
import { useDialog } from '../components/Dialog'
import type { Video, MediaMetadata, Segment } from '../api/types'

export default function Videos() {
  const [videos, setVideos] = useState<Video[]>([])
  const [mediaMap, setMediaMap] = useState<Record<string, MediaMetadata>>({})
  const [segCountMap, setSegCountMap] = useState<Record<string, number>>({})
  const [thumbMap, setThumbMap] = useState<Record<string, string>>({})
  const [loading, setLoading] = useState(true)
  const [q, setQ] = useState('')
  const [sort, setSort] = useState('newest')
  const [uploading, setUploading] = useState(false)
  const [uploadProgress, setUploadProgress] = useState<{ name: string; loaded: number; total: number } | null>(null)
  const uploadAbort = useRef<AbortController | null>(null)
  const [err, setErr] = useState<string | null>(null)
  const { confirm, notify } = useDialog()

  const fetchAll = async () => {
    try {
      setLoading(true)
      setErr(null)
      const v = await client.get<Video[]>('/api/v1/videos/')
      const list = Array.isArray(v) ? v : (v as any).videos || []
      setVideos(list)
      const m: Record<string, MediaMetadata> = {}
      const seg: Record<string, number> = {}
      const thumbs: Record<string, string> = {}
      await Promise.all(list.map(async (vid: Video) => {
        try { m[vid.id] = await client.get<MediaMetadata>(`/api/v1/videos/${vid.id}/media`) } catch {}
        try {
          const s = await client.get<Segment[]>(`/api/v1/videos/${vid.id}/segments`)
          seg[vid.id] = s.length
        } catch { seg[vid.id] = 0 }
        try {
          const frames: any[] = await client.get<any[]>(`/api/v1/videos/${vid.id}/frames`)
          if (frames.length > 0) thumbs[vid.id] = client.imageUrl(`/api/v1/videos/${vid.id}/frames/${frames[0].id}/image`)
        } catch {}
      }))
      setMediaMap(m)
      setSegCountMap(seg)
      setThumbMap(thumbs)
    } catch (e: any) { setErr(e.message) } finally { setLoading(false) }
  }
  useEffect(() => { fetchAll() }, [])
  useEffect(() => () => { uploadAbort.current?.abort() }, [])

  const onUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]; if (!f) return
    if (uploadAbort.current) { e.target.value = ''; return }
    const ctrl = new AbortController()
    uploadAbort.current = ctrl
    setUploading(true)
    setUploadProgress({ name: f.name, loaded: 0, total: f.size })
    try {
      await client.upload('/api/v1/videos/', f, {
        signal: ctrl.signal,
        onProgress: (loaded, total) => setUploadProgress({ name: f.name, loaded, total }),
      })
      await fetchAll()
    } catch (err: any) {
      if (err?.name !== 'AbortError') await notify({ title: 'Upload failed', message: err.message })
    } finally { setUploading(false); setUploadProgress(null); uploadAbort.current = null; e.target.value = '' }
  }

  const cancelUpload = () => { uploadAbort.current?.abort() }

  const onDelete = async (id: string, filename: string) => {
    const ok = await confirm({
      title: 'Delete video',
      message: `Delete "${filename}"? This will permanently delete the video and all its frames, tracks and metadata.`,
      confirmLabel: 'Delete',
      danger: true,
    })
    if (!ok) return
    try { await client.del(`/api/v1/videos/${id}`); setVideos(prev => prev.filter(v => v.id !== id)) } catch (e: any) { await notify({ title: 'Delete failed', message: e.message }) }
  }

  const filtered = useMemo(() => {
    let list = [...videos]
    if (q) list = list.filter(v => v.filename.toLowerCase().includes(q.toLowerCase()))
    if (sort === 'newest') list.sort((a, b) => +new Date(b.created_at) - +new Date(a.created_at))
    else list.sort((a, b) => +new Date(a.created_at) - +new Date(b.created_at))
    return list
  }, [videos, q, sort])

  if (loading) return <div className="card" style={{padding:40,textAlign:'center',color:'var(--muted)'}}>Loading videos…</div>
  if (err) return <div className="card" style={{padding:24}}><div style={{color:'#ef4444',marginBottom:12}}>Error: {err}</div><button className="btn" onClick={fetchAll}>Retry</button></div>

  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', marginBottom: 14 }}>
        <div>
          <h1 style={{ fontSize: 20, fontWeight: 700, margin: '0 0 2px 0', letterSpacing: -0.02 }}>Videos</h1>
          <div style={{ fontSize: 12, color: 'var(--muted)' }}>Browse and manage your videos</div>
        </div>
        <label className="btn btn-primary" style={{ cursor: 'pointer' }}>
          {uploading ? 'Uploading…' : '+ Add Video'}
          <input type="file" accept=".mp4,.mkv,.avi,.mov" style={{ display: 'none' }} onChange={onUpload} />
        </label>
      </div>

      <div style={{ display: 'flex', gap: 10, marginBottom: 16 }}>
        <div className="input-wrap" style={{ maxWidth: 340 }}>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="11" cy="11" r="6" /><path d="M20 20L15 15" /></svg>
          <input className="input" placeholder="Search videos..." value={q} onChange={e => setQ(e.target.value)} />
        </div>
        <select className="select" value={sort} onChange={e => setSort(e.target.value)} style={{ marginLeft: 'auto' }}>
          <option value="newest">Newest first</option>
          <option value="oldest">Oldest first</option>
        </select>
      </div>

      {filtered.length === 0 ? (
        <div className="card" style={{ padding: 40, textAlign: 'center' }}>
          <div style={{ color: 'var(--muted)', fontSize: 13, marginBottom: 8 }}>No videos found</div>
          <div style={{ fontSize: 12, color: 'var(--muted)' }}>Upload a video or add a local source in <Link to="/local-sources" style={{ color: 'var(--accent)', fontWeight: 600 }}>Settings</Link>.</div>
        </div>
      ) : (
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, minmax(0, 1fr))', gap: 14 }}>
          {filtered.map((v) => {
            const m = mediaMap[v.id]
            const thumb = thumbMap[v.id]
            const date = new Date(v.created_at)
            const dateStr = date.toLocaleDateString('en-US', { month: 'short', day: '2-digit', year: 'numeric' }) + ' · ' + date.toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })
            const dur = m?.duration_seconds
              ? `${String(Math.floor(m.duration_seconds / 3600)).padStart(2,'0')}:${String(Math.floor((m.duration_seconds % 3600)/60)).padStart(2,'0')}:${String(Math.floor(m.duration_seconds % 60)).padStart(2,'0')}`
              : '—'
            const fps = m?.video_fps ? `${m.video_fps.toFixed(0)} FPS` : ''
            const res = m?.video_width && m?.video_height ? `${m.video_width}×${m.video_height}` : ''
            const segs = segCountMap[v.id] ?? 0
            const size = (v.size_bytes / 1024 / 1024).toFixed(1)
            const isReady = v.status === 'READY'
            return (
              <div key={v.id} style={{ background: 'white', borderRadius: 12, overflow: 'hidden', border: '1px solid var(--border)', boxShadow: '0 1px 3px rgba(16,24,40,0.06)', display: 'flex', flexDirection: 'column' }}>
                <Link to={`/videos/${v.id}`} style={{ display: 'block', position: 'relative', height: 168, background: '#eef2f0', overflow: 'hidden' }}>
                  {thumb ? (
                    <img src={thumb} alt={v.filename} loading="lazy" style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                  ) : (
                    <div style={{ width: '100%', height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', background: '#f3f4f6' }}>
                      <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#9aa3b1" strokeWidth="1.4"><rect x="2" y="2" width="20" height="15" rx="2" /><circle cx="8" cy="9.5" r="2" /><path d="M2 14l6-4 4 3 4-4 6 5" /></svg>
                    </div>
                  )}
                  <div style={{ position: 'absolute', top: 8, right: 8, display: 'flex', gap: 6, alignItems: 'center' }}>
                    <span style={{ padding: '4px 8px', borderRadius: 20, fontSize: 10, fontWeight: 700, background: isReady ? 'rgba(16,163,127,0.95)' : 'rgba(255,255,255,0.88)', backdropFilter: 'blur(8px)', WebkitBackdropFilter: 'blur(8px)', color: isReady ? 'white' : '#6b7280', border: `1px solid ${isReady ? 'rgba(16,163,127,0.2)' : 'rgba(255,255,255,0.6)'}` }}>
                      {isReady ? 'Processed' : v.status}
                    </span>
                    <span style={{ width: 26, height: 26, borderRadius: 8, background: 'rgba(255,255,255,0.88)', backdropFilter: 'blur(8px)', WebkitBackdropFilter: 'blur(8px)', border: '1px solid rgba(255,255,255,0.6)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke={isReady ? '#10a37f' : '#9aa3b1'} strokeWidth="1.9"><circle cx="12" cy="12" r="8" /><path d="M9 12l2 2 4-4" /></svg>
                    </span>
                  </div>
                </Link>

                <div style={{ padding: '12px 12px 10px', display: 'flex', flexDirection: 'column', gap: 6, flex: 1 }}>
                  <Link to={`/videos/${v.id}`} style={{ fontSize: 12.5, fontWeight: 700, color: 'var(--text)', lineHeight: 1.3, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{v.filename}</Link>

                  <div style={{ display: 'flex', alignItems: 'center', gap: 5, fontSize: 11, color: 'var(--muted)' }}>
                    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7"><rect x="3" y="4" width="18" height="16" rx="2" /><path d="M3 9h18M8 3v4M16 3v4" /></svg>
                    <span>{dateStr}</span>
                  </div>

                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 10px', fontSize: 11, color: 'var(--muted)', lineHeight: 1.4 }}>
                    {res && <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}><svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6"><rect x="2" y="2" width="20" height="15" rx="2" /><circle cx="12" cy="9.5" r="1.5" /></svg>{res} {fps && `· ${fps}`}</span>}
                    {!res && fps && <span>{fps}</span>}
                    <span>{dur !== '—' && `◷ ${dur}`}</span>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: 'var(--muted)' }}>
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: 4 }}>◍ {segs} segments</span>
                    <span>·</span>
                    <span>{size} MB</span>
                  </div>

                  <div style={{ fontSize: 10.5, color: 'var(--muted-2)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', background: '#f8f9f8', border: '1px solid #f0f2f1', borderRadius: 6, padding: '5px 7px', marginTop: 2 }} title={v.source_path || v.storage_key || ''}>
                    {v.source_path || v.storage_key || '—'}
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginTop: 4, paddingTop: 8, borderTop: '1px solid #f3f4f6' }}>
                    <span style={{ fontSize: 10, color: 'var(--muted)', display: 'inline-flex', alignItems: 'center', gap: 4 }}>
                      <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8"><path d="M12 2a7 7 0 0 0-7 7c0 5 7 11 7 11s7-6 7-11a7 7 0 0 0-7-7z" /><circle cx="12" cy="9" r="2" /></svg>
                      {v.source_path?.split('/').slice(-2, -1)[0] || 'Local'}
                    </span>
                    <button onClick={() => onDelete(v.id, v.filename)} title="Delete video" style={{ width: 26, height: 26, borderRadius: 8, background: 'white', border: '1px solid var(--border)', color: '#9aa3b1', display: 'flex', alignItems: 'center', justifyContent: 'center', cursor: 'pointer' }}>
                      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6" /><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" /><path d="M10 11v6M14 11v6" /><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" /></svg>
                    </button>
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}

      <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 12 }}>Showing {filtered.length} videos</div>

      <style>{`@media(max-width:1200px){div[style*="grid-template-columns: repeat(4"]{grid-template-columns:repeat(3,1fr)!important}}@media(max-width:900px){div[style*="grid-template-columns: repeat(4"]{grid-template-columns:repeat(2,1fr)!important}}@media(max-width:560px){div[style*="grid-template-columns: repeat(4"]{grid-template-columns:1fr!important}}`}</style>

      {uploadProgress && (() => {
        const pct = uploadProgress.total > 0 ? Math.min(100, Math.round(uploadProgress.loaded / uploadProgress.total * 100)) : 0
        const mb = (n: number) => `${(n / 1024 / 1024).toFixed(1)} MB`
        return (
          <div className="card" style={{ position: 'fixed', right: 18, bottom: 18, width: 320, zIndex: 60, padding: 14, boxShadow: '0 8px 28px rgba(16,24,40,0.16)' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
              <div style={{ fontSize: 12.5, fontWeight: 700, color: 'var(--text)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: 200 }} title={uploadProgress.name}>
                Uploading {uploadProgress.name}
              </div>
              <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--accent-dark)' }}>{pct}%</div>
            </div>
            <div style={{ height: 8, borderRadius: 6, background: '#eef1f0', overflow: 'hidden', margin: '8px 0 6px 0' }}>
              <div style={{ height: '100%', width: `${pct}%`, borderRadius: 6, background: 'var(--accent)', transition: 'width 0.15s' }} />
            </div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ fontSize: 11, color: 'var(--muted)' }}>{mb(uploadProgress.loaded)} of {mb(uploadProgress.total)}</div>
              <button className="btn btn-sm" onClick={cancelUpload}>Cancel</button>
            </div>
          </div>
        )
      })()}
    </div>
  )
}
