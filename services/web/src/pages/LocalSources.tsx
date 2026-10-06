import { useEffect, useState } from 'react'
import { client } from '../api/client'
import { useDialog } from '../components/Dialog'
import type { LocalSource } from '../api/types'
import { Loading, ErrorState } from '../components/Loading'

export default function LocalSources() {
  const [sources, setSources] = useState<LocalSource[]>([])
  const [err, setErr] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [name, setName] = useState('')
  const [path, setPath] = useState('')
  const [adding, setAdding] = useState(false)
  const { confirm, notify } = useDialog()

  const fetch = async () => {
    try {
      setLoading(true)
      const s = await client.get<LocalSource[]>('/api/v1/local-sources/')
      setSources(s)
      setErr(null)
    } catch (e: any) { setErr(e.message) } finally { setLoading(false) }
  }
  useEffect(() => { fetch() }, [])

  const onAdd = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!name || !path) return
    setAdding(true)
    try {
      await client.post('/api/v1/local-sources/', { name, path })
      setName(''); setPath('')
      await fetch()
    } catch (e: any) { await notify({ title: 'Add source failed', message: e.message }) } finally { setAdding(false) }
  }

  const onDelete = async (id: string) => {
    const ok = await confirm({
      title: 'Delete local source',
      message: 'Delete this local source? This does not delete video files.',
      confirmLabel: 'Delete',
      danger: true,
    })
    if (!ok) return
    try {
      await client.del(`/api/v1/local-sources/${id}`)
      await fetch()
    } catch (e: any) { await notify({ title: 'Delete failed', message: e.message }) }
  }

  if (loading) return <Loading />
  if (err) return <ErrorState error={err} retry={fetch} />

  return (
    <div>
      <div className="card">
        <h3>Add Local Source</h3>
        <form onSubmit={onAdd} style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <input placeholder="Name" value={name} onChange={e => setName(e.target.value)} required />
          <input placeholder="/absolute/path/to/videos" value={path} onChange={e => setPath(e.target.value)} required style={{ flex: 1, minWidth: 200 }} />
          <button className="btn btn-primary" disabled={adding}>{adding ? 'Adding…' : 'Add'}</button>
        </form>
      </div>

      <div className="card">
        <h3>Local Sources ({sources.length})</h3>
        <table className="table">
          <thead><tr><th>Name</th><th>Path</th><th>Enabled</th><th>Created</th><th></th></tr></thead>
          <tbody>
            {sources.map(s => (
              <tr key={s.id}>
                <td>{s.name}</td><td style={{ wordBreak: 'break-all' }}>{s.path}</td><td>{s.enabled ? 'yes' : 'no'}</td><td>{new Date(s.created_at).toLocaleString()}</td>
                <td><button className="btn btn-danger" onClick={() => onDelete(s.id)}>Delete</button></td>
              </tr>
            ))}
          </tbody>
        </table>
        {sources.length === 0 && <div className="empty">No local sources configured</div>}
      </div>
    </div>
  )
}
