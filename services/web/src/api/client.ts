const BASE = (import.meta.env.VITE_API_BASE_URL as string) || ''

async function req<T>(path: string, opts: RequestInit = {}): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    ...opts,
    headers: { 'Content-Type': 'application/json', ...(opts.headers || {}) },
  })
  if (!res.ok) {
    const txt = await res.text()
    let msg = txt
    try { const j = JSON.parse(txt); msg = j.error || txt } catch {}
    throw new Error(`${res.status} ${msg}`)
  }
  if (res.status === 204) return undefined as unknown as T
  const ct = res.headers.get('content-type') || ''
  if (ct.includes('image')) return res as unknown as T
  return res.json() as Promise<T>
}

export const client = {
  get: <T>(p: string) => req<T>(p),
  post: <T>(p: string, body?: unknown) => req<T>(p, { method: 'POST', body: body ? JSON.stringify(body) : undefined }),
  del: <T>(p: string) => req<T>(p, { method: 'DELETE' }),
  upload: <T>(p: string, file: File, opts?: {
    onProgress?: (loaded: number, total: number) => void
    signal?: AbortSignal
  }): Promise<T> => {
    return new Promise<T>((resolve, reject) => {
      const xhr = new XMLHttpRequest()
      xhr.open('POST', `${BASE}${p}`)
      if (opts?.signal) {
        if (opts.signal.aborted) {
          reject(new DOMException('Upload cancelled', 'AbortError'))
          return
        }
        opts.signal.addEventListener('abort', () => xhr.abort(), { once: true })
      }
      xhr.upload.onprogress = (ev) => {
        if (ev.lengthComputable) opts?.onProgress?.(ev.loaded, ev.total)
      }
      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            resolve(JSON.parse(xhr.responseText))
          } catch {
            reject(new Error('Invalid server response'))
          }
        } else {
          let msg = xhr.responseText
          try { const j = JSON.parse(xhr.responseText); msg = j.error || msg } catch {}
          reject(new Error(`${xhr.status} ${msg}`))
        }
      }
      xhr.onerror = () => reject(new Error('Network error during upload'))
      xhr.onabort = () => reject(new DOMException('Upload cancelled', 'AbortError'))
      const fd = new FormData()
      fd.append('file', file)
      xhr.send(fd)
    })
  },
  imageUrl: (path: string) => `${BASE}${path}`,
  base: BASE,
}
