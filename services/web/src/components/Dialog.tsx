import { createContext, useCallback, useContext, useEffect, useRef, useState } from 'react'
import type { ReactNode } from 'react'

export interface ConfirmOptions {
  title: string
  message: string
  confirmLabel?: string
  cancelLabel?: string
  /** Red confirm button for destructive actions (deletes). */
  danger?: boolean
}

export interface NotifyOptions {
  title: string
  message: string
  okLabel?: string
}

interface PendingConfirm extends ConfirmOptions {
  kind: 'confirm'
  resolve: (ok: boolean) => void
}

interface PendingNotify extends NotifyOptions {
  kind: 'notify'
  resolve: () => void
}

type Pending = PendingConfirm | PendingNotify

interface DialogApi {
  /** Themed replacement for window.confirm. Resolves true on confirm, false on cancel/dismiss. */
  confirm: (opts: ConfirmOptions) => Promise<boolean>
  /** Themed replacement for window.alert. Resolves once dismissed. */
  notify: (opts: NotifyOptions) => Promise<void>
}

const DialogContext = createContext<DialogApi | null>(null)

export function useDialog(): DialogApi {
  const api = useContext(DialogContext)
  if (!api) throw new Error('useDialog must be used inside <DialogProvider>')
  return api
}

export function DialogProvider({ children }: { children: ReactNode }) {
  const [pending, setPending] = useState<Pending | null>(null)
  // Guard against a second dialog replacing an unsettled one: serialize.
  const queue = useRef<Pending[]>([])

  const settle = useCallback(() => {
    const next = queue.current.shift()
    setPending(next ?? null)
  }, [])

  const confirm = useCallback((opts: ConfirmOptions) => {
    return new Promise<boolean>((resolve) => {
      const item: Pending = { kind: 'confirm', cancelLabel: 'Cancel', confirmLabel: 'Confirm', ...opts, resolve }
      queue.current.push(item)
      setPending((cur) => cur ?? queue.current.shift() ?? null)
    })
  }, [])

  const notify = useCallback((opts: NotifyOptions) => {
    return new Promise<void>((resolve) => {
      const item: Pending = { kind: 'notify', okLabel: 'OK', ...opts, resolve }
      queue.current.push(item)
      setPending((cur) => cur ?? queue.current.shift() ?? null)
    })
  }, [])

  const onConfirm = useCallback(() => {
    if (pending?.kind === 'confirm') pending.resolve(true)
    else if (pending?.kind === 'notify') pending.resolve()
    settle()
  }, [pending, settle])

  const onCancel = useCallback(() => {
    if (pending?.kind === 'confirm') pending.resolve(false)
    else if (pending?.kind === 'notify') pending.resolve()
    settle()
  }, [pending, settle])

  useEffect(() => {
    if (!pending) return
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onCancel()
    }
    window.addEventListener('keydown', onKey)
    const prev = document.body.style.overflow
    document.body.style.overflow = 'hidden'
    return () => {
      window.removeEventListener('keydown', onKey)
      document.body.style.overflow = prev
    }
  }, [pending, onCancel])

  return (
    <DialogContext.Provider value={{ confirm, notify }}>
      {children}
      {pending && (
        <div className="dlg-overlay" onMouseDown={(e) => { if (e.target === e.currentTarget) onCancel() }}>
          <div className="dlg-box" role="alertdialog" aria-modal="true" aria-label={pending.title}>
            <div className="dlg-icon" data-danger={pending.kind === 'confirm' && (pending as PendingConfirm).danger ? true : undefined}>
              {pending.kind === 'confirm' ? (
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                  <path d="M12 9v4M12 17h.01" /><path d="M10.3 3.9L1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" />
                </svg>
              ) : (
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
                  <circle cx="12" cy="12" r="9" /><path d="M12 8v4M12 16h.01" />
                </svg>
              )}
            </div>
            <h3 className="dlg-title">{pending.title}</h3>
            <p className="dlg-message">{pending.message}</p>
            <div className="dlg-actions">
              {pending.kind === 'confirm' ? (
                <>
                  <button className="btn" onClick={onCancel}>{(pending as PendingConfirm).cancelLabel}</button>
                  <button
                    className={(pending as PendingConfirm).danger ? 'btn btn-danger' : 'btn btn-primary'}
                    onClick={onConfirm}
                    autoFocus
                  >
                    {(pending as PendingConfirm).confirmLabel}
                  </button>
                </>
              ) : (
                <button className="btn btn-primary" onClick={onConfirm} autoFocus>{(pending as PendingNotify).okLabel}</button>
              )}
            </div>
          </div>
        </div>
      )}
    </DialogContext.Provider>
  )
}
