package detector

import (
	"context"
	"log/slog"
	"sync"
)

// Manager owns a standby persistent detector session so video processing can
// start with an already-loaded model. It implements the "one persistent
// detector worker" architecture on top of the existing Session lifecycle:
//
//   - Warmup() pre-loads YOLO once in the background (API startup).
//   - Acquire() hands the warm standby to the caller, which owns and closes
//     it as usual (same lifecycle as NewSession).
//   - After a handover a fresh standby is pre-loaded for the next video, so
//     every video after the first skips the Python/ultralytics cold start.
//
// Nothing about detection behavior changes: sessions, batching and the
// stdin/stdout protocol are exactly the per-video persistent sessions the
// pipeline already used.
type Manager struct {
	y       *YoloDetector
	prewarm bool

	mu       sync.Mutex
	standby  *Session
	warmDone chan struct{}
	rootCtx  context.Context
	cancel   context.CancelFunc
}

// NewManager wraps a YoloDetector with warm-standby management.
func NewManager(y *YoloDetector, prewarm bool) *Manager {
	ctx, cancel := context.WithCancel(context.Background())
	return &Manager{y: y, prewarm: prewarm, rootCtx: ctx, cancel: cancel}
}

// Warmup pre-loads a standby session in the background. Idempotent: a second
// call while a load is running (or a standby exists) is a no-op.
func (m *Manager) Warmup() {
	if !m.prewarm {
		return
	}
	m.mu.Lock()
	if m.standby != nil || m.warmDone != nil {
		m.mu.Unlock()
		return
	}
	done := make(chan struct{})
	m.warmDone = done
	m.mu.Unlock()

	go func() {
		defer close(done)
		slog.Info("detector: prewarming persistent session", "model", m.y.modelPath)
		sess, err := m.y.NewSession(m.rootCtx)
		if err != nil {
			if m.rootCtx.Err() == nil {
				slog.Warn("detector: prewarm failed; sessions will be spawned per video", "error", err)
			}
			return
		}
		slog.Info("detector: prewarm session ready", "model", m.y.modelPath)
		m.mu.Lock()
		// A concurrent Acquire may have skipped waiting for this warmup;
		// storing it as the standby keeps exactly one resident session.
		if m.standby == nil {
			m.standby = sess
			sess = nil
		}
		m.mu.Unlock()
		if sess != nil {
			_ = sess.Close()
		}
	}()
}

// Acquire returns a ready-to-use persistent session: the prewarmed standby
// when available (waiting for an in-flight warmup so no duplicate process is
// spawned), otherwise a freshly started one. Ownership transfers to the
// caller, which closes the session as usual.
func (m *Manager) Acquire(ctx context.Context) (*Session, error) {
	m.mu.Lock()
	if s := m.standby; s != nil {
		m.standby = nil
		m.mu.Unlock()
		slog.Info("detector: acquired prewarmed session")
		// Keep exactly one standby resident for the next video.
		go m.Warmup()
		return s, nil
	}
	done := m.warmDone
	m.mu.Unlock()

	if done != nil {
		select {
		case <-done:
			m.mu.Lock()
			s := m.standby
			m.standby = nil
			m.mu.Unlock()
			if s != nil {
				slog.Info("detector: acquired prewarmed session (after warmup)")
				go m.Warmup()
				return s, nil
			}
		case <-ctx.Done():
			return nil, ctx.Err()
		}
	}
	return m.y.NewSession(ctx)
}

// Close releases the standby and cancels any in-flight warmup.
func (m *Manager) Close() {
	m.cancel()
	m.mu.Lock()
	s := m.standby
	m.standby = nil
	m.mu.Unlock()
	if s != nil {
		if err := s.Close(); err != nil {
			slog.Warn("detector: failed to close prewarm session", "error", err)
		}
	}
}
