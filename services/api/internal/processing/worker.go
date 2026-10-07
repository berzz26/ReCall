package processing

import (
	"context"
	"log/slog"
	"time"

	"github.com/berzz26/recall/services/api/internal/video"
	"github.com/google/uuid"
)

type Worker struct {
	videoService *video.Service
	processor    Processor
	pollInterval time.Duration
	playable     PlayableEnsurer
}

// PlayableEnsurer generates the browser-playable proxy (H.264 sidecar for
// codecs browsers cannot decode, e.g. MPEG-4 Part 2) before processing.
// Implemented by *playable.Service; nil disables the step.
type PlayableEnsurer interface {
	Ensure(ctx context.Context, id uuid.UUID) (state string, key string, err error)
}

func NewWorker(videoService *video.Service, processor Processor, pollInterval time.Duration) *Worker {
	if pollInterval <= 0 {
		pollInterval = 2 * time.Second
	}
	if processor == nil {
		processor = &NoopProcessor{}
	}
	return &Worker{
		videoService: videoService,
		processor:    processor,
		pollInterval: pollInterval,
	}
}

// WithPlayable enables pre-processing proxy generation. A proxy failure only
// warns: the original remains authoritative for the pipeline, so processing
// must never fail just because browsers cannot play the native codec.
func (w *Worker) WithPlayable(p PlayableEnsurer) *Worker {
	w.playable = p
	return w
}

func (w *Worker) Start(ctx context.Context) {
	slog.Info("processing worker started", "pollInterval", w.pollInterval.String())
	if err := w.poll(ctx); err != nil && ctx.Err() != nil {
		slog.Info("processing worker stopped")
		return
	}
	ticker := time.NewTicker(w.pollInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			slog.Info("processing worker stopped")
			return
		case <-ticker.C:
			if err := w.poll(ctx); err != nil {
				if ctx.Err() != nil {
					return
				}
			}
		}
	}
}

func (w *Worker) poll(ctx context.Context) error {
	v, err := w.videoService.ClaimNext(ctx)
	if err != nil {
		return nil
	}
	if v == nil {
		return nil
	}

	slog.Info("video claimed", "id", v.ID.String(), "status", v.Status)
	slog.Info("video processing started", "id", v.ID.String())

	// Browser-playable proxy first: independent of stage success/failure,
	// so even a video that later fails processing can still be watched.
	if w.playable != nil {
		if state, key, err := w.playable.Ensure(ctx, v.ID); err != nil {
			slog.Warn("playable proxy generation failed, continuing with original", "id", v.ID.String(), "error", err)
		} else if state == "ready" {
			slog.Info("playable proxy ensured", "id", v.ID.String(), "playable_key", key)
		}
	}

	err = w.processor.Process(ctx, v)
	if err != nil {
		slog.Error("video processing failed", "id", v.ID.String(), "error", err)
		if _, markErr := w.videoService.MarkFailed(ctx, v.ID, err.Error()); markErr != nil {
			slog.Error("failed to mark video as FAILED", "id", v.ID.String(), "error", markErr)
		} else {
			slog.Info("video marked FAILED", "id", v.ID.String())
		}
		return nil
	}

	if _, err := w.videoService.MarkReady(ctx, v.ID); err != nil {
		slog.Error("failed to mark video as READY", "id", v.ID.String(), "error", err)
		return err
	}
	slog.Info("video marked READY", "id", v.ID.String())
	slog.Info("video processing completed", "id", v.ID.String())
	return nil
}

func (w *Worker) ProcessOne(ctx context.Context) error {
	return w.poll(ctx)
}
