package video_frame

// Parallel extraction: worker pool decodes concurrently while results stay
// in timestamp order. Uses the in-memory fixture from plan_test.go.

import (
	"context"
	"errors"
	"math"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/berzz26/recall/services/api/internal/sampler"
)

// TestParallelExtractionOrderAndContiguity decodes with sleeps rigged so
// later timestamps finish first; results must still come out ordered with
// contiguous frame indices, and workers must actually overlap.
func TestParallelExtractionOrderAndContiguity(t *testing.T) {
	const duration = 40.0 // 20 timestamps on the 2s grid
	fx := newPlanFixture(t, duration)
	ctx := context.Background()
	payload := stubJPEG(t, 8, 6)

	var cur, maxInFlight int32
	fx.svc.WithExtractWorkers(4)
	fx.svc.extractOne = func(_ context.Context, _, _ string, ts float64, _ int) ([]byte, error) {
		c := atomic.AddInt32(&cur, 1)
		for {
			m := atomic.LoadInt32(&maxInFlight)
			if c <= m || atomic.CompareAndSwapInt32(&maxInFlight, m, c) {
				break
			}
		}
		defer atomic.AddInt32(&cur, -1)
		// Later timestamps finish first to force out-of-order completion.
		idx := int(math.Round(ts / 2))
		time.Sleep(time.Duration(20-idx) * 5 * time.Millisecond)
		return payload, nil
	}

	planner := sampler.NewBaselinePlanner(2*time.Second, 1.0)
	plan, err := planner.Plan(fx.vid.ID, duration)
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}

	frames, err := fx.svc.GenerateForVideoWithPlan(ctx, fx.vid, fx.segments, plan, 720, 480)
	if err != nil {
		t.Fatalf("GenerateForVideoWithPlan: %v", err)
	}
	if len(frames) != plan.Len() {
		t.Fatalf("expected %d frames, got %d", plan.Len(), len(frames))
	}
	want := plan.Timestamps()
	for i, f := range frames {
		if f.FrameIndex != i {
			t.Fatalf("frame %d has non-contiguous index %d", i, f.FrameIndex)
		}
		if math.Abs(f.TimestampSeconds-want[i]) > 1e-9 {
			t.Fatalf("frame %d ts %v, want %v", i, f.TimestampSeconds, want[i])
		}
	}
	if maxInFlight < 2 {
		t.Fatalf("expected overlapping decodes, max in-flight was %d", maxInFlight)
	}
}

// TestParallelExtractionErrorFailsClean checks that one failing timestamp
// fails the video with the sequential error message and no partial frames.
func TestParallelExtractionErrorFailsClean(t *testing.T) {
	const duration = 40.0
	fx := newPlanFixture(t, duration)
	ctx := context.Background()
	payload := stubJPEG(t, 8, 6)

	fx.svc.WithExtractWorkers(4)
	fx.svc.extractOne = func(_ context.Context, _, _ string, ts float64, _ int) ([]byte, error) {
		if math.Abs(ts-20.0) < 1e-9 {
			return nil, errors.New("boom")
		}
		return payload, nil
	}

	planner := sampler.NewBaselinePlanner(2*time.Second, 1.0)
	plan, err := planner.Plan(fx.vid.ID, duration)
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}

	_, err = fx.svc.GenerateForVideoWithPlan(ctx, fx.vid, fx.segments, plan, 720, 480)
	if err == nil {
		t.Fatalf("expected failure for undecodable timestamp")
	}
	if !strings.Contains(err.Error(), "ffmpeg failed at timestamp 20.000000") {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(fx.repo.frames) != 0 {
		t.Fatalf("expected no persisted frames after failure, got %d", len(fx.repo.frames))
	}
}

func TestEffectiveWorkers(t *testing.T) {
	svc := &Service{}
	if w := svc.effectiveWorkers(); w < 1 || w > 8 {
		t.Fatalf("auto workers out of range: %d", w)
	}
	svc.WithExtractWorkers(3)
	if w := svc.effectiveWorkers(); w != 3 {
		t.Fatalf("explicit workers = %d, want 3", w)
	}
	svc.WithExtractWorkers(-2)
	if w := svc.effectiveWorkers(); w < 1 || w > 8 {
		t.Fatalf("non-positive workers should fall back to auto, got %d", w)
	}
}
