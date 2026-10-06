package video_frame

// Resilience of frame extraction against timestamps with no decodable frame
// (phantom tail past the last real frame, local gaps). Uses the in-memory
// fixture from plan_test.go with a stub decoder (no ffmpeg, no database).

import (
	"context"
	"math"
	"strings"
	"testing"
	"time"

	"github.com/berzz26/recall/services/api/internal/sampler"
)

// emptyAt returns a stub decoder that yields valid JPEGs except at the given
// timestamps, where it reports an empty decode like the real ffmpeg path.
func emptyAt(t *testing.T, payload []byte, calls *[]float64, empty func(ts float64) bool) extractFunc {
	t.Helper()
	return func(_ context.Context, _, _ string, ts float64, _ int) ([]byte, error) {
		*calls = append(*calls, ts)
		if empty(ts) {
			return nil, &emptyFrameError{Timestamp: ts}
		}
		return payload, nil
	}
}

func sawCall(calls []float64, want float64) bool {
	for _, c := range calls {
		if math.Abs(c-want) < 1e-9 {
			return true
		}
	}
	return false
}

// Reproduces the 300.033s video failure: container duration overshoots the
// last real frame, so the final grid point (300.0) — and the whole step-back
// window behind it — has nothing decodable. The video must succeed with 150
// frames instead of failing entirely.
func TestPhantomTailSkipped(t *testing.T) {
	const duration = 300.033067
	fx := newPlanFixture(t, duration)
	ctx := context.Background()
	payload := stubJPEG(t, 8, 6)

	// Nothing decodable from 298.0 on: the direct attempt at plan ts 298.0
	// fails but recovers via step-back to 297.5, while 300.0 exhausts the
	// whole step-back window and is skipped as phantom tail.
	fx.svc.extractOne = emptyAt(t, payload, fx.calls, func(ts float64) bool {
		return ts >= 298.0-1e-9
	})

	planner := sampler.NewBaselinePlanner(2*time.Second, 1.0)
	plan, err := planner.Plan(fx.vid.ID, duration)
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}
	if plan.Len() != 151 {
		t.Fatalf("expected 151 planned timestamps, got %d", plan.Len())
	}

	frames, err := fx.svc.GenerateForVideoWithPlan(ctx, fx.vid, fx.segments, plan, 720, 480)
	if err != nil {
		t.Fatalf("GenerateForVideoWithPlan should skip the phantom tail, got: %v", err)
	}
	if len(frames) != 150 {
		t.Fatalf("expected 150 frames (tail skipped), got %d", len(frames))
	}
	for i, f := range frames {
		if f.FrameIndex != i {
			t.Fatalf("frame %d has non-contiguous index %d", i, f.FrameIndex)
		}
	}
	last := frames[len(frames)-1]
	if math.Abs(last.TimestampSeconds-298.0) > 1e-9 {
		t.Fatalf("last frame ts %v, want 298.0", last.TimestampSeconds)
	}
	if !sawCall(*fx.calls, 300.0) || !sawCall(*fx.calls, 298.0) {
		t.Fatalf("expected decode attempts at 300.0 and step-back 298.0, got tail %v", (*fx.calls)[len(*fx.calls)-5:])
	}
}

// A single undecodable timestamp away from EOF is recovered via step-back;
// the frame is recorded at the requested timestamp.
func TestStepBackRecovers(t *testing.T) {
	const duration = 104.0
	fx := newPlanFixture(t, duration)
	ctx := context.Background()
	payload := stubJPEG(t, 8, 6)

	fx.svc.extractOne = emptyAt(t, payload, fx.calls, func(ts float64) bool {
		return math.Abs(ts-100.0) < 1e-9
	})

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
	found := false
	for _, f := range frames {
		if math.Abs(f.TimestampSeconds-100.0) < 1e-9 {
			found = true
		}
	}
	if !found {
		t.Fatalf("frame at requested ts 100.0 missing")
	}
	if !sawCall(*fx.calls, 99.5) {
		t.Fatalf("expected step-back decode at 99.5")
	}
}

// A gap wider than the step-back window away from EOF still fails the video,
// preserving the existing strictness where frames should exist.
func TestPersistentGapStillFails(t *testing.T) {
	const duration = 300.0
	fx := newPlanFixture(t, duration)
	ctx := context.Background()
	payload := stubJPEG(t, 8, 6)

	fx.svc.extractOne = emptyAt(t, payload, fx.calls, func(ts float64) bool {
		return ts >= 98.0-1e-9 && ts <= 100.0+1e-9
	})

	planner := sampler.NewBaselinePlanner(2*time.Second, 1.0)
	plan, err := planner.Plan(fx.vid.ID, duration)
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}

	_, err = fx.svc.GenerateForVideoWithPlan(ctx, fx.vid, fx.segments, plan, 720, 480)
	if err == nil {
		t.Fatalf("expected failure for persistent mid-video gap")
	}
	if !strings.Contains(err.Error(), "ffmpeg failed at timestamp 100.000000") {
		t.Fatalf("unexpected error: %v", err)
	}
}

// The typed error keeps the legacy message so log searches are unaffected.
func TestEmptyFrameErrorMessage(t *testing.T) {
	err := &emptyFrameError{Timestamp: 300.0}
	if err.Error() != "ffmpeg produced empty frame at 300.000000" {
		t.Fatalf("message changed: %q", err.Error())
	}
}
