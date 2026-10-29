package video_frame

// Membership semantics for the two independent frame sets: coexistence,
// idempotent regeneration without re-extraction, set isolation under
// replacement, shared-timestamp single-row promotion to 'both', and the
// ExtractAdditional adopt-from-perception path. Uses the in-memory fixture
// from plan_test.go (stubbed decoder, no database, no ffmpeg).

import (
	"context"
	"math"
	"testing"
	"time"

	"github.com/berzz26/recall/services/api/internal/sampler"
)

func epsEqual(a, b float64) bool { return math.Abs(a-b) < frameTSEpsilon }

// frameAt returns the frame of the given timestamp from an (unsorted) list.
func frameAt(frames []VideoFrame, ts float64) *VideoFrame {
	for i := range frames {
		if epsEqual(frames[i].TimestampSeconds, ts) {
			return &frames[i]
		}
	}
	return nil
}

// tsList extracts the timestamps of a frame list in order.
func tsList(frames []VideoFrame) []float64 {
	out := make([]float64, 0, len(frames))
	for _, f := range frames {
		out = append(out, f.TimestampSeconds)
	}
	return out
}

func assertPurposes(t *testing.T, fx *planTestFixture, want map[float64]Purpose) {
	t.Helper()
	for ts, p := range want {
		f := frameAt(fx.repo.frames, ts)
		if f == nil {
			t.Fatalf("no frame at ts %v", ts)
		}
		if f.Purpose != p {
			t.Fatalf("frame at %v has purpose %q, want %q", ts, f.Purpose, p)
		}
	}
}

// TestTwoSetsCoexist: perception and semantic selections can coexist; a
// timestamp selected by both policies becomes exactly one physical row with
// purpose 'both', and each set queries independently.
func TestTwoSetsCoexist(t *testing.T) {
	const duration = 12.0
	fx := newPlanFixture(t, duration)
	ctx := context.Background()

	semFrames, err := fx.svc.GenerateForVideo(ctx, fx.vid, fx.segments, duration, 1280, 720)
	if err != nil {
		t.Fatalf("semantic GenerateForVideo: %v", err)
	}
	if len(semFrames) != 6 {
		t.Fatalf("want 6 semantic frames, got %d", len(semFrames))
	}
	assertPurposes(t, fx, map[float64]Purpose{
		0: PurposeSemantic, 2: PurposeSemantic, 10: PurposeSemantic,
	})

	// The perception grid (default 2s interval) matches the semantic
	// grid 1:1: every row is adopted into the perception set with no
	// re-extraction and no physical duplication.
	perceptionRes, err := fx.svc.GeneratePerceptionForVideo(ctx, fx.vid, fx.segments, duration, 1280, 720)
	if err != nil {
		t.Fatalf("GeneratePerceptionForVideo: %v", err)
	}
	if len(perceptionRes.Frames) != 6 {
		t.Fatalf("want 6 perception frames, got %d", len(perceptionRes.Frames))
	}
	if len(fx.repo.frames) != 6 {
		t.Fatalf("shared timestamps must stay single rows, got %d rows", len(fx.repo.frames))
	}
	for _, ts := range []float64{0, 2, 4, 6, 8, 10} {
		assertPurposes(t, fx, map[float64]Purpose{ts: PurposeBoth})
	}
	for _, id := range perceptionRes.DroppedFrameIDs {
		t.Fatalf("no frames should leave the perception set here, got dropped %s", id)
	}

	semSet, err := fx.svc.repo.GetByVideoIDSet(ctx, fx.vid.ID, SetSemantic)
	if err != nil {
		t.Fatalf("semantic set query: %v", err)
	}
	perSet, err := fx.svc.repo.GetByVideoIDSet(ctx, fx.vid.ID, SetPerception)
	if err != nil {
		t.Fatalf("perception set query: %v", err)
	}
	if len(semSet) != 6 || len(perSet) != 6 {
		t.Fatalf("both sets must serve all 6 rows: semantic %d perception %d", len(semSet), len(perSet))
	}
}

// TestRegenerationIsIdempotent: regenerating a set with the same selection
// reuses every stored frame — no re-extraction, no row churn.
func TestRegenerationIsIdempotent(t *testing.T) {
	const duration = 12.0
	fx := newPlanFixture(t, duration)
	ctx := context.Background()

	if _, err := fx.svc.GeneratePerceptionForVideo(ctx, fx.vid, fx.segments, duration, 1280, 720); err != nil {
		t.Fatalf("perception: %v", err)
	}
	first := fx.callSnapshot()
	if len(first) != 6 {
		t.Fatalf("first perception run should decode 6 timestamps, got %d", len(first))
	}

	plan := sampler.NewBaselinePlanner(2*time.Second, 1.0)
	semPlan, err := plan.Plan(fx.vid.ID, duration)
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	if _, err := fx.svc.GenerateForVideoWithPlan(ctx, fx.vid, fx.segments, semPlan, 1280, 720); err != nil {
		t.Fatalf("semantic: %v", err)
	}
	before := len(fx.callSnapshot())

	// Same selections again: everything matched, nothing extracted.
	if _, err := fx.svc.GeneratePerceptionForVideo(ctx, fx.vid, fx.segments, duration, 1280, 720); err != nil {
		t.Fatalf("perception regen: %v", err)
	}
	semPlan2, err := plan.Plan(fx.vid.ID, duration)
	if err != nil {
		t.Fatalf("plan2: %v", err)
	}
	if _, err := fx.svc.GenerateForVideoWithPlan(ctx, fx.vid, fx.segments, semPlan2, 1280, 720); err != nil {
		t.Fatalf("semantic regen: %v", err)
	}
	after := fx.callSnapshot()
	if len(after) != before {
		t.Fatalf("idempotent regen must not re-extract: %d new decodes", len(after)-before)
	}
	if len(fx.repo.frames) != 6 {
		t.Fatalf("row count changed: %d", len(fx.repo.frames))
	}
	assertPurposes(t, fx, map[float64]Purpose{0: PurposeBoth, 6: PurposeBoth, 10: PurposeBoth})
}

// TestSetReplacementIsolation: replacing one frame set must never delete,
// detach, or invalidate frames belonging exclusively to the other set.
func TestSetReplacementIsolation(t *testing.T) {
	const duration = 12.0
	fx := newPlanFixture(t, duration)
	ctx := context.Background()

	// Baseline: both grids at 2s -> six shared rows.
	if _, err := fx.svc.GeneratePerceptionForVideo(ctx, fx.vid, fx.segments, duration, 1280, 720); err != nil {
		t.Fatalf("perception: %v", err)
	}
	semPlan4, err := sampler.NewBaselinePlanner(4*time.Second, 1.0).Plan(fx.vid.ID, duration)
	if err != nil {
		t.Fatalf("plan: %v", err)
	}

	// Replace semantic with a coarser 4s grid: rows 2, 6, 10 leave the
	// semantic set (demoted to perception-only; rows kept) and 0, 4, 8 stay
	// shared.
	if _, err := fx.svc.GenerateForVideoWithPlan(ctx, fx.vid, fx.segments, semPlan4, 1280, 720); err != nil {
		t.Fatalf("semantic replace: %v", err)
	}
	if len(fx.repo.frames) != 6 {
		t.Fatalf("semantic replacement must not delete perception-only rows, got %d rows", len(fx.repo.frames))
	}
	assertPurposes(t, fx, map[float64]Purpose{
		0: PurposeBoth, 4: PurposeBoth, 8: PurposeBoth,
		2: PurposePerception, 6: PurposePerception, 10: PurposePerception,
	})

	semSet, _ := fx.svc.repo.GetByVideoIDSet(ctx, fx.vid.ID, SetSemantic)
	perSet, _ := fx.svc.repo.GetByVideoIDSet(ctx, fx.vid.ID, SetPerception)
	if len(semSet) != 3 {
		t.Fatalf("semantic set should have 3 members, got %d", len(semSet))
	}
	if len(perSet) != 6 {
		t.Fatalf("perception set must stay intact, got %d members", len(perSet))
	}

	// Densify perception to 1s: rows 2, 6, 10 were demoted to
	// perception-only by the semantic replacement, so they simply confirm
	// their perception membership; new rows extract; semantic membership
	// (0, 4, 8) is untouched — perception regeneration must never grow the
	// semantic set.
	fx.svc.WithPerceptionSampleFPS(1.0)
	perceptionRes, err := fx.svc.GeneratePerceptionForVideo(ctx, fx.vid, fx.segments, duration, 1280, 720)
	if err != nil {
		t.Fatalf("perception densify: %v", err)
	}
	if len(perceptionRes.DroppedFrameIDs) != 0 {
		t.Fatalf("densifying perception must not drop frames, got %d dropped", len(perceptionRes.DroppedFrameIDs))
	}
	if len(fx.repo.frames) != 12 {
		t.Fatalf("want 12 rows after densifying to 1s, got %d", len(fx.repo.frames))
	}
	for _, ts := range []float64{2, 6, 10} {
		assertPurposes(t, fx, map[float64]Purpose{ts: PurposePerception})
	}
	semSet, _ = fx.svc.repo.GetByVideoIDSet(ctx, fx.vid.ID, SetSemantic)
	if len(semSet) != 3 {
		t.Fatalf("semantic set changed during perception regen: %d members", len(semSet))
	}
	ids := make(map[int]bool, 12)
	maxIdx := -1
	for _, f := range fx.repo.frames {
		if ids[f.FrameIndex] {
			t.Fatalf("duplicate frame index %d", f.FrameIndex)
		}
		ids[f.FrameIndex] = true
		if f.FrameIndex > maxIdx {
			maxIdx = f.FrameIndex
		}
	}
	if len(ids) != 12 {
		t.Fatalf("want 12 distinct indexes, got %d", len(ids))
	}
	if maxIdx != 11 {
		t.Fatalf("new perception rows must continue the index range, max %d", maxIdx)
	}
}

// TestPerceptionShrinkDropsSharedFrames: shrinking the perception cadence
// demotes shared rows out of the perception set and reports their IDs so the
// orchestrator can purge their stale production detections.
func TestPerceptionShrinkDropsSharedFrames(t *testing.T) {
	const duration = 12.0
	fx := newPlanFixture(t, duration)
	ctx := context.Background()

	if _, err := fx.svc.GeneratePerceptionForVideo(ctx, fx.vid, fx.segments, duration, 1280, 720); err != nil {
		t.Fatalf("perception: %v", err)
	}
	semPlan, err := sampler.NewBaselinePlanner(2*time.Second, 1.0).Plan(fx.vid.ID, duration)
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	if _, err := fx.svc.GenerateForVideoWithPlan(ctx, fx.vid, fx.segments, semPlan, 1280, 720); err != nil {
		t.Fatalf("semantic: %v", err)
	}
	// All six rows are 'both' now; shrink perception to 4s.
	fx.svc.WithPerceptionSampleFPS(0.25)
	res, err := fx.svc.GeneratePerceptionForVideo(ctx, fx.vid, fx.segments, duration, 1280, 720)
	if err != nil {
		t.Fatalf("perception shrink: %v", err)
	}
	if len(res.Frames) != 3 {
		t.Fatalf("want 3 perception frames after shrink, got %d", len(res.Frames))
	}
	for _, ts := range []float64{2, 6, 10} {
		f := frameAt(fx.repo.frames, ts)
		if f == nil {
			t.Fatalf("shared frame at %v must survive demotion", ts)
		}
		if f.Purpose != PurposeSemantic {
			t.Fatalf("frame at %v demoted to %q, want semantic", ts, f.Purpose)
		}
	}
	if len(res.DroppedFrameIDs) != 3 {
		t.Fatalf("want 3 dropped (demoted) frame ids, got %d", len(res.DroppedFrameIDs))
	}
	for _, id := range res.DroppedFrameIDs {
		found := false
		for _, ts := range []float64{2, 6, 10} {
			if f := frameAt(fx.repo.frames, ts); f != nil && f.ID == id {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("dropped id %s does not correspond to a demoted shared frame", id)
		}
	}
}

// TestExtractAdditionalAdoptsPerceptionFrames: semantic refinement adopts
// perception-only rows (promote to 'both', returned to the refiner), skips
// rows already in the semantic set, and extracts fresh ones without
// re-touching adopted TimestampSeconds.
func TestExtractAdditionalAdoptsPerceptionFrames(t *testing.T) {
	const duration = 12.0
	fx := newPlanFixture(t, duration)
	ctx := context.Background()

	if _, err := fx.svc.GeneratePerceptionForVideo(ctx, fx.vid, fx.segments, duration, 1280, 720); err != nil {
		t.Fatalf("perception: %v", err)
	}
	// Coarser semantic replacement leaves rows 2, 6, 10 perception-only.
	semPlan4, err := sampler.NewBaselinePlanner(4*time.Second, 1.0).Plan(fx.vid.ID, duration)
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	if _, err := fx.svc.GenerateForVideoWithPlan(ctx, fx.vid, fx.segments, semPlan4, 1280, 720); err != nil {
		t.Fatalf("semantic replace: %v", err)
	}
	assertPurposes(t, fx, map[float64]Purpose{6: PurposePerception})
	baseCalls := len(fx.callSnapshot())

	added, err := fx.svc.ExtractAdditional(ctx, fx.vid, fx.segments, []float64{6}, duration, 1280, 720)
	if err != nil {
		t.Fatalf("ExtractAdditional: %v", err)
	}
	if len(added) != 1 || !epsEqual(added[0].TimestampSeconds, 6) {
		t.Fatalf("want adopted row returned, got %v", tsList(added))
	}
	if added[0].Purpose != PurposeBoth {
		t.Fatalf("adopted perception row must be promoted, got %q", added[0].Purpose)
	}
	// No extraction happened: the perception row's asset was reused.
	if len(fx.callSnapshot()) != baseCalls {
		t.Fatalf("adopted frame must not be re-extracted")
	}
	assertPurposes(t, fx, map[float64]Purpose{6: PurposeBoth})

	// Already in the semantic set: skipped, nothing returned.
	again, err := fx.svc.ExtractAdditional(ctx, fx.vid, fx.segments, []float64{6}, duration, 1280, 720)
	if err != nil {
		t.Fatalf("ExtractAdditional again: %v", err)
	}
	if len(again) != 0 {
		t.Fatalf("already-semantic timestamp must be skipped, got %v", tsList(again))
	}

	// Fresh timestamps: extracted as semantic, FrameIndex continues.
	fresh, err := fx.svc.ExtractAdditional(ctx, fx.vid, fx.segments, []float64{9, 1}, duration, 1280, 720)
	if err != nil {
		t.Fatalf("ExtractAdditional fresh: %v", err)
	}
	if len(fresh) != 2 {
		t.Fatalf("want 2 fresh frames, got %v", tsList(fresh))
	}
	if fresh[0].TimestampSeconds >= fresh[1].TimestampSeconds {
		t.Fatalf("result must follow requested timestamp order: %v", tsList(fresh))
	}
	assertPurposes(t, fx, map[float64]Purpose{9: PurposeSemantic, 1: PurposeSemantic})
}
