package sampler

// Unit tests for the coarse selection rule: running maximum, heartbeat,
// activity gating, reset, budget cap, fallback, and timestamp validity.

import (
	"context"
	"math"
	"testing"
	"time"

	"github.com/google/uuid"
)

// synthStatic returns n probe frames at fps spacing with identical means.
func synthStatic(level uint8, n int, fps float64) []ProbeFrame {
	m := make([]uint8, 64)
	for i := range m {
		m[i] = level
	}
	frames := make([]ProbeFrame, 0, n)
	for i := 0; i < n; i++ {
		cp := make([]uint8, 64)
		copy(cp, m)
		frames = append(frames, ProbeFrame{TimestampSeconds: float64(i) / fps, Means: cp})
	}
	return frames
}

// synthSpike returns means identical to base except changedBlocks entries
// shifted by delta.
func synthSpike(base uint8, changedBlocks int, delta uint8) []uint8 {
	m := make([]uint8, 64)
	for i := range m {
		m[i] = base
	}
	for b := 0; b < changedBlocks && b < 64; b++ {
		m[b] = base + delta
	}
	return m
}

func testPlanner() *AdaptiveCoarsePlanner {
	return NewAdaptiveCoarsePlanner(AdaptiveConfig{
		BaselineInterval: 2 * time.Second,
		Beta:             1.0,
		ProbeFPS:         5,
		ProbeSize:        64,
		ProbeGrid:        8,
		NoiseK:           3,
		CoarseInterval:   4 * time.Second,
		MaxGap:           10 * time.Second,
		FKeep:            0.25,
		FFmpegPath:       "ffmpeg",
		PlanTimeout:      time.Minute,
	})
}

func reasonsOf(plan SamplingPlan) map[float64]Reason {
	out := map[float64]Reason{}
	for _, e := range plan.Entries {
		out[e.TimestampSeconds] = e.Reason
	}
	return out
}

// Test 5 — a static video still generates heartbeat samples.
func TestHeartbeatOnStaticVideo(t *testing.T) {
	p := testPlanner()
	frames := synthStatic(100, 5*25, 5) // 25s static
	plan, report, err := p.PlanVideoWithFrames(uuid.New(), 25, frames, 0)
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	if plan.Mode != ModeAdaptive {
		t.Fatalf("mode %q", plan.Mode)
	}
	// Heartbeats at ~10s and ~20s plus initial coarse at 0.
	want := []float64{0, 10, 20}
	got := plan.Timestamps()
	if len(got) != len(want) {
		t.Fatalf("want %v, got %v", want, got)
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-9 {
			t.Fatalf("want %v, got %v", want, got)
		}
	}
	rs := reasonsOf(plan)
	if rs[0] != ReasonCoarse || rs[10] != ReasonHeartbeat || rs[20] != ReasonHeartbeat {
		t.Fatalf("reasons wrong: %v", rs)
	}
	if report == nil || len(report.Points) != len(frames) {
		t.Fatalf("probe series must be retained")
	}
}

// Test 6 — sustained activity past G selects a coarse frame.
func TestCoarseActivitySelection(t *testing.T) {
	p := testPlanner()
	frames := synthStatic(100, 5*25, 5) // 25s
	spike := synthSpike(100, 20, 80)    // 20/64 = 0.3125 >= F_KEEP
	for i := 5; i < len(frames); i++ {  // change persists from t=1.0
		frames[i].Means = spike
	}
	plan, _, err := p.PlanVideoWithFrames(uuid.New(), 25, frames, 0)
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	rs := reasonsOf(plan)
	if rs[4.0] != ReasonCoarse {
		t.Fatalf("want coarse keep at 4.0, got %v", plan.Timestamps())
	}
	// After the 4.0 keep the changed frame becomes the reference, so later
	// frames match it; coverage resumes via heartbeat at 14.0.
	if rs[14.0] != ReasonHeartbeat {
		t.Fatalf("want heartbeat at 14.0, got %v", plan.Timestamps())
	}
}

// Test 7 — activity before G does not trigger immediately, but the running
// maximum remembers it and fires once G elapses.
func TestActivityBeforeGRemembered(t *testing.T) {
	p := testPlanner()
	frames := synthStatic(100, 5*12, 5)
	frames[5].Means = synthSpike(100, 20, 80) // single spike at t=1.0
	plan, report, err := p.PlanVideoWithFrames(uuid.New(), 12, frames, 0)
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	// No selection at the spike itself.
	for _, pt := range report.Points {
		if math.Abs(pt.TimestampSeconds-1.0) < 1e-9 {
			if pt.Selected {
				t.Fatalf("spike at 1.0 must not select before G")
			}
			// Test 4 — running maximum preserves the transient change.
			if math.Abs(pt.RunningMax-20.0/64) > 1e-9 {
				t.Fatalf("running max must hold 0.3125, got %v", pt.RunningMax)
			}
		}
	}
	rs := reasonsOf(plan)
	if rs[4.0] != ReasonCoarse {
		t.Fatalf("remembered activity must fire at G=4.0, got %v", plan.Timestamps())
	}
}

// Test 8 — running_max resets after a selection.
func TestRunningMaxResetAfterSelection(t *testing.T) {
	p := testPlanner()
	frames := synthStatic(100, 5*12, 5)
	frames[5].Means = synthSpike(100, 20, 80) // spike at t=1.0, fires at 4.0
	_, report, err := p.PlanVideoWithFrames(uuid.New(), 12, frames, 0)
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	// Frame right after the 4.0 selection is static vs the new reference:
	// its running max must be back at 0.
	for _, pt := range report.Points {
		if math.Abs(pt.TimestampSeconds-4.2) < 1e-9 && pt.RunningMax != 0 {
			t.Fatalf("running max must reset after selection, got %v", pt.RunningMax)
		}
	}
}

// Test 10 — a configuration that would select hundreds of timestamps is
// still capped at the Phase 1 budget.
func TestBudgetCapEnforced(t *testing.T) {
	p := NewAdaptiveCoarsePlanner(AdaptiveConfig{
		BaselineInterval: 2 * time.Second,
		Beta:             1.0,
		CoarseInterval:   200 * time.Millisecond, // permissive: every frame
		MaxGap:           time.Hour,              // no heartbeat interference
		FKeep:            0,                      // any non-negative max keeps
		FFmpegPath:       "ffmpeg",
		PlanTimeout:      time.Minute,
	})
	const duration = 60.0
	frames := synthStatic(100, 300, 5)
	for i := range frames {
		frames[i].Means = synthSpike(uint8(100+i%50), 64, 10) // always changing
	}
	plan, _, err := p.PlanVideoWithFrames(uuid.New(), duration, frames, 0)
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	budget, _ := BaselineBudget(duration, 2*time.Second, 1.0)
	if len(plan.Entries) != budget.Cap {
		t.Fatalf("want exactly cap=%d, got %d", budget.Cap, len(plan.Entries))
	}
	if plan.Budget.Cap != budget.Cap {
		t.Fatalf("plan must record the Phase 1 budget")
	}
}

// Test 11 — probe failure returns the baseline plan with
// mode = baseline_fallback.
func TestFallbackOnProbeFailure(t *testing.T) {
	p := testPlanner()
	vid := uuid.New()
	const duration = 30.0
	res, err := p.PlanVideo(context.Background(), vid, "/nonexistent/video.mp4", duration)
	if err != nil {
		t.Fatalf("fallback must not error, got %v", err)
	}
	if res.Plan.Mode != ModeBaselineFallback {
		t.Fatalf("mode %q", res.Plan.Mode)
	}
	if res.FallbackReason == "" {
		t.Fatalf("fallback reason must be recorded")
	}
	if res.Probe != nil {
		t.Fatalf("no probe series on fallback")
	}
	base, err := NewBaselinePlanner(2*time.Second, 1.0).Plan(vid, duration)
	if err != nil {
		t.Fatalf("baseline: %v", err)
	}
	got, want := res.Plan.Timestamps(), base.Timestamps()
	if len(got) != len(want) {
		t.Fatalf("fallback must equal baseline: %d vs %d", len(got), len(want))
	}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-9 {
			t.Fatalf("fallback must equal baseline")
		}
	}
}

// Test 12 — selected timestamps are always sorted, unique, and within
// [0, duration); short videos still yield a valid plan.
func TestTimestampValidityAndShortVideos(t *testing.T) {
	p := testPlanner()
	for _, duration := range []float64{0.5, 1.0, 2.0, 25.0} {
		n := int(duration * 5)
		if n < 1 {
			n = 1
		}
		frames := synthStatic(100, n, 5)
		// Inject change so the active path is exercised too.
		if len(frames) > 3 {
			frames[len(frames)-1].Means = synthSpike(100, 40, 80)
		}
		plan, _, err := p.PlanVideoWithFrames(uuid.New(), duration, frames, 1)
		if err != nil {
			t.Fatalf("duration %v: %v", duration, err)
		}
		ts := plan.Timestamps()
		if len(ts) == 0 {
			t.Fatalf("duration %v: empty plan", duration)
		}
		seen := map[float64]bool{}
		for i, x := range ts {
			if x < 0 || x >= duration {
				t.Fatalf("duration %v: out-of-range ts %v", duration, x)
			}
			if i > 0 && x <= ts[i-1] {
				t.Fatalf("duration %v: unsorted %v", duration, ts)
			}
			if seen[x] {
				t.Fatalf("duration %v: duplicate %v", duration, x)
			}
			seen[x] = true
		}
		if ts[0] != 0 {
			t.Fatalf("duration %v: first ts must be 0, got %v", duration, ts[0])
		}
		budget, _ := BaselineBudget(duration, 2*time.Second, 1.0)
		if len(ts) > budget.Cap {
			t.Fatalf("duration %v: exceeds budget", duration)
		}
	}
}

// Score metadata: selections carry the triggering running maximum.
func TestSelectionScoresRecorded(t *testing.T) {
	p := testPlanner()
	frames := synthStatic(100, 5*12, 5)
	for i := 5; i < len(frames); i++ {
		frames[i].Means = synthSpike(100, 20, 80)
	}
	plan, _, err := p.PlanVideoWithFrames(uuid.New(), 12, frames, 0)
	if err != nil {
		t.Fatalf("plan: %v", err)
	}
	for _, e := range plan.Entries {
		if e.Score == nil {
			t.Fatalf("selection at %v lacks probe score", e.TimestampSeconds)
		}
		if e.Reason == ReasonCoarse && e.TimestampSeconds > 0 && *e.Score < DefaultFKeep {
			t.Fatalf("coarse selection below F_KEEP: %v", *e.Score)
		}
	}
}
