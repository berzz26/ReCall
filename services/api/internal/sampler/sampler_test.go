package sampler

import (
	"math"
	"testing"
	"time"

	"github.com/google/uuid"
)

func TestBaselineBudget(t *testing.T) {
	b, err := BaselineBudget(10, 2*time.Second, 1.0)
	if err != nil {
		t.Fatalf("err %v", err)
	}
	if b.BaselineCount != 5 || b.Cap != 5 {
		t.Fatalf("expected N=5 B=5 got %+v", b)
	}

	// 110.94s video: ceil(110.94/2) = 56, matching legacy frame count.
	b, err = BaselineBudget(110.94, 2*time.Second, 1.0)
	if err != nil {
		t.Fatalf("err %v", err)
	}
	if b.BaselineCount != 56 || b.Cap != 56 {
		t.Fatalf("expected N=56 B=56 got %+v", b)
	}

	// Short video still yields 1 frame.
	b, err = BaselineBudget(1, 2*time.Second, 1.0)
	if err != nil {
		t.Fatalf("err %v", err)
	}
	if b.BaselineCount != 1 || b.Cap != 1 {
		t.Fatalf("expected N=1 B=1 got %+v", b)
	}

	// Beta scales the cap: B = ceil(beta * N).
	b, err = BaselineBudget(10, 2*time.Second, 0.5)
	if err != nil {
		t.Fatalf("err %v", err)
	}
	if b.Cap != 3 { // ceil(0.5*5)
		t.Fatalf("expected B=3 got %+v", b)
	}
}

func TestBaselineBudgetInvalid(t *testing.T) {
	for _, tc := range []struct {
		name     string
		duration float64
		interval time.Duration
		beta     float64
	}{
		{"zero duration", 0, 2 * time.Second, 1.0},
		{"negative duration", -1, 2 * time.Second, 1.0},
		{"NaN duration", math.NaN(), 2 * time.Second, 1.0},
		{"zero beta", 10, 2 * time.Second, 0},
		{"negative beta", 10, 2 * time.Second, -0.5},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := BaselineBudget(tc.duration, tc.interval, tc.beta); err == nil {
				t.Fatalf("expected error")
			}
		})
	}
	// Non-positive interval falls back to the 2s default rather than erroring.
	b, err := BaselineBudget(10, 0, 1.0)
	if err != nil {
		t.Fatalf("err %v", err)
	}
	if b.BaselineCount != 5 {
		t.Fatalf("expected fallback N=5 got %+v", b)
	}
}

func TestBaselinePlannerGrid(t *testing.T) {
	p := NewBaselinePlanner(2*time.Second, 1.0)
	plan, err := p.Plan(uuid.New(), 10)
	if err != nil {
		t.Fatalf("err %v", err)
	}
	if plan.Mode != ModeBaseline {
		t.Fatalf("mode %q", plan.Mode)
	}
	exp := []float64{0, 2, 4, 6, 8}
	got := plan.Timestamps()
	if len(got) != len(exp) {
		t.Fatalf("expected %v got %v", exp, got)
	}
	for i, e := range exp {
		if math.Abs(got[i]-e) > 1e-9 {
			t.Fatalf("mismatch %d: %v vs %v", i, got[i], e)
		}
		if plan.Entries[i].Reason != ReasonCoarse {
			t.Fatalf("entry %d reason %q", i, plan.Entries[i].Reason)
		}
	}
	if plan.Budget.Cap != 5 || len(plan.Entries) > plan.Budget.Cap {
		t.Fatalf("budget violated: %+v entries=%d", plan.Budget, len(plan.Entries))
	}
}

func TestBaselinePlannerInvalidDuration(t *testing.T) {
	p := NewBaselinePlanner(2*time.Second, 1.0)
	if _, err := p.Plan(uuid.New(), 0); err == nil {
		t.Fatalf("expected error for 0 duration")
	}
}

func TestNormalizeSorting(t *testing.T) {
	got := NormalizeTimestamps([]float64{9.5, 0.0, 6.2, 2.7}, 20)
	exp := []float64{0.0, 2.7, 6.2, 9.5}
	if len(got) != len(exp) {
		t.Fatalf("expected %v got %v", exp, got)
	}
	for i := range exp {
		if math.Abs(got[i]-exp[i]) > 1e-9 {
			t.Fatalf("mismatch %d: %v vs %v", i, got[i], exp[i])
		}
	}
}

func TestNormalizeDuplicates(t *testing.T) {
	got := NormalizeTimestamps([]float64{2.0, 0.0, 2.0, 4.0, 0.0}, 20)
	if len(got) != 3 || got[0] != 0 || got[1] != 2 || got[2] != 4 {
		t.Fatalf("expected [0 2 4] got %v", got)
	}
	// Near-duplicates within epsilon collapse too.
	got = NormalizeTimestamps([]float64{2.0, 2.0 + 1e-9, 4.0}, 20)
	if len(got) != 2 {
		t.Fatalf("expected 2 got %v", got)
	}
}

func TestNormalizeBoundaries(t *testing.T) {
	const dur = 10.0
	got := NormalizeTimestamps([]float64{-5.0, -1e-9, 0.0, 9.999999, 10.0, 12.0}, dur)
	// -5 dropped, -1e-9 clamped to 0, 0 kept once, 9.999999 kept, >=10 dropped.
	if len(got) != 2 {
		t.Fatalf("expected 2 got %v", got)
	}
	if got[0] != 0 {
		t.Fatalf("expected first 0 got %v", got[0])
	}
	if math.Abs(got[1]-9.999999) > 1e-9 {
		t.Fatalf("expected 9.999999 got %v", got[1])
	}
	// NaN/Inf dropped.
	got = NormalizeTimestamps([]float64{math.NaN(), math.Inf(1), 2.0}, dur)
	if len(got) != 1 || got[0] != 2.0 {
		t.Fatalf("expected [2] got %v", got)
	}
}

func TestNormalizeEmpty(t *testing.T) {
	got := NormalizeTimestamps(nil, 10)
	if len(got) != 0 {
		t.Fatalf("expected empty got %v", got)
	}
	got = NormalizeTimestamps([]float64{}, 10)
	if len(got) != 0 {
		t.Fatalf("expected empty got %v", got)
	}
}

func TestCapTimestamps(t *testing.T) {
	ts := []float64{0, 2, 4, 6, 8}
	capped := CapTimestamps(ts, 3)
	if len(capped) != 3 || capped[2] != 4 {
		t.Fatalf("expected first 3 got %v", capped)
	}
	// Cap above length is a no-op.
	if got := CapTimestamps(ts, 10); len(got) != 5 {
		t.Fatalf("expected 5 got %v", got)
	}
}

func TestPlanWithTimestampsEnforcesCap(t *testing.T) {
	vid := uuid.New()
	budget := Budget{BaselineInterval: 2 * time.Second, Beta: 1.0, BaselineCount: 5, Cap: 2}
	plan := PlanWithTimestamps(vid, 10, []float64{8, 0, 4, 2, 6}, ReasonHeartbeat, budget)
	got := plan.Timestamps()
	if len(got) != 2 || got[0] != 0 || got[1] != 2 {
		t.Fatalf("expected earliest 2 sorted, got %v", got)
	}
	for _, e := range plan.Entries {
		if e.Reason != ReasonHeartbeat {
			t.Fatalf("reason %q", e.Reason)
		}
	}
}
