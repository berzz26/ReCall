package sampler

// Production-robustness unit tests (Phase 5): sampler modes, global-busy
// heuristic, near-static guard, and final plan validation. No fixtures or
// database needed.

import (
	"testing"

	"github.com/google/uuid"
)

func busyReport(fracs []float64) *ProbeReport {
	rep := &ProbeReport{}
	for i, f := range fracs {
		rep.Points = append(rep.Points, ProbePoint{TimestampSeconds: float64(i) * 0.2, ChangedFraction: f})
	}
	rep.FramesDecoded = len(fracs)
	return rep
}

func TestSamplerModesDistinct(t *testing.T) {
	modes := map[string]bool{
		ModeBaseline:         true,
		ModeAdaptive:         true,
		ModeUniformFallback:  true,
		ModeBaselineFallback: true,
	}
	if len(modes) != 4 {
		t.Fatalf("sampler modes must be four distinct strings: %v", modes)
	}
	if SamplerVersion == "" {
		t.Fatalf("sampler version must be set")
	}
}

func TestGloballyBusyMostlyBusy(t *testing.T) {
	// 90% of points at/above threshold -> globally busy.
	var fracs []float64
	for i := 0; i < 100; i++ {
		if i < 90 {
			fracs = append(fracs, 0.5)
		} else {
			fracs = append(fracs, 0.0)
		}
	}
	stats := IsGloballyBusy(busyReport(fracs), DefaultBusyConfig())
	if !stats.Busy {
		t.Fatalf("90%% active must be busy: %+v", stats)
	}
	if stats.BusyFraction != 0.9 || stats.BusyPoints != 90 || stats.TotalPoints != 100 {
		t.Fatalf("busy stats wrong: %+v", stats)
	}
}

func TestGloballyBusyMixedActivity(t *testing.T) {
	// 30% busy, 70% quiet -> NOT globally busy; adaptive planning proceeds.
	var fracs []float64
	for i := 0; i < 100; i++ {
		if i < 30 {
			fracs = append(fracs, 0.8)
		} else {
			fracs = append(fracs, 0.0)
		}
	}
	stats := IsGloballyBusy(busyReport(fracs), DefaultBusyConfig())
	if stats.Busy {
		t.Fatalf("30%% active must not be busy: %+v", stats)
	}
}

func TestGloballyBusyStatic(t *testing.T) {
	// Completely static video is exactly where adaptive sampling saves
	// work: it must NOT fall back.
	var fracs []float64
	for i := 0; i < 100; i++ {
		fracs = append(fracs, 0.0)
	}
	stats := IsGloballyBusy(busyReport(fracs), DefaultBusyConfig())
	if stats.Busy {
		t.Fatalf("static video must not be busy: %+v", stats)
	}
}

func TestGloballyBusyEmptyAndNil(t *testing.T) {
	if s := IsGloballyBusy(nil, DefaultBusyConfig()); s.Busy || s.BusyFraction != 0 {
		t.Fatalf("nil report must not be busy: %+v", s)
	}
	if s := IsGloballyBusy(&ProbeReport{}, DefaultBusyConfig()); s.Busy {
		t.Fatalf("empty report must not be busy: %+v", s)
	}
}

func TestGloballyBusyMeanWouldMislead(t *testing.T) {
	// 10% extremely active (1.0) + 90% static: a mean-based rule could read
	// this as busy; the coverage rule must not.
	var fracs []float64
	for i := 0; i < 100; i++ {
		if i < 10 {
			fracs = append(fracs, 1.0)
		} else {
			fracs = append(fracs, 0.0)
		}
	}
	stats := IsGloballyBusy(busyReport(fracs), DefaultBusyConfig())
	if stats.Busy {
		t.Fatalf("10%% spiky activity must not be busy: %+v", stats)
	}
}

func TestBusyThresholdBoundary(t *testing.T) {
	// Points exactly at threshold count as active (>=).
	rep := busyReport([]float64{DefaultBusyThreshold, DefaultBusyThreshold, 0.0, 0.0})
	stats := ProbeBusyFraction(rep, DefaultBusyThreshold)
	if stats.BusyFraction != 0.5 {
		t.Fatalf("threshold boundary must count (>=): %+v", stats)
	}
}

func TestNearStaticFlickerGuard(t *testing.T) {
	// Phase 4 concern: one isolated single-block flicker (1/64) among
	// static points. Without a floor it ranks 1 and could drive
	// refinement; with the production floor it ranks 0 everywhere.
	var pts []ProbePoint
	for i := 0; i < 50; i++ {
		pts = append(pts, ProbePoint{TimestampSeconds: float64(i) * 0.2})
	}
	pts[25].ChangedFraction = 1.0 / 64.0
	rep := &ProbeReport{Points: pts}
	starts := []float64{0, 2, 4, 6, 8}
	ends := []float64{2, 4, 6, 8, 10}
	plain := ProbeRankForIntervals(rep, starts, ends)
	peak := 0.0
	for _, r := range plain {
		if r > peak {
			peak = r
		}
	}
	if peak != 1.0 {
		t.Fatalf("unfloored rank must expose the flicker at 1.0, got %v", plain)
	}
	floored := ProbeRankForIntervalsWithFloor(rep, starts, ends, DefaultProbeActivityFloor)
	for _, r := range floored {
		if r != 0 {
			t.Fatalf("floored rank must suppress single-block flicker, got %v", floored)
		}
	}
	// Genuine motion (many blocks) survives the floor.
	pts[26].ChangedFraction = 0.5
	kept := ProbeRankForIntervalsWithFloor(rep, starts, ends, DefaultProbeActivityFloor)
	found := false
	for _, r := range kept {
		if r > 0 {
			found = true
		}
	}
	if !found {
		t.Fatalf("floor must not suppress genuine motion: %v", kept)
	}
}

func validTestPlan() SamplingPlan {
	budget := Budget{BaselineInterval: 2000000000, Beta: 1.0, BaselineCount: 5, Cap: 5}
	return SamplingPlan{
		VideoID:         uuid.New(),
		DurationSeconds: 10,
		Mode:            ModeAdaptive,
		Budget:          budget,
		Entries: []PlannedTimestamp{
			{TimestampSeconds: 0, Reason: ReasonCoarse},
			{TimestampSeconds: 4, Reason: ReasonHeartbeat},
			{TimestampSeconds: 8, Reason: ReasonRefinement},
		},
	}
}

func TestValidatePlanOK(t *testing.T) {
	if err := ValidatePlan(validTestPlan()); err != nil {
		t.Fatalf("valid plan rejected: %v", err)
	}
}

func TestValidatePlanFailures(t *testing.T) {
	cases := map[string]func(SamplingPlan) SamplingPlan{
		"empty": func(p SamplingPlan) SamplingPlan {
			p.Entries = nil
			return p
		},
		"unsorted": func(p SamplingPlan) SamplingPlan {
			p.Entries[0], p.Entries[1] = p.Entries[1], p.Entries[0]
			return p
		},
		"duplicate": func(p SamplingPlan) SamplingPlan {
			p.Entries[1] = p.Entries[0]
			return p
		},
		"negative": func(p SamplingPlan) SamplingPlan {
			p.Entries[0].TimestampSeconds = -1
			return p
		},
		"over duration": func(p SamplingPlan) SamplingPlan {
			p.Entries[2].TimestampSeconds = 10
			return p
		},
		"over cap": func(p SamplingPlan) SamplingPlan {
			p.Budget.Cap = 2
			return p
		},
		"bad reason": func(p SamplingPlan) SamplingPlan {
			p.Entries[0].Reason = "mystery"
			return p
		},
		"empty reason": func(p SamplingPlan) SamplingPlan {
			p.Entries[0].Reason = ""
			return p
		},
	}
	for name, mutate := range cases {
		if err := ValidatePlan(mutate(validTestPlan())); err == nil {
			t.Fatalf("%s plan must fail validation", name)
		}
	}
}

func TestUniformFallbackIsBaselineShaped(t *testing.T) {
	// uniform_fallback reuses the baseline planner: same timestamps and
	// budget as baseline, only the mode records the different reason.
	vid := uuid.New()
	base, err := NewBaselinePlanner(DefaultBaselineInterval, DefaultBeta).Plan(vid, 10)
	if err != nil {
		t.Fatalf("baseline: %v", err)
	}
	base.Mode = ModeUniformFallback
	if err := ValidatePlan(base); err != nil {
		t.Fatalf("uniform fallback plan invalid: %v", err)
	}
	want := []float64{0, 2, 4, 6, 8}
	got := base.Timestamps()
	if len(got) != len(want) {
		t.Fatalf("uniform fallback must carry baseline coverage: %v", got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("uniform fallback must carry baseline coverage: %v", got)
		}
	}
}
