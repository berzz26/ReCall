package sampler

// Production robustness for the adaptive sampler (Phase 5).
//
// Safety hierarchy (adaptive sampler is an optimization; the baseline plan
// is the source of truth):
//
//	            ┌───────────────┐
//	            │ Adaptive ON?  │  (SAMPLER_ADAPTIVE)
//	            └───────┬───────┘
//	                    │
//	         no ────────┴────── yes
//	         │                  │
//	         ↓                  ↓
//	     baseline          run probe
//	                            │
//	                  ┌─────────┴─────────┐
//	                  │                   │
//	               failure             success
//	                  │                   │
//	                  ↓                   ↓
//	          baseline_fallback      busy?
//	                                      │
//	                            ┌─────────┴─────────┐
//	                            │                   │
//	                          yes                  no
//	                            │                   │
//	                            ↓                   ↓
//	                   uniform_fallback       adaptive (+refine)
//	                            │                   │
//	                            └─────────┬─────────┘
//	                                      ↓
//	                               validate plan
//	                                      │
//	                            ┌─────────┴─────────┐
//	                            │                   │
//	                          invalid             valid
//	                            │                   │
//	                            ↓                   ↓
//	                    baseline_fallback       downstream
//
// Any adaptive failure discards the adaptive state and falls back to the
// baseline plan; downstream stages only ever consume one complete,
// validated frame set.

import (
	"fmt"
	"math"
)

const (
	// ModeUniformFallback marks a budget-respecting approximately-uniform
	// plan chosen because the video is globally busy: the probe reports
	// persistent activity across most of the timeline, so selective
	// skipping has little opportunity and refinement would waste planning
	// compute. Distinct from ModeBaselineFallback (adaptive planning
	// itself failed) because the reason — and the tuning response — differ.
	ModeUniformFallback = "uniform_fallback"
)

// SamplerVersion identifies the planning algorithm that produced a plan.
// Bump when probe thresholds, gamma, epsilon, busy detection, or
// refinement behavior change, so later analysis can attribute plans.
const SamplerVersion = "adaptive-v1"

// Busy-heuristic defaults. ChangedFraction is the fraction of the 64 probe
// blocks that changed, so 0.25 means a quarter of the frame differs from
// the last kept reference — the same scale as F_KEEP. A point at or above
// the threshold counts as active; a video with most of its timeline active
// is globally busy. Conservative starting points, easy to tune.
const (
	// DefaultBusyThreshold is the per-point ChangedFraction at/above which
	// a probe sample counts as active.
	DefaultBusyThreshold = 0.25
	// DefaultBusyFraction is the fraction of probe points that must be
	// active for the video to count as globally busy.
	DefaultBusyFraction = 0.6
)

// BusyConfig configures global-busy detection. Zero values fall back to the
// defaults above in sanitized form via NewBusyConfig.
type BusyConfig struct {
	// Threshold is the per-point ChangedFraction counting as active.
	Threshold float64
	// Fraction is the active-point share marking the video globally busy.
	Fraction float64
}

// DefaultBusyConfig returns the Phase 5 starting configuration.
func DefaultBusyConfig() BusyConfig {
	return BusyConfig{Threshold: DefaultBusyThreshold, Fraction: DefaultBusyFraction}
}

// NewBusyConfig sanitizes cfg (NaN/negative values revert to defaults).
func NewBusyConfig(cfg BusyConfig) BusyConfig {
	if math.IsNaN(cfg.Threshold) || cfg.Threshold < 0 {
		cfg.Threshold = DefaultBusyThreshold
	}
	if math.IsNaN(cfg.Fraction) || cfg.Fraction < 0 {
		cfg.Fraction = DefaultBusyFraction
	}
	return cfg
}

// BusyStats records the busy evaluation for logging and tests.
type BusyStats struct {
	// BusyPoints counts probe points at/above threshold; TotalPoints is the
	// series length. BusyFraction = BusyPoints / TotalPoints (0 when empty).
	BusyPoints   int
	TotalPoints  int
	BusyFraction float64
	Threshold    float64
	Busy         bool
}

// ProbeBusyFraction returns the share of probe points whose ChangedFraction
// is at/above threshold. Coverage-style (not mean-based): a video that is
// 10% extremely active and 90% static must NOT read as globally busy, while
// a persistently half-active timeline must.
func ProbeBusyFraction(report *ProbeReport, threshold float64) BusyStats {
	stats := BusyStats{Threshold: threshold}
	if report == nil {
		return stats
	}
	stats.TotalPoints = len(report.Points)
	for _, p := range report.Points {
		if p.ChangedFraction >= threshold {
			stats.BusyPoints++
		}
	}
	if stats.TotalPoints > 0 {
		stats.BusyFraction = float64(stats.BusyPoints) / float64(stats.TotalPoints)
	}
	return stats
}

// IsGloballyBusy reports whether the probe series shows persistent activity
// across most of the timeline. A nil/empty series is never busy (no
// evidence); static video is exactly where adaptive sampling should save
// work, so it must NOT fall back.
func IsGloballyBusy(report *ProbeReport, cfg BusyConfig) BusyStats {
	cfg = NewBusyConfig(cfg)
	stats := ProbeBusyFraction(report, cfg.Threshold)
	stats.Busy = stats.TotalPoints > 0 && stats.BusyFraction >= cfg.Fraction
	return stats
}

// ValidatePlan checks a final plan before it reaches production extraction.
// Timestamps must be sorted, unique (dedupeEpsilon), within [0, duration],
// within budget, non-empty for non-empty video, and every entry must carry
// a known selection reason. Anything else is a planning bug: callers fall
// back to baseline rather than repairing silently.
func ValidatePlan(plan SamplingPlan) error {
	ts := plan.Timestamps()
	if len(ts) == 0 {
		return fmt.Errorf("plan has no timestamps")
	}
	for i, t := range ts {
		if math.IsNaN(t) || math.IsInf(t, 0) {
			return fmt.Errorf("timestamp %d is not finite: %v", i, t)
		}
		if t < 0 {
			return fmt.Errorf("timestamp %d negative: %v", i, t)
		}
		if t >= plan.DurationSeconds-1e-9 {
			return fmt.Errorf("timestamp %d exceeds duration %v: %v", i, plan.DurationSeconds, t)
		}
		if i > 0 {
			if t < ts[i-1] {
				return fmt.Errorf("timestamps not sorted at %d: %v < %v", i, t, ts[i-1])
			}
			if math.Abs(t-ts[i-1]) < dedupeEpsilon {
				return fmt.Errorf("duplicate timestamp at %d: %v", i, t)
			}
		}
	}
	if plan.Budget.Cap > 0 && len(ts) > plan.Budget.Cap {
		return fmt.Errorf("plan has %d frames over budget cap %d", len(ts), plan.Budget.Cap)
	}
	for i, e := range plan.Entries {
		switch e.Reason {
		case ReasonCoarse, ReasonHeartbeat, ReasonRefinement:
		default:
			return fmt.Errorf("entry %d has invalid reason %q", i, e.Reason)
		}
	}
	return nil
}
