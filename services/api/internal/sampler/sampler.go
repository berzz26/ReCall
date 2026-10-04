// Package sampler separates frame *selection* (planning) from frame
// *extraction* (decoding).
//
// Pipeline boundary:
//
//	VIDEO
//	  |
//	  v
//	SAMPLER / PLANNER  (this package: decides WHICH timestamps to process)
//	  |
//	  | timestamp list (SamplingPlan)
//	  v
//	FRAME EXTRACTOR    (video_frame.Service: decides HOW to obtain those frames)
//	  |
//	  | VideoFrame objects (unchanged shape)
//	  v
//	YOLO -> tracking -> events -> VLM -> embeddings (untouched)
//
// The sampler knows nothing about YOLO, events, or the VLM. Downstream
// components never need to know why a frame was selected.
package sampler

import (
	"fmt"
	"math"
	"sort"
	"time"

	"github.com/google/uuid"
)

// Selection reasons a timestamp can carry. Phase 1 only produces
// ReasonCoarse (the fixed-interval baseline pass); the remaining values
// reserve the vocabulary for later adaptive phases.
type Reason string

const (
	ReasonCoarse     Reason = "coarse"
	ReasonHeartbeat  Reason = "heartbeat"
	ReasonRefinement Reason = "refinement"
)

// Sampler modes.
const (
	// ModeBaseline reproduces the legacy fixed-interval behavior.
	ModeBaseline = "baseline"
)

// dedupeEpsilon treats two timestamps closer than this as duplicates.
const dedupeEpsilon = 1e-6

// DefaultBaselineInterval preserves the legacy fixed 2-second sampling.
const DefaultBaselineInterval = 2 * time.Second

// DefaultBeta preserves the legacy frame budget exactly.
const DefaultBeta = 1.0

// PlannedTimestamp is one selected timestamp plus inspectable metadata
// describing why it was selected.
type PlannedTimestamp struct {
	TimestampSeconds float64
	Reason           Reason
	// Score is optional selection-score information (nil when the planner
	// does not score, e.g. the uniform baseline planner).
	Score *float64
}

// Budget is the central frame-budget abstraction.
//
//	N_baseline = ceil(duration / baselineInterval)
//	B          = ceil(beta * N_baseline)   (hard cap on frames sent to YOLO)
//
// Phase 1 uses beta = 1.0, therefore B = N_baseline.
type Budget struct {
	// BaselineInterval is the reference fixed sampling interval (2s).
	BaselineInterval time.Duration
	// Beta scales the baseline count. Starts at 1.0.
	Beta float64
	// BaselineCount is N_baseline.
	BaselineCount int
	// Cap is B, the hard cap. The sampler must never emit more than Cap
	// timestamps, and the extractor must never return more than Cap frames.
	Cap int
}

// BaselineBudget computes the budget for a video of the given duration.
// It is the single place where the N_baseline / B calculation lives.
func BaselineBudget(durationSeconds float64, baselineInterval time.Duration, beta float64) (Budget, error) {
	if math.IsNaN(durationSeconds) || math.IsInf(durationSeconds, 0) || durationSeconds <= 0 {
		return Budget{}, fmt.Errorf("video duration unavailable; cannot compute budget")
	}
	iv := baselineInterval
	if iv <= 0 {
		iv = DefaultBaselineInterval
	}
	if iv.Seconds() <= 0 {
		return Budget{}, fmt.Errorf("baseline interval must be > 0")
	}
	if math.IsNaN(beta) || math.IsInf(beta, 0) || beta <= 0 {
		return Budget{}, fmt.Errorf("sampler beta must be > 0, got %v", beta)
	}
	n := int(math.Ceil(durationSeconds / iv.Seconds()))
	if n <= 0 {
		return Budget{}, fmt.Errorf("video duration unavailable; cannot compute budget")
	}
	cap := int(math.Ceil(beta * float64(n)))
	if cap < 1 {
		cap = 1
	}
	return Budget{
		BaselineInterval: iv,
		Beta:             beta,
		BaselineCount:    n,
		Cap:              cap,
	}, nil
}

// SamplingPlan is the explicit, inspectable representation of which
// timestamps the sampler selected for a video.
type SamplingPlan struct {
	VideoID         uuid.UUID
	DurationSeconds float64
	// Mode names the planner that produced this plan ("baseline" for now).
	Mode string
	// Budget records the budget this plan was built under.
	Budget Budget
	// Entries are sorted ascending by timestamp, deduplicated, clamped to
	// [0, duration), and capped at Budget.Cap.
	Entries []PlannedTimestamp
}

// Timestamps returns the selected timestamps in ascending order.
func (p SamplingPlan) Timestamps() []float64 {
	ts := make([]float64, 0, len(p.Entries))
	for _, e := range p.Entries {
		ts = append(ts, e.TimestampSeconds)
	}
	return ts
}

// Len returns the number of selected timestamps.
func (p SamplingPlan) Len() int { return len(p.Entries) }

// NormalizeTimestamps sorts ascending, removes duplicates (within
// dedupeEpsilon), drops out-of-range values, and clamps tiny negative
// values to 0. Timestamps at or beyond durationSeconds are dropped because
// no frame exists there. NaN/Inf are dropped.
func NormalizeTimestamps(timestamps []float64, durationSeconds float64) []float64 {
	if len(timestamps) == 0 {
		return []float64{}
	}
	sorted := make([]float64, 0, len(timestamps))
	for _, t := range timestamps {
		if math.IsNaN(t) || math.IsInf(t, 0) {
			continue
		}
		if t < 0 {
			if t > -dedupeEpsilon {
				t = 0
			} else {
				continue
			}
		}
		if t >= durationSeconds-1e-9 {
			continue
		}
		sorted = append(sorted, t)
	}
	sort.Float64s(sorted)
	out := make([]float64, 0, len(sorted))
	for i, t := range sorted {
		if i > 0 && math.Abs(t-sorted[i-1]) < dedupeEpsilon {
			continue
		}
		out = append(out, t)
	}
	return out
}

// CapTimestamps enforces the hard budget cap, keeping the earliest
// timestamps (entries are ascending).
func CapTimestamps(timestamps []float64, capN int) []float64 {
	if capN < 0 {
		capN = 0
	}
	if len(timestamps) > capN {
		return timestamps[:capN]
	}
	return timestamps
}

// BaselineTimestamps generates the legacy uniform grid:
// 0, interval, 2*interval, ... < duration. This is the canonical
// implementation; video_frame.SampleTimestamps delegates to it so the grid
// is defined in exactly one place.
func BaselineTimestamps(durationSeconds float64, interval time.Duration) ([]float64, error) {
	if interval <= 0 {
		return nil, fmt.Errorf("frame sample interval must be > 0")
	}
	if math.IsNaN(durationSeconds) || math.IsInf(durationSeconds, 0) || durationSeconds <= 0 {
		return nil, fmt.Errorf("video duration unavailable; cannot sample frames")
	}
	iv := interval.Seconds()
	if iv <= 0 {
		return nil, fmt.Errorf("frame sample interval must be > 0")
	}
	var ts []float64
	for t := 0.0; t < durationSeconds-1e-9; t += iv {
		ts = append(ts, t)
	}
	if len(ts) == 0 {
		return nil, fmt.Errorf("video duration unavailable; cannot sample frames")
	}
	return ts, nil
}

// Planner decides which timestamps of a video should be processed.
// It returns timestamps only; it never touches the decoder, YOLO,
// events, or the VLM.
type Planner interface {
	Plan(videoID uuid.UUID, durationSeconds float64) (SamplingPlan, error)
}

// BaselinePlanner reproduces the legacy fixed-interval sampling as an
// explicit plan: uniform grid at BaselineInterval, capped at the budget
// (beta-scaled). With beta = 1.0 the plan is exactly the legacy grid.
type BaselinePlanner struct {
	BaselineInterval time.Duration
	Beta             float64
}

// NewBaselinePlanner builds a planner; non-positive values fall back to the
// legacy defaults (2s interval, beta 1.0).
func NewBaselinePlanner(baselineInterval time.Duration, beta float64) *BaselinePlanner {
	if baselineInterval <= 0 {
		baselineInterval = DefaultBaselineInterval
	}
	if math.IsNaN(beta) || math.IsInf(beta, 0) || beta <= 0 {
		beta = DefaultBeta
	}
	return &BaselinePlanner{BaselineInterval: baselineInterval, Beta: beta}
}

// Plan implements Planner.
func (p *BaselinePlanner) Plan(videoID uuid.UUID, durationSeconds float64) (SamplingPlan, error) {
	budget, err := BaselineBudget(durationSeconds, p.BaselineInterval, p.Beta)
	if err != nil {
		return SamplingPlan{}, err
	}
	raw, err := BaselineTimestamps(durationSeconds, p.BaselineInterval)
	if err != nil {
		return SamplingPlan{}, err
	}
	ts := CapTimestamps(NormalizeTimestamps(raw, durationSeconds), budget.Cap)
	entries := make([]PlannedTimestamp, 0, len(ts))
	for _, t := range ts {
		entries = append(entries, PlannedTimestamp{TimestampSeconds: t, Reason: ReasonCoarse})
	}
	return SamplingPlan{
		VideoID:         videoID,
		DurationSeconds: durationSeconds,
		Mode:            ModeBaseline,
		Budget:          budget,
		Entries:         entries,
	}, nil
}

// PlanWithTimestamps builds an explicit plan from a caller-supplied
// timestamp list (used by tests and, later, adaptive rounds). The list is
// normalized (sorted, deduplicated, range-checked) and hard-capped at the
// budget so the extractor can never exceed B frames.
func PlanWithTimestamps(videoID uuid.UUID, durationSeconds float64, timestamps []float64, reason Reason, budget Budget) SamplingPlan {
	ts := CapTimestamps(NormalizeTimestamps(timestamps, durationSeconds), budget.Cap)
	entries := make([]PlannedTimestamp, 0, len(ts))
	for _, t := range ts {
		entries = append(entries, PlannedTimestamp{TimestampSeconds: t, Reason: reason})
	}
	return SamplingPlan{
		VideoID:         videoID,
		DurationSeconds: durationSeconds,
		Mode:            ModeBaseline,
		Budget:          budget,
		Entries:         entries,
	}
}
