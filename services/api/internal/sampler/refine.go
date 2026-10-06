package sampler

// Priority-based adaptive refinement (Phase 4).
//
// The coarse plan samples more sparsely than the baseline grid. Refinement
// spends the REMAINING frame budget — never more — inserting midpoint frames
// only into intervals where the current sampling appears insufficient:
//
//	priority = max(disagreement, gamma * probe_rank) * gap_length
//
// Loop per round: replay the throwaway tracker from scratch over ALL current
// timestamps, rescore every adjacent interval, rank probe activity per
// interval, select the highest-priority eligible intervals subject to the
// remaining budget, extract + detect ONLY the new midpoints, merge, repeat
// for at most MaxRounds. Unused budget is intentional savings.
//
// The refiner never decodes video or runs YOLO itself: new midpoints are
// obtained through a caller-supplied DetectFunc (extraction + detection stay
// outside the sampler, preserving the planner/extractor boundary and avoiding
// import cycles). It never touches production tracker state.

import (
	"context"
	"fmt"
	"log/slog"
	"math"
	"sort"
	"strconv"

	"github.com/google/uuid"

	"github.com/berzz26/recall/services/api/internal/tracker"
)

// Refinement defaults.
const (
	// DefaultGamma weights the probe-rank prior against disagreement in the
	// max() combination. Untuned starting value.
	DefaultGamma = 0.5
	// DefaultRefineMinGapSeconds (MIN_GAP) is the minimum interval width
	// eligible for splitting. Repeated bisection converges here and stops.
	DefaultRefineMinGapSeconds = 0.5
	// DefaultRefineEpsilon (EPSILON) is the minimum priority eligible for
	// refinement. Small but nonzero: exact-zero priorities (no tracking
	// signal AND no relative probe activity) never refine, while any
	// genuine signal passes. Easy to tune upward to suppress weak
	// probe-only refinement.
	DefaultRefineEpsilon = 0.1
	// DefaultRefineMaxRounds bounds the best-first loop.
	DefaultRefineMaxRounds = 4
	// DefaultProbeActivityFloor is the near-static guard: interval probe
	// activity below this ChangedFraction is treated as zero before rank
	// normalization. ChangedFraction counts changed 8x8 blocks out of 64,
	// so 0.03 (~2 blocks) neutralizes isolated single-block sensor flicker
	// (1/64 ≈ 0.0156) that would otherwise rank 1 and drive refinement on
	// effectively static footage. Genuine motion spans many blocks and is
	// unaffected. Deliberately conservative; tune upward to harden further.
	DefaultProbeActivityFloor = 0.03
)

// RefinementConfig configures the refinement planner. Zero values fall back
// to the defaults above in NewRefiner.
type RefinementConfig struct {
	// Gamma weights the probe rank prior: max(disagreement, gamma*rank).
	Gamma float64
	// MinGapSeconds is the minimum splittable interval width.
	MinGapSeconds float64
	// Epsilon is the minimum refinement priority.
	Epsilon float64
	// MaxRounds bounds the refinement loop.
	MaxRounds int
	// ActivityFloor zeroes interval probe activity below this ChangedFraction
	// before rank normalization (near-static guard). Zero disables it,
	// preserving exact Phase 4 behavior.
	ActivityFloor float64
	// Disagreement configures the Phase 3 throwaway scorer reused each round.
	Disagreement DisagreementConfig
}

// DefaultRefinementConfig returns the Phase 4 starting configuration.
func DefaultRefinementConfig() RefinementConfig {
	return RefinementConfig{
		Gamma:         DefaultGamma,
		MinGapSeconds: DefaultRefineMinGapSeconds,
		Epsilon:       DefaultRefineEpsilon,
		MaxRounds:     DefaultRefineMaxRounds,
		Disagreement:  DefaultDisagreementConfig(),
	}
}

func (c RefinementConfig) sanitized() RefinementConfig {
	if math.IsNaN(c.Gamma) || c.Gamma < 0 {
		c.Gamma = DefaultGamma
	}
	if math.IsNaN(c.MinGapSeconds) || c.MinGapSeconds < 0 {
		c.MinGapSeconds = DefaultRefineMinGapSeconds
	}
	if math.IsNaN(c.Epsilon) || c.Epsilon < 0 {
		c.Epsilon = DefaultRefineEpsilon
	}
	if c.MaxRounds <= 0 {
		c.MaxRounds = DefaultRefineMaxRounds
	}
	if math.IsNaN(c.ActivityFloor) || c.ActivityFloor < 0 {
		c.ActivityFloor = 0
	}
	c.Disagreement = c.Disagreement.sanitized()
	return c
}

// RefinementCandidate is one splittable interval with its full decision
// record, so any selection can be explained after the fact.
type RefinementCandidate struct {
	StartTimestamp float64
	EndTimestamp   float64
	Gap            float64

	Disagreement float64
	ProbeRank    float64
	Priority     float64

	Midpoint float64
}

// SortCandidates orders highest priority first with deterministic
// timestamp tie-breaking (earlier start, then earlier end). Never rely on
// map order or unstable sorts: sort.SliceStable over an explicitly ordered
// less function.
func SortCandidates(cands []RefinementCandidate) {
	sort.SliceStable(cands, func(i, j int) bool {
		if cands[i].Priority != cands[j].Priority {
			return cands[i].Priority > cands[j].Priority
		}
		if cands[i].StartTimestamp != cands[j].StartTimestamp {
			return cands[i].StartTimestamp < cands[j].StartTimestamp
		}
		return cands[i].EndTimestamp < cands[j].EndTimestamp
	})
}

// ProbeRankForIntervals rank-normalizes per-interval probe activity within
// one video: rank = (# intervals with strictly smaller value) / (n-1), so
// quietest -> 0, highest -> 1. All-equal values (including all-zero, the
// static-video case) rank 0 everywhere. A single interval ranks 0: with no
// peer there is no relative basis, so disagreement alone drives.
//
// Interval activity reuses the Phase 2 probe representation: the maximum
// ChangedFraction among probe points strictly inside the interval
// (start < t <= end). If an interval saw significant visual change, its
// prior can raise its refinement priority even when tracking is silent.
// ProbeRankForIntervals rank-normalizes with no activity floor (exact Phase
// 4 behavior). Production refinement passes ActivityFloor via
// ProbeRankForIntervalsWithFloor.
func ProbeRankForIntervals(probe *ProbeReport, starts, ends []float64) []float64 {
	return ProbeRankForIntervalsWithFloor(probe, starts, ends, 0)
}

// ProbeRankForIntervalsWithFloor is ProbeRankForIntervals with a near-static
// guard: per-interval peak activity below floor is treated as zero before
// ranking, so isolated single-block flicker cannot rank 1 and trigger
// refinement on effectively static footage.
func ProbeRankForIntervalsWithFloor(probe *ProbeReport, starts, ends []float64, floor float64) []float64 {
	n := len(starts)
	ranks := make([]float64, n)
	if n == 0 || probe == nil {
		return ranks
	}
	activity := make([]float64, n)
	for i := range starts {
		peak := 0.0
		for _, p := range probe.Points {
			if p.TimestampSeconds > starts[i] && p.TimestampSeconds <= ends[i] {
				if p.ChangedFraction > peak {
					peak = p.ChangedFraction
				}
			}
		}
		if peak < floor {
			peak = 0
		}
		activity[i] = peak
	}
	if n == 1 {
		return ranks
	}
	for i := range activity {
		less := 0
		for j := range activity {
			if activity[j] < activity[i] {
				less++
			}
		}
		ranks[i] = float64(less) / float64(n-1)
	}
	return ranks
}

// PriorityOf implements priority = max(disagreement, gamma*probe_rank)*gap.
func PriorityOf(disagreement, probeRank, gap, gamma float64) float64 {
	sig := disagreement
	if g := gamma * probeRank; g > sig {
		sig = g
	}
	return sig * gap
}

// MidpointOf returns the bisection timestamp for an interval.
func MidpointOf(start, end float64) float64 { return (start + end) / 2 }

// BuildCandidates scores every interval and returns the eligible ones:
// gap > MinGap AND priority > Epsilon, ordered by the deterministic
// priority queue (highest first). Budget truncation happens at selection,
// not here.
func BuildCandidates(scores []IntervalScore, ranks []float64, cfg RefinementConfig) []RefinementCandidate {
	var cands []RefinementCandidate
	for i, s := range scores {
		var rank float64
		if i < len(ranks) {
			rank = ranks[i]
		}
		gap := s.EndTimestamp - s.StartTimestamp
		priority := PriorityOf(s.Disagreement, rank, gap, cfg.Gamma)
		if gap <= cfg.MinGapSeconds || priority <= cfg.Epsilon {
			continue
		}
		cands = append(cands, RefinementCandidate{
			StartTimestamp: s.StartTimestamp,
			EndTimestamp:   s.EndTimestamp,
			Gap:            gap,
			Disagreement:   s.Disagreement,
			ProbeRank:      rank,
			Priority:       priority,
			Midpoint:       MidpointOf(s.StartTimestamp, s.EndTimestamp),
		})
	}
	SortCandidates(cands)
	return cands
}

// MidpointDetections carries YOLO output for one newly inserted timestamp,
// echoed in request order.
type MidpointDetections struct {
	Timestamp  float64
	Detections []tracker.DetectionInput
}

// DetectFunc extracts and detects ONLY the requested midpoint timestamps
// (never reprocessing existing frames) and returns one result per requested
// timestamp, matched by timestamp value (order-independent). Every returned
// timestamp consumes one unit of the remaining YOLO budget.
type DetectFunc func(ctx context.Context, timestamps []float64) ([]MidpointDetections, error)

// RefinementInput is the coarse state refinement starts from.
type RefinementInput struct {
	// Plan is the coarse SamplingPlan (reasons preserved into the output).
	Plan SamplingPlan
	// Probe is the Phase 2 report, reused across rounds (may be nil on
	// baseline_fallback: probe prior is then 0 everywhere).
	Probe *ProbeReport
	// Frames parallels Plan timestamps (any stable FrameIDs); Detections
	// holds every already-YOLO'd frame's detections keyed by those IDs.
	Frames     []tracker.FrameInput
	Detections map[uuid.UUID][]tracker.DetectionInput
}

// RoundInfo records one refinement round for inspection and tests.
type RoundInfo struct {
	Round    int
	Scores   []IntervalScore
	Selected []RefinementCandidate
	Inserted []float64
}

// RefinementResult is the final outcome of adaptive refinement.
type RefinementResult struct {
	// Plan is the final sorted, deduplicated plan; original coarse/heartbeat
	// entries keep their reason, new entries carry ReasonRefinement with
	// the selection priority as Score.
	Plan SamplingPlan
	// Rounds counts insertion rounds performed (0 when nothing qualified).
	Rounds int
	// FramesAdded counts new YOLO'd midpoints; FramesProcessed is the
	// final total (initial + added).
	FramesAdded     int
	FramesProcessed int
	// RemainingBudget is Cap - FramesProcessed (>= 0 by construction).
	RemainingBudget int
	// RefinedIntervals counts intervals ever split; MaxPriority is the
	// highest priority observed across all rounds.
	RefinedIntervals int
	MaxPriority      float64
	// History records every round; FinalScores is the last round's
	// interval scoring (nil when no round ran).
	History     []RoundInfo
	FinalScores []IntervalScore
}

// Refiner runs bounded best-first adaptive refinement.
type Refiner struct {
	cfg RefinementConfig
}

// NewRefiner sanitizes cfg and returns a refinement planner.
func NewRefiner(cfg RefinementConfig) *Refiner {
	return &Refiner{cfg: cfg.sanitized()}
}

// Config returns the sanitized configuration.
func (r *Refiner) Config() RefinementConfig { return r.cfg }

// midpointFrameID deterministically derives a frame ID for an inserted
// timestamp so replay ordering never depends on caller ID schemes.
func midpointFrameID(ts float64) uuid.UUID {
	return uuid.NewSHA1(uuid.NameSpaceOID, []byte("refine:"+strconv.FormatFloat(ts, 'f', 9, 64)))
}

// Refine grows the coarse timestamp set by best-first midpoint insertion.
// Budget invariant: len(initial) + ever-inserted <= Budget.Cap holds at
// every point; selection is truncated to the remaining budget BEFORE any
// extraction/detection, so no YOLO frame is ever spent beyond the cap.
func (r *Refiner) Refine(ctx context.Context, in RefinementInput, detect DetectFunc) (RefinementResult, error) {
	cfg := r.cfg
	plan := in.Plan
	duration := plan.DurationSeconds
	capN := plan.Budget.Cap

	// Mutable state: parallel timestamp/reason lists plus frame/detection
	// maps. Timestamps stay sorted and deduplicated (dedupeEpsilon).
	timestamps := NormalizeTimestamps(plan.Timestamps(), duration)
	reasons := make([]Reason, 0, len(timestamps))
	tsReason := make(map[float64]Reason, len(plan.Entries))
	for _, e := range plan.Entries {
		if _, ok := tsReason[e.TimestampSeconds]; !ok {
			tsReason[e.TimestampSeconds] = e.Reason
		}
	}
	for _, t := range timestamps {
		reason, ok := tsReason[t]
		if !ok {
			reason = ReasonCoarse
		}
		reasons = append(reasons, reason)
	}
	frames := make([]tracker.FrameInput, 0, len(in.Frames))
	byFrame := make(map[uuid.UUID][]tracker.DetectionInput, len(in.Detections))
	for _, f := range in.Frames {
		frames = append(frames, f)
	}
	for fid, dets := range in.Detections {
		cp := make([]tracker.DetectionInput, len(dets))
		copy(cp, dets)
		byFrame[fid] = cp
	}
	// Every timestamp must resolve to exactly one frame.
	aligned := make([]tracker.FrameInput, 0, len(timestamps))
	for _, t := range timestamps {
		var fid uuid.UUID
		found := false
		for _, f := range frames {
			if math.Abs(f.Timestamp-t) < dedupeEpsilon {
				fid = f.FrameID
				found = true
				break
			}
		}
		if !found {
			return RefinementResult{}, fmt.Errorf("refinement: no frame for timestamp %v", t)
		}
		aligned = append(aligned, tracker.FrameInput{FrameID: fid, Timestamp: t})
	}
	frames = aligned

	remaining := capN - len(timestamps)
	if capN > 0 && remaining < 0 {
		remaining = 0
	}
	if capN <= 0 {
		remaining = 1 << 30 // uncapped plan: still bound loop by rounds/eligibility
	}

	var (
		res          RefinementResult
		maxPriority  float64
		refinedTotal int
		initialCount = len(timestamps)
	)

	for round := 1; round <= cfg.MaxRounds; round++ {
		if err := ctx.Err(); err != nil {
			return RefinementResult{}, err
		}
		if remaining <= 0 {
			break
		}
		scores, err := ScoreIntervals(ctx, frames, byFrame, cfg.Disagreement)
		if err != nil {
			return RefinementResult{}, fmt.Errorf("refinement round %d scoring: %w", round, err)
		}
		if len(scores) == 0 {
			break
		}
		starts := make([]float64, len(scores))
		ends := make([]float64, len(scores))
		for i, s := range scores {
			starts[i], ends[i] = s.StartTimestamp, s.EndTimestamp
		}
		ranks := ProbeRankForIntervalsWithFloor(in.Probe, starts, ends, cfg.ActivityFloor)
		cands := BuildCandidates(scores, ranks, cfg)
		eligible := len(cands)
		for _, c := range cands {
			if c.Priority > maxPriority {
				maxPriority = c.Priority
			}
		}
		res.FinalScores = scores
		if len(cands) == 0 {
			break
		}
		// Best-first truncation to the remaining budget BEFORE spending.
		if len(cands) > remaining {
			cands = cands[:remaining]
		}
		// Deduplicate midpoints against existing timestamps (epsilon) and
		// range; drop anything already processed (no duplicate YOLO).
		var want []float64
		for _, c := range cands {
			mid := c.Midpoint
			if math.IsNaN(mid) || math.IsInf(mid, 0) || mid < 0 || mid >= duration-1e-9 {
				continue
			}
			dup := false
			for _, t := range timestamps {
				if math.Abs(t-mid) < dedupeEpsilon {
					dup = true
					break
				}
			}
			for _, w := range want {
				if math.Abs(w-mid) < dedupeEpsilon {
					dup = true
					break
				}
			}
			if !dup {
				want = append(want, mid)
			}
		}
		if len(want) == 0 {
			break
		}
		if detect == nil {
			return RefinementResult{}, fmt.Errorf("refinement: no detection callback for %d midpoints", len(want))
		}
		got, err := detect(ctx, want)
		if err != nil {
			return RefinementResult{}, fmt.Errorf("refinement round %d detect: %w", round, err)
		}
		if len(got) != len(want) {
			return RefinementResult{}, fmt.Errorf("refinement round %d: want %d detections, got %d", round, len(want), len(got))
		}
		// Merge in request order (deterministic regardless of callback
		// order): match each result to its requested timestamp by value.
		// Extractors may sort internally, so positional alignment is not
		// assumed.
		type merged struct {
			ts   float64
			dets []tracker.DetectionInput
		}
		used := make([]bool, len(got))
		ordered := make([]merged, 0, len(got))
		for _, w := range want {
			found := false
			for j := range got {
				if !used[j] && math.Abs(got[j].Timestamp-w) < dedupeEpsilon {
					used[j] = true
					ordered = append(ordered, merged{ts: w, dets: got[j].Detections})
					found = true
					break
				}
			}
			if !found {
				return RefinementResult{}, fmt.Errorf("refinement round %d: no detections for requested %v", round, w)
			}
		}
		for j := range got {
			if !used[j] {
				return RefinementResult{}, fmt.Errorf("refinement round %d: unexpected detections for %v", round, got[j].Timestamp)
			}
		}
		var inserted []float64
		for _, m := range ordered {
			fid := midpointFrameID(m.ts)
			frames = append(frames, tracker.FrameInput{FrameID: fid, Timestamp: m.ts})
			stamped := make([]tracker.DetectionInput, 0, len(m.dets))
			for _, d := range m.dets {
				d.FrameID = fid
				d.Timestamp = m.ts
				stamped = append(stamped, d)
			}
			byFrame[fid] = stamped
			timestamps = append(timestamps, m.ts)
			reasons = append(reasons, ReasonRefinement)
			inserted = append(inserted, m.ts)
		}
		// Re-sort timestamps+reasons together; frames are resorted by the
		// scorer replay each round, so only the parallel metadata needs it.
		idx := make([]int, len(timestamps))
		for i := range idx {
			idx[i] = i
		}
		sort.SliceStable(idx, func(a, b int) bool { return timestamps[idx[a]] < timestamps[idx[b]] })
		sortedTs := make([]float64, len(timestamps))
		sortedRe := make([]Reason, len(reasons))
		for i, j := range idx {
			sortedTs[i] = timestamps[j]
			sortedRe[i] = reasons[j]
		}
		timestamps, reasons = sortedTs, sortedRe

		remaining -= len(inserted)
		res.Rounds++
		refinedTotal += len(cands)
		res.History = append(res.History, RoundInfo{
			Round:    round,
			Scores:   scores,
			Selected: append([]RefinementCandidate(nil), cands...),
			Inserted: append([]float64(nil), inserted...),
		})
		slog.Info("sampler: refinement round",
			"video_id", plan.VideoID.String(),
			"round", round,
			"current_frame_count", len(timestamps),
			"remaining_budget", remaining,
			"candidate_interval_count", len(scores),
			"eligible_interval_count", eligible,
			"selected_interval_count", len(cands),
			"selected_midpoints", inserted,
			"max_priority", maxPriority,
		)
		for _, c := range cands {
			slog.Debug("sampler: refinement candidate",
				"video_id", plan.VideoID.String(),
				"round", round,
				"start", c.StartTimestamp, "end", c.EndTimestamp, "gap", c.Gap,
				"disagreement", c.Disagreement, "probe_rank", c.ProbeRank,
				"priority", c.Priority, "midpoint", c.Midpoint,
			)
		}
		if remaining <= 0 {
			break
		}
	}

	// Final plan: preserved reasons, refinement entries scored by priority.
	// Recompute per-timestamp priority for refinement entries from the last
	// selection when available (informational only).
	entries := make([]PlannedTimestamp, 0, len(timestamps))
	for i, t := range timestamps {
		e := PlannedTimestamp{TimestampSeconds: t, Reason: reasons[i]}
		if reasons[i] == ReasonRefinement {
			// Score carries the selection priority when known.
			for _, h := range res.History {
				for _, c := range h.Selected {
					if math.Abs(c.Midpoint-t) < dedupeEpsilon {
						p := c.Priority
						e.Score = &p
					}
				}
			}
			if e.Score == nil {
				z := 0.0
				e.Score = &z
			}
		} else if orig := findPlanScore(plan, t); orig != nil {
			e.Score = orig
		}
		entries = append(entries, e)
	}
	res.Plan = SamplingPlan{
		VideoID:         plan.VideoID,
		DurationSeconds: duration,
		Mode:            plan.Mode,
		Budget:          plan.Budget,
		Entries:         entries,
	}
	res.FramesAdded = len(timestamps) - initialCount
	res.FramesProcessed = len(timestamps)
	if capN > 0 {
		res.RemainingBudget = capN - len(timestamps)
		if res.RemainingBudget < 0 {
			res.RemainingBudget = 0
		}
	} else {
		res.RemainingBudget = 0
	}
	res.RefinedIntervals = refinedTotal
	res.MaxPriority = maxPriority
	return res, nil
}

// findPlanScore carries over the coarse Score for preserved entries.
func findPlanScore(plan SamplingPlan, ts float64) *float64 {
	for _, e := range plan.Entries {
		if math.Abs(e.TimestampSeconds-ts) < dedupeEpsilon {
			return e.Score
		}
	}
	return nil
}
