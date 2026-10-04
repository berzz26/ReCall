package sampler

// Temporal disagreement scorer (Phase 3).
//
// Given the Phase 2 coarse timestamps plus the YOLO detections for those
// frames, the scorer replays detections through an isolated throwaway
// ByteTrack path and produces one disagreement score per adjacent interval:
//
//	"Tracking behavior between these two sampled frames suggests that the
//	current temporal spacing may be too coarse."
//
// The score is a tracking-continuity signal, NOT a semantic event signal.
// It never claims an event happened or that footage is interesting.
//
// Formula (weights configurable, starting values below):
//
//	raw = W_BIRTH*births + W_DEATH*deaths + W_WEAK*weak + W_FRAGMENT*fragments
//	d   = raw / max(detections_A, detections_B, 1)
//
// Births and deaths carry full weight: identity discontinuity is the
// primary under-sampling symptom. Weak (second-stage) matches and
// fragments carry partial weight: the tracker bridged the gap, but only
// just. Raw IoU is deliberately NOT the score: at 3-4s spacing the same
// person routinely moves far enough that box IoU collapses, while track
// identity may still continue.
//
// The scorer never decodes video, never invokes FFmpeg or YOLO, and never
// touches production tracker state. It replays from scratch on every call,
// so later refinement rounds can simply resubmit the full updated set.

import (
	"context"
	"fmt"
	"math"
	"sort"

	"github.com/google/uuid"

	"github.com/berzz26/recall/services/api/internal/tracker"
)

// IntervalScore is the disagreement record for one adjacent timestamp pair.
// Raw event counts are kept alongside the score for debugging.
type IntervalScore struct {
	StartTimestamp float64
	EndTimestamp   float64
	GapSeconds     float64

	DetectionsStart int
	DetectionsEnd   int

	Births      int
	Deaths      int
	WeakMatches int
	Fragments   int

	Disagreement float64
}

// DisagreementWeights scales the event counts in the raw score. Starting
// values are judgment calls, not tuned constants: births/deaths (identity
// breaks) matter more than weak/fragment continuations.
type DisagreementWeights struct {
	Birth    float64
	Death    float64
	Weak     float64
	Fragment float64
}

// DisagreementConfig configures the scorer. Threshold defaults mirror the
// production ByteTrack configuration so throwaway judgments stay consistent
// with production tracking.
type DisagreementConfig struct {
	HighThreshold  float64
	LowThreshold   float64
	MatchThreshold float64
	FuseScore      bool
	// LostSeconds is the wall-clock lost-track allowance (cf. production
	// count-based TrackBuffer).
	LostSeconds float64
	// FragMaxGapSeconds bounds death->birth linkage in time.
	FragMaxGapSeconds float64
	// FragMaxDistance bounds death->birth linkage in normalized center
	// distance. Conservative: only near-continuations count.
	FragMaxDistance float64
	Weights         DisagreementWeights
}

// DefaultDisagreementConfig returns the Phase 3 starting configuration.
func DefaultDisagreementConfig() DisagreementConfig {
	return DisagreementConfig{
		HighThreshold:     0.50,
		LowThreshold:      0.10,
		MatchThreshold:    0.30,
		FuseScore:         true,
		LostSeconds:       10.0,
		FragMaxGapSeconds: 8.0,
		FragMaxDistance:   0.15,
		Weights:           DisagreementWeights{Birth: 1.0, Death: 1.0, Weak: 0.5, Fragment: 0.75},
	}
}

func (c DisagreementConfig) sanitized() DisagreementConfig {
	if c.HighThreshold <= 0 {
		c.HighThreshold = 0.50
	}
	if c.LowThreshold < 0 {
		c.LowThreshold = 0.10
	}
	if c.MatchThreshold <= 0 {
		c.MatchThreshold = 0.30
	}
	if c.LostSeconds <= 0 {
		c.LostSeconds = 10.0
	}
	if c.FragMaxGapSeconds <= 0 {
		c.FragMaxGapSeconds = 8.0
	}
	if c.FragMaxDistance <= 0 {
		c.FragMaxDistance = 0.15
	}
	if c.Weights.Birth == 0 && c.Weights.Death == 0 && c.Weights.Weak == 0 && c.Weights.Fragment == 0 {
		c.Weights = DisagreementWeights{Birth: 1.0, Death: 1.0, Weak: 0.5, Fragment: 0.75}
	}
	return c
}

const tsEpsilon = 1e-9

func sameTs(a, b float64) bool { return math.Abs(a-b) < tsEpsilon }

// ScoreIntervals replays detections through a fresh throwaway tracker and
// scores every adjacent timestamp interval. Timestamps need not be uniformly
// spaced; the tracker predicts with the actual delta. Frames with no
// detections on both ends score 0 (the Phase 2 probe covers those regions).
//
// The input shapes mirror tracker.Track: frames carry FrameID+Timestamp and
// detectionsByFrame is keyed by FrameID. Detections below the low-confidence
// cutoff are excluded before analysis to avoid confidence-flicker births.
func ScoreIntervals(ctx context.Context, frames []tracker.FrameInput, detectionsByFrame map[uuid.UUID][]tracker.DetectionInput, cfg DisagreementConfig) ([]IntervalScore, error) {
	cfg = cfg.sanitized()
	if len(frames) < 2 {
		return []IntervalScore{}, nil
	}

	// Conservative confidence cutoff, applied before any lifecycle logic.
	filtered := make(map[uuid.UUID][]tracker.DetectionInput, len(detectionsByFrame))
	for fid, dets := range detectionsByFrame {
		kept := make([]tracker.DetectionInput, 0, len(dets))
		for _, d := range dets {
			if d.Confidence >= cfg.LowThreshold {
				kept = append(kept, d)
			}
		}
		if len(kept) > 0 {
			filtered[fid] = kept
		}
	}

	replay, err := tracker.ReplayThrowaway(ctx, frames, filtered, tracker.ThrowawayConfig{
		HighThreshold:  cfg.HighThreshold,
		LowThreshold:   cfg.LowThreshold,
		MatchThreshold: cfg.MatchThreshold,
		FuseScore:      cfg.FuseScore,
		LostSeconds:    cfg.LostSeconds,
	})
	if err != nil {
		return nil, fmt.Errorf("throwaway replay: %w", err)
	}
	if len(replay.Frames) < 2 {
		return []IntervalScore{}, nil
	}

	counts := make([]int, len(replay.Frames))
	for i, rf := range replay.Frames {
		counts[i] = len(filtered[rf.FrameID])
	}

	weakAt := make([]int, len(replay.Frames))
	for i, rf := range replay.Frames {
		for _, a := range rf.Associations {
			if a.Stage == tracker.StageSecond {
				weakAt[i]++
			}
		}
	}

	scores := make([]IntervalScore, 0, len(replay.Frames)-1)
	for i := 0; i+1 < len(replay.Frames); i++ {
		tA, tB := replay.Frames[i].Timestamp, replay.Frames[i+1].Timestamp
		nA, nB := counts[i], counts[i+1]

		var births, deaths []tracker.TrackLife
		for _, life := range replay.Tracks {
			if sameTs(life.FirstSeen, tB) {
				births = append(births, life)
			}
			if sameTs(life.LastSeen, tA) {
				deaths = append(deaths, life)
			}
		}
		sort.Slice(births, func(a, b int) bool { return births[a].Index < births[b].Index })
		sort.Slice(deaths, func(a, b int) bool { return deaths[a].Index < deaths[b].Index })

		fragments := countFragments(deaths, births, tB-tA, cfg.FragMaxGapSeconds, cfg.FragMaxDistance)

		var d float64
		if nA == 0 && nB == 0 {
			d = 0
		} else {
			denom := nA
			if nB > denom {
				denom = nB
			}
			if denom < 1 {
				denom = 1
			}
			raw := cfg.Weights.Birth*float64(len(births)) +
				cfg.Weights.Death*float64(len(deaths)) +
				cfg.Weights.Weak*float64(weakAt[i+1]) +
				cfg.Weights.Fragment*float64(fragments)
			d = raw / float64(denom)
		}

		scores = append(scores, IntervalScore{
			StartTimestamp:  tA,
			EndTimestamp:    tB,
			GapSeconds:      tB - tA,
			DetectionsStart: nA,
			DetectionsEnd:   nB,
			Births:          len(births),
			Deaths:          len(deaths),
			WeakMatches:     weakAt[i+1],
			Fragments:       fragments,
			Disagreement:    d,
		})
	}
	return scores, nil
}

// countFragments links deaths to births within one interval: same label, a
// short time gap, and nearby positions plausibly continue the same object.
// Pairing is greedy one-to-one in track-index order (deterministic) and
// conservative by design — it is a continuity hint, not re-identification.
func countFragments(deaths, births []tracker.TrackLife, gap, maxGap, maxDist float64) int {
	if gap > maxGap || len(deaths) == 0 || len(births) == 0 {
		return 0
	}
	used := make([]bool, len(births))
	frags := 0
	for _, d := range deaths {
		for j, b := range births {
			if used[j] || d.Label != b.Label {
				continue
			}
			if centerDist(d.LastBBox, b.FirstBBox) <= maxDist {
				used[j] = true
				frags++
				break
			}
		}
	}
	return frags
}

// centerDist is the normalized center distance between two boxes.
func centerDist(a, b tracker.DetectionInput) float64 {
	ax := a.BBoxX + a.BBoxWidth/2
	ay := a.BBoxY + a.BBoxHeight/2
	bx := b.BBoxX + b.BBoxWidth/2
	by := b.BBoxY + b.BBoxHeight/2
	dx := ax - bx
	dy := ay - by
	return math.Sqrt(dx*dx + dy*dy)
}
