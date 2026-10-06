package tracker

// Throwaway ByteTrack replay for the adaptive sampler's temporal
// disagreement scorer.
//
// This path is fully isolated from production tracking: it operates on a
// fresh, function-local track set on every invocation and never touches any
// production tracker instance or persisted state. Production Track() is
// unchanged.
//
// Reused verbatim from the production ByteTrack implementation:
// Kalman initiate/predict/update, Mahalanobis gating, IoU cost with optional
// score fusion, label-aware matching, Hungarian assignment, and the two
// association stages (high-confidence first stage, low-confidence second
// stage). A second-stage association is the tracker's own "weak association"
// distinction and is reported as such.
//
// Deliberate throwaway differences from production:
//   - Event emission: every association records its stage; track births
//     (first observation) and the last observation of each track are
//     reported so the sampler can attribute births/deaths per interval.
//     No MinHits confirmation gating is applied to events: a track that
//     appears at frame B is a birth for interval A->B even if it is never
//     seen again.
//   - Time-aware lost handling: the production TrackBuffer counts tracker
//     updates, but 3 updates can mean 1.5s or 12s under adaptive sampling.
//     Here a lost track remains matchable for LostSeconds (wall-clock,
//     measured from its last match) and is then removed.
//   - Prediction uses the actual timestamp delta (dt clamp is a wide
//     numerical safety bound only, not a behavioral one). Coarse-plan gaps
//     are already bounded by the heartbeat, so dt is naturally small.

import (
	"context"
	"math"
	"sort"

	"github.com/google/uuid"
)

// AssociationStage records which ByteTrack association stage produced a
// match. StageSecond is the tracker's native weak-association signal: the
// detection only matched through the low-confidence second-stage path.
type AssociationStage int

const (
	StageFirst AssociationStage = iota + 1
	StageSecond
)

// ThrowawayConfig configures the isolated replay. Threshold defaults mirror
// the production ByteTrack configuration (TRACKER_HIGH_THRESHOLD 0.50,
// TRACKER_LOW_THRESHOLD 0.10, TRACKER_MATCH_THRESHOLD 0.30, fuse on) so the
// throwaway judgments stay consistent with production tracking behavior.
// LostSeconds replaces the count-based TrackBuffer: production buffer 5 at
// the 2s baseline grid ≈ 10s of wall-clock tolerance.
type ThrowawayConfig struct {
	HighThreshold  float64
	LowThreshold   float64
	MatchThreshold float64
	FuseScore      bool
	// LostSeconds is how long a lost track stays matchable after its last
	// match, in seconds.
	LostSeconds float64
	// MaxDtSeconds is a numerical safety clamp on prediction delta only.
	MaxDtSeconds float64
}

// DefaultThrowawayConfig mirrors production thresholds with a 10s
// lost-track allowance (≈ TrackBuffer 5 × 2s baseline spacing).
func DefaultThrowawayConfig() ThrowawayConfig {
	return ThrowawayConfig{
		HighThreshold:  0.50,
		LowThreshold:   0.10,
		MatchThreshold: 0.30,
		FuseScore:      true,
		LostSeconds:    10.0,
		MaxDtSeconds:   60.0,
	}
}

// ReplayAssociation links one detection to a throwaway track.
type ReplayAssociation struct {
	DetectionID uuid.UUID
	TrackIndex  int
	Stage       AssociationStage
	// NewTrack is true when this detection created the track (birth).
	NewTrack bool
}

// ReplayFrame is the per-frame outcome in chronological order.
type ReplayFrame struct {
	FrameID      uuid.UUID
	Timestamp    float64
	Associations []ReplayAssociation
}

// TrackLife summarizes one throwaway track's whole observed lifespan.
type TrackLife struct {
	Index     int
	Label     string
	FirstSeen float64
	LastSeen  float64
	FirstBBox DetectionInput
	LastBBox  DetectionInput
}

// ReplayResult is the complete throwaway replay: per-frame associations plus
// per-track lifespans. Callers attribute interval events from these.
type ReplayResult struct {
	Frames []ReplayFrame
	Tracks []TrackLife
}

// ReplayThrowaway replays detections through a fresh tracker from scratch.
// Frames are sorted chronologically (timestamp, then FrameID); detections
// within a frame are ordered by confidence desc, UUID asc — the same
// deterministic conventions as production Track().
func ReplayThrowaway(ctx context.Context, frames []FrameInput, detectionsByFrame map[uuid.UUID][]DetectionInput, cfg ThrowawayConfig) (ReplayResult, error) {
	if cfg.HighThreshold <= 0 {
		cfg.HighThreshold = 0.50
	}
	if cfg.LowThreshold < 0 {
		cfg.LowThreshold = 0.10
	}
	if cfg.MatchThreshold <= 0 {
		cfg.MatchThreshold = 0.30
	}
	if cfg.LostSeconds <= 0 {
		cfg.LostSeconds = 10.0
	}
	if cfg.MaxDtSeconds <= 0 {
		cfg.MaxDtSeconds = 60.0
	}

	ordered := make([]FrameInput, len(frames))
	copy(ordered, frames)
	sort.Slice(ordered, func(i, j int) bool {
		if ordered[i].Timestamp == ordered[j].Timestamp {
			return ordered[i].FrameID.String() < ordered[j].FrameID.String()
		}
		return ordered[i].Timestamp < ordered[j].Timestamp
	})

	active := make(map[int]*btTrack)
	nextIndex := 0
	var result ReplayResult

	for _, fr := range ordered {
		select {
		case <-ctx.Done():
			return ReplayResult{}, ctx.Err()
		default:
		}

		// Drop tracks lost longer than the wall-clock allowance (seconds,
		// not update counts).
		for idx, tr := range active {
			if tr.State == StateLost || tr.State == StateNew {
				if fr.Timestamp-tr.LastMatchedTime > cfg.LostSeconds {
					tr.State = StateRemoved
					delete(active, idx)
				}
			}
		}

		// Predict surviving tracks with the actual timestamp delta.
		for _, tr := range active {
			if tr.State != StateTracked && tr.State != StateLost && tr.State != StateNew {
				continue
			}
			dt := fr.Timestamp - tr.LastMatchedTime
			if dt < 0 {
				dt = 0
			}
			if dt > cfg.MaxDtSeconds {
				dt = cfg.MaxDtSeconds
			}
			if dt > 1e-9 {
				tr.Mean, tr.Cov = kalmanPredict(tr.Mean, tr.Cov, dt)
			}
			tr.PredictedBBox = xyahToTLWH(tr.Mean, tr.Label)
		}

		dets := detectionsByFrame[fr.FrameID]
		var highDets, lowDets []DetectionInput
		for _, d := range dets {
			if d.Confidence >= cfg.HighThreshold {
				highDets = append(highDets, d)
			} else if d.Confidence >= cfg.LowThreshold {
				lowDets = append(lowDets, d)
			}
		}
		sort.Slice(highDets, func(i, j int) bool {
			if highDets[i].Confidence == highDets[j].Confidence {
				return highDets[i].ID.String() < highDets[j].ID.String()
			}
			return highDets[i].Confidence > highDets[j].Confidence
		})
		sort.Slice(lowDets, func(i, j int) bool {
			if lowDets[i].Confidence == lowDets[j].Confidence {
				return lowDets[i].ID.String() < lowDets[j].ID.String()
			}
			return lowDets[i].Confidence > lowDets[j].Confidence
		})

		var candidates []*btTrack
		for _, tr := range active {
			if tr.State == StateTracked || tr.State == StateLost || tr.State == StateNew {
				candidates = append(candidates, tr)
			}
		}
		sort.Slice(candidates, func(i, j int) bool { return candidates[i].Index < candidates[j].Index })

		var rf ReplayFrame
		rf.FrameID = fr.FrameID
		rf.Timestamp = fr.Timestamp
		matchedTrack := make(map[int]bool)
		matchedHigh := make(map[int]bool)

		// First stage: all live tracks vs high-confidence detections.
		if len(candidates) > 0 && len(highDets) > 0 {
			cost := buildCostMatrixByteTrack(candidates, highDets, cfg.MatchThreshold, cfg.FuseScore)
			assign := hungarianRect(cost, len(candidates), len(highDets))
			for i, tr := range candidates {
				j := assign[i]
				if j < 0 || j >= len(highDets) || cost[i][j] >= INF/2 {
					continue
				}
				applyThrowawayMatch(tr, highDets[j], fr.Timestamp)
				rf.Associations = append(rf.Associations, ReplayAssociation{
					DetectionID: highDets[j].ID, TrackIndex: tr.Index, Stage: StageFirst,
				})
				matchedTrack[tr.Index] = true
				matchedHigh[j] = true
			}
		}

		// Second stage: still-unmatched Tracked tracks vs low detections.
		// This is the weak-association path.
		var remaining []*btTrack
		for _, tr := range candidates {
			if !matchedTrack[tr.Index] && tr.State == StateTracked {
				remaining = append(remaining, tr)
			}
		}
		sort.Slice(remaining, func(i, j int) bool { return remaining[i].Index < remaining[j].Index })
		if len(remaining) > 0 && len(lowDets) > 0 {
			cost := buildCostMatrixByteTrack(remaining, lowDets, cfg.MatchThreshold, cfg.FuseScore)
			assign := hungarianRect(cost, len(remaining), len(lowDets))
			for i, tr := range remaining {
				j := assign[i]
				if j < 0 || j >= len(lowDets) || cost[i][j] >= INF/2 {
					continue
				}
				applyThrowawayMatch(tr, lowDets[j], fr.Timestamp)
				rf.Associations = append(rf.Associations, ReplayAssociation{
					DetectionID: lowDets[j].ID, TrackIndex: tr.Index, Stage: StageSecond,
				})
				matchedTrack[tr.Index] = true
			}
		}

		// Unmatched tracks go lost.
		for _, tr := range candidates {
			if matchedTrack[tr.Index] {
				continue
			}
			if tr.State == StateTracked {
				tr.State = StateLost
			}
		}

		// Births: unmatched high-confidence detections start new tracks.
		for idx, d := range highDets {
			if matchedHigh[idx] {
				continue
			}
			z := tlwhToXyah(d)
			mean, cov := kalmanInitiate(z)
			newIdx := nextIndex
			nextIndex++
			tr := &btTrack{
				Index:           newIdx,
				Label:           d.Label,
				State:           StateNew,
				Mean:            mean,
				Cov:             cov,
				LastBBox:        d,
				PredictedBBox:   d,
				Hits:            1,
				HitCount:        1,
				StartTimestamp:  fr.Timestamp,
				LastTimestamp:   fr.Timestamp,
				LastMatchedTime: fr.Timestamp,
			}
			active[newIdx] = tr
			rf.Associations = append(rf.Associations, ReplayAssociation{
				DetectionID: d.ID, TrackIndex: newIdx, Stage: StageFirst, NewTrack: true,
			})
		}

		sort.Slice(rf.Associations, func(i, j int) bool {
			return rf.Associations[i].DetectionID.String() < rf.Associations[j].DetectionID.String()
		})
		result.Frames = append(result.Frames, rf)
	}

	// Lifespans from association history (deterministic track-index order).
	lives := make(map[int]*TrackLife)
	for _, rf := range result.Frames {
		for _, a := range rf.Associations {
			l, ok := lives[a.TrackIndex]
			if !ok {
				l = &TrackLife{Index: a.TrackIndex, FirstSeen: rf.Timestamp}
				lives[a.TrackIndex] = l
			}
			l.LastSeen = rf.Timestamp
		}
	}
	// Bboxes and labels from the underlying detections.
	byDet := make(map[uuid.UUID]DetectionInput)
	for _, dets := range detectionsByFrame {
		for _, d := range dets {
			byDet[d.ID] = d
		}
	}
	orderedIdx := make([]int, 0, len(lives))
	for idx := range lives {
		orderedIdx = append(orderedIdx, idx)
	}
	sort.Ints(orderedIdx)
	for _, idx := range orderedIdx {
		l := lives[idx]
		for _, rf := range result.Frames {
			for _, a := range rf.Associations {
				if a.TrackIndex != idx {
					continue
				}
				d := byDet[a.DetectionID]
				if rf.Timestamp == l.FirstSeen && l.FirstBBox.ID == uuid.Nil {
					l.Label = d.Label
					l.FirstBBox = d
				}
				if rf.Timestamp == l.LastSeen {
					l.LastBBox = d
				}
			}
		}
		result.Tracks = append(result.Tracks, *l)
	}
	return result, nil
}

// applyThrowawayMatch folds an associated detection into a throwaway track.
func applyThrowawayMatch(tr *btTrack, d DetectionInput, timestamp float64) {
	z := tlwhToXyah(d)
	tr.Mean, tr.Cov = kalmanUpdate(tr.Mean, tr.Cov, z)
	tr.LastBBox = d
	tr.PredictedBBox = d
	tr.State = StateTracked
	tr.Hits++
	tr.HitCount++
	tr.Misses = 0
	tr.LastMatchedTime = timestamp
	tr.LastTimestamp = timestamp
}

// centerDistance returns the normalized center distance between two boxes.
func centerDistance(a, b DetectionInput) float64 {
	ax := a.BBoxX + a.BBoxWidth/2
	ay := a.BBoxY + a.BBoxHeight/2
	bx := b.BBoxX + b.BBoxWidth/2
	by := b.BBoxY + b.BBoxHeight/2
	dx := ax - bx
	dy := ay - by
	return math.Sqrt(dx*dx + dy*dy)
}
