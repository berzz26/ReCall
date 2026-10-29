package tracker

import (
	"context"
	"fmt"

	"github.com/google/uuid"
)

type DetectionInput struct {
	ID         uuid.UUID
	Label      string
	Confidence float64
	BBoxX      float64
	BBoxY      float64
	BBoxWidth  float64
	BBoxHeight float64
	FrameID    uuid.UUID
	Timestamp  float64
}

type FrameInput struct {
	FrameID   uuid.UUID
	Timestamp float64
}

type TrackAssignment struct {
	DetectionID uuid.UUID
	TrackIndex  int
	TrackID     uuid.UUID // to be filled after persistence
}

type Tracker interface {
	Track(ctx context.Context, frames []FrameInput, detectionsByFrame map[uuid.UUID][]DetectionInput) (map[uuid.UUID]int, error)
	Name() string
	Version() string
}

// Simple result: map detectionID -> trackIndex

// SparseParams tunes the sparse-video association gating in ByteTrack
// (sampled frames can be seconds apart; see bytetrack.go for semantics).
// Zero values fall back to the built-in defaults, so passing the zero
// struct keeps the tuned production behavior.
type SparseParams struct {
	// DtThreshold: seconds between frames before sparse gating takes over
	// from the dense IoU/Mahalanobis gates (TRACKER_SPARSE_DT_THRESHOLD).
	DtThreshold float64
	// AllowFloor: normalized center-distance floor for the drift gate,
	// covering slow/stationary objects (TRACKER_SPARSE_ALLOW_FLOOR).
	AllowFloor float64
	// MaxSpeed: normalized frame units per second of plausible object
	// motion, headroom over typical walking/crossing speed
	// (TRACKER_SPARSE_MAX_SPEED).
	MaxSpeed float64
	// ReattachGap: maximum seconds since a dead track's last observation
	// that a later track may still be re-attached to it
	// (TRACKER_REATTACH_MAX_GAP).
	ReattachGap float64
}

func (p SparseParams) withDefaults() SparseParams {
	if p.DtThreshold <= 0 {
		p.DtThreshold = sparseDtThreshold
	}
	if p.AllowFloor < 0 {
		p.AllowFloor = sparseAllowFloor
	}
	if p.MaxSpeed <= 0 {
		p.MaxSpeed = sparseMaxSpeed
	}
	if p.ReattachGap <= 0 {
		p.ReattachGap = reattachMaxGap
	}
	return p
}

// New creates a tracker based on TRACKER_TYPE.
// trackerType must be "iou" or "bytetrack"; any other value returns an error.
// For "iou", the ByteTrack thresholds/buffer are ignored and defaults (0.2 / 5s) are used.
// For "bytetrack", the provided thresholds, buffer and sparse-gating params are used.
// fuseScore and minHits are ByteTrack-only.
func New(trackerType string, highThreshold, lowThreshold, matchThreshold float64, trackBuffer int, fuseScore bool, minHits int, sparse SparseParams) (Tracker, error) {
	switch trackerType {
	case "iou":
		return NewIoUTracker(), nil
	case "bytetrack":
		return NewByteTrack(highThreshold, lowThreshold, matchThreshold, trackBuffer, fuseScore, minHits, sparse), nil
	default:
		return nil, fmt.Errorf("invalid TRACKER_TYPE %q: must be one of [iou, bytetrack]", trackerType)
	}
}

// NewWithDefaults keeps backward compatibility for callers that used the old 4-param factory.
func NewWith4Params(trackerType string, highThreshold, lowThreshold, matchThreshold float64, trackBuffer int) (Tracker, error) {
	return New(trackerType, highThreshold, lowThreshold, matchThreshold, trackBuffer, true, 2, SparseParams{})
}
