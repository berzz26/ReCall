package video_frame

import (
	"time"

	"github.com/google/uuid"
)

// Purpose is the frame-set designation of a stored frame row. It
// distinguishes which sampling policy selected the frame so perception and
// semantic frame sets can coexist for one video without overwriting each
// other (see migration 000017):
//
//	perception: selected by the regular fixed-rate perception sampler
//	            (PERCEPTION_SAMPLE_FPS); consumed by production YOLO,
//	            ByteTrack, events, and the frame UI.
//	semantic:   selected by the adaptive sampler; consumed by VLM
//	            description generation.
//	both:       member of both sets. Timestamps shared between the two
//	            policies reuse a single physical row (and storage asset);
//	            sharing never overwrites the other set's membership.
type Purpose string

const (
	// PurposeBoth is the DB column default and the backfill value for
	// frames persisted before frame-set separation (they served both
	// consumers). Inserts of a shared timestamp also receive it.
	PurposeBoth Purpose = "both"
	// PurposePerception marks perception-set-only rows.
	PurposePerception Purpose = "perception"
	// PurposeSemantic marks semantic-set-only rows.
	PurposeSemantic Purpose = "semantic"
)

// resolve normalizes the zero/insert convenience value.
func (p Purpose) resolve() Purpose {
	if p == "" {
		return PurposeBoth
	}
	return p
}

// Owns reports whether a row with purpose p belongs to the queried set
// (membership identity is preserved even when a semantic timestamp
// coincides with a perception timestamp, because 'both' is its own value).
// The zero value "" means "all sets" (the legacy pre-separation rows).
func (p Purpose) Owns(set Set) bool {
	if p == "" {
		return true
	}
	switch set {
	case SetPerception:
		return p == PurposePerception || p == PurposeBoth
	case SetSemantic:
		return p == PurposeSemantic || p == PurposeBoth
	default: // empty set: every row
		return true
	}
}

// Set is the query-side name for a frame-set membership: the perception
// consumer set or the semantic consumer set.
type Set string

const (
	SetPerception Set = "perception"
	SetSemantic   Set = "semantic"
)

// purposesForSet returns the row-space purposes that are members of the
// given set; it is the canonical mapping behind repository membership
// queries.
func purposesForSet(set Set) []Purpose {
	switch set {
	case SetPerception:
		return []Purpose{PurposePerception, PurposeBoth}
	case SetSemantic:
		return []Purpose{PurposeSemantic, PurposeBoth}
	default: // empty: every purpose
		return []Purpose{PurposePerception, PurposeSemantic, PurposeBoth}
	}
}

type VideoFrame struct {
	ID               uuid.UUID `json:"id"`
	VideoID          uuid.UUID `json:"video_id"`
	SegmentID        uuid.UUID `json:"segment_id"`
	FrameIndex       int       `json:"frame_index"`
	TimestampSeconds float64   `json:"timestamp_seconds"`
	StorageKey       string    `json:"storage_key"`
	Width            int       `json:"width"`
	Height           int       `json:"height"`
	Purpose          Purpose   `json:"purpose"`
	CreatedAt        time.Time `json:"created_at"`
	UpdatedAt        time.Time `json:"updated_at"`
}
