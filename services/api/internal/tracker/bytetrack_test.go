package tracker

import (
	"context"
	"testing"

	"github.com/google/uuid"
)

// detBox builds a deformable detection with label/confidence/position.
func detBox(label string, conf, x float64) DetectionInput {
	return DetectionInput{ID: uuid.New(), Label: label, Confidence: conf, BBoxX: x, BBoxY: 0.3, BBoxWidth: 0.2, BBoxHeight: 0.5}
}

// trackOver runs the production ByteTrack over the given sampled frames,
// returning detectionID -> trackIndex assignments.
func trackOver(t *testing.T, timestamps []float64, detsPerFrame [][]DetectionInput) map[uuid.UUID]int {
	t.Helper()
	trk, err := New("bytetrack", 0.5, 0.1, 0.2, 1, true, 2, SparseParams{})
	if err != nil {
		t.Fatalf("tracker.New: %v", err)
	}
	var frames []FrameInput
	byFrame := make(map[uuid.UUID][]DetectionInput)
	for i, ts := range timestamps {
		fid := uuid.New()
		frames = append(frames, FrameInput{FrameID: fid, Timestamp: ts})
		byFrame[fid] = detsPerFrame[i]
	}
	assignments, err := trk.Track(context.Background(), frames, byFrame)
	if err != nil {
		t.Fatalf("Track: %v", err)
	}
	return assignments
}

// A person moving at realistic speed across multi-second sampled gaps stays on
// one track: the sparse gate must keep the identity (plain IoU would fail).
func TestSparseGateKeepsIdentity(t *testing.T) {
	// center moves 0.24 over 5s (~ walking speed): IoU = 0 over the gap.
	a := detBox("person", 0.9, 0.10)
	b := detBox("person", 0.9, 0.34)
	assignments := trackOver(t, []float64{0, 5}, [][]DetectionInput{{a}, {b}})
	ai, bi := assignments[a.ID], assignments[b.ID]
	if ai != bi {
		t.Fatalf("want same track across 5s gap at walking speed, got %d vs %d", ai, bi)
	}
}

// A track sampled one frame, then missed by YOLO for two frames, dies and a new
// detection later must be re-attached to the dead track within the gap limit.
func TestReattachMergesDiedTrack(t *testing.T) {
	// Timeline (TrackBuffer=1): hits at 0s and 3s, YOLO misses at 9s and 21s
	// (track dies), object re-detected at 32s ~ same region.
	a0 := detBox("person", 0.9, 0.10)
	a3 := detBox("person", 0.9, 0.12)
	b32 := detBox("person", 0.9, 0.14)
	assignments := trackOver(t,
		[]float64{0, 3, 9, 21, 32},
		[][]DetectionInput{{a0}, {a3}, {}, {}, {b32}},
	)
	ia0, ib32 := assignments[a0.ID], assignments[b32.ID]
	if ia0 != ib32 {
		t.Fatalf("want re-detected object at 32s re-attached to dead track, got %d vs %d", ia0, ib32)
	}
	if assignments[a3.ID] != ia0 {
		t.Fatalf("want mid-track observation 3s on the same track: %d vs %d", assignments[a3.ID], ia0)
	}
}

// A YOLO category flip between frames must not break identity (soft label gate):
// car and truck share the vehicle confusion group.
func TestSoftLabelKeepsIdentity(t *testing.T) {
	a := detBox("car", 0.9, 0.10)
	b := detBox("truck", 0.9, 0.12)
	assignments := trackOver(t, []float64{0, 1}, [][]DetectionInput{{a}, {b}})
	ai, bi := assignments[a.ID], assignments[b.ID]
	if ai != bi {
		t.Fatalf("want car->truck label flip to stay on one track, got %d vs %d", ai, bi)
	}
}
