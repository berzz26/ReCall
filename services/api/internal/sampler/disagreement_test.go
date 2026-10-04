package sampler

// Unit tests for the temporal disagreement scorer (Phase 3, Tests 1-10
// except the Kalman unit check, which lives in tracker/throwaway_test.go).

import (
	"context"
	"reflect"
	"testing"

	"github.com/google/uuid"

	"github.com/berzz26/recall/services/api/internal/tracker"
)

func det(frameID uuid.UUID, ts float64, label string, conf, x, y, w, h float64) tracker.DetectionInput {
	return tracker.DetectionInput{
		ID: uuid.New(), Label: label, Confidence: conf,
		BBoxX: x, BBoxY: y, BBoxWidth: w, BBoxHeight: h,
		FrameID: frameID, Timestamp: ts,
	}
}

// scoreFixture builds scorer input from per-frame detection lists.
func scoreFixture(timestamps []float64, perFrame [][]tracker.DetectionInput) ([]tracker.FrameInput, map[uuid.UUID][]tracker.DetectionInput) {
	frames := make([]tracker.FrameInput, 0, len(timestamps))
	byFrame := make(map[uuid.UUID][]tracker.DetectionInput)
	for i, ts := range timestamps {
		fid := uuid.New()
		frames = append(frames, tracker.FrameInput{FrameID: fid, Timestamp: ts})
		var dets []tracker.DetectionInput
		if i < len(perFrame) {
			for _, d := range perFrame[i] {
				d.FrameID = fid
				d.Timestamp = ts
				dets = append(dets, d)
			}
		}
		if len(dets) > 0 {
			byFrame[fid] = dets
		}
	}
	return frames, byFrame
}

func box(label string, conf, x float64) tracker.DetectionInput {
	return tracker.DetectionInput{ID: uuid.New(), Label: label, Confidence: conf,
		BBoxX: x, BBoxY: 0.3, BBoxWidth: 0.2, BBoxHeight: 0.2}
}

// Test 1 — zero detections on both ends scores 0.
func TestNoDetectionsZeroDisagreement(t *testing.T) {
	frames, byFrame := scoreFixture([]float64{0, 4}, [][]tracker.DetectionInput{{}, {}})
	scores, err := ScoreIntervals(context.Background(), frames, byFrame, DefaultDisagreementConfig())
	if err != nil {
		t.Fatalf("score: %v", err)
	}
	if len(scores) != 1 {
		t.Fatalf("want 1 interval, got %d", len(scores))
	}
	s := scores[0]
	if s.Disagreement != 0 || s.Births != 0 || s.Deaths != 0 || s.WeakMatches != 0 || s.Fragments != 0 {
		t.Fatalf("empty interval must score 0: %+v", s)
	}
	if s.GapSeconds != 4 || s.DetectionsStart != 0 || s.DetectionsEnd != 0 {
		t.Fatalf("interval metadata wrong: %+v", s)
	}
}

// Test 2 — a stable tracked object yields (near-)zero disagreement.
func TestStableObjectLowDisagreement(t *testing.T) {
	frames, byFrame := scoreFixture([]float64{0, 4}, [][]tracker.DetectionInput{
		{box("person", 0.9, 0.3)},
		{box("person", 0.9, 0.3)},
	})
	scores, err := ScoreIntervals(context.Background(), frames, byFrame, DefaultDisagreementConfig())
	if err != nil {
		t.Fatalf("score: %v", err)
	}
	s := scores[0]
	if s.Births != 0 || s.Deaths != 0 || s.WeakMatches != 0 {
		t.Fatalf("stable object must not birth/die: %+v", s)
	}
	if s.Disagreement != 0 {
		t.Fatalf("stable object disagreement want 0, got %v", s.Disagreement)
	}
}

// Test 3 — object appears at B: birth > 0.
func TestObjectBirth(t *testing.T) {
	frames, byFrame := scoreFixture([]float64{0, 4}, [][]tracker.DetectionInput{
		{},
		{box("person", 0.9, 0.3)},
	})
	scores, err := ScoreIntervals(context.Background(), frames, byFrame, DefaultDisagreementConfig())
	if err != nil {
		t.Fatalf("score: %v", err)
	}
	s := scores[0]
	if s.Births != 1 {
		t.Fatalf("want 1 birth, got %+v", s)
	}
	if s.Disagreement != 1.0 {
		t.Fatalf("want d=1.0, got %v", s.Disagreement)
	}
}

// Test 4 — object disappears at B: death > 0.
func TestObjectDeath(t *testing.T) {
	frames, byFrame := scoreFixture([]float64{0, 4}, [][]tracker.DetectionInput{
		{box("person", 0.9, 0.3)},
		{},
	})
	scores, err := ScoreIntervals(context.Background(), frames, byFrame, DefaultDisagreementConfig())
	if err != nil {
		t.Fatalf("score: %v", err)
	}
	s := scores[0]
	if s.Deaths != 1 {
		t.Fatalf("want 1 death, got %+v", s)
	}
	if s.Disagreement != 1.0 {
		t.Fatalf("want d=1.0, got %v", s.Disagreement)
	}
}

// Test 5 — an established track matching only a low-confidence detection
// takes the second-stage (weak) path.
func TestWeakMatchSecondStage(t *testing.T) {
	frames, byFrame := scoreFixture([]float64{0, 4, 8}, [][]tracker.DetectionInput{
		{box("person", 0.9, 0.3)},
		{box("person", 0.9, 0.3)},
		{box("person", 0.3, 0.3)}, // low confidence: second stage only
	})
	scores, err := ScoreIntervals(context.Background(), frames, byFrame, DefaultDisagreementConfig())
	if err != nil {
		t.Fatalf("score: %v", err)
	}
	if len(scores) != 2 {
		t.Fatalf("want 2 intervals, got %d", len(scores))
	}
	if scores[0].WeakMatches != 0 || scores[0].Disagreement != 0 {
		t.Fatalf("first interval must be clean: %+v", scores[0])
	}
	if scores[1].WeakMatches != 1 {
		t.Fatalf("want 1 weak match, got %+v", scores[1])
	}
	if scores[1].Disagreement != 0.5 {
		t.Fatalf("want d=0.5*1/1=0.5, got %v", scores[1].Disagreement)
	}
}

// Test 6 — a death followed by a nearby birth of the same label counts a
// fragment (boxes too far apart to match, close enough to link).
func TestFragmentation(t *testing.T) {
	a := box("person", 0.9, 0.10)
	b := box("person", 0.9, 0.22) // center shift 0.12: IoU ~0.25, dist 0.12
	frames, byFrame := scoreFixture([]float64{0, 4}, [][]tracker.DetectionInput{{a}, {b}})
	scores, err := ScoreIntervals(context.Background(), frames, byFrame, DefaultDisagreementConfig())
	if err != nil {
		t.Fatalf("score: %v", err)
	}
	s := scores[0]
	if s.Births != 1 || s.Deaths != 1 {
		t.Fatalf("want 1 birth + 1 death, got %+v", s)
	}
	if s.Fragments != 1 {
		t.Fatalf("want 1 fragment, got %+v", s)
	}
	want := 1.0 + 1.0 + 0.75 // W_BIRTH + W_DEATH + W_FRAGMENT over max(1,1)
	if s.Disagreement != want {
		t.Fatalf("want d=%v, got %v", want, s.Disagreement)
	}
}

// Test 7 — the tracker reasons in wall-clock seconds, not update counts: a
// static object across a 1s gap stays clean, while the same detections 30s
// apart (beyond the lost allowance) break identity.
func TestVariableGapsUseWallClock(t *testing.T) {
	for _, tc := range []struct {
		name string
		gap  float64
		want float64
	}{
		{"1s gap stays tracked", 1.0, 0},
		{"30s gap breaks identity", 30.0, 2.0}, // birth + death over max(1,1)
	} {
		t.Run(tc.name, func(t *testing.T) {
			frames, byFrame := scoreFixture([]float64{0, tc.gap}, [][]tracker.DetectionInput{
				{box("person", 0.9, 0.3)},
				{box("person", 0.9, 0.3)},
			})
			scores, err := ScoreIntervals(context.Background(), frames, byFrame, DefaultDisagreementConfig())
			if err != nil {
				t.Fatalf("score: %v", err)
			}
			if scores[0].Disagreement != tc.want {
				t.Fatalf("want d=%v, got %+v", tc.want, scores[0])
			}
		})
	}
}

// Test 8 — identical inputs produce identical outputs.
func TestReplayDeterminism(t *testing.T) {
	build := func() ([]tracker.FrameInput, map[uuid.UUID][]tracker.DetectionInput) {
		return scoreFixture([]float64{8, 0, 4, 12}, [][]tracker.DetectionInput{
			{box("person", 0.9, 0.3), box("car", 0.8, 0.6)},
			{},
			{box("person", 0.9, 0.32)},
			{box("car", 0.85, 0.61)},
		})
	}
	f1, d1 := build()
	f2, d2 := build()
	s1, err := ScoreIntervals(context.Background(), f1, d1, DefaultDisagreementConfig())
	if err != nil {
		t.Fatalf("score: %v", err)
	}
	s2, err := ScoreIntervals(context.Background(), f2, d2, DefaultDisagreementConfig())
	if err != nil {
		t.Fatalf("score: %v", err)
	}
	if !reflect.DeepEqual(s1, s2) {
		t.Fatalf("nondeterministic:\n%+v\n%+v", s1, s2)
	}
}

// Test 9 — N timestamps yield exactly N-1 interval scores with boundaries.
func TestMultipleIntervals(t *testing.T) {
	frames, byFrame := scoreFixture(
		[]float64{0, 4, 8, 12, 16},
		[][]tracker.DetectionInput{{}, {}, {}, {}, {}},
	)
	scores, err := ScoreIntervals(context.Background(), frames, byFrame, DefaultDisagreementConfig())
	if err != nil {
		t.Fatalf("score: %v", err)
	}
	if len(scores) != 4 {
		t.Fatalf("want 4 intervals, got %d", len(scores))
	}
	for i, s := range scores {
		if s.StartTimestamp != float64(i*4) || s.EndTimestamp != float64(i*4+4) || s.GapSeconds != 4 {
			t.Fatalf("interval %d boundaries wrong: %+v", i, s)
		}
	}
	if _, err := ScoreIntervals(context.Background(), frames[:1], byFrame, DefaultDisagreementConfig()); err != nil {
		t.Fatalf("single frame: %v", err)
	}
}

// Test 10 — scoring never mutates production tracker state: production
// assignments before and after scoring are identical.
func TestNoProductionTrackerMutation(t *testing.T) {
	prod, err := tracker.New("bytetrack", 0.5, 0.1, 0.3, 5, true, 2)
	if err != nil {
		t.Fatalf("tracker.New: %v", err)
	}
	frames, byFrame := scoreFixture([]float64{0, 4, 8}, [][]tracker.DetectionInput{
		{box("person", 0.9, 0.3)},
		{box("person", 0.9, 0.31)},
		{box("person", 0.9, 0.32)},
	})
	before, err := prod.Track(context.Background(), frames, byFrame)
	if err != nil {
		t.Fatalf("production track: %v", err)
	}
	if _, err := ScoreIntervals(context.Background(), frames, byFrame, DefaultDisagreementConfig()); err != nil {
		t.Fatalf("score: %v", err)
	}
	after, err := prod.Track(context.Background(), frames, byFrame)
	if err != nil {
		t.Fatalf("production track: %v", err)
	}
	if !reflect.DeepEqual(before, after) {
		t.Fatalf("production tracker output changed by scorer:\n%v\n%v", before, after)
	}
}
