package tracker

// Unit checks for the throwaway replay path: time-aware prediction scaling
// and per-event emission. Behavioral interval tests live in
// sampler/disagreement_test.go.

import (
	"context"
	"math"
	"testing"

	"github.com/google/uuid"
)

// The Kalman prediction must propagate with the ACTUAL delta: a 4s gap is
// four times the temporal propagation of a 1s gap, not "one frame".
func TestKalmanPredictUsesActualDelta(t *testing.T) {
	var mean [8]float64
	mean[0] = 0.3 // cx
	mean[4] = 0.1 // vx
	var cov [8][8]float64
	for i := 0; i < 8; i++ {
		cov[i][i] = 1e-4
	}
	m1, _ := kalmanPredict(mean, cov, 1.0)
	m4, _ := kalmanPredict(mean, cov, 4.0)
	d1 := m1[0] - mean[0]
	d4 := m4[0] - mean[0]
	if math.Abs(d4/d1-4.0) > 1e-9 {
		t.Fatalf("dt=4 must propagate 4x dt=1, got %v vs %v", d4, d1)
	}
}

// Lost handling is wall-clock based: identical single-miss sequences behave
// differently at 1s vs 30s gaps.
func TestThrowawayLostIsWallClock(t *testing.T) {
	run := func(gap float64) ReplayResult {
		fa, fb := uuid.New(), uuid.New()
		mkDet := func(fid uuid.UUID, ts float64) DetectionInput {
			return DetectionInput{ID: uuid.New(), Label: "person", Confidence: 0.9,
				BBoxX: 0.3, BBoxY: 0.3, BBoxWidth: 0.2, BBoxHeight: 0.2,
				FrameID: fid, Timestamp: ts}
		}
		frames := []FrameInput{{FrameID: fa, Timestamp: 0}, {FrameID: fb, Timestamp: gap}}
		dets := map[uuid.UUID][]DetectionInput{
			fa: {mkDet(fa, 0)},
			fb: {mkDet(fb, gap)},
		}
		res, err := ReplayThrowaway(context.Background(), frames, dets, DefaultThrowawayConfig())
		if err != nil {
			t.Fatalf("replay: %v", err)
		}
		return res
	}
	// 1s gap: one track, two associations, first stage.
	short := run(1.0)
	if len(short.Tracks) != 1 {
		t.Fatalf("1s gap: want 1 track, got %d", len(short.Tracks))
	}
	// 30s gap (beyond LostSeconds=10): track removed, redetection is new.
	long := run(30.0)
	if len(long.Tracks) != 2 {
		t.Fatalf("30s gap: want 2 tracks (death+birth), got %d", len(long.Tracks))
	}
}

// Replay emits stage information and creation flags for the scorer.
func TestReplayEmitsStagesAndBirths(t *testing.T) {
	fa, fb, fc := uuid.New(), uuid.New(), uuid.New()
	mkDet := func(fid uuid.UUID, ts, conf float64) DetectionInput {
		return DetectionInput{ID: uuid.New(), Label: "person", Confidence: conf,
			BBoxX: 0.3, BBoxY: 0.3, BBoxWidth: 0.2, BBoxHeight: 0.2,
			FrameID: fid, Timestamp: ts}
	}
	frames := []FrameInput{
		{FrameID: fa, Timestamp: 0},
		{FrameID: fb, Timestamp: 4},
		{FrameID: fc, Timestamp: 8},
	}
	dets := map[uuid.UUID][]DetectionInput{
		fa: {mkDet(fa, 0, 0.9)},
		fb: {mkDet(fb, 4, 0.9)},
		fc: {mkDet(fc, 8, 0.3)}, // low confidence -> second stage
	}
	res, err := ReplayThrowaway(context.Background(), frames, dets, DefaultThrowawayConfig())
	if err != nil {
		t.Fatalf("replay: %v", err)
	}
	if len(res.Frames) != 3 {
		t.Fatalf("want 3 frames, got %d", len(res.Frames))
	}
	if !res.Frames[0].Associations[0].NewTrack {
		t.Fatalf("first detection must be flagged as birth")
	}
	var sawSecond bool
	for _, a := range res.Frames[2].Associations {
		if a.Stage == StageSecond {
			sawSecond = true
		}
		if a.NewTrack {
			t.Fatalf("low-confidence detection must not create a track")
		}
	}
	if !sawSecond {
		t.Fatalf("low-confidence continuation must use the second stage")
	}
}
