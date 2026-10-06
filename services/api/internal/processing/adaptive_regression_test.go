package processing

// Real-video regression coverage for the Phase 5 adaptive boundary.
// Uses the real AdaptiveCoarsePlanner visual probe (ffmpeg-gated) with
// in-memory extraction and deterministic scripted detections, so no
// database is needed. Skips gracefully without fixtures or ffmpeg.
// The user runs these manually and reviews the logged §31 tables for
// qualitative behavior (static → large savings, busy → little/none).

import (
	"context"
	"os"
	"os/exec"
	"reflect"
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/berzz26/recall/services/api/internal/sampler"
	"github.com/berzz26/recall/services/api/internal/tracker"
	"github.com/berzz26/recall/services/api/internal/video_frame"
	"github.com/berzz26/recall/services/api/internal/video_segment"
)

func fixtureDuration(t *testing.T, path string) float64 {
	t.Helper()
	out, err := exec.Command("ffprobe", "-v", "error", "-show_entries",
		"format=duration", "-of", "csv=p=0", path).Output()
	if err != nil {
		t.Skipf("ffprobe failed for %s: %v", path, err)
	}
	dur, err := strconv.ParseFloat(strings.TrimSpace(string(out)), 64)
	if err != nil || dur <= 0 {
		t.Skipf("bad duration for %s: %q", path, out)
	}
	return dur
}

func regressionStack(shadow bool, busy sampler.BusyConfig) (*stubExtractor, *stubDetector, AdaptiveSamplerConfig) {
	ext := &stubExtractor{}
	det := &stubDetector{fn: func(video_frame.VideoFrame) []tracker.DetectionInput {
		return []tracker.DetectionInput{personBox(0.9)}
	}}
	cfg := AdaptiveSamplerConfig{
		Busy:             busy,
		Timeout:          5 * time.Minute,
		Shadow:           shadow,
		BaselineInterval: 2 * time.Second,
		Beta:             1.0,
	}
	return ext, det, cfg
}

func checkRegressionInvariants(t *testing.T, mode string, plan *sampler.SamplingPlan, stats *AdaptiveStats, duration string, cap int) {
	t.Helper()
	if plan == nil {
		t.Fatalf("[%s] %s: expected a plan", mode, duration)
	}
	if stats.Mode != mode {
		t.Fatalf("[%s] want mode %q, got %q (reason %q)", duration, mode, stats.Mode, stats.FallbackReason)
	}
	ts := plan.Timestamps()
	if len(ts) == 0 || len(ts) > cap {
		t.Fatalf("[%s] frame count %d outside (0, %d]", duration, len(ts), cap)
	}
	for i, x := range ts {
		if invalidTs(x) {
			t.Fatalf("[%s] bad timestamp %v", duration, x)
		}
		if i > 0 && x <= ts[i-1] {
			t.Fatalf("[%s] timestamps not strictly sorted", duration)
		}
	}
	if err := sampler.ValidatePlan(*plan); err != nil {
		t.Fatalf("[%s] final plan invalid: %v", duration, err)
	}
	if stats.FinalFrames != len(ts) {
		t.Fatalf("[%s] stats/plan count mismatch", duration)
	}
	t.Logf("[%s] mode=%s duration=%.1fs baseline=%d coarse=%d refined=%d final=%d savings=%d rounds=%d maxpri=%.3f planms=%d",
		duration, stats.Mode, stats.DurationSeconds, stats.BaselineFrames,
		stats.CoarseFrames, stats.RefinementFrames, stats.FinalFrames,
		stats.EstimatedSavings, stats.Rounds, stats.MaxPriority, stats.TotalMs)
}

func invalidTs(x float64) bool {
	return x != x || x < 0 // NaN or negative; upper bound checked by ValidatePlan
}

// runProductionTrackOnce proves the final timestamp sequence feeds the real
// production ByteTrack deterministically (refinement never touches it).
func runProductionTrackOnce(t *testing.T, ts []float64) {
	t.Helper()
	trk, err := tracker.New("bytetrack", 0.5, 0.1, 0.3, 5, true, 2)
	if err != nil {
		t.Fatalf("tracker.New: %v", err)
	}
	var frames []tracker.FrameInput
	byFrame := make(map[uuid.UUID][]tracker.DetectionInput)
	for _, x := range ts {
		fid := uuid.New()
		frames = append(frames, tracker.FrameInput{FrameID: fid, Timestamp: x})
		byFrame[fid] = []tracker.DetectionInput{personBox(0.9)}
	}
	ctx := context.Background()
	first, err := trk.Track(ctx, frames, byFrame)
	if err != nil {
		t.Fatalf("production Track: %v", err)
	}
	second, err := trk.Track(ctx, frames, byFrame)
	if err != nil {
		t.Fatalf("production Track: %v", err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("production tracker nondeterministic over final plan")
	}
}

// TestAdaptiveRegressionFixtures runs the orchestrator end to end on every
// available fixture in adaptive, shadow, busy-forced (uniform), and
// fallback modes.
func TestAdaptiveRegressionFixtures(t *testing.T) {
	if _, err := os.Stat("/usr/bin/ffmpeg"); err != nil {
		t.Skip("ffmpeg not available")
	}
	fixtures := []string{
		"/home/berzz/recallTestVideo/video1.mp4",
		"/home/berzz/recallTestVideo/video2.mp4",
	}
	seen := false
	for _, path := range fixtures {
		if _, err := os.Stat(path); err != nil {
			t.Logf("fixture missing, skipping: %s", path)
			continue
		}
		seen = true
		duration := fixtureDuration(t, path)
		budget, err := sampler.BaselineBudget(duration, 2*time.Second, 1.0)
		if err != nil {
			t.Fatalf("budget: %v", err)
		}

		// Mode 1: full adaptive (real probe, real planner, real refiner).
		func() {
			v := testVideo()
			segs := []video_segment.VideoSegment{
				{ID: uuid.New(), VideoID: v.ID, SegmentIndex: 0, StartTime: 0, EndTime: duration, Duration: duration},
			}
			ext, det, cfg := regressionStack(false, sampler.DefaultBusyConfig())
			a, err := NewAdaptiveSampler(ext, det,
				sampler.NewAdaptiveCoarsePlanner(sampler.DefaultAdaptiveConfig()),
				sampler.NewRefiner(sampler.DefaultRefinementConfig()), cfg)
			if err != nil {
				t.Fatalf("orchestrator: %v", err)
			}
			plan, stats, err := a.PlanAndExtract(context.Background(), v, segs, duration, 64, 64, path, false)
			if err != nil {
				t.Fatalf("[%s] adaptive: %v", path, err)
			}
			if stats.Mode != sampler.ModeAdaptive && stats.Mode != sampler.ModeUniformFallback && stats.Mode != sampler.ModeBaselineFallback {
				t.Fatalf("[%s] unexpected mode %q", path, stats.Mode)
			}
			checkRegressionInvariants(t, stats.Mode, plan, stats, path, budget.Cap)
			runProductionTrackOnce(t, plan.Timestamps())
		}()

		// Mode 2: shadow (baseline authoritative, adaptive recorded).
		func() {
			v := testVideo()
			segs := []video_segment.VideoSegment{
				{ID: uuid.New(), VideoID: v.ID, SegmentIndex: 0, StartTime: 0, EndTime: duration, Duration: duration},
			}
			ext, det, cfg := regressionStack(true, sampler.DefaultBusyConfig())
			a, err := NewAdaptiveSampler(ext, det,
				sampler.NewAdaptiveCoarsePlanner(sampler.DefaultAdaptiveConfig()),
				sampler.NewRefiner(sampler.DefaultRefinementConfig()), cfg)
			if err != nil {
				t.Fatalf("orchestrator: %v", err)
			}
			plan, stats, err := a.PlanAndExtract(context.Background(), v, segs, duration, 64, 64, path, false)
			if err != nil {
				t.Fatalf("[%s] shadow: %v", path, err)
			}
			checkRegressionInvariants(t, sampler.ModeBaseline, plan, stats, path, budget.Cap)
			if !stats.Shadow {
				t.Fatalf("[%s] shadow flag must be recorded", path)
			}
		}()

		// Mode 3: busy-forced uniform fallback (zero thresholds => everything busy).
		func() {
			v := testVideo()
			segs := []video_segment.VideoSegment{
				{ID: uuid.New(), VideoID: v.ID, SegmentIndex: 0, StartTime: 0, EndTime: duration, Duration: duration},
			}
			ext, det, cfg := regressionStack(false, sampler.BusyConfig{Threshold: 0, Fraction: 0})
			a, err := NewAdaptiveSampler(ext, det,
				sampler.NewAdaptiveCoarsePlanner(sampler.DefaultAdaptiveConfig()),
				sampler.NewRefiner(sampler.DefaultRefinementConfig()), cfg)
			if err != nil {
				t.Fatalf("orchestrator: %v", err)
			}
			plan, stats, err := a.PlanAndExtract(context.Background(), v, segs, duration, 64, 64, path, false)
			if err != nil {
				t.Fatalf("[%s] busy-forced: %v", path, err)
			}
			checkRegressionInvariants(t, sampler.ModeUniformFallback, plan, stats, path, budget.Cap)
		}()

		// Mode 4: planner-level fallback (simulated probe failure).
		func() {
			v := testVideo()
			segs := testSegments(v.ID)
			fb := sampler.PlanWithTimestamps(v.ID, testDuration, []float64{0, 2, 4, 6, 8, 10, 12, 14, 16, 18}, sampler.ReasonCoarse, testBudget())
			fb.Mode = sampler.ModeBaselineFallback
			planner := &stubPlanner{result: sampler.AdaptivePlanResult{Plan: fb, FallbackReason: "simulated"}}
			ext := &stubExtractor{}
			det := &stubDetector{fn: func(video_frame.VideoFrame) []tracker.DetectionInput { return nil }}
			a := newTestOrchestrator(ext, det, planner, nil)
			plan, stats, err := a.PlanAndExtract(context.Background(), v, segs, testDuration, 64, 64, "/tmp/fake.mp4", false)
			if err != nil {
				t.Fatalf("fallback: %v", err)
			}
			checkRegressionInvariants(t, sampler.ModeBaselineFallback, plan, stats, "/tmp/fake.mp4", testCap)
		}()
	}
	if !seen {
		t.Skip("no fixtures available")
	}
}
