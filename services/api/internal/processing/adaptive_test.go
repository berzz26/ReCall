package processing

// Production-boundary tests for the Phase 5 adaptive orchestrator (feature-
// flag matrix, fallbacks, timeout, cancellation, shadow, resume, panic).
// All dependencies are fakes: no database, no ffmpeg, no fixtures needed.

import (
	"context"
	"errors"
	"math"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/berzz26/recall/services/api/internal/sampler"
	"github.com/berzz26/recall/services/api/internal/tracker"
	"github.com/berzz26/recall/services/api/internal/video"
	"github.com/berzz26/recall/services/api/internal/video_frame"
	"github.com/berzz26/recall/services/api/internal/video_segment"
)

const (
	testDuration = 20.0
	testCap      = 10 // ceil(20/2), beta 1.0
)

func testBudget() sampler.Budget {
	return sampler.Budget{BaselineInterval: 2 * time.Second, Beta: 1.0, BaselineCount: testCap, Cap: testCap}
}

func testVideo() *video.Video {
	src := video.SourceTypeLocal
	path := "/tmp/fake.mp4"
	return &video.Video{ID: uuid.New(), Filename: "fake.mp4", SourceType: src, SourcePath: &path}
}

func testSegments(vid uuid.UUID) []video_segment.VideoSegment {
	return []video_segment.VideoSegment{
		{ID: uuid.New(), VideoID: vid, SegmentIndex: 0, StartTime: 0, EndTime: testDuration, Duration: testDuration},
	}
}

// stubPlanner returns a canned coarse result (or error/panic) for matrix tests.
type stubPlanner struct {
	result         sampler.AdaptivePlanResult
	err            error
	panicVal       any
	blockUntilDone bool
	calls          int
}

func (s *stubPlanner) PlanVideo(ctx context.Context, _ uuid.UUID, _ string, _ float64) (sampler.AdaptivePlanResult, error) {
	s.calls++
	if s.panicVal != nil {
		panic(s.panicVal)
	}
	if s.blockUntilDone {
		<-ctx.Done()
		return sampler.AdaptivePlanResult{}, ctx.Err()
	}
	return s.result, s.err
}

// stubExtractor serves frames from memory and records every call.
type stubExtractor struct {
	existing        []video_frame.VideoFrame
	genPlans        []sampler.SamplingPlan
	additionalCalls [][]float64
	additionalErr   error
}

func framesFor(vid uuid.UUID, segID uuid.UUID, ts []float64) []video_frame.VideoFrame {
	out := make([]video_frame.VideoFrame, 0, len(ts))
	for i, t := range ts {
		out = append(out, video_frame.VideoFrame{
			ID: uuid.New(), VideoID: vid, SegmentID: segID, FrameIndex: i,
			TimestampSeconds: t, StorageKey: "k", Width: 64, Height: 64,
		})
	}
	return out
}

func (s *stubExtractor) GenerateForVideoWithPlan(_ context.Context, v *video.Video, _ []video_segment.VideoSegment, plan sampler.SamplingPlan, _, _ int) ([]video_frame.VideoFrame, error) {
	s.genPlans = append(s.genPlans, plan)
	ts := plan.Timestamps()
	seg := uuid.New()
	if len(s.existing) > 0 {
		seg = s.existing[0].SegmentID
	}
	return framesFor(v.ID, seg, ts), nil
}

func (s *stubExtractor) ExtractAdditional(_ context.Context, v *video.Video, _ []video_segment.VideoSegment, timestamps []float64, _ float64, _, _ int) ([]video_frame.VideoFrame, error) {
	cp := append([]float64(nil), timestamps...)
	s.additionalCalls = append(s.additionalCalls, cp)
	if s.additionalErr != nil {
		return nil, s.additionalErr
	}
	var seg uuid.UUID
	if len(s.existing) > 0 {
		seg = s.existing[0].SegmentID
	} else {
		seg = uuid.New()
	}
	return framesFor(v.ID, seg, timestamps), nil
}

func (s *stubExtractor) GetByVideoID(_ context.Context, _ uuid.UUID) ([]video_frame.VideoFrame, error) {
	return s.existing, nil
}

// stubDetector returns canned planning detections and records calls.
type stubDetector struct {
	fn    func(f video_frame.VideoFrame) []tracker.DetectionInput
	calls int
}

func (s *stubDetector) DetectFrames(_ context.Context, _ uuid.UUID, frames []video_frame.VideoFrame) ([]tracker.DetectionInput, error) {
	s.calls++
	var out []tracker.DetectionInput
	for _, f := range frames {
		for _, d := range s.fn(f) {
			d.FrameID = f.ID
			d.Timestamp = f.TimestampSeconds
			out = append(out, d)
		}
	}
	return out, nil
}

func personBox(conf float64) tracker.DetectionInput {
	return tracker.DetectionInput{ID: uuid.New(), Label: "person", Confidence: conf,
		BBoxX: 0.3, BBoxY: 0.3, BBoxWidth: 0.2, BBoxHeight: 0.2}
}

func newTestOrchestrator(ext *stubExtractor, det *stubDetector, planner *stubPlanner, mutate func(*AdaptiveSamplerConfig)) *AdaptiveSampler {
	cfg := AdaptiveSamplerConfig{
		Busy:             sampler.DefaultBusyConfig(),
		Timeout:          30 * time.Second,
		Shadow:           false,
		BaselineInterval: 2 * time.Second,
		Beta:             1.0,
	}
	if mutate != nil {
		mutate(&cfg)
	}
	refiner := sampler.NewRefiner(sampler.RefinementConfig{
		Gamma:         sampler.DefaultGamma,
		MinGapSeconds: sampler.DefaultRefineMinGapSeconds,
		Epsilon:       sampler.DefaultRefineEpsilon,
		MaxRounds:     2,
		ActivityFloor: 0, // matrix tests use exact Phase 4 behavior
		Disagreement:  sampler.DefaultDisagreementConfig(),
	})
	a, err := NewAdaptiveSampler(ext, det, planner, refiner, cfg)
	if err != nil {
		panic(err)
	}
	return a
}

func adaptiveStubResult(vid uuid.UUID, ts []float64, probe *sampler.ProbeReport) sampler.AdaptivePlanResult {
	plan := sampler.PlanWithTimestamps(vid, testDuration, ts, sampler.ReasonCoarse, testBudget())
	plan.Mode = sampler.ModeAdaptive
	return sampler.AdaptivePlanResult{Plan: plan, Probe: probe}
}

func TestOrchestratorRequiresDeps(t *testing.T) {
	ext, det, planner := &stubExtractor{}, &stubDetector{}, &stubPlanner{}
	refiner := sampler.NewRefiner(sampler.DefaultRefinementConfig())
	if _, err := NewAdaptiveSampler(nil, det, planner, refiner, AdaptiveSamplerConfig{}); err == nil {
		t.Fatalf("nil extractor must fail fast")
	}
	if _, err := NewAdaptiveSampler(ext, nil, planner, refiner, AdaptiveSamplerConfig{}); err == nil {
		t.Fatalf("nil detector must fail fast")
	}
	if _, err := NewAdaptiveSampler(ext, det, nil, refiner, AdaptiveSamplerConfig{}); err == nil {
		t.Fatalf("nil planner must fail fast")
	}
	if _, err := NewAdaptiveSampler(ext, det, planner, nil, AdaptiveSamplerConfig{}); err == nil {
		t.Fatalf("nil refiner must fail fast")
	}
}

func TestAdaptivePathInvokesPlanner(t *testing.T) {
	v := testVideo()
	segs := testSegments(v.ID)
	planner := &stubPlanner{result: adaptiveStubResult(v.ID, []float64{0, 10}, nil)}
	ext := &stubExtractor{}
	det := &stubDetector{fn: func(video_frame.VideoFrame) []tracker.DetectionInput { return nil }}
	a := newTestOrchestrator(ext, det, planner, nil)

	plan, stats, err := a.PlanAndExtract(context.Background(), v, segs, testDuration, 64, 64, "/tmp/fake.mp4", false)
	if err != nil {
		t.Fatalf("PlanAndExtract: %v", err)
	}
	if planner.calls != 1 {
		t.Fatalf("planner must be invoked once, got %d", planner.calls)
	}
	if stats.Mode != sampler.ModeAdaptive {
		t.Fatalf("mode must be adaptive, got %q", stats.Mode)
	}
	if stats.Version != sampler.SamplerVersion || stats.Version == "" {
		t.Fatalf("version must be logged, got %q", stats.Version)
	}
	if plan == nil || len(plan.Timestamps()) == 0 {
		t.Fatalf("adaptive plan must be returned")
	}
	if len(plan.Timestamps()) > testCap {
		t.Fatalf("budget exceeded: %d > %d", len(plan.Timestamps()), testCap)
	}
	if stats.FinalFrames != len(plan.Timestamps()) {
		t.Fatalf("stats must match plan")
	}
}

func TestBaselineFallbackOnProbeFailure(t *testing.T) {
	v := testVideo()
	segs := testSegments(v.ID)
	fb := sampler.PlanWithTimestamps(v.ID, testDuration, []float64{0, 2, 4, 6, 8, 10, 12, 14, 16, 18}, sampler.ReasonCoarse, testBudget())
	fb.Mode = sampler.ModeBaselineFallback
	planner := &stubPlanner{result: sampler.AdaptivePlanResult{Plan: fb, FallbackReason: "visual probe failed: boom"}}
	ext := &stubExtractor{}
	det := &stubDetector{fn: func(video_frame.VideoFrame) []tracker.DetectionInput { return nil }}
	a := newTestOrchestrator(ext, det, planner, nil)

	plan, stats, err := a.PlanAndExtract(context.Background(), v, segs, testDuration, 64, 64, "/tmp/fake.mp4", false)
	if err != nil {
		t.Fatalf("fallback must not error: %v", err)
	}
	if stats.Mode != sampler.ModeBaselineFallback {
		t.Fatalf("mode must be baseline_fallback, got %q", stats.Mode)
	}
	if stats.FallbackReason == "" {
		t.Fatalf("fallback reason must be recorded")
	}
	if det.calls != 0 {
		t.Fatalf("no planning YOLO on fallback path, got %d calls", det.calls)
	}
	if len(ext.additionalCalls) != 0 {
		t.Fatalf("no midpoint extraction on fallback path")
	}
	if plan == nil || len(plan.Timestamps()) != testCap {
		t.Fatalf("baseline plan must carry full coverage")
	}
}

func TestUniformFallbackOnBusy(t *testing.T) {
	v := testVideo()
	segs := testSegments(v.ID)
	var pts []sampler.ProbePoint
	for i := 0; i < 50; i++ {
		frac := 0.0
		if i < 45 {
			frac = 0.5
		}
		pts = append(pts, sampler.ProbePoint{TimestampSeconds: float64(i) * 0.2, ChangedFraction: frac})
	}
	probe := &sampler.ProbeReport{Points: pts, FramesDecoded: len(pts)}
	planner := &stubPlanner{result: adaptiveStubResult(v.ID, []float64{0, 10}, probe)}
	ext := &stubExtractor{}
	det := &stubDetector{fn: func(video_frame.VideoFrame) []tracker.DetectionInput { return nil }}
	a := newTestOrchestrator(ext, det, planner, nil)

	plan, stats, err := a.PlanAndExtract(context.Background(), v, segs, testDuration, 64, 64, "/tmp/fake.mp4", false)
	if err != nil {
		t.Fatalf("uniform fallback must not error: %v", err)
	}
	if stats.Mode != sampler.ModeUniformFallback {
		t.Fatalf("mode must be uniform_fallback, got %q", stats.Mode)
	}
	if stats.BusyFraction < 0.6 {
		t.Fatalf("busy fraction must be recorded, got %v", stats.BusyFraction)
	}
	if plan == nil || len(plan.Timestamps()) > testCap {
		t.Fatalf("uniform plan must respect budget")
	}
	if det.calls != 0 || len(ext.additionalCalls) != 0 {
		t.Fatalf("uniform path must not refine: det=%d additional=%d", det.calls, len(ext.additionalCalls))
	}
}

func TestPlanningTimeoutFallback(t *testing.T) {
	v := testVideo()
	segs := testSegments(v.ID)
	planner := &stubPlanner{blockUntilDone: true}
	ext := &stubExtractor{}
	det := &stubDetector{fn: func(video_frame.VideoFrame) []tracker.DetectionInput { return nil }}
	a := newTestOrchestrator(ext, det, planner, func(c *AdaptiveSamplerConfig) {
		c.Timeout = 50 * time.Millisecond
	})

	_, stats, err := a.PlanAndExtract(context.Background(), v, segs, testDuration, 64, 64, "/tmp/fake.mp4", false)
	if err != nil {
		t.Fatalf("timeout must fall back, not error: %v", err)
	}
	if stats.Mode != sampler.ModeBaselineFallback {
		t.Fatalf("mode must be baseline_fallback, got %q", stats.Mode)
	}
}

func TestCancellationPropagates(t *testing.T) {
	v := testVideo()
	segs := testSegments(v.ID)
	planner := &stubPlanner{result: adaptiveStubResult(v.ID, []float64{0, 10}, nil)}
	ext := &stubExtractor{}
	det := &stubDetector{fn: func(video_frame.VideoFrame) []tracker.DetectionInput { return nil }}
	a := newTestOrchestrator(ext, det, planner, nil)

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, _, err := a.PlanAndExtract(ctx, v, segs, testDuration, 64, 64, "/tmp/fake.mp4", false)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("cancellation must propagate, got %v", err)
	}
	if len(ext.genPlans) != 0 || len(ext.additionalCalls) != 0 {
		t.Fatalf("cancelled run must not extract")
	}
}

func TestInvalidPlanFallback(t *testing.T) {
	v := testVideo()
	segs := testSegments(v.ID)
	bad := sampler.PlanWithTimestamps(v.ID, testDuration, []float64{0, 4}, sampler.ReasonCoarse, testBudget())
	bad.Mode = sampler.ModeAdaptive
	bad.Entries[0], bad.Entries[1] = bad.Entries[1], bad.Entries[0] // unsorted
	planner := &stubPlanner{result: sampler.AdaptivePlanResult{Plan: bad, Probe: &sampler.ProbeReport{}}}
	ext := &stubExtractor{}
	det := &stubDetector{fn: func(video_frame.VideoFrame) []tracker.DetectionInput { return nil }}
	a := newTestOrchestrator(ext, det, planner, nil)

	plan, stats, err := a.PlanAndExtract(context.Background(), v, segs, testDuration, 64, 64, "/tmp/fake.mp4", false)
	if err != nil {
		t.Fatalf("invalid plan must fall back, not error: %v", err)
	}
	if stats.Mode != sampler.ModeBaselineFallback {
		t.Fatalf("mode must be baseline_fallback, got %q", stats.Mode)
	}
	if plan == nil || len(plan.Timestamps()) != testCap {
		t.Fatalf("fallback must carry baseline coverage")
	}
}

func TestRefinementFailureFallback(t *testing.T) {
	v := testVideo()
	segs := testSegments(v.ID)
	// Coarse [0,8] with a birth at 8: disagreement 1.0, gap 8 -> eligible.
	planner := &stubPlanner{result: func() sampler.AdaptivePlanResult {
		r := adaptiveStubResult(v.ID, []float64{0, 8}, &sampler.ProbeReport{})
		return r
	}()}
	ext := &stubExtractor{additionalErr: errors.New("ffmpeg exploded")}
	det := &stubDetector{fn: func(f video_frame.VideoFrame) []tracker.DetectionInput {
		if math.Abs(f.TimestampSeconds-8) < 1e-6 {
			return []tracker.DetectionInput{personBox(0.9)}
		}
		return nil
	}}
	a := newTestOrchestrator(ext, det, planner, nil)

	plan, stats, err := a.PlanAndExtract(context.Background(), v, segs, testDuration, 64, 64, "/tmp/fake.mp4", false)
	if err != nil {
		t.Fatalf("refinement failure must fall back, not error: %v", err)
	}
	if stats.Mode != sampler.ModeBaselineFallback {
		t.Fatalf("mode must be baseline_fallback, got %q", stats.Mode)
	}
	if plan == nil || len(plan.Timestamps()) != testCap {
		t.Fatalf("fallback must carry baseline coverage")
	}
}

func TestShadowKeepsBaselineAuthoritative(t *testing.T) {
	v := testVideo()
	segs := testSegments(v.ID)
	planner := &stubPlanner{result: adaptiveStubResult(v.ID, []float64{0, 10}, nil)}
	ext := &stubExtractor{}
	det := &stubDetector{fn: func(video_frame.VideoFrame) []tracker.DetectionInput { return nil }}
	a := newTestOrchestrator(ext, det, planner, func(c *AdaptiveSamplerConfig) {
		c.Shadow = true
	})

	plan, stats, err := a.PlanAndExtract(context.Background(), v, segs, testDuration, 64, 64, "/tmp/fake.mp4", false)
	if err != nil {
		t.Fatalf("shadow must not error: %v", err)
	}
	if stats.Mode != sampler.ModeBaseline {
		t.Fatalf("shadow run mode must be baseline, got %q", stats.Mode)
	}
	if !stats.Shadow || stats.ShadowAdaptiveFrames != 2 {
		t.Fatalf("shadow must record would-be adaptive count: %+v", stats)
	}
	if det.calls != 0 || len(ext.additionalCalls) != 0 {
		t.Fatalf("shadow must not refine or planning-YOLO")
	}
	if plan == nil || len(plan.Timestamps()) != testCap {
		t.Fatalf("shadow must extract baseline coverage")
	}
}

func TestResumeSkipsPlanning(t *testing.T) {
	v := testVideo()
	segs := testSegments(v.ID)
	planner := &stubPlanner{result: adaptiveStubResult(v.ID, []float64{0, 10}, nil)}
	ext := &stubExtractor{existing: framesFor(v.ID, uuid.New(), []float64{0, 2, 4})}
	det := &stubDetector{fn: func(video_frame.VideoFrame) []tracker.DetectionInput { return nil }}
	a := newTestOrchestrator(ext, det, planner, nil)

	plan, stats, err := a.PlanAndExtract(context.Background(), v, segs, testDuration, 64, 64, "/tmp/fake.mp4", true)
	if err != nil {
		t.Fatalf("resume must not error: %v", err)
	}
	if plan != nil || !stats.Resumed || stats.FinalFrames != 3 {
		t.Fatalf("resume must preserve existing frames untouched: %+v", stats)
	}
	if planner.calls != 0 || len(ext.genPlans) != 0 {
		t.Fatalf("resume must not plan or extract")
	}
}

func TestPanicFallback(t *testing.T) {
	v := testVideo()
	segs := testSegments(v.ID)
	planner := &stubPlanner{panicVal: "simulated planner panic"}
	ext := &stubExtractor{}
	det := &stubDetector{fn: func(video_frame.VideoFrame) []tracker.DetectionInput { return nil }}
	a := newTestOrchestrator(ext, det, planner, nil)

	plan, stats, err := a.PlanAndExtract(context.Background(), v, segs, testDuration, 64, 64, "/tmp/fake.mp4", false)
	if err != nil {
		t.Fatalf("panic must fall back, not error: %v", err)
	}
	if stats.Mode != sampler.ModeBaselineFallback {
		t.Fatalf("mode must be baseline_fallback, got %q", stats.Mode)
	}
	if plan == nil || len(plan.Timestamps()) != testCap {
		t.Fatalf("panic fallback must carry baseline coverage")
	}
}

func TestProcessorWiringDefaults(t *testing.T) {
	p := NewFFprobeProcessor("ffprobe", time.Second, nil, nil)
	if p.adaptiveSampler != nil {
		t.Fatalf("adaptive sampler must be nil by default (baseline preserved)")
	}
	ext, det, planner := &stubExtractor{}, &stubDetector{}, &stubPlanner{}
	a := newTestOrchestrator(ext, det, planner, nil)
	if p.WithAdaptiveSampler(a) != p {
		t.Fatalf("setter must chain")
	}
	if p.adaptiveSampler == nil {
		t.Fatalf("setter must install orchestrator")
	}
}
