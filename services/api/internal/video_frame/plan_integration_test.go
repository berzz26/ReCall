package video_frame

// Integration coverage for the planner -> extractor contract against a real
// video fixture and the real ffmpeg seeking decoder (no mocks).
// Skips gracefully when the fixture or ffmpeg is unavailable, mirroring the
// existing processing/frame_test.go conventions.

import (
	"context"
	"math"
	"os"
	"reflect"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/berzz26/recall/services/api/internal/detector"
	"github.com/berzz26/recall/services/api/internal/sampler"
	"github.com/berzz26/recall/services/api/internal/storage"
	"github.com/berzz26/recall/services/api/internal/tracker"
	"github.com/berzz26/recall/services/api/internal/video"
	"github.com/berzz26/recall/services/api/internal/video_segment"
)

const integrationFixture = "/home/berzz/recallTestVideo/video1.mp4"

func newIntegrationFixture(t *testing.T, duration float64) (*Service, *fakeRepo, storage.Storage, *video.Video, []video_segment.VideoSegment) {
	t.Helper()
	if _, err := os.Stat(integrationFixture); err != nil {
		t.Skipf("video fixture not found: %v", err)
	}
	if _, err := os.Stat("/usr/bin/ffmpeg"); err != nil {
		t.Skip("ffmpeg not available")
	}
	dir := t.TempDir()
	store, err := storage.NewLocalStorage(dir)
	if err != nil {
		t.Fatalf("storage: %v", err)
	}
	srcType := video.SourceTypeLocal
	srcPath := integrationFixture
	v := &video.Video{ID: uuid.New(), Filename: "video1.mp4", SourceType: srcType, SourcePath: &srcPath}
	vid := v.ID
	segs := []video_segment.VideoSegment{
		{ID: uuid.New(), VideoID: vid, SegmentIndex: 0, StartTime: 0, EndTime: duration / 2, Duration: duration / 2},
		{ID: uuid.New(), VideoID: vid, SegmentIndex: 1, StartTime: duration / 2, EndTime: duration, Duration: duration - duration/2},
	}
	repo := &fakeRepo{}
	svc := NewService(nil, store, 2*time.Second, "ffmpeg", 30*time.Second, 85)
	svc.repo = repo // real decoder (extractSingleFrame), in-memory persistence
	return svc, repo, store, v, segs
}

func TestIntegrationExplicitTimestamps(t *testing.T) {
	const duration = 20.0
	svc, _, store, v, segs := newIntegrationFixture(t, duration)
	ctx := context.Background()

	budget, err := sampler.BaselineBudget(duration, 2*time.Second, 1.0)
	if err != nil {
		t.Fatalf("budget: %v", err)
	}
	// Non-uniform timestamps: the extractor must return exactly these.
	want := []float64{0.0, 2.7, 6.2, 9.5}
	plan := sampler.PlanWithTimestamps(v.ID, duration, want, sampler.ReasonCoarse, budget)

	frames, err := svc.GenerateForVideoWithPlan(ctx, v, segs, plan, 1280, 720)
	if err != nil {
		t.Fatalf("GenerateForVideoWithPlan: %v", err)
	}
	if len(frames) != len(want) {
		t.Fatalf("expected %d frames got %d", len(want), len(frames))
	}
	for i, f := range frames {
		if math.Abs(f.TimestampSeconds-want[i]) > 1e-9 {
			t.Fatalf("frame %d ts %v want %v", i, f.TimestampSeconds, want[i])
		}
		if f.FrameIndex != i {
			t.Fatalf("frame %d index %d", i, f.FrameIndex)
		}
		if f.Width <= 0 || f.Height <= 0 {
			t.Fatalf("frame %d invalid dims %dx%d", i, f.Width, f.Height)
		}
		if ok, _ := store.Exists(ctx, f.StorageKey); !ok {
			t.Fatalf("frame %d file missing", i)
		}
	}
}

func TestIntegrationBaselineParity(t *testing.T) {
	// The refactored GenerateForVideo (plan-backed) must still produce the
	// legacy uniform grid.
	const duration = 12.0
	svc, _, _, v, segs := newIntegrationFixture(t, duration)
	ctx := context.Background()

	frames, err := svc.GenerateForVideo(ctx, v, segs, duration, 1280, 720)
	if err != nil {
		t.Fatalf("GenerateForVideo: %v", err)
	}
	exp := []float64{0, 2, 4, 6, 8, 10}
	if len(frames) != len(exp) {
		t.Fatalf("expected %d frames got %d", len(exp), len(frames))
	}
	for i, e := range exp {
		if math.Abs(frames[i].TimestampSeconds-e) > 1e-9 {
			t.Fatalf("frame %d ts %v want %v", i, frames[i].TimestampSeconds, e)
		}
	}
}

// TestIntegrationAdaptiveCoarse exercises the Phase 2 path end to end on a
// real fixture: video -> visual probe -> adaptive planner ->
// GenerateForVideoWithPlan (real ffmpeg extraction, in-memory persistence).
// Downstream shape and behavior are unchanged; only the timestamp plan is
// adaptive.
func TestIntegrationAdaptiveCoarse(t *testing.T) {
	const duration = 110.94 // ffprobe duration of video1.mp4
	svc, _, store, v, _ := newIntegrationFixture(t, duration)
	ctx := context.Background()

	// Segments covering the whole video (30s like the pipeline default).
	var segs []video_segment.VideoSegment
	start := 0.0
	for i := 0; start < duration-1e-9; i++ {
		end := start + 30.0
		if end > duration {
			end = duration
		}
		segs = append(segs, video_segment.VideoSegment{
			ID: uuid.New(), VideoID: v.ID, SegmentIndex: i,
			StartTime: start, EndTime: end, Duration: end - start,
		})
		start = end
	}

	planner := sampler.NewAdaptiveCoarsePlanner(sampler.DefaultAdaptiveConfig())
	svc.WithVideoPlanner(planner)

	frames, err := svc.GenerateForVideo(ctx, v, segs, duration, 1270, 720)
	if err != nil {
		t.Fatalf("GenerateForVideo (adaptive): %v", err)
	}
	if len(frames) == 0 {
		t.Fatalf("adaptive plan produced no frames")
	}

	budget, err := sampler.BaselineBudget(duration, 2*time.Second, 1.0)
	if err != nil {
		t.Fatalf("budget: %v", err)
	}
	if len(frames) > budget.Cap {
		t.Fatalf("adaptive selected %d frames, budget cap %d", len(frames), budget.Cap)
	}
	t.Logf("adaptive: %d frames vs baseline budget %d (saved ~%d)",
		len(frames), budget.BaselineCount, budget.BaselineCount-len(frames))

	seen := map[float64]bool{}
	for i, f := range frames {
		ts := f.TimestampSeconds
		if ts < 0 || ts >= duration {
			t.Fatalf("frame %d ts %v out of range", i, ts)
		}
		if i > 0 && ts <= frames[i-1].TimestampSeconds {
			t.Fatalf("frames not strictly sorted at %d", i)
		}
		if seen[ts] {
			t.Fatalf("duplicate ts %v", ts)
		}
		seen[ts] = true
		// VideoFrame shape unchanged for downstream consumers.
		if f.VideoID != v.ID || f.SegmentID == uuid.Nil || f.FrameIndex != i {
			t.Fatalf("frame %d identity broken: %+v", i, f)
		}
		if f.Width <= 0 || f.Height <= 0 {
			t.Fatalf("frame %d invalid dims", i)
		}
		if ok, _ := store.Exists(ctx, f.StorageKey); !ok {
			t.Fatalf("frame %d file missing", i)
		}
	}
}

// TestIntegrationDisagreementFlow is the Phase 3 integration (no refinement):
// adaptive coarse plan -> GenerateForVideoWithPlan (real ffmpeg) -> YOLO
// (fake analyzer with a designed disappearance) -> throwaway disagreement
// scorer -> IntervalScore[]. Production tracking is untouched.
func TestIntegrationDisagreementFlow(t *testing.T) {
	const fixture = "/home/berzz/recallTestVideo/video2.mp4"
	const duration = 11.907
	if _, err := os.Stat(fixture); err != nil {
		t.Skipf("video fixture not found: %v", err)
	}
	if _, err := os.Stat("/usr/bin/ffmpeg"); err != nil {
		t.Skip("ffmpeg not available")
	}
	ctx := context.Background()
	dir := t.TempDir()
	store, err := storage.NewLocalStorage(dir)
	if err != nil {
		t.Fatalf("storage: %v", err)
	}
	srcType := video.SourceTypeLocal
	srcPath := fixture
	v := &video.Video{ID: uuid.New(), Filename: "video2.mp4", SourceType: srcType, SourcePath: &srcPath}
	segs := []video_segment.VideoSegment{
		{ID: uuid.New(), VideoID: v.ID, SegmentIndex: 0, StartTime: 0, EndTime: duration, Duration: duration},
	}
	repo := &fakeRepo{}
	svc := NewService(nil, store, 2*time.Second, "ffmpeg", 30*time.Second, 85)
	svc.repo = repo

	// Phase 2 coarse plan from the real probe.
	planner := sampler.NewAdaptiveCoarsePlanner(sampler.DefaultAdaptiveConfig())
	pres, err := planner.PlanVideo(ctx, v.ID, fixture, duration)
	if err != nil {
		t.Fatalf("PlanVideo: %v", err)
	}
	if pres.Plan.Mode != sampler.ModeAdaptive {
		t.Fatalf("want adaptive plan, got %q (fallback %q)", pres.Plan.Mode, pres.FallbackReason)
	}
	if pres.Plan.Len() < 2 {
		t.Fatalf("need >=2 coarse timestamps, got %d", pres.Plan.Len())
	}

	frames, err := svc.GenerateForVideoWithPlan(ctx, v, segs, pres.Plan, 1280, 720)
	if err != nil {
		t.Fatalf("GenerateForVideoWithPlan: %v", err)
	}

	// Fake YOLO: stable person while ts < 6s, then it disappears.
	analyzer := &detector.SingleFakeAnalyzer{
		Fn: func(fr detector.FrameInput) []detector.DetectionResult {
			if fr.Timestamp < 6.0 {
				return []detector.DetectionResult{
					{Label: "person", Confidence: 0.9, BBoxX: 0.3, BBoxY: 0.3, BBoxWidth: 0.2, BBoxHeight: 0.2},
				}
			}
			return nil
		},
	}
	var detInputs []detector.FrameInput
	for _, f := range frames {
		detInputs = append(detInputs, detector.FrameInput{
			FrameID: f.ID, VideoID: f.VideoID, SegmentID: f.SegmentID,
			Timestamp: f.TimestampSeconds, Width: f.Width, Height: f.Height,
			StorageKey: f.StorageKey,
		})
	}
	results, err := analyzer.AnalyzeBatch(ctx, detInputs)
	if err != nil {
		t.Fatalf("AnalyzeBatch: %v", err)
	}

	var tframes []tracker.FrameInput
	byFrame := make(map[uuid.UUID][]tracker.DetectionInput)
	for _, f := range frames {
		tframes = append(tframes, tracker.FrameInput{FrameID: f.ID, Timestamp: f.TimestampSeconds})
		for _, r := range results[f.ID] {
			byFrame[f.ID] = append(byFrame[f.ID], tracker.DetectionInput{
				ID: uuid.New(), Label: r.Label, Confidence: r.Confidence,
				BBoxX: r.BBoxX, BBoxY: r.BBoxY, BBoxWidth: r.BBoxWidth, BBoxHeight: r.BBoxHeight,
				FrameID: f.ID, Timestamp: f.TimestampSeconds,
			})
		}
	}
	scores, err := sampler.ScoreIntervals(ctx, tframes, byFrame, sampler.DefaultDisagreementConfig())
	if err != nil {
		t.Fatalf("ScoreIntervals: %v", err)
	}
	if len(scores) != len(frames)-1 {
		t.Fatalf("want %d interval scores, got %d", len(frames)-1, len(scores))
	}
	// The designed disappearance must surface as a death in exactly the
	// interval crossing 6s.
	deathIntervals := 0
	for i, s := range scores {
		if s.StartTimestamp >= s.EndTimestamp || s.GapSeconds <= 0 {
			t.Fatalf("interval %d malformed: %+v", i, s)
		}
		if i > 0 && s.StartTimestamp != scores[i-1].EndTimestamp {
			t.Fatalf("intervals not contiguous at %d", i)
		}
		if s.StartTimestamp < 6.0 && s.EndTimestamp >= 6.0 {
			deathIntervals++
			if s.Deaths != 1 {
				t.Fatalf("crossing interval must record the death: %+v", s)
			}
		}
	}
	if deathIntervals != 1 {
		t.Fatalf("want exactly 1 interval crossing 6s, got %d", deathIntervals)
	}
	again, err := sampler.ScoreIntervals(ctx, tframes, byFrame, sampler.DefaultDisagreementConfig())
	if err != nil {
		t.Fatalf("ScoreIntervals: %v", err)
	}
	if !reflect.DeepEqual(scores, again) {
		t.Fatalf("scorer nondeterministic")
	}
	t.Logf("disagreement flow: %d coarse frames -> %d intervals, death captured", len(frames), len(scores))
}

// TestIntegrationRefinementFlow is the Phase 4 end-to-end integration on a
// real fixture (no refinement in production yet):
// adaptive coarse plan -> coarse extraction -> fake YOLO -> disagreement ->
// priority -> midpoint insertion -> new extraction (real ffmpeg) -> new YOLO
// -> rescore -> final plan -> production ByteTrack once on the final
// sequence. Production tracking state is never used for planning.
func TestIntegrationRefinementFlow(t *testing.T) {
	const fixture = "/home/berzz/recallTestVideo/video2.mp4"
	const duration = 11.907
	if _, err := os.Stat(fixture); err != nil {
		t.Skipf("video fixture not found: %v", err)
	}
	if _, err := os.Stat("/usr/bin/ffmpeg"); err != nil {
		t.Skip("ffmpeg not available")
	}
	ctx := context.Background()
	dir := t.TempDir()
	store, err := storage.NewLocalStorage(dir)
	if err != nil {
		t.Fatalf("storage: %v", err)
	}
	srcType := video.SourceTypeLocal
	srcPath := fixture
	v := &video.Video{ID: uuid.New(), Filename: "video2.mp4", SourceType: srcType, SourcePath: &srcPath}
	segs := []video_segment.VideoSegment{
		{ID: uuid.New(), VideoID: v.ID, SegmentIndex: 0, StartTime: 0, EndTime: duration, Duration: duration},
	}
	repo := &fakeRepo{}
	svc := NewService(nil, store, 2*time.Second, "ffmpeg", 30*time.Second, 85)
	svc.repo = repo

	// 1. Coarse plan from the real probe.
	planner := sampler.NewAdaptiveCoarsePlanner(sampler.DefaultAdaptiveConfig())
	pres, err := planner.PlanVideo(ctx, v.ID, fixture, duration)
	if err != nil {
		t.Fatalf("PlanVideo: %v", err)
	}
	if pres.Plan.Mode != sampler.ModeAdaptive {
		t.Fatalf("want adaptive plan, got %q", pres.Plan.Mode)
	}
	coarseTs := pres.Plan.Timestamps()
	if len(coarseTs) < 2 {
		t.Fatalf("need >=2 coarse timestamps, got %v", coarseTs)
	}

	// 2. Coarse extraction (real ffmpeg).
	coarseFrames, err := svc.GenerateForVideoWithPlan(ctx, v, segs, pres.Plan, 1280, 720)
	if err != nil {
		t.Fatalf("GenerateForVideoWithPlan: %v", err)
	}

	// 3. Fake YOLO with a designed disappearance at 6s.
	analyzer := &detector.SingleFakeAnalyzer{
		Fn: func(fr detector.FrameInput) []detector.DetectionResult {
			if fr.Timestamp < 6.0 {
				return []detector.DetectionResult{
					{Label: "person", Confidence: 0.9, BBoxX: 0.3, BBoxY: 0.3, BBoxWidth: 0.2, BBoxHeight: 0.2},
				}
			}
			return nil
		},
	}
	analyze := func(frames []VideoFrame) (map[uuid.UUID][]tracker.DetectionInput, error) {
		var inputs []detector.FrameInput
		for _, f := range frames {
			inputs = append(inputs, detector.FrameInput{
				FrameID: f.ID, VideoID: f.VideoID, SegmentID: f.SegmentID,
				Timestamp: f.TimestampSeconds, Width: f.Width, Height: f.Height,
				StorageKey: f.StorageKey,
			})
		}
		results, err := analyzer.AnalyzeBatch(ctx, inputs)
		if err != nil {
			return nil, err
		}
		byFrame := make(map[uuid.UUID][]tracker.DetectionInput)
		for _, f := range frames {
			for _, r := range results[f.ID] {
				byFrame[f.ID] = append(byFrame[f.ID], tracker.DetectionInput{
					ID: uuid.New(), Label: r.Label, Confidence: r.Confidence,
					BBoxX: r.BBoxX, BBoxY: r.BBoxY, BBoxWidth: r.BBoxWidth, BBoxHeight: r.BBoxHeight,
					FrameID: f.ID, Timestamp: f.TimestampSeconds,
				})
			}
		}
		return byFrame, nil
	}
	coarseByFrame, err := analyze(coarseFrames)
	if err != nil {
		t.Fatalf("analyze coarse: %v", err)
	}

	// 4-6. Refine: midpoints extracted for real, YOLO only on new frames.
	budget, err := sampler.BaselineBudget(duration, 2*time.Second, 1.0)
	if err != nil {
		t.Fatalf("budget: %v", err)
	}
	yoloFrames := len(coarseFrames)
	var tframes []tracker.FrameInput
	for _, f := range coarseFrames {
		tframes = append(tframes, tracker.FrameInput{FrameID: f.ID, Timestamp: f.TimestampSeconds})
	}
	refiner := sampler.NewRefiner(sampler.DefaultRefinementConfig())
	res, err := refiner.Refine(ctx, sampler.RefinementInput{
		Plan: pres.Plan, Probe: pres.Probe, Frames: tframes, Detections: coarseByFrame,
	}, func(ctx context.Context, midpoints []float64) ([]sampler.MidpointDetections, error) {
		// No already-processed timestamp may be re-extracted/re-YOLO'd.
		for _, m := range midpoints {
			for _, c := range coarseTs {
				if math.Abs(m-c) < 1e-6 {
					t.Fatalf("refinement re-requested coarse timestamp %v", m)
				}
			}
		}
		newFrames, err := svc.ExtractAdditional(ctx, v, segs, midpoints, duration, 1280, 720)
		if err != nil {
			return nil, err
		}
		if len(newFrames) != len(midpoints) {
			t.Fatalf("want %d midpoint frames, got %d", len(midpoints), len(newFrames))
		}
		for _, f := range newFrames {
			if ok, _ := store.Exists(ctx, f.StorageKey); !ok {
				t.Fatalf("midpoint file missing: %s", f.StorageKey)
			}
		}
		yoloFrames += len(newFrames)
		if yoloFrames > budget.Cap {
			t.Fatalf("YOLO budget exceeded mid-refinement: %d > %d", yoloFrames, budget.Cap)
		}
		md, err := analyze(newFrames)
		if err != nil {
			return nil, err
		}
		var out []sampler.MidpointDetections
		for _, f := range newFrames {
			out = append(out, sampler.MidpointDetections{Timestamp: f.TimestampSeconds, Detections: md[f.ID]})
		}
		return out, nil
	})
	if err != nil {
		t.Fatalf("Refine: %v", err)
	}

	// 7. Final plan checks.
	if res.FramesAdded < 1 {
		t.Fatalf("designed death must trigger refinement, added %d", res.FramesAdded)
	}
	finalTs := res.Plan.Timestamps()
	for i := 1; i < len(finalTs); i++ {
		if finalTs[i]-finalTs[i-1] < 1e-6 {
			t.Fatalf("final plan not sorted/deduplicated: %v", finalTs)
		}
	}
	for _, ts := range finalTs {
		if ts < 0 || ts >= duration {
			t.Fatalf("final ts out of range: %v", ts)
		}
	}
	if yoloFrames != len(finalTs) {
		t.Fatalf("YOLO frames %d != final plan %d", yoloFrames, len(finalTs))
	}
	if yoloFrames > budget.Cap {
		t.Fatalf("total YOLO frames %d exceed budget %d", yoloFrames, budget.Cap)
	}
	midWithDets := 0
	for _, e := range res.Plan.Entries {
		if e.Reason == sampler.ReasonRefinement {
			midWithDets++
		}
	}
	if midWithDets == 0 {
		t.Fatalf("no refinement entries in final plan")
	}

	// 8. Production ByteTrack once on the final sequence; planning must not
	// have mutated it (deterministic re-run matches).
	prod, err := tracker.New("bytetrack", 0.5, 0.1, 0.3, 5, true, 2)
	if err != nil {
		t.Fatalf("tracker.New: %v", err)
	}
	allFrames, err := svc.GetByVideoID(ctx, v.ID)
	if err != nil {
		t.Fatalf("GetByVideoID: %v", err)
	}
	var pframes []tracker.FrameInput
	pbyFrame := make(map[uuid.UUID][]tracker.DetectionInput)
	for _, f := range allFrames {
		pframes = append(pframes, tracker.FrameInput{FrameID: f.ID, Timestamp: f.TimestampSeconds})
	}
	// Re-run fake YOLO over all final frames for the production pass input.
	pmd, err := analyze(allFrames)
	if err != nil {
		t.Fatalf("analyze final: %v", err)
	}
	for k, ds := range pmd {
		pbyFrame[k] = ds
	}
	first, err := prod.Track(ctx, pframes, pbyFrame)
	if err != nil {
		t.Fatalf("production Track: %v", err)
	}
	second, err := prod.Track(ctx, pframes, pbyFrame)
	if err != nil {
		t.Fatalf("production Track: %v", err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("production tracker mutated by refinement flow")
	}
	t.Logf("refinement flow: %d coarse + %d refined = %d YOLO frames (cap %d), %d rounds",
		len(coarseTs), res.FramesAdded, yoloFrames, budget.Cap, res.Rounds)
}
