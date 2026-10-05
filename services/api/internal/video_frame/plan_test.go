package video_frame

import (
	"bytes"
	"context"
	"image"
	"image/color"
	"image/jpeg"
	"math"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/berzz26/recall/services/api/internal/sampler"
	"github.com/berzz26/recall/services/api/internal/storage"
	"github.com/berzz26/recall/services/api/internal/video"
	"github.com/berzz26/recall/services/api/internal/video_segment"
)

// fakeRepo is an in-memory frameRepository: no database required.
type fakeRepo struct {
	frames []VideoFrame
}

func (f *fakeRepo) GetByVideoID(_ context.Context, videoID uuid.UUID) ([]VideoFrame, error) {
	var out []VideoFrame
	for _, fr := range f.frames {
		if fr.VideoID == videoID {
			out = append(out, fr)
		}
	}
	if out == nil {
		out = []VideoFrame{}
	}
	return out, nil
}

func (f *fakeRepo) DeleteByVideoID(_ context.Context, videoID uuid.UUID) error {
	kept := f.frames[:0]
	for _, fr := range f.frames {
		if fr.VideoID != videoID {
			kept = append(kept, fr)
		}
	}
	f.frames = kept
	return nil
}

func (f *fakeRepo) CreateBatch(_ context.Context, frames []VideoFrame) ([]VideoFrame, error) {
	var saved []VideoFrame
	for _, fr := range frames {
		if fr.ID == uuid.Nil {
			fr.ID = uuid.New()
		}
		f.frames = append(f.frames, fr)
		saved = append(saved, fr)
	}
	if saved == nil {
		saved = []VideoFrame{}
	}
	return saved, nil
}

// stubJPEG returns minimal valid JPEG bytes with the given dimensions.
func stubJPEG(t *testing.T, w, h int) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.Set(x, y, color.RGBA{uint8(x % 256), uint8(y % 256), 128, 255})
		}
	}
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, img, &jpeg.Options{Quality: 80}); err != nil {
		t.Fatalf("encode jpeg: %v", err)
	}
	return buf.Bytes()
}

type planTestFixture struct {
	svc      *Service
	repo     *fakeRepo
	store    storage.Storage
	dir      string
	vid      *video.Video
	segments []video_segment.VideoSegment
	calls    *[]float64 // timestamps seen by the stub decoder
}

// newPlanFixture builds a Service whose decoder is stubbed (no ffmpeg)
// and whose persistence is in-memory (no database).
func newPlanFixture(t *testing.T, duration float64) *planTestFixture {
	t.Helper()
	dir := t.TempDir()
	store, err := storage.NewLocalStorage(dir)
	if err != nil {
		t.Fatalf("storage: %v", err)
	}
	// Stand-in source file: the stub decoder ignores content, but path
	// resolution still stats it like the real pipeline.
	src := filepath.Join(dir, "src.mp4")
	if err := os.WriteFile(src, []byte("fake-video-bytes"), 0644); err != nil {
		t.Fatalf("src: %v", err)
	}
	srcType := video.SourceTypeLocal
	v := &video.Video{ID: uuid.New(), Filename: "test.mp4", SourceType: srcType, SourcePath: &src}

	vid := v.ID
	segs := []video_segment.VideoSegment{
		{ID: uuid.New(), VideoID: vid, SegmentIndex: 0, StartTime: 0, EndTime: duration / 2, Duration: duration / 2},
		{ID: uuid.New(), VideoID: vid, SegmentIndex: 1, StartTime: duration / 2, EndTime: duration, Duration: duration - duration/2},
	}

	repo := &fakeRepo{}
	svc := NewService(nil, store, 2*time.Second, "ffmpeg", 30*time.Second, 85)
	svc.repo = repo
	var calls []float64
	payload := stubJPEG(t, 8, 6)
	svc.extractOne = func(_ context.Context, _, _ string, ts float64, _ int) ([]byte, error) {
		calls = append(calls, ts)
		return payload, nil
	}
	return &planTestFixture{svc: svc, repo: repo, store: store, dir: dir, vid: v, segments: segs, calls: &calls}
}

func TestPlannerExtractorContract(t *testing.T) {
	const duration = 10.0
	fx := newPlanFixture(t, duration)
	ctx := context.Background()

	planner := sampler.NewBaselinePlanner(2*time.Second, 1.0)
	plan, err := planner.Plan(fx.vid.ID, duration)
	if err != nil {
		t.Fatalf("Plan: %v", err)
	}

	frames, err := fx.svc.GenerateForVideoWithPlan(ctx, fx.vid, fx.segments, plan, 1280, 720)
	if err != nil {
		t.Fatalf("GenerateForVideoWithPlan: %v", err)
	}
	if len(frames) != plan.Len() {
		t.Fatalf("expected %d frames got %d", plan.Len(), len(frames))
	}
	want := plan.Timestamps()
	if len(*fx.calls) != len(want) {
		t.Fatalf("decoder called %d times, want %d (batch, one call per timestamp)", len(*fx.calls), len(want))
	}
	for i, f := range frames {
		// VideoFrame shape unchanged for downstream consumers.
		if f.VideoID != fx.vid.ID {
			t.Fatalf("frame %d video mismatch", i)
		}
		if f.SegmentID == uuid.Nil {
			t.Fatalf("frame %d missing segment", i)
		}
		if f.FrameIndex != i {
			t.Fatalf("frame %d index %d", i, f.FrameIndex)
		}
		if math.Abs(f.TimestampSeconds-want[i]) > 1e-9 {
			t.Fatalf("frame %d ts %v want %v", i, f.TimestampSeconds, want[i])
		}
		if f.Width <= 0 || f.Height <= 0 {
			t.Fatalf("frame %d dims %dx%d", i, f.Width, f.Height)
		}
		if ok, _ := fx.store.Exists(ctx, f.StorageKey); !ok {
			t.Fatalf("frame %d file missing %s", i, f.StorageKey)
		}
		if math.Abs((*fx.calls)[i]-want[i]) > 1e-9 {
			t.Fatalf("decoder call %d got %v want %v", i, (*fx.calls)[i], want[i])
		}
	}
}

func TestExtractorSortsAndDedupes(t *testing.T) {
	const duration = 20.0
	fx := newPlanFixture(t, duration)
	ctx := context.Background()

	budget, err := sampler.BaselineBudget(duration, 2*time.Second, 1.0)
	if err != nil {
		t.Fatalf("budget: %v", err)
	}
	plan := sampler.PlanWithTimestamps(fx.vid.ID, duration,
		[]float64{9.5, 2.7, 2.7, 0.0, 6.2, 0.0}, sampler.ReasonCoarse, budget)

	frames, err := fx.svc.GenerateForVideoWithPlan(ctx, fx.vid, fx.segments, plan, 1280, 720)
	if err != nil {
		t.Fatalf("GenerateForVideoWithPlan: %v", err)
	}
	var got []float64
	for _, f := range frames {
		got = append(got, f.TimestampSeconds)
	}
	exp := []float64{0.0, 2.7, 6.2, 9.5}
	if len(got) != len(exp) {
		t.Fatalf("expected %v got %v", exp, got)
	}
	for i := range exp {
		if math.Abs(got[i]-exp[i]) > 1e-9 {
			t.Fatalf("mismatch %d: %v vs %v", i, got[i], exp[i])
		}
	}
}

func TestExtractorBoundaries(t *testing.T) {
	const duration = 10.0
	fx := newPlanFixture(t, duration)
	ctx := context.Background()

	budget, err := sampler.BaselineBudget(duration, 2*time.Second, 1.0)
	if err != nil {
		t.Fatalf("budget: %v", err)
	}
	plan := sampler.PlanWithTimestamps(fx.vid.ID, duration,
		[]float64{-3.0, 0.0, 9.999, 10.0, 15.0}, sampler.ReasonCoarse, budget)

	frames, err := fx.svc.GenerateForVideoWithPlan(ctx, fx.vid, fx.segments, plan, 1280, 720)
	if err != nil {
		t.Fatalf("GenerateForVideoWithPlan: %v", err)
	}
	if len(frames) != 2 {
		t.Fatalf("expected [0, 9.999], got %d frames", len(frames))
	}
	if frames[0].TimestampSeconds != 0 || math.Abs(frames[1].TimestampSeconds-9.999) > 1e-9 {
		t.Fatalf("unexpected timestamps %v %v", frames[0].TimestampSeconds, frames[1].TimestampSeconds)
	}
}

func TestExtractorEmptyPlan(t *testing.T) {
	const duration = 10.0
	fx := newPlanFixture(t, duration)
	ctx := context.Background()

	budget, err := sampler.BaselineBudget(duration, 2*time.Second, 1.0)
	if err != nil {
		t.Fatalf("budget: %v", err)
	}
	plan := sampler.PlanWithTimestamps(fx.vid.ID, duration, nil, sampler.ReasonCoarse, budget)

	frames, err := fx.svc.GenerateForVideoWithPlan(ctx, fx.vid, fx.segments, plan, 1280, 720)
	if err != nil {
		t.Fatalf("GenerateForVideoWithPlan: %v", err)
	}
	if len(frames) != 0 {
		t.Fatalf("expected 0 frames got %d", len(frames))
	}
	if len(*fx.calls) != 0 {
		t.Fatalf("decoder should not run for empty plan, got %d calls", len(*fx.calls))
	}
}

func TestExtractorNeverExceedsPlan(t *testing.T) {
	const duration = 200.0
	fx := newPlanFixture(t, duration)
	ctx := context.Background()

	// Hand-built oversized plan under a small budget: extractor must cap.
	budget := sampler.Budget{BaselineInterval: 2 * time.Second, Beta: 1.0, BaselineCount: 100, Cap: 5}
	var many []float64
	for i := 0; i < 100; i++ {
		many = append(many, float64(i))
	}
	plan := sampler.PlanWithTimestamps(fx.vid.ID, duration, many, sampler.ReasonRefinement, budget)

	frames, err := fx.svc.GenerateForVideoWithPlan(ctx, fx.vid, fx.segments, plan, 1280, 720)
	if err != nil {
		t.Fatalf("GenerateForVideoWithPlan: %v", err)
	}
	if len(frames) != 5 {
		t.Fatalf("expected hard cap of 5 frames, got %d", len(frames))
	}
	if len(frames) > len(plan.Entries) {
		t.Fatalf("frame count exceeds plan")
	}
}

func TestGenerateForVideoUsesBaselinePlan(t *testing.T) {
	const duration = 10.0
	fx := newPlanFixture(t, duration)
	ctx := context.Background()

	frames, err := fx.svc.GenerateForVideo(ctx, fx.vid, fx.segments, duration, 1280, 720)
	if err != nil {
		t.Fatalf("GenerateForVideo: %v", err)
	}
	// Legacy grid for 10s at 2s: [0 2 4 6 8].
	exp := []float64{0, 2, 4, 6, 8}
	if len(frames) != len(exp) {
		t.Fatalf("expected %d frames got %d", len(exp), len(frames))
	}
	for i, e := range exp {
		if math.Abs(frames[i].TimestampSeconds-e) > 1e-9 {
			t.Fatalf("frame %d ts %v want %v", i, frames[i].TimestampSeconds, e)
		}
	}
}

// TestExtractAdditionalSkipsExisting verifies additive midpoint extraction:
// existing timestamps are not re-decoded, FrameIndex/storage keys continue,
// and duplicates within the request are collapsed.
func TestExtractAdditionalSkipsExisting(t *testing.T) {
	const duration = 10.0
	fx := newPlanFixture(t, duration)
	ctx := context.Background()

	budget, err := sampler.BaselineBudget(duration, 2*time.Second, 1.0)
	if err != nil {
		t.Fatalf("budget: %v", err)
	}
	plan := sampler.PlanWithTimestamps(fx.vid.ID, duration, []float64{0, 4, 8}, sampler.ReasonCoarse, budget)
	if _, err := fx.svc.GenerateForVideoWithPlan(ctx, fx.vid, fx.segments, plan, 1280, 720); err != nil {
		t.Fatalf("GenerateForVideoWithPlan: %v", err)
	}
	*fx.calls = nil // reset decoder call recording

	added, err := fx.svc.ExtractAdditional(ctx, fx.vid, fx.segments, []float64{2, 4, 4.0, 6}, duration, 1280, 720)
	if err != nil {
		t.Fatalf("ExtractAdditional: %v", err)
	}
	if len(added) != 2 {
		t.Fatalf("want 2 new frames (2 and 6), got %d", len(added))
	}
	if len(*fx.calls) != 2 || (*fx.calls)[0] != 2 || (*fx.calls)[1] != 6 {
		t.Fatalf("decoder must run only for new timestamps, got %v", *fx.calls)
	}
	if added[0].FrameIndex != 3 || added[1].FrameIndex != 4 {
		t.Fatalf("FrameIndex must continue after existing: %+v", added)
	}
	all, err := fx.svc.GetByVideoID(ctx, fx.vid.ID)
	if err != nil {
		t.Fatalf("GetByVideoID: %v", err)
	}
	if len(all) != 5 {
		t.Fatalf("want 5 total frames, got %d", len(all))
	}
	keys := map[string]bool{}
	for _, f := range all {
		if keys[f.StorageKey] {
			t.Fatalf("duplicate storage key %s", f.StorageKey)
		}
		keys[f.StorageKey] = true
	}
}
