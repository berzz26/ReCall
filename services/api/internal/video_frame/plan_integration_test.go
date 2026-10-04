package video_frame

// Integration coverage for the planner -> extractor contract against a real
// video fixture and the real ffmpeg seeking decoder (no mocks).
// Skips gracefully when the fixture or ffmpeg is unavailable, mirroring the
// existing processing/frame_test.go conventions.

import (
	"context"
	"math"
	"os"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/berzz26/recall/services/api/internal/sampler"
	"github.com/berzz26/recall/services/api/internal/storage"
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
