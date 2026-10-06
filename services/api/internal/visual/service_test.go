package visual

import (
	"bytes"
	"context"
	"os"
	"testing"
	"time"

	"github.com/berzz26/recall/pkg/database"
	"github.com/berzz26/recall/services/api/internal/detection"
	"github.com/berzz26/recall/services/api/internal/detector"
	"github.com/berzz26/recall/services/api/internal/storage"
	"github.com/berzz26/recall/services/api/internal/video"
	"github.com/berzz26/recall/services/api/internal/video_frame"
	"github.com/berzz26/recall/services/api/internal/video_media"
	"github.com/berzz26/recall/services/api/internal/video_segment"
	"github.com/google/uuid"
)

func newTestVisualDeps(t *testing.T, analyzer detector.VisualAnalyzer) (*Service, *video_frame.Service, *video_segment.Service, *video_media.Service, *database.Service, storage.Storage, string) {
	t.Helper()
	url := os.Getenv("DATABASE_URL")
	if url == "" {
		url = "postgres://recall:recall@localhost:5436/recall?sslmode=disable"
	}
	db, err := database.New(url)
	if err != nil {
		t.Skipf("db not available: %v", err)
	}
	dir := t.TempDir()
	store, _ := storage.NewLocalStorage(dir)
	detRepo := detection.NewRepository(db.DB)
	frameRepo := video_frame.NewRepository(db.DB)
	vs := NewService(detRepo, frameRepo, store, analyzer, 0.25, "test-detector", "1")
	frameSvc := video_frame.NewService(frameRepo, store, 2*time.Second, "ffmpeg", 30*time.Second, 85)
	segSvc := video_segment.NewService(video_segment.NewRepository(db.DB), 30*time.Second)
	mediaSvc := video_media.NewService(video_media.NewRepository(db.DB))
	return vs, frameSvc, segSvc, mediaSvc, db, store, dir
}

func TestFakeAnalyzerPersists(t *testing.T) {
	fake := &detector.FakeVisualAnalyzer{}
	vs, frameSvc, segSvc, mediaSvc, db, store, dir := newTestVisualDeps(t, fake)
	defer func() { db.Close(); os.RemoveAll(dir) }()
	ctx := context.Background()
	videoSvc := video.NewServiceWithConfig(video.NewRepository(db.DB), store, 1<<30)
	v, err := videoSvc.CreateVideo(ctx, "test.mp4", func() *string { s := uuid.New().String(); return &s }(), func() *string { s := "video/mp4"; return &s }(), func() *int64 { i := int64(100); return &i }(), func() *video.SourceType { s := video.SourceTypeUpload; return &s }(), nil)
	if err != nil {
		t.Fatalf("CreateVideo: %v", err)
	}
	defer videoSvc.DeleteVideo(ctx, v.ID)
	meta := &video_media.MediaMetadata{VideoID: v.ID}
	dur := 10.0
	meta.DurationSeconds = &dur
	w := 1270
	h := 720
	meta.VideoWidth = &w
	meta.VideoHeight = &h
	video_media.NewRepository(db.DB).Upsert(ctx, meta)
	segs, err := segSvc.GenerateForVideo(ctx, v.ID, 10)
	if err != nil {
		t.Fatalf("segments: %v", err)
	}
	frameRepo := video_frame.NewRepository(db.DB)
	jpeg := []byte{0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xD9}
	var frames []video_frame.VideoFrame
	for i := 0; i < 5; i++ {
		ts := float64(i * 2)
		seg := video_frame.FindSegment(segs, ts)
		if seg == nil {
			seg = &segs[0]
		}
		key := "videos/" + v.ID.String() + "/frames/00000" + string(rune('0'+i)) + ".jpg"
		if err := store.Save(ctx, key, bytes.NewReader(jpeg)); err != nil {
			t.Fatalf("save: %v", err)
		}
		f := video_frame.VideoFrame{VideoID: v.ID, SegmentID: seg.ID, FrameIndex: i, TimestampSeconds: ts, StorageKey: key, Width: 640, Height: 480}
		frames = append(frames, f)
	}
	if _, err := frameRepo.CreateBatch(ctx, frames); err != nil {
		t.Fatalf("CreateBatch frames: %v", err)
	}
	defer func() {
		vs.DeleteByVideoID(ctx, v.ID)
		frameSvc.DeleteByVideoID(ctx, v.ID)
		segSvc.DeleteByVideoID(ctx, v.ID)
		mediaSvc.DeleteByVideoID(ctx, v.ID)
	}()

	saved, err := vs.AnalyzeVideo(ctx, v.ID)
	if err != nil {
		t.Fatalf("AnalyzeVideo: %v", err)
	}
	if len(saved) != 10 {
		t.Fatalf("expected 10 got %d", len(saved))
	}
	if fake.Calls != 1 {
		t.Fatalf("expected 1 call got %d", fake.Calls)
	}
	saved2, err := vs.AnalyzeVideo(ctx, v.ID)
	if err != nil {
		t.Fatalf("second: %v", err)
	}
	if len(saved2) != 10 {
		t.Fatalf("idempotency failed %d", len(saved2))
	}
}

func TestThresholdFiltering(t *testing.T) {
	url := os.Getenv("DATABASE_URL")
	if url == "" {
		url = "postgres://recall:recall@localhost:5436/recall?sslmode=disable"
	}
	db, err := database.New(url)
	if err != nil {
		t.Skipf("db not available: %v", err)
	}
	defer db.Close()
	dir := t.TempDir()
	store, _ := storage.NewLocalStorage(dir)
	defer os.RemoveAll(dir)
	ctx := context.Background()
	videoSvc := video.NewServiceWithConfig(video.NewRepository(db.DB), store, 1<<30)
	v, _ := videoSvc.CreateVideo(ctx, "thresh.mp4", func() *string { s := uuid.New().String(); return &s }(), func() *string { s := "video/mp4"; return &s }(), func() *int64 { i := int64(100); return &i }(), func() *video.SourceType { s := video.SourceTypeUpload; return &s }(), nil)
	defer videoSvc.DeleteVideo(ctx, v.ID)
	segSvc := video_segment.NewService(video_segment.NewRepository(db.DB), 30*time.Second)
	segs, _ := segSvc.GenerateForVideo(ctx, v.ID, 10)
	frameRepo := video_frame.NewRepository(db.DB)
	jpeg := []byte{0xFF, 0xD8, 0xFF, 0xD9}
	key := "videos/" + v.ID.String() + "/frames/000000.jpg"
	store.Save(ctx, key, bytes.NewReader(jpeg))
	f := video_frame.VideoFrame{VideoID: v.ID, SegmentID: segs[0].ID, FrameIndex: 0, TimestampSeconds: 0, StorageKey: key, Width: 640, Height: 480}
	savedFrames, _ := frameRepo.CreateBatch(ctx, []video_frame.VideoFrame{f})
	fid := savedFrames[0].ID
	fake := &detector.FakeVisualAnalyzer{Results: map[uuid.UUID][]detector.DetectionResult{
		fid: {{Label: "person", Confidence: 0.1, BBoxX: 0.1, BBoxY: 0.1, BBoxWidth: 0.1, BBoxHeight: 0.1}},
	}}
	vs := NewService(detection.NewRepository(db.DB), frameRepo, store, fake, 0.5, "test", "1")
	saved, err := vs.AnalyzeVideo(ctx, v.ID)
	if err != nil {
		t.Fatalf("Analyze: %v", err)
	}
	if len(saved) != 0 {
		t.Fatalf("threshold should filter, got %d", len(saved))
	}
}

// TestDetectFramesEphemeral verifies planning-only detection: same filtering
// as production AnalyzeVideo, but nothing is persisted to the detection
// table. DB-gated like the other visual tests.
func TestDetectFramesEphemeral(t *testing.T) {
	fake := &detector.FakeVisualAnalyzer{}
	vs, frameSvc, segSvc, mediaSvc, db, store, dir := newTestVisualDeps(t, fake)
	defer func() { db.Close(); os.RemoveAll(dir) }()
	ctx := context.Background()
	videoSvc := video.NewServiceWithConfig(video.NewRepository(db.DB), store, 1<<30)
	v, err := videoSvc.CreateVideo(ctx, "ephemeral.mp4", func() *string { s := uuid.New().String(); return &s }(), func() *string { s := "video/mp4"; return &s }(), func() *int64 { i := int64(100); return &i }(), func() *video.SourceType { s := video.SourceTypeUpload; return &s }(), nil)
	if err != nil {
		t.Fatalf("CreateVideo: %v", err)
	}
	defer videoSvc.DeleteVideo(ctx, v.ID)
	dur := 10.0
	meta := &video_media.MediaMetadata{VideoID: v.ID, DurationSeconds: &dur}
	video_media.NewRepository(db.DB).Upsert(ctx, meta)
	segs, err := segSvc.GenerateForVideo(ctx, v.ID, 10)
	if err != nil {
		t.Fatalf("segments: %v", err)
	}
	frameRepo := video_frame.NewRepository(db.DB)
	jpeg := []byte{0xFF, 0xD8, 0xFF, 0xE0, 0x00, 0x10, 0x4A, 0x46, 0x49, 0x46, 0x00, 0x01, 0x01, 0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x00, 0xFF, 0xD9}
	var frames []video_frame.VideoFrame
	for i := 0; i < 3; i++ {
		ts := float64(i * 2)
		seg := video_frame.FindSegment(segs, ts)
		if seg == nil {
			seg = &segs[0]
		}
		key := "videos/" + v.ID.String() + "/frames/eph000" + string(rune('0'+i)) + ".jpg"
		if err := store.Save(ctx, key, bytes.NewReader(jpeg)); err != nil {
			t.Fatalf("save: %v", err)
		}
		frames = append(frames, video_frame.VideoFrame{VideoID: v.ID, SegmentID: seg.ID, FrameIndex: i, TimestampSeconds: ts, StorageKey: key, Width: 640, Height: 480})
	}
	if _, err := frameRepo.CreateBatch(ctx, frames); err != nil {
		t.Fatalf("CreateBatch frames: %v", err)
	}
	defer func() {
		vs.DeleteByVideoID(ctx, v.ID)
		frameSvc.DeleteByVideoID(ctx, v.ID)
		segSvc.DeleteByVideoID(ctx, v.ID)
		mediaSvc.DeleteByVideoID(ctx, v.ID)
	}()

	got, err := vs.DetectFrames(ctx, v.ID, frames)
	if err != nil {
		t.Fatalf("DetectFrames: %v", err)
	}
	// Fake analyzer returns 2 detections per frame.
	if len(got) != 6 {
		t.Fatalf("expected 6 planning detections, got %d", len(got))
	}
	byFrame := map[uuid.UUID]int{}
	for _, d := range got {
		if d.ID == uuid.Nil {
			t.Fatalf("planning detection needs fresh ID")
		}
		byFrame[d.FrameID]++
	}
	for _, f := range frames {
		if byFrame[f.ID] != 2 {
			t.Fatalf("frame %s: want 2 detections, got %d", f.ID, byFrame[f.ID])
		}
	}
	// Nothing persisted: the detection table must stay empty.
	stored, err := vs.GetByVideoID(ctx, v.ID)
	if err != nil {
		t.Fatalf("GetByVideoID: %v", err)
	}
	if len(stored) != 0 {
		t.Fatalf("planning detections must not persist, found %d", len(stored))
	}
	// Empty input is a no-op, not an error.
	empty, err := vs.DetectFrames(ctx, v.ID, nil)
	if err != nil || len(empty) != 0 {
		t.Fatalf("empty DetectFrames must return empty, got %v %v", empty, err)
	}
}
