package video_frame

import (
	"context"
	"os"
	"testing"
	"time"

	"github.com/berzz26/recall/pkg/database"
	"github.com/berzz26/recall/services/api/internal/video"
	"github.com/berzz26/recall/services/api/internal/video_segment"
	"github.com/google/uuid"
)

func newTestRepo(t *testing.T) (*Repository, *database.Service) {
	t.Helper()
	url := os.Getenv("DATABASE_URL")
	if url == "" {
		url = "postgres://recall:recall@localhost:5436/recall?sslmode=disable"
	}
	db, err := database.New(url)
	if err != nil {
		t.Skipf("db not available: %v", err)
	}
	t.Cleanup(func() { db.Close() })
	return NewRepository(db.DB), db
}

func createVideoAndSegment(t *testing.T, ctx context.Context, db *database.Service) (uuid.UUID, uuid.UUID) {
	t.Helper()
	repo := video.NewRepository(db.DB)
	v, err := repo.Create(ctx, "test.mp4", uuid.New().String(), "video/mp4", 100, video.SourceTypeUpload, nil)
	if err != nil {
		t.Fatalf("Create video: %v", err)
	}
	segRepo := video_segment.NewRepository(db.DB)
	segs, err := video_segment.BuildSegments(v.ID, 60, 30*time.Second)
	if err != nil {
		t.Fatalf("BuildSegments: %v", err)
	}
	saved, err := segRepo.ReplaceForVideo(ctx, v.ID, segs)
	if err != nil {
		t.Fatalf("ReplaceForVideo: %v", err)
	}
	return v.ID, saved[0].ID
}

func TestCreateGetDelete(t *testing.T) {
	repo, db := newTestRepo(t)
	ctx := context.Background()
	vid, segID := createVideoAndSegment(t, ctx, db)
	defer video.NewRepository(db.DB).Delete(ctx, vid)

	f := VideoFrame{VideoID: vid, SegmentID: segID, FrameIndex: 0, TimestampSeconds: 0, StorageKey: "videos/" + vid.String() + "/frames/000000.jpg", Width: 1280, Height: 720}
	created, err := repo.Create(ctx, f)
	if err != nil {
		t.Fatalf("Create: %v", err)
	}
	if created.FrameIndex != 0 {
		t.Fatalf("index mismatch")
	}
	list, err := repo.GetByVideoID(ctx, vid)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	if len(list) != 1 {
		t.Fatalf("expected 1 got %d", len(list))
	}
	if err := repo.DeleteByVideoID(ctx, vid); err != nil {
		t.Fatalf("Delete: %v", err)
	}
	list2, _ := repo.GetByVideoID(ctx, vid)
	if len(list2) != 0 {
		t.Fatalf("expected 0 after delete got %d", len(list2))
	}
}

func TestCreateBatch(t *testing.T) {
	repo, db := newTestRepo(t)
	ctx := context.Background()
	vid, segID := createVideoAndSegment(t, ctx, db)
	defer video.NewRepository(db.DB).Delete(ctx, vid)

	frames := []VideoFrame{
		{VideoID: vid, SegmentID: segID, FrameIndex: 0, TimestampSeconds: 0, StorageKey: "videos/" + vid.String() + "/frames/000000.jpg", Width: 640, Height: 480},
		{VideoID: vid, SegmentID: segID, FrameIndex: 1, TimestampSeconds: 2, StorageKey: "videos/" + vid.String() + "/frames/000001.jpg", Width: 640, Height: 480},
	}
	saved, err := repo.CreateBatch(ctx, frames)
	if err != nil {
		t.Fatalf("CreateBatch: %v", err)
	}
	if len(saved) != 2 {
		t.Fatalf("expected 2 got %d", len(saved))
	}
	list, _ := repo.GetByVideoID(ctx, vid)
	if len(list) != 2 {
		t.Fatalf("expected 2 got %d", len(list))
	}
	if list[0].FrameIndex != 0 || list[1].FrameIndex != 1 {
		t.Fatalf("ordering mismatch")
	}
}

func TestUniqueConstraint(t *testing.T) {
	repo, db := newTestRepo(t)
	ctx := context.Background()
	vid, segID := createVideoAndSegment(t, ctx, db)
	defer video.NewRepository(db.DB).Delete(ctx, vid)

	f := VideoFrame{VideoID: vid, SegmentID: segID, FrameIndex: 0, TimestampSeconds: 0, StorageKey: "videos/" + vid.String() + "/frames/000000.jpg", Width: 100, Height: 100}
	if _, err := repo.Create(ctx, f); err != nil {
		t.Fatalf("first create: %v", err)
	}
	if _, err := repo.Create(ctx, f); err == nil {
		t.Fatalf("expected unique violation")
	}
}

func TestCascadeDelete(t *testing.T) {
	repo, db := newTestRepo(t)
	ctx := context.Background()
	vid, segID := createVideoAndSegment(t, ctx, db)

	f := VideoFrame{VideoID: vid, SegmentID: segID, FrameIndex: 0, TimestampSeconds: 0, StorageKey: "videos/" + vid.String() + "/frames/000000.jpg", Width: 320, Height: 240}
	if _, err := repo.Create(ctx, f); err != nil {
		t.Fatalf("Create: %v", err)
	}
	if err := video.NewRepository(db.DB).Delete(ctx, vid); err != nil {
		t.Fatalf("Delete video: %v", err)
	}
	list, _ := repo.GetByVideoID(ctx, vid)
	if len(list) != 0 {
		t.Fatalf("expected cascade delete got %d", len(list))
	}
}

func TestGetByVideoIDOrder(t *testing.T) {
	repo, db := newTestRepo(t)
	ctx := context.Background()
	vid, segID := createVideoAndSegment(t, ctx, db)
	defer video.NewRepository(db.DB).Delete(ctx, vid)

	for i := 2; i >= 0; i-- {
		f := VideoFrame{VideoID: vid, SegmentID: segID, FrameIndex: i, TimestampSeconds: float64(i * 2), StorageKey: "videos/" + vid.String() + "/frames/" + string(rune('0'+i)) + ".jpg", Width: 100, Height: 100}
		if _, err := repo.Create(ctx, f); err != nil {
			t.Fatalf("Create %d: %v", i, err)
		}
	}
	list, _ := repo.GetByVideoID(ctx, vid)
	if len(list) != 3 || list[0].FrameIndex != 0 || list[2].FrameIndex != 2 {
		t.Fatalf("ordering failed %v", list)
	}
}

func TestGetByVideoIDChronologicalAfterRefinement(t *testing.T) {
	// Refinement midpoints are appended with insertion-order FrameIndex
	// values that do not match temporal order (e.g. coarse [0s, 10s] take
	// indices 0,1, then midpoint 5s takes index 2). Reads must still come
	// back in timestamp order.
	repo, db := newTestRepo(t)
	ctx := context.Background()
	vid, segID := createVideoAndSegment(t, ctx, db)
	defer video.NewRepository(db.DB).Delete(ctx, vid)

	seed := []VideoFrame{
		{VideoID: vid, SegmentID: segID, FrameIndex: 0, TimestampSeconds: 0, StorageKey: "videos/" + vid.String() + "/frames/000000.jpg", Width: 100, Height: 100},
		{VideoID: vid, SegmentID: segID, FrameIndex: 1, TimestampSeconds: 10, StorageKey: "videos/" + vid.String() + "/frames/000001.jpg", Width: 100, Height: 100},
		{VideoID: vid, SegmentID: segID, FrameIndex: 2, TimestampSeconds: 5, StorageKey: "videos/" + vid.String() + "/frames/000002.jpg", Width: 100, Height: 100},
	}
	if _, err := repo.CreateBatch(ctx, seed); err != nil {
		t.Fatalf("CreateBatch: %v", err)
	}
	got, err := repo.GetByVideoID(ctx, vid)
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	wantTs := []float64{0, 5, 10}
	if len(got) != len(wantTs) {
		t.Fatalf("expected %d got %d", len(wantTs), len(got))
	}
	for i, want := range wantTs {
		if got[i].TimestampSeconds != want {
			t.Fatalf("position %d: want ts %v got ts %v (index %d)", i, want, got[i].TimestampSeconds, got[i].FrameIndex)
		}
	}
	segList, err := repo.GetBySegmentID(ctx, segID)
	if err != nil {
		t.Fatalf("GetBySegmentID: %v", err)
	}
	for i, want := range wantTs {
		if segList[i].TimestampSeconds != want {
			t.Fatalf("segment position %d: want ts %v got ts %v", i, want, segList[i].TimestampSeconds)
		}
	}
}
