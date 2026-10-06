package video

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/berzz26/recall/pkg/database"
	"github.com/berzz26/recall/services/api/internal/storage"
)

func newTestServiceWithStorage(t *testing.T) (*Service, *storage.LocalStorage, func()) {
	t.Helper()
	databaseURL := os.Getenv("DATABASE_URL")
	if databaseURL == "" {
		databaseURL = "postgres://recall:recall@localhost:5436/recall?sslmode=disable"
	}
	db, err := database.New(databaseURL)
	if err != nil {
		t.Skipf("database not available: %v", err)
	}
	dir := t.TempDir()
	store, err := storage.NewLocalStorage(dir)
	if err != nil {
		t.Fatalf("storage failed: %v", err)
	}
	repo := NewRepository(db.DB)
	svc := NewServiceWithStorage(repo, store)
	cleanup := func() {
		db.Close()
		os.RemoveAll(dir)
	}
	return svc, store, cleanup
}

func TestUploadSuccess(t *testing.T) {
	svc, store, cleanup := newTestServiceWithStorage(t)
	defer cleanup()
	ctx := context.Background()

	data := []byte("fake video content for testing")
	hash := sha256.Sum256(data)
	expectedHash := hex.EncodeToString(hash[:])

	reader := bytes.NewReader(data)
	v, err := svc.UploadVideo(ctx, "test.mp4", reader, "video/mp4")
	if err != nil {
		t.Fatalf("UploadVideo failed: %v", err)
	}
	if v.SourceType != SourceTypeUpload {
		t.Fatalf("expected UPLOAD got %s", v.SourceType)
	}
	if v.StorageKey == nil || *v.StorageKey == "" {
		t.Fatalf("storage_key should be set")
	}
	expectedKey := "videos/" + v.ID.String() + "/original.mp4"
	if *v.StorageKey != expectedKey {
		t.Fatalf("expected key %s got %s", expectedKey, *v.StorageKey)
	}
	if v.Status != StatusUploaded {
		t.Fatalf("expected UPLOADED got %s", v.Status)
	}
	if v.ContentHash != expectedHash {
		t.Fatalf("hash mismatch expected %s got %s", expectedHash, v.ContentHash)
	}
	if v.SizeBytes != int64(len(data)) {
		t.Fatalf("size mismatch expected %d got %d", len(data), v.SizeBytes)
	}
	if v.SourcePath != nil {
		t.Fatalf("source_path should be nil for UPLOAD")
	}

	exists, _ := store.Exists(ctx, *v.StorageKey)
	if !exists {
		t.Fatalf("file should exist in storage")
	}

	fetched, err := svc.GetVideo(ctx, v.ID)
	if err != nil {
		t.Fatalf("GetVideo failed: %v", err)
	}
	if fetched.StorageKey == nil || *fetched.StorageKey != expectedKey {
		t.Fatalf("persisted storage_key mismatch")
	}

	if err := svc.DeleteVideo(ctx, v.ID); err != nil {
		t.Fatalf("Delete failed: %v", err)
	}
	exists, _ = store.Exists(ctx, *v.StorageKey)
	if exists {
		t.Fatalf("file should be deleted after DeleteVideo")
	}
}

func TestUploadUnsupportedType(t *testing.T) {
	svc, _, cleanup := newTestServiceWithStorage(t)
	defer cleanup()
	ctx := context.Background()
	_, err := svc.UploadVideo(ctx, "bad.txt", bytes.NewReader([]byte("data")), "text/plain")
	if err == nil {
		t.Fatalf("expected error for unsupported type")
	}
}

// cancelAfterFirstRead yields one chunk, then blocks until ctx is done —
// simulating a client disconnect mid-upload.
type cancelAfterFirstRead struct {
	ctx   context.Context
	first bool
}

func (r *cancelAfterFirstRead) Read(p []byte) (int, error) {
	if !r.first {
		r.first = true
		return copy(p, []byte("partial-data")), nil
	}
	<-r.ctx.Done()
	return 0, r.ctx.Err()
}

func TestUploadCancelledDeletesRow(t *testing.T) {
	svc, _, cleanup := newTestServiceWithStorage(t)
	defer cleanup()
	bg := context.Background()

	before, err := svc.ListVideos(bg, 100, 0)
	if err != nil {
		t.Fatalf("ListVideos failed: %v", err)
	}

	ctx, cancel := context.WithCancel(bg)
	done := make(chan error, 1)
	go func() {
		_, err := svc.UploadVideo(ctx, "cancel.mp4", &cancelAfterFirstRead{ctx: ctx}, "video/mp4")
		done <- err
	}()
	time.Sleep(time.Second)
	cancel()
	if err := <-done; err == nil {
		t.Fatalf("expected error after cancel")
	}

	after, err := svc.ListVideos(bg, 100, 0)
	if err != nil {
		t.Fatalf("ListVideos failed: %v", err)
	}
	if len(after) != len(before) {
		t.Fatalf("cancelled upload left %d extra video row(s)", len(after)-len(before))
	}
}

func TestDeleteLocalDoesNotDeleteSource(t *testing.T) {
	svc, store, cleanup := newTestServiceWithStorage(t)
	defer cleanup()
	ctx := context.Background()

	tmpFile, err := os.CreateTemp("", "localvideo*.mp4")
	if err != nil {
		t.Fatalf("temp file: %v", err)
	}
	tmpFile.Write([]byte("local content"))
	tmpFile.Close()
	path := tmpFile.Name()
	defer os.Remove(path)

	pathCopy := path
	v, err := svc.CreateVideo(ctx, "local.mp4", strPtr2("hash123"), strPtr2("video/mp4"), int64Ptr(123), sourcePtr(SourceTypeLocal), &pathCopy)
	if err != nil {
		t.Fatalf("CreateVideo LOCAL failed: %v", err)
	}
	if v.StorageKey != nil {
		t.Fatalf("storage_key should be nil for LOCAL")
	}

	if err := svc.DeleteVideo(ctx, v.ID); err != nil {
		t.Fatalf("Delete failed: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("source file should not be deleted")
	}
	_ = store
}

func TestIngestLocalDirectory(t *testing.T) {
	svc, _, cleanup := newTestServiceWithStorage(t)
	defer cleanup()
	ctx := context.Background()

	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "a.mp4"), []byte("a"), 0644)
	os.WriteFile(filepath.Join(dir, "b.mkv"), []byte("b"), 0644)
	os.WriteFile(filepath.Join(dir, "c.txt"), []byte("ignore"), 0644)
	sub := filepath.Join(dir, "sub")
	os.MkdirAll(sub, 0755)
	os.WriteFile(filepath.Join(sub, "d.mov"), []byte("d"), 0644)

	count, err := svc.IngestLocalDirectory(ctx, dir)
	if err != nil {
		t.Fatalf("Ingest failed: %v", err)
	}
	if count != 3 {
		t.Fatalf("expected 3 ingested got %d", count)
	}

	count2, err := svc.IngestLocalDirectory(ctx, dir)
	if err != nil {
		t.Fatalf("second Ingest failed: %v", err)
	}
	if count2 != 0 {
		t.Fatalf("expected 0 on second ingest, got %d", count2)
	}

	list, err := svc.ListVideos(ctx, 100, 0)
	if err != nil {
		t.Fatalf("List failed: %v", err)
	}
	found := 0
	for _, v := range list {
		if v.SourceType == SourceTypeLocal && v.SourcePath != nil && (*v.SourcePath == filepath.Join(dir, "a.mp4") || *v.SourcePath == filepath.Join(dir, "b.mkv") || *v.SourcePath == filepath.Join(sub, "d.mov")) {
			found++
		}
	}
	if found != 3 {
		t.Fatalf("expected 3 LOCAL videos in list, found %d", found)
	}
}
