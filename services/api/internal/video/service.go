package video

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/berzz26/recall/services/api/internal/storage"
	"github.com/google/uuid"
)

type Service struct {
	repo          *Repository
	storage       storage.Storage
	maxUploadSize int64
}

func NewService(repo *Repository) *Service {
	return &Service{repo: repo, storage: nil, maxUploadSize: 1 << 30}
}

func NewServiceWithStorage(repo *Repository, store storage.Storage) *Service {
	return &Service{repo: repo, storage: store, maxUploadSize: 1 << 30}
}

func NewServiceWithConfig(repo *Repository, store storage.Storage, maxUploadSize int64) *Service {
	if maxUploadSize <= 0 {
		maxUploadSize = 1 << 30
	}
	return &Service{repo: repo, storage: store, maxUploadSize: maxUploadSize}
}

func (s *Service) SetStorage(store storage.Storage) {
	s.storage = store
}

func (s *Service) SetMaxUploadSize(size int64) {
	s.maxUploadSize = size
}

var allowedExtensions = map[string]bool{
	".mp4": true,
	".mkv": true,
	".avi": true,
	".mov": true,
}

func isAllowedExtension(ext string) bool {
	return allowedExtensions[strings.ToLower(ext)]
}

func mimeFromExtension(ext string) string {
	switch strings.ToLower(ext) {
	case ".mp4":
		return "video/mp4"
	case ".mkv":
		return "video/x-matroska"
	case ".avi":
		return "video/x-msvideo"
	case ".mov":
		return "video/quicktime"
	default:
		return "application/octet-stream"
	}
}

func hashFile(path string) (string, int64, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", 0, err
	}
	defer f.Close()
	h := sha256.New()
	n, err := io.Copy(h, f)
	if err != nil {
		return "", 0, err
	}
	return hex.EncodeToString(h.Sum(nil)), n, nil
}

func (s *Service) CreateVideo(ctx context.Context, filename string, contentHash *string, mimeType *string, sizeBytes *int64, sourceType *SourceType, sourcePath *string) (*Video, error) {
	if filename == "" {
		return nil, errors.New("filename is required")
	}

	st := SourceTypeUpload
	if sourceType != nil {
		st = *sourceType
	}
	switch st {
	case SourceTypeLocal, SourceTypeUpload:
	default:
		return nil, fmt.Errorf("invalid source_type %s", st)
	}

	if st == SourceTypeLocal && (sourcePath == nil || *sourcePath == "") {
		return nil, errors.New("source_path is required for LOCAL source_type")
	}
	if st == SourceTypeUpload {
		sourcePath = nil
	}

	if st == SourceTypeLocal {
		ext := filepath.Ext(filename)
		if !isAllowedExtension(ext) {
			ext2 := filepath.Ext(*sourcePath)
			if !isAllowedExtension(ext2) {
				return nil, fmt.Errorf("unsupported file type %s", ext)
			}
		}

		absPath, err := filepath.Abs(*sourcePath)
		if err != nil {
			return nil, fmt.Errorf("invalid source_path")
		}
		info, err := os.Stat(absPath)
		if err != nil {
			if os.IsNotExist(err) {
				return nil, fmt.Errorf("source_path does not exist: %w", err)
			}
			return nil, fmt.Errorf("cannot access source_path: %w", err)
		}
		if !info.Mode().IsRegular() {
			return nil, errors.New("source_path is not a regular file")
		}
		if info.Size() == 0 {
			return nil, errors.New("file is empty")
		}
		f, err := os.Open(absPath)
		if err != nil {
			return nil, fmt.Errorf("cannot open source_path: %w", err)
		}
		f.Close()

		hash, size, err := hashFile(absPath)
		if err != nil {
			return nil, fmt.Errorf("failed to hash file: %w", err)
		}
		mime := mimeFromExtension(filepath.Ext(absPath))
		if mime == "application/octet-stream" {
			mime = mimeFromExtension(filepath.Ext(filename))
		}
		mtime := info.ModTime()
		absCopy := absPath
		v, err := s.repo.CreateWithMtime(ctx, filename, hash, mime, size, st, &absCopy, &mtime, StatusUploaded)
		if err != nil {
			return nil, fmt.Errorf("create video: %w", err)
		}
		return v, nil
	}

	ch := ""
	if contentHash != nil {
		ch = *contentHash
	}
	mt := ""
	if mimeType != nil {
		mt = *mimeType
		if mt == "" {
			mt = "application/octet-stream"
		}
	}
	sb := int64(0)
	if sizeBytes != nil {
		sb = *sizeBytes
		if sb < 0 {
			return nil, errors.New("size_bytes must be >= 0")
		}
	}

	v, err := s.repo.Create(ctx, filename, ch, mt, sb, st, sourcePath)
	if err != nil {
		return nil, fmt.Errorf("create video: %w", err)
	}
	return v, nil
}

func (s *Service) UploadVideo(ctx context.Context, filename string, reader io.Reader, mimeType string) (*Video, error) {
	if filename == "" {
		return nil, errors.New("filename is required")
	}
	ext := filepath.Ext(filename)
	if ext == "" {
		ext = ".mp4"
	}
	if !isAllowedExtension(ext) {
		return nil, fmt.Errorf("unsupported file type %s", ext)
	}
	if reader == nil {
		return nil, errors.New("reader is required")
	}
	if s.storage == nil {
		return nil, errors.New("storage not configured")
	}

	mimeType = mimeFromExtension(ext)

	limitReader := io.LimitReader(reader, s.maxUploadSize+1)

	v, err := s.repo.Create(ctx, filename, "", mimeType, 0, SourceTypeUpload, nil)
	if err != nil {
		return nil, fmt.Errorf("create video: %w", err)
	}

	storageKey := fmt.Sprintf("videos/%s/original%s", v.ID.String(), strings.ToLower(ext))

	pr, pw := io.Pipe()
	hash := sha256.New()
	tee := io.TeeReader(limitReader, hash)
	var written int64
	var copyErr error
	done := make(chan struct{})
	go func() {
		defer close(done)
		n, err := io.Copy(pw, tee)
		written = n
		copyErr = err
		pw.CloseWithError(err)
	}()

	err = s.storage.Save(ctx, storageKey, pr)
	<-done
	if copyErr != nil && err == nil {
		err = copyErr
	}
	if err != nil {
		// Detached ctx for cleanup: the request ctx may already be
		// cancelled (client disconnect), which would abort the cleanup
		// queries themselves.
		cleanupCtx := context.WithoutCancel(ctx)
		s.storage.Delete(cleanupCtx, storageKey)
		if ctx.Err() != nil {
			// Client went away mid-upload (e.g. Cancel pressed): remove
			// the row entirely so a cancelled upload doesn't linger as
			// FAILED in the video list.
			_ = s.repo.Delete(cleanupCtx, v.ID)
		} else {
			s.repo.UpdateStatus(cleanupCtx, v.ID, StatusFailed)
		}
		return nil, fmt.Errorf("save file: %w", err)
	}

	if written == 0 {
		s.storage.Delete(ctx, storageKey)
		s.repo.UpdateStatus(ctx, v.ID, StatusFailed)
		return nil, errors.New("empty file")
	}
	if written > s.maxUploadSize {
		s.storage.Delete(ctx, storageKey)
		s.repo.UpdateStatus(ctx, v.ID, StatusFailed)
		return nil, fmt.Errorf("file too large: %d bytes exceeds limit %d", written, s.maxUploadSize)
	}

	hashStr := hex.EncodeToString(hash.Sum(nil))
	size := written

	updated, err := s.repo.UpdateUpload(ctx, v.ID, storageKey, hashStr, mimeType, size, StatusUploaded)
	if err != nil {
		s.storage.Delete(ctx, storageKey)
		s.repo.UpdateStatus(ctx, v.ID, StatusFailed)
		return nil, fmt.Errorf("update video: %w", err)
	}
	return updated, nil
}

func (s *Service) GetVideo(ctx context.Context, id uuid.UUID) (*Video, error) {
	if id == uuid.Nil {
		return nil, errors.New("invalid id")
	}

	v, err := s.repo.GetByID(ctx, id)
	if err != nil {
		return nil, fmt.Errorf("get video %s: %w", id.String(), err)
	}
	return v, nil
}

func (s *Service) ListVideos(ctx context.Context, limit, offset int) ([]Video, error) {
	if limit <= 0 {
		limit = 20
	}
	if limit > 100 {
		limit = 100
	}
	if offset < 0 {
		offset = 0
	}

	videos, err := s.repo.List(ctx, limit, offset)
	if err != nil {
		return nil, fmt.Errorf("list videos: %w", err)
	}
	if videos == nil {
		videos = []Video{}
	}
	return videos, nil
}

func (s *Service) GetByContentHash(ctx context.Context, hash string) ([]Video, error) {
	if hash == "" {
		return nil, errors.New("hash is required")
	}
	return s.repo.GetByContentHash(ctx, hash)
}

func (s *Service) ClaimNext(ctx context.Context) (*Video, error) {
	v, err := s.repo.ClaimNext(ctx)
	if err != nil {
		return nil, err
	}
	return v, nil
}

func (s *Service) MarkReady(ctx context.Context, id uuid.UUID) (*Video, error) {
	return s.repo.MarkReady(ctx, id)
}

func (s *Service) MarkFailed(ctx context.Context, id uuid.UUID, errMsg string) (*Video, error) {
	return s.repo.MarkFailed(ctx, id, errMsg)
}

func (s *Service) ExistsBySourcePath(ctx context.Context, path string) (bool, error) {
	return s.repo.ExistsBySourcePath(ctx, path)
}

func (s *Service) CreateLocalFromPath(ctx context.Context, path string) (*Video, error) {
	abs, err := filepath.Abs(path)
	if err != nil {
		return nil, err
	}
	filename := filepath.Base(abs)
	return s.CreateVideo(ctx, filename, nil, nil, nil, func() *SourceType { s := SourceTypeLocal; return &s }(), &abs)
}

func (s *Service) DeleteVideo(ctx context.Context, id uuid.UUID) error {
	if id == uuid.Nil {
		return errors.New("invalid id")
	}

	v, err := s.repo.GetByID(ctx, id)
	if err != nil {
		return fmt.Errorf("get video %s: %w", id.String(), err)
	}

	// Collect frame storage keys before DB cascade deletes them
	var frameKeys []string
	if s.storage != nil {
		if keys, err := s.repo.ListFrameStorageKeys(ctx, id); err == nil {
			frameKeys = keys
		}
	}

	if err := s.repo.Delete(ctx, id); err != nil {
		return fmt.Errorf("delete video %s: %w", id.String(), err)
	}

	// Delete original video file (only for uploads — never delete user's local source file)
	if v.SourceType == SourceTypeUpload && v.StorageKey != nil && *v.StorageKey != "" && s.storage != nil {
		_ = s.storage.Delete(ctx, *v.StorageKey)
	}

	// Delete the browser-playable proxy if one was generated (lives under
	// storage regardless of source type).
	if v.PlayableKey != nil && *v.PlayableKey != "" && s.storage != nil {
		_ = s.storage.Delete(ctx, *v.PlayableKey)
	}

	// Delete frame images from storage (DB rows cascade via ON DELETE CASCADE)
	if s.storage != nil {
		for _, k := range frameKeys {
			_ = s.storage.Delete(ctx, k)
		}
		// Also try prefix delete for any stray frames (e.g., if DB already cascaded before we listed)
		// Storage is filesystem; listing via DB is sufficient, but be defensive
	}

	return nil
}

func (s *Service) UpdateStatus(ctx context.Context, id uuid.UUID, status Status) (*Video, error) {
	if id == uuid.Nil {
		return nil, errors.New("invalid id")
	}
	switch status {
	case StatusUploading, StatusUploaded, StatusProcessing, StatusReady, StatusFailed:
	default:
		return nil, fmt.Errorf("invalid status %s", status)
	}

	v, err := s.repo.UpdateStatus(ctx, id, status)
	if err != nil {
		return nil, fmt.Errorf("update video %s status: %w", id.String(), err)
	}
	return v, nil
}

func (s *Service) IngestLocalDirectory(ctx context.Context, dir string) (int, error) {
	abs, err := filepath.Abs(dir)
	if err != nil {
		return 0, err
	}
	info, err := os.Stat(abs)
	if err != nil {
		return 0, err
	}
	if !info.IsDir() {
		return 0, fmt.Errorf("not a directory")
	}

	var count int
	err = filepath.WalkDir(abs, func(path string, d os.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			return nil
		}
		ext := strings.ToLower(filepath.Ext(path))
		if !isAllowedExtension(ext) {
			return nil
		}
		exists, err := s.repo.ExistsBySourcePath(ctx, path)
		if err != nil {
			return nil
		}
		if exists {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return nil
		}
		if !info.Mode().IsRegular() {
			return nil
		}
		if info.Size() == 0 {
			return nil
		}
		hash, _, err := hashFile(path)
		if err != nil {
			return nil
		}
		filename := filepath.Base(path)
		mime := mimeFromExtension(ext)
		mtime := info.ModTime()
		_, err = s.repo.CreateWithMtime(ctx, filename, hash, mime, info.Size(), SourceTypeLocal, &path, &mtime, StatusUploaded)
		if err != nil {
			return nil
		}
		count++
		return nil
	})
	if err != nil {
		return count, err
	}
	return count, nil
}
