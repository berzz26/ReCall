package video

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
)

type Repository struct {
	db *pgxpool.Pool
}

func NewRepository(db *pgxpool.Pool) *Repository {
	return &Repository{db: db}
}

const videoFields = `id, filename, content_hash, mime_type, size_bytes, source_type, source_path, storage_key, playable_key, source_mtime, processing_error, status, created_at, updated_at`

func scanVideo(row interface{ Scan(dest ...any) error }) (*Video, error) {
	var v Video
	err := row.Scan(
		&v.ID,
		&v.Filename,
		&v.ContentHash,
		&v.MimeType,
		&v.SizeBytes,
		&v.SourceType,
		&v.SourcePath,
		&v.StorageKey,
		&v.PlayableKey,
		&v.SourceMtime,
		&v.ProcessingError,
		&v.Status,
		&v.CreatedAt,
		&v.UpdatedAt,
	)
	if err != nil {
		return nil, err
	}
	return &v, nil
}

func (r *Repository) Create(ctx context.Context, filename, contentHash, mimeType string, sizeBytes int64, sourceType SourceType, sourcePath *string) (*Video, error) {
	query := fmt.Sprintf(`
		INSERT INTO videos (filename, content_hash, mime_type, size_bytes, source_type, source_path, status)
		VALUES ($1, $2, $3, $4, $5, $6, $7)
		RETURNING %s
	`, videoFields)

	row := r.db.QueryRow(ctx, query, filename, contentHash, mimeType, sizeBytes, sourceType, sourcePath, StatusUploading)
	return scanVideo(row)
}

func (r *Repository) CreateWithMtime(ctx context.Context, filename, contentHash, mimeType string, sizeBytes int64, sourceType SourceType, sourcePath *string, sourceMtime *time.Time, status Status) (*Video, error) {
	query := fmt.Sprintf(`
		INSERT INTO videos (filename, content_hash, mime_type, size_bytes, source_type, source_path, source_mtime, status)
		VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
		RETURNING %s
	`, videoFields)

	row := r.db.QueryRow(ctx, query, filename, contentHash, mimeType, sizeBytes, sourceType, sourcePath, sourceMtime, status)
	return scanVideo(row)
}

func (r *Repository) GetByID(ctx context.Context, id uuid.UUID) (*Video, error) {
	query := fmt.Sprintf(`SELECT %s FROM videos WHERE id = $1`, videoFields)
	row := r.db.QueryRow(ctx, query, id)
	return scanVideo(row)
}

func (r *Repository) List(ctx context.Context, limit, offset int) ([]Video, error) {
	if limit <= 0 {
		limit = 20
	}
	if limit > 100 {
		limit = 100
	}
	if offset < 0 {
		offset = 0
	}

	query := fmt.Sprintf(`SELECT %s FROM videos ORDER BY created_at DESC LIMIT $1 OFFSET $2`, videoFields)
	rows, err := r.db.Query(ctx, query, limit, offset)
	if err != nil {
		return nil, err
	}
	defer rows.Close()

	var list []Video
	for rows.Next() {
		v, err := scanVideo(rows)
		if err != nil {
			return nil, err
		}
		list = append(list, *v)
	}
	if list == nil {
		list = []Video{}
	}
	return list, nil
}

func (r *Repository) Delete(ctx context.Context, id uuid.UUID) error {
	_, err := r.db.Exec(ctx, `DELETE FROM videos WHERE id = $1`, id)
	return err
}

func (r *Repository) UpdateStatus(ctx context.Context, id uuid.UUID, status Status) (*Video, error) {
	query := fmt.Sprintf(`
		UPDATE videos SET status = $2, updated_at = now()
		WHERE id = $1
		RETURNING %s
	`, videoFields)
	row := r.db.QueryRow(ctx, query, id, status)
	return scanVideo(row)
}

func (r *Repository) UpdateUpload(ctx context.Context, id uuid.UUID, storageKey, contentHash, mimeType string, sizeBytes int64, status Status) (*Video, error) {
	query := fmt.Sprintf(`
		UPDATE videos SET storage_key = $2, content_hash = $3, mime_type = $4, size_bytes = $5, status = $6, updated_at = now()
		WHERE id = $1
		RETURNING %s
	`, videoFields)
	row := r.db.QueryRow(ctx, query, id, storageKey, contentHash, mimeType, sizeBytes, status)
	return scanVideo(row)
}

// SetPlayableKey records the storage key of the browser-playable proxy.
func (r *Repository) SetPlayableKey(ctx context.Context, id uuid.UUID, playableKey string) (*Video, error) {
	query := fmt.Sprintf(`
		UPDATE videos SET playable_key = $2, updated_at = now()
		WHERE id = $1
		RETURNING %s
	`, videoFields)
	row := r.db.QueryRow(ctx, query, id, playableKey)
	return scanVideo(row)
}

func (r *Repository) GetBySourcePath(ctx context.Context, sourcePath string) (*Video, error) {
	query := fmt.Sprintf(`SELECT %s FROM videos WHERE source_path = $1 AND source_type = $2 LIMIT 1`, videoFields)
	row := r.db.QueryRow(ctx, query, sourcePath, SourceTypeLocal)
	return scanVideo(row)
}

func (r *Repository) ExistsBySourcePath(ctx context.Context, sourcePath string) (bool, error) {
	var exists bool
	err := r.db.QueryRow(ctx, `SELECT EXISTS(SELECT 1 FROM videos WHERE source_path = $1 AND source_type = $2)`, sourcePath, SourceTypeLocal).Scan(&exists)
	return exists, err
}

func (r *Repository) GetByContentHash(ctx context.Context, hash string) ([]Video, error) {
	query := fmt.Sprintf(`SELECT %s FROM videos WHERE content_hash = $1 ORDER BY created_at DESC`, videoFields)
	rows, err := r.db.Query(ctx, query, hash)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var list []Video
	for rows.Next() {
		v, err := scanVideo(rows)
		if err != nil {
			return nil, err
		}
		list = append(list, *v)
	}
	if list == nil {
		list = []Video{}
	}
	return list, nil
}

func (r *Repository) ClaimNext(ctx context.Context) (*Video, error) {
	query := fmt.Sprintf(`
		UPDATE videos SET status = $1, updated_at = now()
		WHERE id = (
			SELECT id FROM videos WHERE status = $2 ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1
		)
		RETURNING %s
	`, videoFields)
	row := r.db.QueryRow(ctx, query, StatusProcessing, StatusUploaded)
	v, err := scanVideo(row)
	if err != nil {
		if strings.Contains(err.Error(), "no rows") {
			return nil, nil
		}
		return nil, err
	}
	return v, nil
}

func (r *Repository) MarkReady(ctx context.Context, id uuid.UUID) (*Video, error) {
	query := fmt.Sprintf(`
		UPDATE videos SET status = $2, processing_error = NULL, updated_at = now()
		WHERE id = $1
		RETURNING %s
	`, videoFields)
	row := r.db.QueryRow(ctx, query, id, StatusReady)
	return scanVideo(row)
}

func (r *Repository) MarkFailed(ctx context.Context, id uuid.UUID, errMsg string) (*Video, error) {
	query := fmt.Sprintf(`
		UPDATE videos SET status = $2, processing_error = $3, updated_at = now()
		WHERE id = $1
		RETURNING %s
	`, videoFields)
	row := r.db.QueryRow(ctx, query, id, StatusFailed, errMsg)
	return scanVideo(row)
}

func (r *Repository) ListFrameStorageKeys(ctx context.Context, videoID uuid.UUID) ([]string, error) {
	rows, err := r.db.Query(ctx, `SELECT storage_key FROM video_frames WHERE video_id = $1`, videoID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var keys []string
	for rows.Next() {
		var k string
		if err := rows.Scan(&k); err != nil {
			return nil, err
		}
		keys = append(keys, k)
	}
	return keys, rows.Err()
}

func (r *Repository) Count(ctx context.Context) (int64, error) {
	var count int64
	err := r.db.QueryRow(ctx, `SELECT COUNT(*) FROM videos`).Scan(&count)
	return count, err
}
