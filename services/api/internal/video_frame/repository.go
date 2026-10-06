package video_frame

import (
	"context"
	"fmt"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
)

type Repository struct {
	db *pgxpool.Pool
}

func NewRepository(db *pgxpool.Pool) *Repository {
	return &Repository{db: db}
}

const fields = `id, video_id, segment_id, frame_index, timestamp_seconds, storage_key, width, height, created_at, updated_at`

func scan(row interface{ Scan(dest ...any) error }) (*VideoFrame, error) {
	var f VideoFrame
	err := row.Scan(&f.ID, &f.VideoID, &f.SegmentID, &f.FrameIndex, &f.TimestampSeconds, &f.StorageKey, &f.Width, &f.Height, &f.CreatedAt, &f.UpdatedAt)
	if err != nil {
		return nil, err
	}
	return &f, nil
}

func (r *Repository) GetByVideoID(ctx context.Context, videoID uuid.UUID) ([]VideoFrame, error) {
	// Chronological order, not insertion order: refinement midpoints are
	// appended out of sequence (FrameIndex reflects extraction order), so
	// timestamp_seconds is the canonical ordering key with frame_index
	// only as a deterministic tiebreak.
	query := fmt.Sprintf(`SELECT %s FROM video_frames WHERE video_id = $1 ORDER BY timestamp_seconds ASC, frame_index ASC`, fields)
	rows, err := r.db.Query(ctx, query, videoID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var list []VideoFrame
	for rows.Next() {
		f, err := scan(rows)
		if err != nil {
			return nil, err
		}
		list = append(list, *f)
	}
	if list == nil {
		list = []VideoFrame{}
	}
	return list, rows.Err()
}

func (r *Repository) DeleteByVideoID(ctx context.Context, videoID uuid.UUID) error {
	_, err := r.db.Exec(ctx, `DELETE FROM video_frames WHERE video_id = $1`, videoID)
	return err
}

func (r *Repository) GetBySegmentID(ctx context.Context, segmentID uuid.UUID) ([]VideoFrame, error) {
	// See GetByVideoID: chronological order with frame_index tiebreak.
	query := fmt.Sprintf(`SELECT %s FROM video_frames WHERE segment_id = $1 ORDER BY timestamp_seconds ASC, frame_index ASC`, fields)
	rows, err := r.db.Query(ctx, query, segmentID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var list []VideoFrame
	for rows.Next() {
		f, err := scan(rows)
		if err != nil {
			return nil, err
		}
		list = append(list, *f)
	}
	if list == nil {
		list = []VideoFrame{}
	}
	return list, rows.Err()
}

func (r *Repository) DeleteBySegmentID(ctx context.Context, segmentID uuid.UUID) error {
	_, err := r.db.Exec(ctx, `DELETE FROM video_frames WHERE segment_id = $1`, segmentID)
	return err
}

func (r *Repository) CreateBatch(ctx context.Context, frames []VideoFrame) ([]VideoFrame, error) {
	if len(frames) == 0 {
		return []VideoFrame{}, nil
	}
	tx, err := r.db.Begin(ctx)
	if err != nil {
		return nil, err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	var result []VideoFrame
	for _, f := range frames {
		query := fmt.Sprintf(`INSERT INTO video_frames (video_id, segment_id, frame_index, timestamp_seconds, storage_key, width, height) VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING %s`, fields)
		row := tx.QueryRow(ctx, query, f.VideoID, f.SegmentID, f.FrameIndex, f.TimestampSeconds, f.StorageKey, f.Width, f.Height)
		saved, err := scan(row)
		if err != nil {
			return nil, err
		}
		result = append(result, *saved)
	}
	if err := tx.Commit(ctx); err != nil {
		return nil, err
	}
	return result, nil
}

func (r *Repository) Create(ctx context.Context, f VideoFrame) (*VideoFrame, error) {
	query := fmt.Sprintf(`INSERT INTO video_frames (video_id, segment_id, frame_index, timestamp_seconds, storage_key, width, height) VALUES ($1,$2,$3,$4,$5,$6,$7) RETURNING %s`, fields)
	row := r.db.QueryRow(ctx, query, f.VideoID, f.SegmentID, f.FrameIndex, f.TimestampSeconds, f.StorageKey, f.Width, f.Height)
	return scan(row)
}
