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

const fields = `id, video_id, segment_id, frame_index, timestamp_seconds, storage_key, width, height, purpose, created_at, updated_at`

func scan(row interface{ Scan(dest ...any) error }) (*VideoFrame, error) {
	var f VideoFrame
	err := row.Scan(&f.ID, &f.VideoID, &f.SegmentID, &f.FrameIndex, &f.TimestampSeconds, &f.StorageKey, &f.Width, &f.Height, &f.Purpose, &f.CreatedAt, &f.UpdatedAt)
	if err != nil {
		return nil, err
	}
	return &f, nil
}

// GetByVideoID returns every frame of the video regardless of frame-set
// membership (perception + semantic + both).
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

// GetByVideoIDSet returns the frames of one frame set (the perception set
// or the semantic set). Rows whose purpose is 'both' belong to both sets;
// rows are never duplicated across sets.
func (r *Repository) GetByVideoIDSet(ctx context.Context, videoID uuid.UUID, set Set) ([]VideoFrame, error) {
	query := fmt.Sprintf(`SELECT %s FROM video_frames WHERE video_id = $1 AND purpose = ANY($2) ORDER BY timestamp_seconds ASC, frame_index ASC`, fields)
	return r.queryFrames(ctx, query, videoID, purposesForSet(set))
}

// GetBySegmentID returns every frame of the segment regardless of frame-set
// membership.
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

// GetBySegmentIDSet returns the frames of one frame set within a segment
// (perception for production detection, semantic for VLM evidence).
func (r *Repository) GetBySegmentIDSet(ctx context.Context, segmentID uuid.UUID, set Set) ([]VideoFrame, error) {
	query := fmt.Sprintf(`SELECT %s FROM video_frames WHERE segment_id = $1 AND purpose = ANY($2) ORDER BY timestamp_seconds ASC, frame_index ASC`, fields)
	return r.queryFrames(ctx, query, segmentID, purposesForSet(set))
}

func (r *Repository) queryFrames(ctx context.Context, query string, key any, purposes []Purpose) ([]VideoFrame, error) {
	texts := make([]string, len(purposes))
	for i, p := range purposes {
		texts[i] = string(p)
	}
	rows, err := r.db.Query(ctx, query, key, texts)
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

func (r *Repository) DeleteBySegmentID(ctx context.Context, segmentID uuid.UUID) error {
	_, err := r.db.Exec(ctx, `DELETE FROM video_frames WHERE segment_id = $1`, segmentID)
	return err
}

// UpdatePurpose rewrites the frame-set designation of one row. Membership
// replacement (GenerateForVideoWithPlan / GeneratePerceptionForVideo) uses
// it to demote shared rows dropped from a set and to promote rows shared
// with the regenerated set.
func (r *Repository) UpdatePurpose(ctx context.Context, frameID uuid.UUID, purpose Purpose) error {
	tag, err := r.db.Exec(ctx, `UPDATE video_frames SET purpose = $2, updated_at = now() WHERE id = $1`, frameID, purpose)
	if err != nil {
		return err
	}
	if tag.RowsAffected() == 0 {
		return fmt.Errorf("video_frame: update purpose: frame %s not found", frameID)
	}
	return nil
}

// DeleteByIDs removes specific frame rows (detections attached to them
// cascade via FK). Used by membership replacement to drop exactly the rows
// of the regenerated set without touching the other set.
func (r *Repository) DeleteByIDs(ctx context.Context, ids []uuid.UUID) error {
	if len(ids) == 0 {
		return nil
	}
	texts := make([]string, len(ids))
	for i, id := range ids {
		texts[i] = id.String()
	}
	_, err := r.db.Exec(ctx, `DELETE FROM video_frames WHERE id = ANY($1)`, texts)
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
		query := fmt.Sprintf(`INSERT INTO video_frames (video_id, segment_id, frame_index, timestamp_seconds, storage_key, width, height, purpose) VALUES ($1,$2,$3,$4,$5,$6,$7, CASE WHEN $8::text = '' THEN 'both'::text ELSE $8::text END) RETURNING %s`, fields)
		row := tx.QueryRow(ctx, query, f.VideoID, f.SegmentID, f.FrameIndex, f.TimestampSeconds, f.StorageKey, f.Width, f.Height, f.Purpose)
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
	query := fmt.Sprintf(`INSERT INTO video_frames (video_id, segment_id, frame_index, timestamp_seconds, storage_key, width, height, purpose) VALUES ($1,$2,$3,$4,$5,$6,$7, CASE WHEN $8::text = '' THEN 'both'::text ELSE $8::text END) RETURNING %s`, fields)
	row := r.db.QueryRow(ctx, query, f.VideoID, f.SegmentID, f.FrameIndex, f.TimestampSeconds, f.StorageKey, f.Width, f.Height, f.Purpose)
	return scan(row)
}
