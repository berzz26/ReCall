package detection

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

const fields = `id, video_id, segment_id, frame_id, label, confidence, bbox_x, bbox_y, bbox_width, bbox_height, detector_name, detector_version, created_at, updated_at`

func scan(row interface{ Scan(dest ...any) error }) (*Detection, error) {
	var d Detection
	err := row.Scan(&d.ID, &d.VideoID, &d.SegmentID, &d.FrameID, &d.Label, &d.Confidence, &d.BBoxX, &d.BBoxY, &d.BBoxWidth, &d.BBoxHeight, &d.DetectorName, &d.DetectorVersion, &d.CreatedAt, &d.UpdatedAt)
	if err != nil {
		return nil, err
	}
	return &d, nil
}

func (r *Repository) CreateBatch(ctx context.Context, detections []Detection) ([]Detection, error) {
	if len(detections) == 0 {
		return []Detection{}, nil
	}
	tx, err := r.db.Begin(ctx)
	if err != nil {
		return nil, err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	var result []Detection
	for _, d := range detections {
		query := fmt.Sprintf(`INSERT INTO video_frame_detections (video_id, segment_id, frame_id, label, confidence, bbox_x, bbox_y, bbox_width, bbox_height, detector_name, detector_version) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11) RETURNING %s`, fields)
		row := tx.QueryRow(ctx, query, d.VideoID, d.SegmentID, d.FrameID, d.Label, d.Confidence, d.BBoxX, d.BBoxY, d.BBoxWidth, d.BBoxHeight, d.DetectorName, d.DetectorVersion)
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

func (r *Repository) GetByFrameID(ctx context.Context, frameID uuid.UUID) ([]Detection, error) {
	query := fmt.Sprintf(`SELECT %s FROM video_frame_detections WHERE frame_id = $1 ORDER BY created_at ASC, id ASC`, fields)
	rows, err := r.db.Query(ctx, query, frameID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var list []Detection
	for rows.Next() {
		d, err := scan(rows)
		if err != nil {
			return nil, err
		}
		list = append(list, *d)
	}
	if list == nil {
		list = []Detection{}
	}
	return list, rows.Err()
}

func (r *Repository) GetByVideoID(ctx context.Context, videoID uuid.UUID) ([]Detection, error) {
	query := fmt.Sprintf(`SELECT %s FROM video_frame_detections WHERE video_id = $1 ORDER BY created_at ASC, id ASC`, fields)
	rows, err := r.db.Query(ctx, query, videoID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var list []Detection
	for rows.Next() {
		d, err := scan(rows)
		if err != nil {
			return nil, err
		}
		list = append(list, *d)
	}
	if list == nil {
		list = []Detection{}
	}
	return list, rows.Err()
}

func (r *Repository) DeleteByVideoID(ctx context.Context, videoID uuid.UUID) error {
	_, err := r.db.Exec(ctx, `DELETE FROM video_frame_detections WHERE video_id = $1`, videoID)
	return err
}

func (r *Repository) DeleteByFrameID(ctx context.Context, frameID uuid.UUID) error {
	_, err := r.db.Exec(ctx, `DELETE FROM video_frame_detections WHERE frame_id = $1`, frameID)
	return err
}

// DeleteByFrameIDs removes the detections of multiple frames in one round
// trip (batch version of DeleteByFrameID). Used by the orchestrator to clean
// stale production detections on frames that left the perception set during
// a perception regeneration.
func (r *Repository) DeleteByFrameIDs(ctx context.Context, frameIDs []uuid.UUID) error {
	if len(frameIDs) == 0 {
		return nil
	}
	ids := make([]string, len(frameIDs))
	for i, id := range frameIDs {
		ids[i] = id.String()
	}
	_, err := r.db.Exec(ctx, `DELETE FROM video_frame_detections WHERE frame_id = ANY($1)`, ids)
	return err
}

func (r *Repository) GetBySegmentID(ctx context.Context, segmentID uuid.UUID) ([]Detection, error) {
	query := fmt.Sprintf(`SELECT %s FROM video_frame_detections WHERE segment_id = $1 ORDER BY created_at ASC, id ASC`, fields)
	rows, err := r.db.Query(ctx, query, segmentID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var list []Detection
	for rows.Next() {
		d, err := scan(rows)
		if err != nil {
			return nil, err
		}
		list = append(list, *d)
	}
	if list == nil {
		list = []Detection{}
	}
	return list, rows.Err()
}

func (r *Repository) DeleteBySegmentID(ctx context.Context, segmentID uuid.UUID) error {
	_, err := r.db.Exec(ctx, `DELETE FROM video_frame_detections WHERE segment_id = $1`, segmentID)
	return err
}
