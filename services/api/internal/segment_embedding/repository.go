package segment_embedding

import (
	"context"
	"fmt"
	"os"
	"strings"

	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
)

type Repository struct {
	db *pgxpool.Pool
}

func NewRepository(db *pgxpool.Pool) *Repository {
	return &Repository{db: db}
}

func scanEmbedding(row interface{ Scan(dest ...any) error }) (*Embedding, error) {
	var e Embedding
	var vecStr string
	// pgvector returns vector as string "[0,1,...]" when scanning to string, but we can scan via pgx with vector type
	// Use string scan then parse? Instead use pgvector support: driver returns []float32 via special type?
	// Fallback: scan as string and parse
	var embeddingRaw string
	err := row.Scan(&e.ID, &e.VideoID, &e.SegmentID, &e.DescriptionID, &embeddingRaw, &e.ModelName, &e.ModelVersion, &e.CreatedAt, &e.UpdatedAt)
	if err != nil {
		return nil, err
	}
	// embeddingRaw is like "[0.1,0.2,...]"
	_ = vecStr
	// Parse not needed for Replace flows; SearchSimilar parses similarity separately
	// For simplicity, leave Embedding empty here; callers that need vector can query via string parsing if needed
	// But we can try to parse
	e.Embedding = parseVector(embeddingRaw)
	return &e, nil
}

func parseVector(s string) []float32 {
	s = strings.TrimSpace(s)
	s = strings.TrimPrefix(s, "[")
	s = strings.TrimSuffix(s, "]")
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]float32, 0, len(parts))
	for _, p := range parts {
		var f float64
		_, err := fmt.Sscanf(strings.TrimSpace(p), "%f", &f)
		if err == nil {
			out = append(out, float32(f))
		}
	}
	return out
}

const embeddingFields = `id, video_id, segment_id, description_id, embedding, model_name, model_version, created_at, updated_at`

func (r *Repository) ReplaceForVideo(ctx context.Context, videoID uuid.UUID, embeddings []Embedding) ([]Embedding, error) {
	tx, err := r.db.Begin(ctx)
	if err != nil {
		return nil, err
	}
	defer func() { _ = tx.Rollback(ctx) }()
	// Delete existing for this video+model (idempotent)
	// We need to delete all embeddings for video where model matches those in input
	// Simpler: delete all for video (as we re-generate all)
	if _, err := tx.Exec(ctx, `DELETE FROM video_segment_embeddings WHERE video_id = $1`, videoID); err != nil {
		return nil, err
	}
	var out []Embedding
	for _, e := range embeddings {
		vecStr := vectorToString(e.Embedding)
		query := fmt.Sprintf(`INSERT INTO video_segment_embeddings (video_id, segment_id, description_id, embedding, model_name, model_version) VALUES ($1,$2,$3,$4::vector,$5,$6) RETURNING %s`, embeddingFields)
		row := tx.QueryRow(ctx, query, e.VideoID, e.SegmentID, e.DescriptionID, vecStr, e.ModelName, e.ModelVersion)
		saved, err := scanEmbedding(row)
		if err != nil {
			return nil, err
		}
		out = append(out, *saved)
	}
	if err := tx.Commit(ctx); err != nil {
		return nil, err
	}
	return out, nil
}

func vectorToString(v []float32) string {
	var sb strings.Builder
	sb.WriteString("[")
	for i, f := range v {
		if i > 0 {
			sb.WriteString(",")
		}
		sb.WriteString(fmt.Sprintf("%f", f))
	}
	sb.WriteString("]")
	return sb.String()
}

func (r *Repository) GetByVideoID(ctx context.Context, videoID uuid.UUID) ([]Embedding, error) {
	query := fmt.Sprintf(`SELECT %s FROM video_segment_embeddings WHERE video_id = $1 ORDER BY created_at ASC`, embeddingFields)
	rows, err := r.db.Query(ctx, query, videoID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var list []Embedding
	for rows.Next() {
		e, err := scanEmbedding(rows)
		if err != nil {
			return nil, err
		}
		list = append(list, *e)
	}
	if list == nil {
		list = []Embedding{}
	}
	return list, rows.Err()
}

func (r *Repository) GetBySegmentID(ctx context.Context, segmentID uuid.UUID) ([]Embedding, error) {
	query := fmt.Sprintf(`SELECT %s FROM video_segment_embeddings WHERE segment_id = $1`, embeddingFields)
	rows, err := r.db.Query(ctx, query, segmentID)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var list []Embedding
	for rows.Next() {
		e, err := scanEmbedding(rows)
		if err != nil {
			return nil, err
		}
		list = append(list, *e)
	}
	if list == nil {
		list = []Embedding{}
	}
	return list, rows.Err()
}

// SearchSimilar performs pgvector cosine search.
func (r *Repository) SearchSimilar(ctx context.Context, queryVec []float32, limit int, videoID *uuid.UUID) ([]SearchResult, error) {
	if len(queryVec) != 384 {
		return nil, fmt.Errorf("invalid query vector dimension %d != 384", len(queryVec))
	}
	vecStr := vectorToString(queryVec)
	// Build query with optional video filter
	query := `
		SELECT
			e.id, e.video_id, e.segment_id, e.description_id, e.embedding::text, e.model_name, e.model_version, e.created_at, e.updated_at,
			d.description,
			COALESCE(s.start_time, 0), COALESCE(s.end_time, 0),
			1 - (e.embedding <=> $1::vector) AS similarity
		FROM video_segment_embeddings e
		JOIN video_segment_descriptions d ON d.id = e.description_id
		LEFT JOIN video_segments s ON s.id = e.segment_id
		WHERE 1=1
	`
	args := []interface{}{vecStr}
	argIdx := 2
	if videoID != nil {
		query += fmt.Sprintf(" AND e.video_id = $%d", argIdx)
		args = append(args, *videoID)
		argIdx++
	}
	query += fmt.Sprintf(" ORDER BY e.embedding <=> $1::vector LIMIT $%d", argIdx)
	args = append(args, limit)

	queryFile := "../search_query.txt"
	embeddingStr := vectorToString(queryVec)
	if err := os.WriteFile(queryFile, []byte(embeddingStr), 0644); err != nil {
		return nil, fmt.Errorf("failed to write search query to file: %w", err)
	}

	rows, err := r.db.Query(ctx, query, args...)
	if err != nil {
		return nil, err
	}
	defer rows.Close()
	var results []SearchResult
	for rows.Next() {
		var e Embedding
		var embeddingStr string
		var desc string
		var start, end float64
		var similarity float64
		err := rows.Scan(&e.ID, &e.VideoID, &e.SegmentID, &e.DescriptionID, &embeddingStr, &e.ModelName, &e.ModelVersion, &e.CreatedAt, &e.UpdatedAt, &desc, &start, &end, &similarity)
		if err != nil {
			return nil, err
		}
		e.Embedding = parseVector(embeddingStr)
		results = append(results, SearchResult{
			Embedding:   e,
			Description: desc,
			StartTime:   start,
			EndTime:     end,
			Similarity:  similarity,
		})
	}
	if results == nil {
		results = []SearchResult{}
	}
	return results, rows.Err()
}
