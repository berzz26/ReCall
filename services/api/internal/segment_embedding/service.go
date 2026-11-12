package segment_embedding

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/berzz26/recall/services/api/internal/embedding"
	"github.com/berzz26/recall/services/api/internal/segment_description"
	"github.com/google/uuid"
)

const (
	ModelName    = "BAAI/bge-small-en-v1.5"
	ModelVersion = "v1.5"
)

type Service struct {
	repo         *Repository
	descRepo     *segment_description.Repository
	embedder     embedding.TextEmbedder
	modelName    string
	modelVersion string
}

func NewService(repo *Repository, descRepo *segment_description.Repository, embedder embedding.TextEmbedder, modelName, modelVersion string) *Service {
	if modelName == "" {
		modelName = ModelName
	}
	if modelVersion == "" {
		modelVersion = ModelVersion
	}
	return &Service{repo: repo, descRepo: descRepo, embedder: embedder, modelName: modelName, modelVersion: modelVersion}
}

func (s *Service) GenerateForVideo(ctx context.Context, videoID uuid.UUID) ([]Embedding, error) {
	start := time.Now()
	descs, err := s.descRepo.GetByVideoID(ctx, videoID)
	if err != nil {
		return nil, err
	}
	if len(descs) == 0 {
		slog.Info("embedding: no descriptions, skipping embedding generation (optional)", "video_id", videoID.String())
		return []Embedding{}, nil
	}
	for _, d := range descs {
		if d.Description == "" {
			return nil, fmt.Errorf("empty description for segment %s", d.SegmentID)
		}
	}
	// Prepare batch
	ids := make([]string, len(descs))
	texts := make([]string, len(descs))
	for i, d := range descs {
		ids[i] = d.ID.String()
		texts[i] = d.Description
	}
	slog.Info("embedding: start", "video_id", videoID.String(), "descriptions", len(descs), "model", s.modelName)

	// Use embedder batch via type assertion for WithIDs if available
	var embeddings map[string][]float32
	if be, ok := s.embedder.(interface {
		EmbedPassagesWithIDs(ctx context.Context, ids []string, texts []string) (map[string][]float32, error)
	}); ok {
		embeddings, err = be.EmbedPassagesWithIDs(ctx, ids, texts)
	} else {
		// fallback: embed sequentially? but spec requires batch; use EmbedPassages and map by order
		vecs, err2 := s.embedder.EmbedPassages(ctx, texts)
		if err2 != nil {
			return nil, err2
		}
		embeddings = make(map[string][]float32, len(ids))
		for i, id := range ids {
			embeddings[id] = vecs[i]
		}
		err = nil
	}
	if err != nil {
		return nil, err
	}
	// Validate and build embeddings
	var toPersist []Embedding
	for _, d := range descs {
		vec, ok := embeddings[d.ID.String()]
		if !ok {
			return nil, fmt.Errorf("missing embedding for description %s", d.ID)
		}
		if len(vec) != 384 {
			return nil, fmt.Errorf("invalid embedding dimension for %s: %d", d.ID, len(vec))
		}
		toPersist = append(toPersist, Embedding{
			VideoID:       d.VideoID,
			SegmentID:     d.SegmentID,
			DescriptionID: d.ID,
			Embedding:     vec,
			ModelName:     s.modelName,
			ModelVersion:  s.modelVersion,
		})
	}
	saved, err := s.repo.ReplaceForVideo(ctx, videoID, toPersist)
	if err != nil {
		return nil, err
	}
	slog.Info("embedding: complete", "video_id", videoID.String(), "embeddings", len(saved), "duration_ms", time.Since(start).Milliseconds())
	return saved, nil
}

func (s *Service) SearchSimilar(ctx context.Context, queryVec []float32, limit int, videoID *uuid.UUID) ([]SearchResult, error) {
	return s.repo.SearchSimilar(ctx, queryVec, limit, videoID)
}

func (s *Service) GetByVideoID(ctx context.Context, videoID uuid.UUID) ([]Embedding, error) {
	return s.repo.GetByVideoID(ctx, videoID)
}
