package rerank

import (
	"context"
)

// Reranker scores (query, document) pairs with a cross-encoder.
// Scores are relevance logits: higher means more relevant. Implementations
// must preserve input order in the returned slice.
type Reranker interface {
	Rerank(ctx context.Context, query string, docs []string) ([]float64, error)
}
