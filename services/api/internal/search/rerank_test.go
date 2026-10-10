package search

import (
	"context"
	"errors"
	"testing"

	"github.com/berzz26/recall/services/api/internal/rerank"
	"github.com/berzz26/recall/services/api/internal/segment_embedding"
	"github.com/google/uuid"
)

func testCandidates(descs []string, sims []float64) []segment_embedding.SearchResult {
	out := make([]segment_embedding.SearchResult, len(descs))
	for i, d := range descs {
		out[i] = segment_embedding.SearchResult{
			Embedding:   segment_embedding.Embedding{VideoID: uuid.New(), SegmentID: uuid.New()},
			Description: d,
			Similarity:  sims[i],
		}
	}
	return out
}

func TestRerankReordersDescending(t *testing.T) {
	svc := NewService(nil, nil, nil, nil, 50, 10, 50, 0).WithReranker(&rerank.FakeReranker{
		Scores: []float64{0.1, 0.9, 0.5},
	})
	cands := testCandidates(
		[]string{"alpha full description", "bravo full description", "charlie full description"},
		[]float64{0.9, 0.8, 0.7},
	)
	reordered, scores := svc.rerankCandidates(context.Background(), "original query", cands)
	if len(reordered) != 3 {
		t.Fatalf("expected 3 candidates, got %d", len(reordered))
	}
	// Fake scores: idx1 (0.9) > idx2 (0.5) > idx0 (0.1).
	if reordered[0].Description != "bravo full description" ||
		reordered[1].Description != "charlie full description" ||
		reordered[2].Description != "alpha full description" {
		t.Fatalf("wrong rerank order: %q, %q, %q",
			reordered[0].Description, reordered[1].Description, reordered[2].Description)
	}
	if scores == nil {
		t.Fatalf("expected score map")
	}
	if got := scores[reordered[0].Embedding.SegmentID]; got != 0.9 {
		t.Fatalf("top score must be 0.9, got %v", got)
	}
}

func TestRerankPassesQueryAndFullDescriptions(t *testing.T) {
	fake := &rerank.FakeReranker{Scores: []float64{0.2, 0.8}}
	svc := NewService(nil, nil, nil, nil, 50, 10, 50, 0).WithReranker(fake)
	cands := testCandidates([]string{"full desc one", "full desc two"}, []float64{0.9, 0.8})
	_, _ = svc.rerankCandidates(context.Background(), "my search query", cands)
	if fake.LastQuery != "my search query" {
		t.Fatalf("reranker must receive original query, got %q", fake.LastQuery)
	}
	if len(fake.LastDocs) != 2 || fake.LastDocs[0] != "full desc one" || fake.LastDocs[1] != "full desc two" {
		t.Fatalf("reranker must receive full descriptions, got %q", fake.LastDocs)
	}
	if fake.Calls != 1 {
		t.Fatalf("expected 1 rerank call, got %d", fake.Calls)
	}
}

func TestRerankFallbackOnError(t *testing.T) {
	svc := NewService(nil, nil, nil, nil, 50, 10, 50, 0).WithReranker(&rerank.FakeReranker{
		Err: errors.New("model down"),
	})
	cands := testCandidates([]string{"a", "b"}, []float64{0.9, 0.8})
	reordered, scores := svc.rerankCandidates(context.Background(), "q", cands)
	if reordered[0].Description != "a" || reordered[1].Description != "b" {
		t.Fatalf("error fallback must preserve vector order")
	}
	if scores != nil {
		t.Fatalf("error fallback must return nil score map")
	}
}

func TestRerankDisabledPreservesOrder(t *testing.T) {
	svc := NewService(nil, nil, nil, nil, 50, 10, 50, 0) // no reranker
	cands := testCandidates([]string{"a", "b", "c"}, []float64{0.9, 0.8, 0.7})
	reordered, scores := svc.rerankCandidates(context.Background(), "q", cands)
	for i := range cands {
		if reordered[i].Description != cands[i].Description {
			t.Fatalf("nil reranker must preserve order")
		}
	}
	if scores != nil {
		t.Fatalf("nil reranker must return nil score map")
	}
}

func TestRerankStableOnTies(t *testing.T) {
	svc := NewService(nil, nil, nil, nil, 50, 10, 50, 0).WithReranker(&rerank.FakeReranker{
		Scores: []float64{0.5, 0.5, 0.9},
	})
	cands := testCandidates([]string{"a", "b", "c"}, []float64{0.9, 0.8, 0.7})
	reordered, _ := svc.rerankCandidates(context.Background(), "q", cands)
	// c first (0.9), then a,b tie in original vector order.
	if reordered[0].Description != "c" || reordered[1].Description != "a" || reordered[2].Description != "b" {
		t.Fatalf("ties must preserve vector order, got %q %q %q",
			reordered[0].Description, reordered[1].Description, reordered[2].Description)
	}
}

func TestRerankScoreRange(t *testing.T) {
	if _, _, ok := rerankScoreRange(nil); ok {
		t.Fatalf("nil map must report ok=false")
	}
	if _, _, ok := rerankScoreRange(map[uuid.UUID]float64{}); ok {
		t.Fatalf("empty map must report ok=false")
	}
	scores := map[uuid.UUID]float64{
		uuid.New(): 2.5,
		uuid.New(): -1.0,
		uuid.New(): 7.0,
	}
	min, max, ok := rerankScoreRange(scores)
	if !ok || min != -1.0 || max != 7.0 {
		t.Fatalf("expected min=-1 max=7, got %v %v %v", min, max, ok)
	}
}

func TestNormalizeRerankScore(t *testing.T) {
	if got := normalizeRerankScore(7.0, -1.0, 7.0); got != 1.0 {
		t.Fatalf("top must normalize to 1, got %v", got)
	}
	if got := normalizeRerankScore(-1.0, -1.0, 7.0); got != 0.0 {
		t.Fatalf("bottom must normalize to 0, got %v", got)
	}
	if got := normalizeRerankScore(3.0, -1.0, 7.0); got != 0.5 {
		t.Fatalf("midpoint must normalize to 0.5, got %v", got)
	}
	// Degenerate pool: identical scores normalize to 1, never NaN.
	if got := normalizeRerankScore(4.0, 4.0, 4.0); got != 1.0 {
		t.Fatalf("degenerate pool must normalize to 1, got %v", got)
	}
	// Clamping guards against float drift.
	if got := normalizeRerankScore(99.0, -1.0, 7.0); got != 1.0 {
		t.Fatalf("above max must clamp to 1, got %v", got)
	}
	if got := normalizeRerankScore(-99.0, -1.0, 7.0); got != 0.0 {
		t.Fatalf("below min must clamp to 0, got %v", got)
	}
}
