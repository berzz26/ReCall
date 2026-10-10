package rerank

import (
	"context"
	"fmt"
)

// FakeReranker is a deterministic test double. Scores are returned
// verbatim when set; otherwise a descending index order is produced.
type FakeReranker struct {
	Scores []float64
	Err    error
	Calls  int
	// LastQuery/LastDocs capture the most recent call for assertions.
	LastQuery string
	LastDocs  []string
}

func (f *FakeReranker) Rerank(_ context.Context, query string, docs []string) ([]float64, error) {
	f.Calls++
	f.LastQuery = query
	f.LastDocs = append([]string(nil), docs...)
	if f.Err != nil {
		return nil, f.Err
	}
	if f.Scores != nil {
		if len(f.Scores) != len(docs) {
			return nil, fmt.Errorf("fake reranker score count %d != docs %d", len(f.Scores), len(docs))
		}
		return append([]float64(nil), f.Scores...), nil
	}
	out := make([]float64, len(docs))
	for i := range out {
		out[i] = float64(len(docs) - i)
	}
	return out, nil
}
