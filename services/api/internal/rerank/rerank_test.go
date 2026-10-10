package rerank

import (
	"context"
	"testing"
	"time"
)

func TestNewDefaults(t *testing.T) {
	r := NewCrossEncoderReranker("", "", "", 0)
	if r.pythonPath != "python3" {
		t.Fatalf("python default must be python3, got %q", r.pythonPath)
	}
	if r.modelName != DefaultModel {
		t.Fatalf("model default must be %q, got %q", DefaultModel, r.modelName)
	}
	if r.timeout != 2*time.Minute {
		t.Fatalf("timeout default must be 2m, got %v", r.timeout)
	}
	if DefaultModel != "mixedbread-ai/mxbai-rerank-xsmall-v1" {
		t.Fatalf("default model must be mxbai-rerank-xsmall-v1, got %q", DefaultModel)
	}
}

func TestRerankEmptyDocs(t *testing.T) {
	r := NewCrossEncoderReranker("python3", "workers/rerank/rerank.py", "", 0)
	scores, err := r.Rerank(context.Background(), "query", nil)
	if err != nil || len(scores) != 0 {
		t.Fatalf("empty docs must return empty scores, got %v %v", scores, err)
	}
}

func TestRerankEmptyQuery(t *testing.T) {
	r := NewCrossEncoderReranker("python3", "workers/rerank/rerank.py", "", 0)
	if _, err := r.Rerank(context.Background(), "  ", []string{"doc"}); err == nil {
		t.Fatalf("empty query must error")
	}
}

func TestFakeReranker(t *testing.T) {
	f := &FakeReranker{Scores: []float64{0.3, 0.7}}
	scores, err := f.Rerank(context.Background(), "q", []string{"a", "b"})
	if err != nil || scores[0] != 0.3 || scores[1] != 0.7 {
		t.Fatalf("fake must return configured scores, got %v %v", scores, err)
	}
	if f.LastQuery != "q" || len(f.LastDocs) != 2 {
		t.Fatalf("fake must capture query/docs")
	}
}
