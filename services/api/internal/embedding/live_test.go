package embedding

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestLivePersistentWorker(t *testing.T) {
	if os.Getenv("RECALL_EMBEDDING_LIVE_TEST") != "1" {
		t.Skip("set RECALL_EMBEDDING_LIVE_TEST=1 to run against the local ONNX artifact")
	}
	root, err := filepath.Abs("../../../../")
	if err != nil {
		t.Fatal(err)
	}
	onnxDir := filepath.Join(root, "workers/embedding/models/bge-small-en-v1.5")
	if _, err := os.Stat(filepath.Join(onnxDir, "model.onnx")); err != nil {
		t.Skipf("artifact missing: %v", err)
	}
	python := os.Getenv("EMBEDDING_PYTHON_PATH")
	if python == "" {
		python = "python3"
	}
	e := NewBGEEmbedder(Options{
		PythonPath:  python,
		ScriptPath:  filepath.Join(root, "workers/embedding/embed.py"),
		Backend:     "onnx",
		ONNXDir:     onnxDir,
		ONNXThreads: 2,
		BatchSize:   4,
		Timeout:     30 * time.Second,
	})
	e.StartPersistent(context.Background())
	defer e.Close()
	if e.sess == nil {
		t.Fatal("persistent session was not established")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 20*time.Second)
	defer cancel()
	vec, err := e.EmbedQuery(ctx, "how to cook pasta")
	if err != nil {
		t.Fatal(err)
	}
	if len(vec) != 384 {
		t.Fatalf("dim = %d, want 384", len(vec))
	}
	vecs, err := e.EmbedPassages(ctx, []string{"hello world", "a person walks down the street"})
	if err != nil {
		t.Fatal(err)
	}
	if len(vecs) != 2 || len(vecs[0]) != 384 || len(vecs[1]) != 384 {
		t.Fatalf("unexpected passage result %dx%d", len(vecs), len(vecs[0]))
	}
}
