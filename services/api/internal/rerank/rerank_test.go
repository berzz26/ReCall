package rerank

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"
)

func TestNewDefaults(t *testing.T) {
	r := NewCrossEncoderReranker(Options{})
	if r.pythonPath != "python3" {
		t.Fatalf("python default must be python3, got %q", r.pythonPath)
	}
	if r.modelName != DefaultModel {
		t.Fatalf("model default must be %q, got %q", DefaultModel, r.modelName)
	}
	if r.backend != "torch" {
		t.Fatalf("backend default must be torch, got %q", r.backend)
	}
	if r.quantization != "none" {
		t.Fatalf("quantization default must be none, got %q", r.quantization)
	}
	if r.onnxDir != DefaultONNXDir {
		t.Fatalf("onnx dir default must be %q, got %q", DefaultONNXDir, r.onnxDir)
	}
	if r.onnxThreads != DefaultONNXThreads {
		t.Fatalf("onnx threads default must be %d, got %d", DefaultONNXThreads, r.onnxThreads)
	}
	if r.batchSize != DefaultBatchSize {
		t.Fatalf("batch size default must be %d, got %d", DefaultBatchSize, r.batchSize)
	}
	if r.timeout != 2*time.Minute {
		t.Fatalf("timeout default must be 2m, got %v", r.timeout)
	}
	if DefaultModel != "mixedbread-ai/mxbai-rerank-xsmall-v1" {
		t.Fatalf("default model must be mxbai-rerank-xsmall-v1, got %q", DefaultModel)
	}
}

func TestWorkerArgsIncludeBackendOptions(t *testing.T) {
	r := NewCrossEncoderReranker(Options{
		ScriptPath:   "workers/rerank/rerank.py",
		ModelName:    "mixedbread-ai/mxbai-rerank-xsmall-v1",
		Backend:      "onnx",
		Quantization: "int8",
		ONNXDir:      "/tmp/rerank-artifact",
		ONNXThreads:  3,
		BatchSize:    7,
	})
	args := strings.Join(r.workerArgs(false, "/tmp/in.json", "/tmp/out.json"), " ")
	for _, want := range []string{
		"--backend onnx",
		"--quantization int8",
		"--onnx-dir /tmp/rerank-artifact",
		"--onnx-threads 3",
		"--batch-size 7",
		"--input /tmp/in.json",
		"--output /tmp/out.json",
	} {
		if !strings.Contains(args, want) {
			t.Fatalf("worker args missing %q: %s", want, args)
		}
	}
	if strings.Contains(args, "--persistent") {
		t.Fatalf("one-shot args must not contain --persistent: %s", args)
	}
	persistentArgs := strings.Join(r.workerArgs(true, "", ""), " ")
	if !strings.Contains(persistentArgs, "--persistent") {
		t.Fatalf("persistent args missing --persistent: %s", persistentArgs)
	}
}

func TestRerankEmptyDocs(t *testing.T) {
	r := NewCrossEncoderReranker(Options{PythonPath: "python3", ScriptPath: "workers/rerank/rerank.py"})
	scores, err := r.Rerank(context.Background(), "query", nil)
	if err != nil || len(scores) != 0 {
		t.Fatalf("empty docs must return empty scores, got %v %v", scores, err)
	}
}

func TestRerankEmptyQuery(t *testing.T) {
	r := NewCrossEncoderReranker(Options{PythonPath: "python3", ScriptPath: "workers/rerank/rerank.py"})
	if _, err := r.Rerank(context.Background(), "  ", []string{"doc"}); err == nil {
		t.Fatalf("empty query must error")
	}
}

func TestUnavailableRerankerFailsFast(t *testing.T) {
	r := NewCrossEncoderReranker(Options{PythonPath: "python3", ScriptPath: "workers/rerank/rerank.py"})
	r.started = true
	r.unavailable = errors.New("artifact missing")
	_, err := r.Rerank(context.Background(), "query", []string{"doc"})
	if err == nil || !strings.Contains(err.Error(), "artifact missing") {
		t.Fatalf("unavailable reranker must return the startup error, got %v", err)
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
