package embedding

import (
	"strings"
	"testing"
)

func TestEmbedderWorkerArgs(t *testing.T) {
	e := NewBGEEmbedder(Options{
		PythonPath:  "python3",
		ScriptPath:  "workers/embedding/embed.py",
		ModelName:   DefaultModel,
		Backend:     "onnx",
		ONNXDir:     "/tmp/bge-artifact",
		ONNXThreads: 3,
		BatchSize:   7,
	})

	args := strings.Join(e.workerArgs(false, "/tmp/in.json", "/tmp/out.json", "query"), " ")
	for _, want := range []string{
		"--backend onnx",
		"--model " + DefaultModel,
		"--onnx-dir /tmp/bge-artifact",
		"--onnx-threads 3",
		"--batch-size 7",
		"--input /tmp/in.json",
		"--output /tmp/out.json",
		"--mode query",
	} {
		if !strings.Contains(args, want) {
			t.Fatalf("worker args missing %q: %s", want, args)
		}
	}
	if strings.Contains(args, "--persistent") {
		t.Fatalf("one-shot args must not contain --persistent: %s", args)
	}

	persistentArgs := strings.Join(e.workerArgs(true, "", "", ""), " ")
	if !strings.Contains(persistentArgs, "--persistent") {
		t.Fatalf("persistent args missing --persistent: %s", persistentArgs)
	}
	if strings.Contains(persistentArgs, "--input") {
		t.Fatalf("persistent args must not contain one-shot input flags: %s", persistentArgs)
	}
	if !strings.Contains(persistentArgs, "--backend onnx") {
		t.Fatalf("persistent args missing backend: %s", persistentArgs)
	}
}

func TestNewBGEEmbedderDefaults(t *testing.T) {
	e := NewBGEEmbedder(Options{})
	if e.backend != DefaultBackend {
		t.Fatalf("backend = %q, want %q", e.backend, DefaultBackend)
	}
	if e.onnxDir != DefaultONNXDir {
		t.Fatalf("onnx dir = %q, want %q", e.onnxDir, DefaultONNXDir)
	}
	if e.onnxThreads != DefaultONNXThreads {
		t.Fatalf("onnx threads = %d, want %d", e.onnxThreads, DefaultONNXThreads)
	}
	if e.batchSize != DefaultBatchSize {
		t.Fatalf("batch size = %d, want %d", e.batchSize, DefaultBatchSize)
	}
	if e.timeout <= 0 {
		t.Fatalf("timeout must be positive")
	}
}
