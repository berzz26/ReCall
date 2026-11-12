// Package onnxrt initializes the shared ONNX Runtime C library once per
// process. Both the native embedding and rerank backends run inference
// in-process (no Python subprocess), so the environment must be created
// exactly once before any session is built.
package onnxrt

import (
	"fmt"
	"sync"

	ort "github.com/yalue/onnxruntime_go"
)

var (
	mu      sync.Mutex
	started bool
	initErr error
)

// Ensure initializes the ONNX Runtime shared library. Safe to call from
// multiple consumers (embedding, rerank): the first call wins and later
// calls are no-ops. runtimePath may be empty to rely on the system search
// path.
func Ensure(runtimePath string) error {
	mu.Lock()
	defer mu.Unlock()
	if started {
		return initErr
	}
	started = true
	if runtimePath != "" {
		ort.SetSharedLibraryPath(runtimePath)
	}
	if err := ort.InitializeEnvironment(); err != nil {
		initErr = fmt.Errorf("initialize onnxruntime: %w", err)
	}
	return initErr
}
