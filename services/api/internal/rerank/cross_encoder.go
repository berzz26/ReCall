package rerank

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	// DefaultModel is the cross-encoder used for the reranking stage.
	DefaultModel = "mixedbread-ai/mxbai-rerank-xsmall-v1"
	// DefaultModelVersion tags rows/logs when the model is swapped.
	DefaultModelVersion = "v1"
	// DefaultBackend preserves the PyTorch inference path unless ONNX is
	// explicitly selected.
	DefaultBackend = "torch"
	// DefaultQuantization keeps full-precision inference unless INT8 is
	// explicitly selected.
	DefaultQuantization = "none"
	// DefaultONNXDir is where prepare_onnx.py writes artifacts by default.
	DefaultONNXDir = "workers/rerank/models/mxbai-rerank-xsmall-v1"
	// DefaultONNXThreads keeps ONNX Runtime from monopolising the host.
	DefaultONNXThreads = 2
	// DefaultBatchSize matches sentence-transformers' CrossEncoder default.
	DefaultBatchSize = 32

	persistentStartupTimeout = 300 * time.Second
)

// Options configures the subprocess-backed cross-encoder reranker.
type Options struct {
	PythonPath   string
	ScriptPath   string
	ModelName    string
	Backend      string // "torch" or "onnx"
	Quantization string // "none" or "int8" (ONNX only)
	ONNXDir      string
	ONNXThreads  int
	BatchSize    int
	Timeout      time.Duration
}

// CrossEncoderReranker invokes workers/rerank/rerank.py.
// It prefers a persistent subprocess (model loaded once) and falls back
// to a single one-shot execution only when no persistent worker was started.
type CrossEncoderReranker struct {
	pythonPath   string
	scriptPath   string
	modelName    string
	backend      string
	quantization string
	onnxDir      string
	onnxThreads  int
	batchSize    int
	timeout      time.Duration

	mu            sync.Mutex
	sess          *persistSession
	started       bool
	unavailable   error
	restartBudget int
	recoverMu     sync.Mutex
}

type persistSession struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout *bufio.Reader
	mu     sync.Mutex
	closed bool
	cancel context.CancelFunc
	done   chan error
}

func NewCrossEncoderReranker(opts Options) *CrossEncoderReranker {
	pythonPath := opts.PythonPath
	if pythonPath == "" {
		pythonPath = "python3"
	}
	scriptPath := opts.ScriptPath
	if scriptPath == "" {
		scriptPath = "workers/rerank/rerank.py"
	}
	modelName := opts.ModelName
	if modelName == "" {
		modelName = DefaultModel
	}
	backend := strings.ToLower(strings.TrimSpace(opts.Backend))
	if backend == "" {
		backend = DefaultBackend
	}
	quantization := strings.ToLower(strings.TrimSpace(opts.Quantization))
	if quantization == "" {
		quantization = DefaultQuantization
	}
	onnxDir := opts.ONNXDir
	if onnxDir == "" {
		onnxDir = DefaultONNXDir
	}
	onnxThreads := opts.ONNXThreads
	if onnxThreads <= 0 {
		onnxThreads = DefaultONNXThreads
	}
	batchSize := opts.BatchSize
	if batchSize <= 0 {
		batchSize = DefaultBatchSize
	}
	timeout := opts.Timeout
	if timeout <= 0 {
		timeout = 2 * time.Minute
	}
	if _, err := os.Stat(scriptPath); err != nil {
		if abs, err2 := filepath.Abs(scriptPath); err2 == nil {
			if _, err3 := os.Stat(abs); err3 == nil {
				scriptPath = abs
			}
		}
		if _, err := os.Stat(scriptPath); err != nil {
			alt := "/home/berzz/recall/workers/rerank/rerank.py"
			if _, err2 := os.Stat(alt); err2 == nil {
				scriptPath = alt
			}
		}
	} else {
		if abs, err := filepath.Abs(scriptPath); err == nil {
			scriptPath = abs
		}
	}
	return &CrossEncoderReranker{
		pythonPath:   pythonPath,
		scriptPath:   scriptPath,
		modelName:    modelName,
		backend:      backend,
		quantization: quantization,
		onnxDir:      onnxDir,
		onnxThreads:  onnxThreads,
		batchSize:    batchSize,
		timeout:      timeout,
	}
}

// StartPersistent launches the model-once worker. Startup is best-effort so
// the API still boots when the artifact/backend is broken; in that case
// reranking is disabled (searches keep vector order) instead of repeatedly
// reloading a model per request. Call once from main().
func (r *CrossEncoderReranker) StartPersistent(ctx context.Context) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.started {
		return
	}
	r.started = true
	sess, err := r.launchPersistent(ctx)
	if err != nil {
		wrapped := fmt.Errorf("rerank persistent worker failed to start (backend=%s, quantization=%s): %w; see rerank subprocess logs for the artifact/backend error", r.backend, r.quantization, err)
		r.unavailable = wrapped
		slog.Error("rerank: persistent worker failed to start; reranking disabled until API restart",
			"error", err, "backend", r.backend, "quantization", r.quantization)
		return
	}
	r.sess = sess
	r.restartBudget = 1
	slog.Info("rerank: persistent worker ready",
		"script", r.scriptPath, "model", r.modelName, "backend", r.backend,
		"quantization", r.quantization, "onnx_dir", r.onnxDir)
}

// Close stops the persistent worker if running.
func (r *CrossEncoderReranker) Close() {
	r.mu.Lock()
	s := r.sess
	r.sess = nil
	r.mu.Unlock()
	closePersistSession(s)
}

func closePersistSession(s *persistSession) {
	if s == nil {
		return
	}
	s.mu.Lock()
	if s.closed {
		s.mu.Unlock()
		return
	}
	s.closed = true
	_, _ = s.stdin.Write([]byte("{\"command\":\"exit\"}\n"))
	_ = s.stdin.Close()
	s.mu.Unlock()

	select {
	case <-s.done:
	case <-time.After(5 * time.Second):
		s.cancel()
		if s.cmd.Process != nil {
			_ = s.cmd.Process.Kill()
		}
		select {
		case <-s.done:
		case <-time.After(2 * time.Second):
		}
	}
}

func (r *CrossEncoderReranker) workerArgs(persistent bool, inputPath, outputPath string) []string {
	args := []string{r.scriptPath}
	if persistent {
		args = append(args, "--persistent")
	} else {
		args = append(args, "--input", inputPath, "--output", outputPath)
	}
	if r.modelName != "" {
		args = append(args, "--model", r.modelName)
	}
	args = append(args, "--backend", r.backend)
	args = append(args, "--quantization", r.quantization)
	args = append(args, "--onnx-dir", r.onnxDir)
	args = append(args, "--onnx-threads", strconv.Itoa(r.onnxThreads))
	args = append(args, "--batch-size", strconv.Itoa(r.batchSize))
	return args
}

func (r *CrossEncoderReranker) launchPersistent(ctx context.Context) (*persistSession, error) {
	if _, err := os.Stat(r.scriptPath); err != nil {
		return nil, fmt.Errorf("rerank script not found: %w", err)
	}
	sessCtx, cancel := context.WithCancel(context.Background())
	args := r.workerArgs(true, "", "")
	cmd := exec.CommandContext(sessCtx, r.pythonPath, args...)
	stdin, err := cmd.StdinPipe()
	if err != nil {
		cancel()
		return nil, err
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		return nil, err
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		cancel()
		return nil, err
	}
	go func() {
		br := bufio.NewReader(stderr)
		for {
			line, err := br.ReadString('\n')
			if err != nil {
				return
			}
			if trimmed := strings.TrimSpace(line); trimmed != "" {
				if strings.Contains(trimmed, `"level": "ERROR"`) || strings.Contains(trimmed, `"level":"ERROR"`) {
					slog.Error("rerank subprocess: " + trimmed)
				} else {
					slog.Info("rerank subprocess: " + trimmed)
				}
			}
		}
	}()
	if err := cmd.Start(); err != nil {
		cancel()
		return nil, err
	}
	s := &persistSession{cmd: cmd, stdin: stdin, stdout: bufio.NewReader(stdout), cancel: cancel, done: make(chan error, 1)}
	go func() { s.done <- cmd.Wait() }()
	readyCh := make(chan string, 1)
	errCh := make(chan error, 1)
	go func() {
		line, err := s.stdout.ReadString('\n')
		if err != nil {
			errCh <- err
			return
		}
		readyCh <- line
	}()
	timeout, cancelT := context.WithTimeout(ctx, persistentStartupTimeout)
	defer cancelT()
	select {
	case <-timeout.Done():
		_ = cmd.Process.Kill()
		cancel()
		return nil, fmt.Errorf("persistent rerank startup timeout: %w", timeout.Err())
	case err := <-errCh:
		cancel()
		return nil, fmt.Errorf("persistent rerank failed to start: %w", err)
	case line := <-readyCh:
		trimmed := strings.TrimSpace(line)
		var resp map[string]any
		if err := json.Unmarshal([]byte(trimmed), &resp); err != nil {
			_ = cmd.Process.Kill()
			cancel()
			return nil, fmt.Errorf("persistent rerank bad ready line %q: %w", trimmed, err)
		}
		if st, _ := resp["status"].(string); st != "ready" {
			_ = cmd.Process.Kill()
			cancel()
			return nil, fmt.Errorf("persistent rerank unexpected ready status %q", trimmed)
		}
		return s, nil
	}
}

type rerankInputItem struct {
	ID   string `json:"id"`
	Text string `json:"text"`
}

type rerankFileInput struct {
	Query string            `json:"query"`
	Items []rerankInputItem `json:"items"`
}

type rerankFileOutput struct {
	Items []rerankOutputItem `json:"items"`
}

type rerankOutputItem struct {
	ID    string  `json:"id"`
	Score float64 `json:"score"`
}

func (r *CrossEncoderReranker) Rerank(ctx context.Context, query string, docs []string) ([]float64, error) {
	if strings.TrimSpace(query) == "" {
		return nil, fmt.Errorf("empty query")
	}
	if len(docs) == 0 {
		return []float64{}, nil
	}
	items := make([]rerankInputItem, len(docs))
	ids := make([]string, len(docs))
	for i, d := range docs {
		// Cross-encoder needs text; empty descriptions score lowest
		// rather than failing the whole batch.
		if strings.TrimSpace(d) == "" {
			d = " "
			docs[i] = d
		}
		id := fmt.Sprintf("doc-%d", i)
		items[i] = rerankInputItem{ID: id, Text: d}
		ids[i] = id
	}
	m, err := r.scoreBatch(ctx, query, items)
	if err != nil {
		return nil, err
	}
	out := make([]float64, len(docs))
	for i, id := range ids {
		s, ok := m[id]
		if !ok {
			return nil, fmt.Errorf("missing rerank score for %s", id)
		}
		out[i] = s
	}
	return out, nil
}

func (r *CrossEncoderReranker) scoreBatch(ctx context.Context, query string, items []rerankInputItem) (map[string]float64, error) {
	m, sess, err := r.persistentRerank(ctx, query, items)
	if err == nil {
		return m, nil
	}
	if ctx.Err() != nil {
		r.invalidateSession(sess)
		return nil, err
	}
	r.invalidateSession(sess)

	r.mu.Lock()
	started := r.started
	unavailable := r.unavailable
	r.mu.Unlock()
	if !started {
		// Backward-compatible path for callers that did not call
		// StartPersistent (tests/one-off use).
		return r.oneShotRerank(ctx, query, items)
	}
	if unavailable != nil {
		return nil, unavailable
	}
	if err := r.recover(ctx); err != nil {
		return nil, err
	}
	m, sess, err = r.persistentRerank(ctx, query, items)
	if err != nil {
		r.invalidateSession(sess)
		r.setUnavailable(err)
		return nil, err
	}
	return m, nil
}

func (r *CrossEncoderReranker) invalidateSession(s *persistSession) {
	if s == nil {
		return
	}
	r.mu.Lock()
	if r.sess == s {
		r.sess = nil
	}
	r.mu.Unlock()
	closePersistSession(s)
}

func (r *CrossEncoderReranker) setUnavailable(err error) {
	r.mu.Lock()
	if r.unavailable == nil {
		r.unavailable = err
	}
	r.mu.Unlock()
}

// recover performs at most one persistent-worker restart for the process
// lifetime. This bounds the cost of a broken worker: subsequent searches fail
// fast and fall back to vector order until the API is restarted.
func (r *CrossEncoderReranker) recover(ctx context.Context) error {
	r.recoverMu.Lock()
	defer r.recoverMu.Unlock()

	r.mu.Lock()
	if r.sess != nil {
		r.mu.Unlock()
		return nil
	}
	if r.unavailable != nil {
		err := r.unavailable
		r.mu.Unlock()
		return err
	}
	if r.restartBudget <= 0 {
		r.mu.Unlock()
		return fmt.Errorf("rerank persistent worker unavailable; restart budget exhausted")
	}
	r.restartBudget--
	r.mu.Unlock()

	slog.Warn("rerank: restarting persistent worker after failure",
		"backend", r.backend, "quantization", r.quantization)
	sess, err := r.launchPersistent(ctx)
	if err != nil {
		if ctx.Err() != nil {
			// Caller went away mid-restart; restore the budget and do not
			// permanently disable reranking because of a cancellation.
			r.mu.Lock()
			r.restartBudget++
			r.mu.Unlock()
			return err
		}
		wrapped := fmt.Errorf("rerank persistent worker restart failed: %w", err)
		r.mu.Lock()
		r.unavailable = wrapped
		r.mu.Unlock()
		slog.Error("rerank: persistent worker restart failed; reranking disabled", "error", err)
		return wrapped
	}
	r.mu.Lock()
	r.sess = sess
	r.mu.Unlock()
	slog.Info("rerank: persistent worker recovered", "backend", r.backend, "quantization", r.quantization)
	return nil
}

func (r *CrossEncoderReranker) persistentRerank(ctx context.Context, query string, items []rerankInputItem) (map[string]float64, *persistSession, error) {
	r.mu.Lock()
	s := r.sess
	r.mu.Unlock()
	if s == nil {
		return nil, nil, fmt.Errorf("no persistent session")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, s, fmt.Errorf("persistent session closed")
	}
	tmpDir, err := os.MkdirTemp("", "rerank-persist-*")
	if err != nil {
		return nil, s, err
	}
	defer os.RemoveAll(tmpDir)
	inputPath := filepath.Join(tmpDir, "input.json")
	outputPath := filepath.Join(tmpDir, "output.json")
	data, _ := json.Marshal(rerankFileInput{Query: query, Items: items})
	if err := os.WriteFile(inputPath, data, 0644); err != nil {
		return nil, s, err
	}
	req, _ := json.Marshal(map[string]any{"input": inputPath, "output": outputPath})
	req = append(req, '\n')
	if _, err := s.stdin.Write(req); err != nil {
		return nil, s, err
	}
	respCh := make(chan string, 1)
	errCh := make(chan error, 1)
	go func() {
		line, err := s.stdout.ReadString('\n')
		if err != nil {
			errCh <- err
			return
		}
		respCh <- line
	}()
	timeoutCtx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()
	select {
	case <-timeoutCtx.Done():
		return nil, s, fmt.Errorf("rerank timeout: %w", timeoutCtx.Err())
	case err := <-errCh:
		return nil, s, fmt.Errorf("persistent rerank read failed: %w", err)
	case line := <-respCh:
		var resp map[string]any
		if err := json.Unmarshal([]byte(strings.TrimSpace(line)), &resp); err != nil {
			return nil, s, fmt.Errorf("bad persistent response %q: %w", strings.TrimSpace(line), err)
		}
		if st, _ := resp["status"].(string); st != "ok" {
			msg, _ := resp["error"].(string)
			return nil, s, fmt.Errorf("persistent rerank failed: %s", msg)
		}
		m, err := readScores(outputPath)
		return m, s, err
	}
}

func (r *CrossEncoderReranker) oneShotRerank(ctx context.Context, query string, items []rerankInputItem) (map[string]float64, error) {
	tmpDir, err := os.MkdirTemp("", "rerank-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmpDir)
	inputPath := filepath.Join(tmpDir, "input.json")
	outputPath := filepath.Join(tmpDir, "output.json")
	data, _ := json.Marshal(rerankFileInput{Query: query, Items: items})
	if err := os.WriteFile(inputPath, data, 0644); err != nil {
		return nil, err
	}
	timeoutCtx, cancel := context.WithTimeout(ctx, r.timeout)
	defer cancel()
	args := r.workerArgs(false, inputPath, outputPath)
	cmd := exec.CommandContext(timeoutCtx, r.pythonPath, args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	cmd.Stdout = &stderr
	if err := cmd.Run(); err != nil {
		msg := stderr.String()
		if len(msg) > 1000 {
			msg = msg[len(msg)-1000:]
		}
		if timeoutCtx.Err() == context.DeadlineExceeded {
			return nil, fmt.Errorf("rerank timeout: %w", timeoutCtx.Err())
		}
		return nil, fmt.Errorf("rerank failed: %s: %w", msg, err)
	}
	return readScores(outputPath)
}

func readScores(outputPath string) (map[string]float64, error) {
	outData, err := os.ReadFile(outputPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read rerank output: %w", err)
	}
	var out rerankFileOutput
	if err := json.Unmarshal(outData, &out); err != nil {
		return nil, fmt.Errorf("failed to parse rerank output: %w", err)
	}
	m := make(map[string]float64, len(out.Items))
	for _, it := range out.Items {
		m[it.ID] = it.Score
	}
	return m, nil
}
