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
	"strings"
	"sync"
	"time"
)

const (
	// DefaultModel is the cross-encoder used for the reranking stage.
	DefaultModel = "mixedbread-ai/mxbai-rerank-xsmall-v1"
	// DefaultModelVersion tags rows/logs when the model is swapped.
	DefaultModelVersion = "v1"
)

// CrossEncoderReranker invokes workers/rerank/rerank.py.
// It prefers a persistent subprocess (model loaded once) and falls back
// to one-shot execution if the persistent worker is unavailable.
type CrossEncoderReranker struct {
	pythonPath string
	scriptPath string
	modelName  string
	timeout    time.Duration

	mu      sync.Mutex
	sess    *persistSession
	started bool
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

func NewCrossEncoderReranker(pythonPath, scriptPath, modelName string, timeout time.Duration) *CrossEncoderReranker {
	if pythonPath == "" {
		pythonPath = "python3"
	}
	if scriptPath == "" {
		scriptPath = "workers/rerank/rerank.py"
	}
	if modelName == "" {
		modelName = DefaultModel
	}
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
	return &CrossEncoderReranker{pythonPath: pythonPath, scriptPath: scriptPath, modelName: modelName, timeout: timeout}
}

// StartPersistent launches the model-once worker. Best-effort: failures are
// logged and one-shot mode is used instead. Call once from main().
func (r *CrossEncoderReranker) StartPersistent(ctx context.Context) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.started {
		return
	}
	r.started = true
	sess, err := r.launchPersistent(ctx)
	if err != nil {
		slog.Warn("rerank: persistent worker unavailable, using one-shot", "error", err)
		return
	}
	r.sess = sess
	slog.Info("rerank: persistent worker ready", "script", r.scriptPath, "model", r.modelName)
}

//python3 Close stops the persistent worker if running.
func (r *CrossEncoderReranker) Close() {
	r.mu.Lock()
	s := r.sess
	r.sess = nil
	r.mu.Unlock()
	if s == nil {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return
	}
	s.closed = true
	_, _ = s.stdin.Write([]byte("{\"command\":\"exit\"}\n"))
	_ = s.stdin.Close()
	select {
	case <-s.done:
	case <-time.After(5 * time.Second):
		s.cancel()
		if s.cmd.Process != nil {
			_ = s.cmd.Process.Kill()
		}
		<-s.done
	}
}

func (r *CrossEncoderReranker) launchPersistent(ctx context.Context) (*persistSession, error) {
	if _, err := os.Stat(r.scriptPath); err != nil {
		return nil, fmt.Errorf("rerank script not found: %w", err)
	}
	sessCtx, cancel := context.WithCancel(context.Background())
	args := []string{r.scriptPath, "--persistent"}
	if r.modelName != "" {
		args = append(args, "--model", r.modelName)
	}
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
				slog.Info("rerank subprocess: " + trimmed)
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
	timeout, cancelT := context.WithTimeout(ctx, 300*time.Second)
	defer cancelT()
	select {
	case <-timeout.Done():
		_ = cmd.Process.Kill()
		return nil, fmt.Errorf("persistent rerank startup timeout: %w", timeout.Err())
	case err := <-errCh:
		return nil, fmt.Errorf("persistent rerank failed to start: %w", err)
	case line := <-readyCh:
		trimmed := strings.TrimSpace(line)
		var resp map[string]any
		if err := json.Unmarshal([]byte(trimmed), &resp); err != nil {
			_ = cmd.Process.Kill()
			return nil, fmt.Errorf("persistent rerank bad ready line %q: %w", trimmed, err)
		}
		if st, _ := resp["status"].(string); st != "ready" {
			_ = cmd.Process.Kill()
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
	Query string           `json:"query"`
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
	if m, err := r.persistentRerank(ctx, query, items); err == nil {
		return m, nil
	} else {
		r.mu.Lock()
		hasSess := r.sess != nil
		r.mu.Unlock()
		if hasSess {
			slog.Warn("rerank: persistent batch failed, falling back to one-shot", "error", err)
		}
	}
	return r.oneShotRerank(ctx, query, items)
}

func (r *CrossEncoderReranker) persistentRerank(ctx context.Context, query string, items []rerankInputItem) (map[string]float64, error) {
	r.mu.Lock()
	s := r.sess
	r.mu.Unlock()
	if s == nil {
		return nil, fmt.Errorf("no persistent session")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, fmt.Errorf("persistent session closed")
	}
	tmpDir, err := os.MkdirTemp("", "rerank-persist-*")
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
	req, _ := json.Marshal(map[string]any{"input": inputPath, "output": outputPath})
	req = append(req, '\n')
	if _, err := s.stdin.Write(req); err != nil {
		return nil, err
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
		return nil, fmt.Errorf("rerank timeout: %w", timeoutCtx.Err())
	case err := <-errCh:
		return nil, fmt.Errorf("persistent rerank read failed: %w", err)
	case line := <-respCh:
		var resp map[string]any
		if err := json.Unmarshal([]byte(strings.TrimSpace(line)), &resp); err != nil {
			return nil, fmt.Errorf("bad persistent response %q: %w", strings.TrimSpace(line), err)
		}
		if st, _ := resp["status"].(string); st != "ok" {
			msg, _ := resp["error"].(string)
			return nil, fmt.Errorf("persistent rerank failed: %s", msg)
		}
		return readScores(outputPath)
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
	args := []string{r.scriptPath, "--input", inputPath, "--output", outputPath}
	if r.modelName != "" {
		args = append(args, "--model", r.modelName)
	}
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
