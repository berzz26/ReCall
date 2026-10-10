package embedding

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

	"github.com/google/uuid"
)

const (
	DefaultModel        = "BAAI/bge-small-en-v1.5"
	DefaultModelVersion = "v1.5"
	QueryPrefix         = "Represent this sentence for searching relevant passages: "
)

// BGEEmbedder invokes workers/embedding/embed.py.
// It prefers a persistent subprocess (model loaded once) and falls back
// to one-shot execution if the persistent worker is unavailable.
type BGEEmbedder struct {
	pythonPath string
	scriptPath string
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

func NewBGEEmbedder(pythonPath, scriptPath string, timeout time.Duration) *BGEEmbedder {
	if pythonPath == "" {
		pythonPath = "python3"
	}
	if scriptPath == "" {
		scriptPath = "workers/embedding/embed.py"
	}
	if timeout <= 0 {
		timeout = 5 * time.Minute
	}
	// Resolve script path similarly to detector
	if _, err := os.Stat(scriptPath); err != nil {
		if abs, err2 := filepath.Abs(scriptPath); err2 == nil {
			if _, err3 := os.Stat(abs); err3 == nil {
				scriptPath = abs
			}
		}
		if _, err := os.Stat(scriptPath); err != nil {
			alt := "/home/berzz/recall/workers/embedding/embed.py"
			if _, err2 := os.Stat(alt); err2 == nil {
				scriptPath = alt
			}
		}
	} else {
		if abs, err := filepath.Abs(scriptPath); err == nil {
			scriptPath = abs
		}
	}
	return &BGEEmbedder{pythonPath: pythonPath, scriptPath: scriptPath, timeout: timeout}
}

// StartPersistent launches the model-once worker. Best-effort: failures are
// logged and one-shot mode is used instead. Call once from main().
func (b *BGEEmbedder) StartPersistent(ctx context.Context) {
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.started {
		return
	}
	b.started = true
	sess, err := b.launchPersistent(ctx)
	if err != nil {
		slog.Warn("embedding: persistent worker unavailable, using one-shot", "error", err)
		return
	}
	b.sess = sess
	slog.Info("embedding: persistent worker ready", "script", b.scriptPath)
}

// Close stops the persistent worker if running.
func (b *BGEEmbedder) Close() {
	b.mu.Lock()
	s := b.sess
	b.sess = nil
	b.mu.Unlock()
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

func (b *BGEEmbedder) launchPersistent(ctx context.Context) (*persistSession, error) {
	if _, err := os.Stat(b.scriptPath); err != nil {
		return nil, fmt.Errorf("embedding script not found: %w", err)
	}
	sessCtx, cancel := context.WithCancel(context.Background())
	cmd := exec.CommandContext(sessCtx, b.pythonPath, b.scriptPath, "--persistent")
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
				slog.Info("embedding subprocess: " + trimmed)
			}
		}
	}()
	if err := cmd.Start(); err != nil {
		cancel()
		return nil, err
	}
	s := &persistSession{cmd: cmd, stdin: stdin, stdout: bufio.NewReader(stdout), cancel: cancel, done: make(chan error, 1)}
	go func() { s.done <- cmd.Wait() }()
	// Wait for ready with timeout
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
	timeout, cancelT := context.WithTimeout(ctx, 180*time.Second)
	defer cancelT()
	select {
	case <-timeout.Done():
		_ = cmd.Process.Kill()
		return nil, fmt.Errorf("persistent embedding startup timeout: %w", timeout.Err())
	case err := <-errCh:
		return nil, fmt.Errorf("persistent embedding failed to start: %w", err)
	case line := <-readyCh:
		trimmed := strings.TrimSpace(line)
		var resp map[string]any
		if err := json.Unmarshal([]byte(trimmed), &resp); err != nil {
			_ = cmd.Process.Kill()
			return nil, fmt.Errorf("persistent embedding bad ready line %q: %w", trimmed, err)
		}
		if st, _ := resp["status"].(string); st != "ready" {
			_ = cmd.Process.Kill()
			return nil, fmt.Errorf("persistent embedding unexpected ready status %q", trimmed)
		}
		return s, nil
	}
}

type embedInput struct {
	Items []embedInputItem `json:"items"`
	Mode  string           `json:"mode,omitempty"`
}

type embedInputItem struct {
	ID   string `json:"id"`
	Text string `json:"text"`
}

type embedOutput struct {
	Items []embedOutputItem `json:"items"`
}

type embedOutputItem struct {
	ID        string    `json:"id"`
	Embedding []float32 `json:"embedding"`
}

func (b *BGEEmbedder) persistentEmbed(ctx context.Context, items []embedInputItem, mode string) (map[string][]float32, error) {
	b.mu.Lock()
	s := b.sess
	b.mu.Unlock()
	if s == nil {
		return nil, fmt.Errorf("no persistent session")
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.closed {
		return nil, fmt.Errorf("persistent session closed")
	}
	tmpDir, err := os.MkdirTemp("", "bge-persist-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmpDir)
	inputPath := filepath.Join(tmpDir, "input.json")
	outputPath := filepath.Join(tmpDir, "output.json")
	data, _ := json.Marshal(embedInput{Items: items, Mode: mode})
	if err := os.WriteFile(inputPath, data, 0644); err != nil {
		return nil, err
	}
	req, _ := json.Marshal(map[string]any{"input": inputPath, "output": outputPath, "mode": mode})
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
	timeoutCtx, cancel := context.WithTimeout(ctx, b.timeout)
	defer cancel()
	select {
	case <-timeoutCtx.Done():
		return nil, fmt.Errorf("embedding timeout: %w", timeoutCtx.Err())
	case err := <-errCh:
		return nil, fmt.Errorf("persistent embedding read failed: %w", err)
	case line := <-respCh:
		var resp map[string]any
		if err := json.Unmarshal([]byte(strings.TrimSpace(line)), &resp); err != nil {
			return nil, fmt.Errorf("bad persistent response %q: %w", strings.TrimSpace(line), err)
		}
		if st, _ := resp["status"].(string); st != "ok" {
			msg, _ := resp["error"].(string)
			return nil, fmt.Errorf("persistent embedding failed: %s", msg)
		}
		outData, err := os.ReadFile(outputPath)
		if err != nil {
			return nil, fmt.Errorf("failed to read embedding output: %w", err)
		}
		var out embedOutput
		if err := json.Unmarshal(outData, &out); err != nil {
			return nil, fmt.Errorf("failed to parse embedding output: %w", err)
		}
		return toMap(out)
	}
}

func toMap(out embedOutput) (map[string][]float32, error) {
	m := make(map[string][]float32, len(out.Items))
	for _, it := range out.Items {
		if len(it.Embedding) != 384 {
			return nil, fmt.Errorf("invalid embedding dimension for %s: %d != 384", it.ID, len(it.Embedding))
		}
		m[it.ID] = it.Embedding
	}
	return m, nil
}

func (b *BGEEmbedder) embedBatch(ctx context.Context, items []embedInputItem, mode string) (map[string][]float32, error) {
	if len(items) == 0 {
		return map[string][]float32{}, nil
	}
	// Prefer persistent worker; fall back to one-shot on any error.
	if m, err := b.persistentEmbed(ctx, items, mode); err == nil {
		return m, nil
	} else {
		b.mu.Lock()
		hasSess := b.sess != nil
		b.mu.Unlock()
		if hasSess {
			slog.Warn("embedding: persistent batch failed, falling back to one-shot", "error", err)
		}
	}
	tmpDir, err := os.MkdirTemp("", "bge-*")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(tmpDir)
	inputPath := filepath.Join(tmpDir, "input.json")
	outputPath := filepath.Join(tmpDir, "output.json")
	in := embedInput{Items: items, Mode: mode}
	data, _ := json.Marshal(in)
	if err := os.WriteFile(inputPath, data, 0644); err != nil {
		return nil, err
	}
	timeoutCtx, cancel := context.WithTimeout(ctx, b.timeout)
	defer cancel()
	cmd := exec.CommandContext(timeoutCtx, b.pythonPath, b.scriptPath, "--input", inputPath, "--output", outputPath, "--mode", mode)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	cmd.Stdout = &stderr
	if err := cmd.Run(); err != nil {
		msg := stderr.String()
		if len(msg) > 1000 {
			msg = msg[len(msg)-1000:]
		}
		if timeoutCtx.Err() == context.DeadlineExceeded {
			return nil, fmt.Errorf("embedding timeout: %w", timeoutCtx.Err())
		}
		return nil, fmt.Errorf("embedding failed: %s: %w", msg, err)
	}
	outData, err := os.ReadFile(outputPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read embedding output: %w", err)
	}
	var out embedOutput
	if err := json.Unmarshal(outData, &out); err != nil {
		return nil, fmt.Errorf("failed to parse embedding output: %w", err)
	}
	return toMap(out)
}

func (b *BGEEmbedder) EmbedPassages(ctx context.Context, texts []string) ([][]float32, error) {
	items := make([]embedInputItem, len(texts))
	for i, t := range texts {
		if t == "" {
			return nil, fmt.Errorf("empty passage text at index %d", i)
		}
		items[i] = embedInputItem{ID: uuid.NewString(), Text: t}
	}
	m, err := b.embedBatch(ctx, items, "passage")
	if err != nil {
		return nil, err
	}
	out := make([][]float32, len(texts))
	for i, it := range items {
		v, ok := m[it.ID]
		if !ok {
			return nil, fmt.Errorf("missing embedding for %s", it.ID)
		}
		out[i] = v
	}
	return out, nil
}

func (b *BGEEmbedder) EmbedQuery(ctx context.Context, query string) ([]float32, error) {
	if query == "" {
		return nil, fmt.Errorf("empty query")
	}
	// BGE convention: prefix query
	id := uuid.NewString()
	items := []embedInputItem{{ID: id, Text: query}}
	// embed.py will add prefix when mode=query
	m, err := b.embedBatch(ctx, items, "query")
	if err != nil {
		return nil, err
	}
	v, ok := m[id]
	if !ok {
		return nil, fmt.Errorf("missing query embedding")
	}
	return v, nil
}

// EmbedPassagesWithIDs is helper for service layer to keep ID mapping.
func (b *BGEEmbedder) EmbedPassagesWithIDs(ctx context.Context, ids []string, texts []string) (map[string][]float32, error) {
	if len(ids) != len(texts) {
		return nil, fmt.Errorf("ids/texts length mismatch")
	}
	items := make([]embedInputItem, len(ids))
	for i := range ids {
		if texts[i] == "" {
			return nil, fmt.Errorf("empty text for id %s", ids[i])
		}
		items[i] = embedInputItem{ID: ids[i], Text: texts[i]}
	}
	return b.embedBatch(ctx, items, "passage")
}
