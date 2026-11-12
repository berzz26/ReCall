package rerank

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/berzz26/recall/services/api/internal/onnxrt"
	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
	ort "github.com/yalue/onnxruntime_go"
)

const (
	// NativeBackend is the in-process ONNX Runtime backend: the Go server
	// loads the prepared local artifact directly (no Python subprocess).
	NativeBackend = "native"
)

// NativeOptions configures the in-process reranker.
type NativeOptions struct {
	ModelName    string
	Quantization string // "none" or "int8" (selects the artifact graph)
	ONNXDir      string
	RuntimePath  string
	ONNXThreads  int
	BatchSize    int
}

// NativeReranker implements Reranker by running the local cross-encoder ONNX
// graph in-process. Scoring semantics mirror workers/rerank/rerank.py's ONNX
// backend: (query, document) pairs tokenized with longest-first truncation at
// the artifact's max_length, batch padded to the batch's longest sequence,
// raw logits returned (no sigmoid/softmax) — the search service only relies
// on ordering.
type NativeReranker struct {
	modelName    string
	quantization string
	onnxDir      string
	runtimePath  string
	onnxThreads  int
	batchSize    int

	mu         sync.Mutex
	session    *ort.DynamicAdvancedSession
	tok        *tokenizer.Tokenizer
	inputNames []string
	outName    string
	maxTokens  int

	loaded bool
	closed bool
}

// NewNativeReranker returns an unloaded reranker; call StartPersistent before
// first use.
func NewNativeReranker(opts NativeOptions) *NativeReranker {
	if opts.ModelName == "" {
		opts.ModelName = DefaultModel
	}
	if opts.Quantization == "" {
		opts.Quantization = DefaultQuantization
	}
	if opts.ONNXDir == "" {
		opts.ONNXDir = DefaultONNXDir
	}
	if opts.RuntimePath == "" {
		opts.RuntimePath = embeddingDefaultORTRuntimePath
	}
	if opts.ONNXThreads <= 0 {
		opts.ONNXThreads = DefaultONNXThreads
	}
	if opts.BatchSize <= 0 {
		opts.BatchSize = DefaultBatchSize
	}
	return &NativeReranker{
		modelName:    opts.ModelName,
		quantization: opts.Quantization,
		onnxDir:      opts.ONNXDir,
		runtimePath:  opts.RuntimePath,
		onnxThreads:  opts.ONNXThreads,
		batchSize:    opts.BatchSize,
	}
}

// StartPersistent loads the model in-process (idempotent). Startup is
// best-effort: on failure the reranker is marked unavailable so searches
// keep vector order instead of failing, exactly like the subprocess backend.
func (n *NativeReranker) StartPersistent(ctx context.Context) {
	_ = ctx
	if err := n.Load(); err != nil {
		slog.Error("rerank: native session failed to load; reranking disabled until API restart",
			"error", err, "model", n.modelName, "quantization", n.quantization)
	}
}

// Close destroys the in-process session.
func (n *NativeReranker) Close() error {
	n.mu.Lock()
	defer n.mu.Unlock()
	if n.closed {
		return nil
	}
	n.closed = true
	if n.session != nil {
		_ = n.session.Destroy()
		n.session = nil
	}
	if n.tok != nil {
		n.tok = nil
	}
	return nil
}

// Load initializes ONNX Runtime, the tokenizer and the session using the
// artifact selected by quantization from rerank_artifact.json.
func (n *NativeReranker) Load() error {
	n.mu.Lock()
	defer n.mu.Unlock()
	if n.loaded {
		return nil
	}
	if n.closed {
		return fmt.Errorf("rerank: native reranker already closed")
	}
	if err := onnxrt.Ensure(n.runtimePath); err != nil {
		return fmt.Errorf("rerank: %w", err)
	}

	onnxDir, err := filepath.Abs(n.onnxDir)
	if err != nil {
		return fmt.Errorf("rerank: resolve onnx dir: %w", err)
	}
	metadataPath := filepath.Join(onnxDir, "rerank_artifact.json")
	raw, err := os.ReadFile(metadataPath)
	if err != nil {
		return fmt.Errorf("rerank: artifact metadata %s: %w", metadataPath, err)
	}
	var meta struct {
		SourceModel string `json:"source_model"`
		MaxLength   int    `json:"max_length"`
		Artifacts   map[string]struct {
			File string `json:"file"`
		} `json:"artifacts"`
	}
	if err := json.Unmarshal(raw, &meta); err != nil {
		return fmt.Errorf("rerank: bad rerank_artifact.json: %w", err)
	}
	if meta.SourceModel != n.modelName {
		return fmt.Errorf("rerank: artifact at %s was prepared for model %q, but RERANKER_MODEL is %q", onnxDir, meta.SourceModel, n.modelName)
	}
	artifact, ok := meta.Artifacts[n.quantization]
	if !ok || artifact.File == "" {
		available := make([]string, 0, len(meta.Artifacts))
		for k := range meta.Artifacts {
			available = append(available, k)
		}
		return fmt.Errorf("rerank: artifact for quantization=%q missing in %s (available: %s)", n.quantization, onnxDir, strings.Join(available, ", "))
	}
	modelPath := filepath.Join(onnxDir, artifact.File)
	if _, err := os.Stat(modelPath); err != nil {
		return fmt.Errorf("rerank: ONNX graph %s: %w", modelPath, err)
	}

	tokenizerPath := filepath.Join(onnxDir, "tokenizer.json")
	if _, err := os.Stat(tokenizerPath); err != nil {
		return fmt.Errorf("rerank: tokenizer artifact %s: %w", tokenizerPath, err)
	}
	tok, err := pretrained.FromFile(tokenizerPath)
	if err != nil {
		return fmt.Errorf("rerank: load tokenizer: %w", err)
	}
	maxLen := meta.MaxLength
	if maxLen <= 0 || maxLen > 512 {
		maxLen = 512
	}
	// Mirror sentence-transformers via rerank.py: padding=True (per-batch
	// longest, applied manually below) and truncation="longest_first".
	tok.WithTruncation(&tokenizer.TruncationParams{
		MaxLength: maxLen,
		Strategy:  tokenizer.LongestFirst,
	})

	inputs, outputs, err := ort.GetInputOutputInfo(modelPath)
	if err != nil {
		return fmt.Errorf("rerank: inspect model: %w", err)
	}
	inputNames := make([]string, len(inputs))
	for i, in := range inputs {
		inputNames[i] = in.Name
	}
	known := map[string]bool{"input_ids": true, "attention_mask": true, "token_type_ids": true}
	for _, name := range inputNames {
		if !known[name] {
			return fmt.Errorf("rerank: unsupported ONNX model input %q", name)
		}
	}
	if len(outputs) != 1 {
		return fmt.Errorf("rerank: expected exactly one model output, got %d", len(outputs))
	}

	options, err := ort.NewSessionOptions()
	if err != nil {
		return fmt.Errorf("rerank: create session options: %w", err)
	}
	if err := options.SetIntraOpNumThreads(n.onnxThreads); err != nil {
		_ = options.Destroy()
		return fmt.Errorf("rerank: set ONNX threads: %w", err)
	}
	session, err := ort.NewDynamicAdvancedSession(modelPath, inputNames, []string{outputs[0].Name}, options)
	_ = options.Destroy()
	if err != nil {
		return fmt.Errorf("rerank: create model session: %w", err)
	}

	n.session = session
	n.tok = tok
	n.inputNames = inputNames
	n.outName = outputs[0].Name
	n.maxTokens = maxLen
	n.loaded = true
	slog.Info("rerank: native ONNX session ready",
		"onnx_dir", onnxDir, "artifact", artifact.File, "max_length", maxLen, "threads", n.onnxThreads)
	return nil
}

func (n *NativeReranker) requireLoaded() error {
	n.mu.Lock()
	defer n.mu.Unlock()
	if n.closed {
		return fmt.Errorf("rerank: native reranker closed")
	}
	if !n.loaded {
		return fmt.Errorf("rerank: native reranker not loaded")
	}
	return nil
}

// Rerank scores (query, document) pairs and returns raw logits, matching the
// subprocess backend's contract with the search service.
func (n *NativeReranker) Rerank(ctx context.Context, query string, docs []string) ([]float64, error) {
	if strings.TrimSpace(query) == "" {
		return nil, fmt.Errorf("empty query")
	}
	if len(docs) == 0 {
		return []float64{}, nil
	}
	if err := n.requireLoaded(); err != nil {
		return nil, err
	}
	n.mu.Lock()
	defer n.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	out := make([]float64, 0, len(docs))
	for base := 0; base < len(docs); base += n.batchSize {
		end := base + n.batchSize
		if end > len(docs) {
			end = len(docs)
		}
		scores, err := n.scoreChunk(query, docs[base:end])
		if err != nil {
			return nil, err
		}
		out = append(out, scores...)
	}
	return out, nil
}

// scoreChunk encodes one batch of pairs (padded to the batch's own longest
// sequence) and runs the session.
func (n *NativeReranker) scoreChunk(query string, docs []string) ([]float64, error) {
	encs := make([]*tokenizer.Encoding, len(docs))
	maxLen := 0
	for i, doc := range docs {
		text := doc
		if strings.TrimSpace(text) == "" {
			// Cross-encoder needs text; empty descriptions score lowest
			// rather than failing the whole batch (matches rerank.py).
			text = " "
		}
		enc, err := n.tok.Encode(tokenizer.NewDualEncodeInput(
			tokenizer.NewInputSequence(query),
			tokenizer.NewInputSequence(text),
		), true)
		if err != nil {
			return nil, fmt.Errorf("rerank: tokenize pair %d: %w", i, err)
		}
		encs[i] = enc
		if enc.Len() > maxLen {
			maxLen = enc.Len()
		}
	}

	ids := make([]int64, len(docs)*maxLen)
	masks := make([]int64, len(ids))
	types := make([]int64, len(ids))
	for i, enc := range encs {
		if len(enc.AttentionMask) != len(enc.Ids) {
			return nil, fmt.Errorf("rerank: tokenizer returned mismatched ids and attention mask lengths")
		}
		for j, id := range enc.Ids {
			if id < 0 || uint64(id) > maxTokenID {
				return nil, fmt.Errorf("rerank: invalid token id %d", id)
			}
			ids[i*maxLen+j] = int64(id)
			masks[i*maxLen+j] = int64(enc.AttentionMask[j])
			if j < len(enc.TypeIds) {
				types[i*maxLen+j] = int64(enc.TypeIds[j])
			}
		}
	}

	shape := ort.NewShape(int64(len(docs)), int64(maxLen))
	ti, err := ort.NewTensor(shape, ids)
	if err != nil {
		return nil, err
	}
	defer ti.Destroy()
	tm, err := ort.NewTensor(shape, masks)
	if err != nil {
		return nil, err
	}
	defer tm.Destroy()
	tt, err := ort.NewTensor(shape, types)
	if err != nil {
		return nil, err
	}
	defer tt.Destroy()

	values := map[string]ort.Value{"input_ids": ti, "attention_mask": tm, "token_type_ids": tt}
	inputVals := make([]ort.Value, len(n.inputNames))
	for i, name := range n.inputNames {
		v, ok := values[name]
		if !ok {
			return nil, fmt.Errorf("rerank: unsupported model input %q", name)
		}
		inputVals[i] = v
	}
	outs := []ort.Value{nil}
	if err := n.session.Run(inputVals, outs); err != nil {
		return nil, fmt.Errorf("rerank: inference batch: %w", err)
	}
	tensor, ok := outs[0].(*ort.Tensor[float32])
	if !ok {
		if outs[0] != nil {
			outs[0].Destroy()
		}
		return nil, fmt.Errorf("rerank: unexpected output type")
	}
	defer tensor.Destroy()
	data := tensor.GetData()
	dims := tensor.GetShape()
	// Expected [batch, 1] for a single-label cross-encoder (matches rerank.py).
	if len(dims) == 2 && dims[1] == 1 {
		// ok
	} else {
		return nil, fmt.Errorf("rerank: unexpected logits shape %v; expected [batch, 1]", dims)
	}
	if len(data) != len(docs) {
		return nil, fmt.Errorf("rerank: got %d scores for %d pairs", len(data), len(docs))
	}
	scores := make([]float64, len(docs))
	for i, v := range data {
		scores[i] = float64(v)
	}
	return scores, nil
}

const maxTokenID = 1<<31 - 1

// embeddingDefaultORTRuntimePath mirrors the embedding package default so
// both backends share one runtime library when ONNXRUNTIME_LIB_PATH is unset.
const embeddingDefaultORTRuntimePath = "third_party/onnxruntime/onnxruntime-linux-x64-1.31.1/lib/libonnxruntime.so"
