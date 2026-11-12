package embedding

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"sync"

	"github.com/berzz26/recall/services/api/internal/onnxrt"
	ort "github.com/yalue/onnxruntime_go"
	"github.com/sugarme/tokenizer"
	"github.com/sugarme/tokenizer/pretrained"
)

const (
	// NativeBackend is the in-process ONNX Runtime backend: the Go server
	// loads the local artifact directly (no Python subprocess, no numpy).
	// Cold start stays under a second on warm page cache.
	NativeBackend = "native"
	// DefaultORTRuntimePath is the bundled ONNX Runtime shared library used
	// when ONNXRUNTIME_LIB_PATH is not set.
	DefaultORTRuntimePath = "third_party/onnxruntime/onnxruntime-linux-x64-1.31.1/lib/libonnxruntime.so"
)

// NativeOptions configures the in-process embedder.
type NativeOptions struct {
	ONNXDir     string
	RuntimePath string
	ONNXThreads int
	BatchSize   int
}

// NativeEmbedder implements TextEmbedder by running the local BGE ONNX graph
// in-process. Semantics mirror workers/embedding/embed.py's OnnxEmbedder:
// artifact metadata selects CLS or mean pooling, queries get the BGE
// retrieval prefix, vectors are L2-normalized, and each batch is padded to
// its own longest sequence so stored vectors stay byte-identical with the
// Python worker's output.
type NativeEmbedder struct {
	onnxDir     string
	runtimePath string
	onnxThreads int
	batchSize   int

	mu         sync.Mutex
	session    *ort.DynamicAdvancedSession
	tok        *tokenizer.Tokenizer
	inputNames []string
	outName    string
	pooling    string
	maxTokens  int
	dimensions int

	loaded bool
	closed bool
}

// NewNativeEmbedder returns an unloaded embedder; call StartPersistent (or
// Load) before first use.
func NewNativeEmbedder(opts NativeOptions) *NativeEmbedder {
	if opts.ONNXDir == "" {
		opts.ONNXDir = DefaultONNXDir
	}
	if opts.RuntimePath == "" {
		opts.RuntimePath = DefaultORTRuntimePath
	}
	if opts.ONNXThreads <= 0 {
		opts.ONNXThreads = DefaultONNXThreads
	}
	if opts.BatchSize <= 0 {
		opts.BatchSize = DefaultBatchSize
	}
	return &NativeEmbedder{
		onnxDir:     opts.ONNXDir,
		runtimePath: opts.RuntimePath,
		onnxThreads: opts.ONNXThreads,
		batchSize:   opts.BatchSize,
	}
}

// Load initializes ONNX Runtime, the tokenizer and the session. It mirrors
// the Python worker's semantics, including the artifact metadata check.
func (n *NativeEmbedder) Load() (err error) {
	n.mu.Lock()
	defer n.mu.Unlock()
	if n.loaded {
		return nil
	}
	if n.closed {
		return fmt.Errorf("embedding: native embedder already closed")
	}
	if err := onnxrt.Ensure(n.runtimePath); err != nil {
		return fmt.Errorf("embedding: %w", err)
	}

	onnxDir, err := filepath.Abs(n.onnxDir)
	if err != nil {
		return fmt.Errorf("embedding: resolve onnx dir: %w", err)
	}
	modelPath := filepath.Join(onnxDir, "model.onnx")
	tokenizerPath := filepath.Join(onnxDir, "tokenizer.json")
	for _, p := range []string{modelPath, tokenizerPath} {
		if _, err := os.Stat(p); err != nil {
			return fmt.Errorf("embedding: artifact %s: %w", p, err)
		}
	}

	// Artifact metadata drives pooling mode, max length and dimension checks
	// exactly like embed.py does.
	meta := struct {
		Pooling    string `json:"pooling"`
		MaxLength  int    `json:"max_length"`
		Dimensions int    `json:"dimensions"`
	}{Pooling: "cls", MaxLength: 512, Dimensions: 384}
	if raw, err := os.ReadFile(filepath.Join(onnxDir, "embedding_artifact.json")); err == nil {
		if err := json.Unmarshal(raw, &meta); err != nil {
			return fmt.Errorf("embedding: bad embedding_artifact.json: %w", err)
		}
	}
	if meta.Pooling != "cls" && meta.Pooling != "mean" {
		return fmt.Errorf("embedding: unsupported pooling mode %q", meta.Pooling)
	}
	maxLen := meta.MaxLength
	if maxLen <= 0 || maxLen > 512 {
		maxLen = 512
	}

	inputs, outputs, err := ort.GetInputOutputInfo(modelPath)
	if err != nil {
		return fmt.Errorf("embedding: inspect model: %w", err)
	}
	inputNames := make([]string, len(inputs))
	for i, in := range inputs {
		inputNames[i] = in.Name
	}
	known := map[string]bool{"input_ids": true, "attention_mask": true, "token_type_ids": true}
	for _, name := range inputNames {
		if !known[name] {
			return fmt.Errorf("embedding: unsupported ONNX model input %q", name)
		}
	}
	if len(outputs) != 1 || len(outputs[0].Dimensions) != 3 || outputs[0].Dimensions[2] != int64(meta.Dimensions) {
		return fmt.Errorf("embedding: expected token-level output with dimension %d", meta.Dimensions)
	}

	tok, err := pretrained.FromFile(tokenizerPath)
	if err != nil {
		return fmt.Errorf("embedding: load tokenizer: %w", err)
	}
	// Match embed.py: enable_truncation(max_length) then manual per-batch
	// padding to the chunk's longest sequence (padding=True semantics).
	tok.WithTruncation(&tokenizer.TruncationParams{
		MaxLength: maxLen,
		Strategy:  tokenizer.LongestFirst,
	})

	options, err := ort.NewSessionOptions()
	if err != nil {
		return fmt.Errorf("embedding: create session options: %w", err)
	}
	if err := options.SetIntraOpNumThreads(n.onnxThreads); err != nil {
		_ = options.Destroy()
		return fmt.Errorf("embedding: set ONNX threads: %w", err)
	}
	session, err := ort.NewDynamicAdvancedSession(modelPath, inputNames, []string{outputs[0].Name}, options)
	_ = options.Destroy()
	if err != nil {
		return fmt.Errorf("embedding: create model session: %w", err)
	}

	n.session = session
	n.tok = tok
	n.inputNames = inputNames
	n.outName = outputs[0].Name
	n.pooling = meta.Pooling
	n.maxTokens = maxLen
	n.dimensions = meta.Dimensions
	n.loaded = true
	slog.Info("embedding: native ONNX session ready",
		"onnx_dir", onnxDir, "pooling", n.pooling, "max_length", maxLen, "threads", n.onnxThreads)
	return nil
}

// StartPersistent satisfies the BGEEmbedder call pattern in main.go. For the
// native backend the model loads in-process; startup errors are returned and
// also remembered for Close-time logging. The provided context is unused
// because loading cannot be cancelled mid-session-creation.
func (n *NativeEmbedder) StartPersistent(ctx context.Context) {
	_ = ctx
	if err := n.Load(); err != nil {
		slog.Error("embedding: native session failed to load", "error", err)
	}
}

// Close destroys the in-process session.
func (n *NativeEmbedder) Close() error {
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

func (n *NativeEmbedder) requireLoaded() error {
	n.mu.Lock()
	defer n.mu.Unlock()
	if n.closed {
		return fmt.Errorf("embedding: native embedder closed")
	}
	if !n.loaded {
		return fmt.Errorf("embedding: native embedder not loaded")
	}
	return nil
}

// EmbedPassages embeds texts with no query prefix.
func (n *NativeEmbedder) EmbedPassages(ctx context.Context, texts []string) ([][]float32, error) {
	for i, t := range texts {
		if t == "" {
			return nil, fmt.Errorf("empty passage text at index %d", i)
		}
	}
	return n.encode(ctx, texts)
}

// EmbedQuery embeds a single query with the BGE retrieval prefix.
func (n *NativeEmbedder) EmbedQuery(ctx context.Context, query string) ([]float32, error) {
	if query == "" {
		return nil, fmt.Errorf("empty query")
	}
	vecs, err := n.encode(ctx, []string{QueryPrefix + query})
	if err != nil {
		return nil, err
	}
	return vecs[0], nil
}

// EmbedPassagesWithIDs embeds keeping caller IDs, matching BGEEmbedder.
func (n *NativeEmbedder) EmbedPassagesWithIDs(ctx context.Context, ids []string, texts []string) (map[string][]float32, error) {
	if len(ids) != len(texts) {
		return nil, fmt.Errorf("ids/texts length mismatch")
	}
	vecs, err := n.EmbedPassages(ctx, texts)
	if err != nil {
		return nil, err
	}
	m := make(map[string][]float32, len(ids))
	for i, id := range ids {
		m[id] = vecs[i]
	}
	return m, nil
}

func (n *NativeEmbedder) encode(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return [][]float32{}, nil
	}
	if err := n.requireLoaded(); err != nil {
		return nil, err
	}
	n.mu.Lock()
	defer n.mu.Unlock()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	out := make([][]float32, 0, len(texts))
	for base := 0; base < len(texts); base += n.batchSize {
		end := base + n.batchSize
		if end > len(texts) {
			end = len(texts)
		}
		chunk := texts[base:end]
		vecs, err := n.embedChunk(chunk)
		if err != nil {
			return nil, err
		}
		out = append(out, vecs...)
	}
	return out, nil
}

// embedChunk tokenizes one batch (padding to the batch's own longest
// sequence), runs the session and pools + normalizes, mirroring embed.py.
func (n *NativeEmbedder) embedChunk(texts []string) ([][]float32, error) {
	encs := make([]*tokenizer.Encoding, len(texts))
	maxLen := 0
	for i, text := range texts {
		enc, err := n.tok.EncodeSingle(text, true)
		if err != nil {
			return nil, fmt.Errorf("embedding: tokenize: %w", err)
		}
		encs[i] = enc
		if enc.Len() > maxLen {
			maxLen = enc.Len()
		}
	}

	ids := make([]int64, len(texts)*maxLen)
	masks := make([]int64, len(ids))
	types := make([]int64, len(ids))
	for i, enc := range encs {
		if len(enc.AttentionMask) != len(enc.Ids) {
			return nil, fmt.Errorf("embedding: tokenizer returned mismatched ids and attention mask lengths")
		}
		for j, id := range enc.Ids {
			if id < 0 || uint64(id) > math.MaxInt32 {
				return nil, fmt.Errorf("embedding: invalid token id %d", id)
			}
			ids[i*maxLen+j] = int64(id)
			masks[i*maxLen+j] = int64(enc.AttentionMask[j])
			if j < len(enc.TypeIds) {
				types[i*maxLen+j] = int64(enc.TypeIds[j])
			}
		}
	}

	shape := ort.NewShape(int64(len(texts)), int64(maxLen))
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
			return nil, fmt.Errorf("embedding: unsupported model input %q", name)
		}
		inputVals[i] = v
	}
	outs := []ort.Value{nil}
	if err := n.session.Run(inputVals, outs); err != nil {
		return nil, fmt.Errorf("embedding: inference batch: %w", err)
	}
	tensor, ok := outs[0].(*ort.Tensor[float32])
	if !ok {
		if outs[0] != nil {
			outs[0].Destroy()
		}
		return nil, fmt.Errorf("embedding: unexpected output type")
	}
	defer tensor.Destroy()
	data := tensor.GetData()
	dims := tensor.GetShape()
	if len(dims) != 3 || dims[2] != int64(n.dimensions) || len(data) != len(texts)*maxLen*n.dimensions {
		return nil, fmt.Errorf("embedding: unexpected output shape %v", dims)
	}

	vectors := make([][]float32, len(texts))
	for row := range texts {
		var pooled []float32
		if n.pooling == "cls" {
			pooled = make([]float32, n.dimensions)
			copy(pooled, data[row*maxLen*n.dimensions:(row*maxLen+1)*n.dimensions])
		} else {
			// Mean pooling weighted by the attention mask.
			pooled = make([]float32, n.dimensions)
			var count float32
			for token := 0; token < maxLen; token++ {
				if masks[row*maxLen+token] == 0 {
					continue
				}
				count++
				for d := 0; d < n.dimensions; d++ {
					pooled[d] += data[(row*maxLen+token)*n.dimensions+d]
				}
			}
			if count == 0 {
				return nil, fmt.Errorf("embedding: tokenizer produced an all-padding row")
			}
			for d := range pooled {
				pooled[d] /= count
			}
		}
		var norm float64
		for _, v := range pooled {
			norm += float64(v) * float64(v)
		}
		norm = math.Sqrt(norm)
		if norm < 1e-12 {
			return nil, fmt.Errorf("embedding: invalid vector norm %f", norm)
		}
		for d := range pooled {
			pooled[d] /= float32(norm)
		}
		vectors[row] = pooled
	}
	return vectors, nil
}
