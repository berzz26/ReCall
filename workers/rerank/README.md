# Rerank worker

Cross-encoder second stage for ReCall search. The worker scores
`(query, full segment description)` pairs and returns raw logits; the Go
service handles ordering and min-max normalization for display.

Default model: `mixedbread-ai/mxbai-rerank-xsmall-v1`.

## Backends

| Backend | Runtime | When to use |
|---|---|---|
| `torch` (default) | PyTorch via `sentence-transformers.CrossEncoder`, CPU | No artifact preparation required. |
| `onnx` | ONNX Runtime `CPUExecutionProvider`, CPU | ONNX Runtime inference; requires a prepared artifact. |

Both backends use the same tokenizer behavior (`padding=True`,
`truncation="longest_first"`, model max length, normally 512) and return raw
logits without sigmoid/softmax. The torch path pins
`activation_fn=torch.nn.Identity()` so its scores match the ONNX path; ranking
is unchanged because sigmoid is monotonic.

INT8 dynamic quantization keeps the same interface but changes logit values and
may reorder candidates whose scores are close; it is opt-in and should be
validated against your own data before use in production.

Select with:

```bash
RERANKER_BACKEND=torch              # default
RERANKER_BACKEND=onnx
RERANKER_QUANTIZATION=none          # FP32 ONNX graph
RERANKER_QUANTIZATION=int8          # dynamic INT8 graph
```

`RERANKER_QUANTIZATION=int8` requires `RERANKER_BACKEND=onnx`; the API rejects
the combination at config load. Quantization is never selected implicitly.

Other settings:

| Variable | Default | Meaning |
|---|---|---|
| `RERANKER_ONNX_DIR` | `workers/rerank/models/mxbai-rerank-xsmall-v1` | Prepared artifact directory. |
| `RERANKER_ONNX_THREADS` | `2` | Intra-op threads; inter-op is fixed at 1. |
| `RERANKER_BATCH_SIZE` | `32` | Pairs per inference batch. |
| `RERANKER_TIMEOUT` | `2m` | Per-batch subprocess timeout. |

## Dependencies

PyTorch backend only:

```bash
pip install -r workers/rerank/requirements.txt
```

ONNX backend and artifact preparation (also installs the base requirements):

```bash
pip install -r workers/rerank/requirements-onnx.txt
```

The requirements pin the versions this integration was written against:
Torch 2.6.0, Transformers 5.17.0, Sentence Transformers 6.0.1,
ONNX 1.18.0, ONNX Runtime 1.23.2.

Sentence Transformers' own `backend="onnx"` support is not used here:
it requires `optimum-onnx`, whose current release constrains
`transformers<4.58`, while Sentence Transformers 6.0.1 requires
`transformers>=5`. The worker therefore uses a thin ONNX Runtime session plus
the Transformers tokenizer, which keeps tokenization/truncation identical to
the PyTorch path.

## Prepare the artifact

`make run` prepares the artifact automatically:

```text
make run
  └── prepare-rerank           (skipped when reranking is off or backend=torch)
        └── prepare_onnx.py --ensure
```

`--ensure` is idempotent: it reads only `rerank_artifact.json` and exits (roughly
a second) when a valid graph already exists for the configured source model,
opset, and `RERANKER_QUANTIZATION`. When the artifact is missing, stale
(different model/opset), or lacks the requested quantization, it re-exports —
so switching `RERANKER_QUANTIZATION=none` ↔ `int8` cleanly prepares what the
runtime needs. Settings come from `.env` / the environment via the Makefile's
`rerank_env` plumbing; command-line values such as
`make prepare-rerank RERANKER_QUANTIZATION=int8` take precedence.

Preparation still never runs inside the API process or a search request; `make`
invokes it as a separate process before the server starts.

Standalone invocation still works, and flags left unset default to the
`RERANKER_*` environment variables:

```bash
# FP32 only (reads RERANKER_MODEL/RERANKER_QUANTIZATION/RERANKER_ONNX_DIR)
python3 workers/rerank/prepare_onnx.py

# Explicit FP32 + a separate dynamically quantized INT8 graph
python3 workers/rerank/prepare_onnx.py \
  --model mixedbread-ai/mxbai-rerank-xsmall-v1 \
  --output-dir workers/rerank/models/mxbai-rerank-xsmall-v1 \
  --quantization int8
```

Other flags: `--ensure` (skip when valid, re-prepare otherwise), `--opset`
(default 17), `--max-length` (default model value, capped at 512), `--threads`
(validation session), `--force` (replace an existing directory).

Resulting layout:

```text
workers/rerank/models/mxbai-rerank-xsmall-v1/
├── rerank_artifact.json       # source model, versions, files, hashes
├── tokenizer.json
├── tokenizer_config.json
└── onnx/
    ├── model.onnx             # FP32 (always written)
    └── model_int8.onnx        # when --quantization int8
```

The tokenizer is saved without the model's `config.json`; the runtime loads it
standalone (`AutoTokenizer.from_pretrained` needs only the tokenizer files).

The directory is git-ignored. `prepare_onnx.py` builds in a temporary
directory and publishes atomically, so a failed run does not leave a partial
artifact behind.

## Runtime behavior and failure policy

The worker keeps the existing stdin/stdout protocol and persistent lifecycle:

- Model/tokenizer/session load once at child-process startup.
- Readiness is `{"status":"ready","model_load_ms":...}` on stdout; all
  diagnostics go to stderr.
- Batches are scored in `RERANKER_BATCH_SIZE` chunks; the session is reused.
- Go serializes batches through one persistent child, so request/response
  matching is not concurrent inside the worker.

Failure handling is deliberately bounded:

- If the persistent worker cannot start at API boot (missing artifact,
  incompatible graph, missing dependency), the API still boots. Reranking is
  disabled with a clear error and searches keep pure vector order.
- If the worker fails during a batch, Go discards the child and allows at most
  one restart for the process lifetime. If that restart fails, further searches
  fail fast (vector order) until the API is restarted.
- There is no unbounded per-request one-shot fallback. One-shot execution
  remains available for direct CLI use and for callers that never started a
  persistent worker.
