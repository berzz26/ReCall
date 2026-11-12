#!/usr/bin/env python3
"""BGE embedding worker.

Embeds passages or queries with BAAI/bge-small-en-v1.5 (384 dim, CLS-pooled,
L2-normalized).

Two backends are supported:

* ``onnx`` (default): loads a prepared local ONNX graph plus its tokenizer via
  ONNX Runtime and the Rust ``tokenizers`` library. No PyTorch import, no
  Hugging Face network access at startup. This is the fast cold-start path.
* ``torch``: legacy sentence-transformers backend. Kept for compatibility;
  it imports PyTorch and resolves the model id through Hugging Face.

One-shot:
    python3 workers/embedding/embed.py --input in.json --output out.json \
        [--mode passage|query] [--backend onnx|torch]

Persistent (model loaded once, batches via stdin/stdout):
    python3 workers/embedding/embed.py --persistent [--backend ...]

The ONNX artifact is produced by ``workers/embedding/prepare_onnx.py`` and is
intentionally kept out of git.
"""
import argparse
import json
import os
import sys
import time

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
DEFAULT_BACKEND = "onnx"
DEFAULT_ONNX_DIR = "workers/embedding/models/bge-small-en-v1.5"
DEFAULT_ONNX_THREADS = 4
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_LENGTH = 512


def _env(name, fallback):
    value = (os.environ.get(name) or "").strip()
    return value or fallback


def _env_int(name, fallback):
    value = (os.environ.get(name) or "").strip()
    if not value:
        return fallback
    try:
        return int(value)
    except ValueError:
        raise SystemExit(f"FATAL: {name} must be an integer, got {value!r}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=False, default="", help="input JSON file")
    p.add_argument("--output", required=False, default="", help="output JSON file")
    p.add_argument("--mode", choices=["passage", "query"], default=None, help="embedding mode")
    p.add_argument("--persistent", action="store_true", help="Run in persistent mode: load model once, handle batches via stdin/stdout")
    p.add_argument("--backend", choices=["onnx", "torch"],
                   default=_env("EMBEDDING_BACKEND", DEFAULT_BACKEND).lower(),
                   help="inference backend (default: EMBEDDING_BACKEND or onnx)")
    p.add_argument("--model", default=_env("EMBEDDING_MODEL", MODEL_NAME),
                   help="model id (torch backend only)")
    p.add_argument("--onnx-dir", default=_env("EMBEDDING_ONNX_DIR", DEFAULT_ONNX_DIR),
                   help="directory containing model.onnx and tokenizer.json")
    p.add_argument("--onnx-threads", type=int,
                   default=_env_int("EMBEDDING_ONNX_THREADS", DEFAULT_ONNX_THREADS),
                   help="ONNX Runtime intra-op threads")
    p.add_argument("--batch-size", type=int,
                   default=_env_int("EMBEDDING_BATCH_SIZE", DEFAULT_BATCH_SIZE),
                   help="texts embedded per inference batch")
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH,
                   help="tokenizer truncation length (capped at 512)")
    return p.parse_args()


def log(msg, **fields):
    rec = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": "INFO", "component": "embedding", "msg": msg}
    if fields:
        rec.update(fields)
    print(json.dumps(rec), file=sys.stderr, flush=True)


def err(msg, **fields):
    rec = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": "ERROR", "component": "embedding", "msg": msg}
    if fields:
        rec.update(fields)
    print(json.dumps(rec), file=sys.stderr, flush=True)


class OnnxEmbedder:
    """BGE inference through ONNX Runtime + tokenizers, CPU only."""

    kind = "onnx"

    def __init__(self, onnx_dir, threads, max_length, batch_size):
        import numpy as np
        import onnxruntime as ort
        from tokenizers import Tokenizer

        onnx_dir = os.path.abspath(onnx_dir)
        model_path = os.path.join(onnx_dir, "model.onnx")
        tokenizer_path = os.path.join(onnx_dir, "tokenizer.json")
        for path in (model_path, tokenizer_path):
            if not os.path.isfile(path):
                raise RuntimeError(
                    f"missing ONNX artifact file {path}; run "
                    f"workers/embedding/prepare_onnx.py --ensure (or set EMBEDDING_ONNX_DIR)"
                )

        metadata = {}
        metadata_path = os.path.join(onnx_dir, "embedding_artifact.json")
        if os.path.isfile(metadata_path):
            try:
                with open(metadata_path) as f:
                    metadata = json.load(f)
            except Exception as e:
                raise RuntimeError(f"failed to read {metadata_path}: {e}") from e

        pooling = str(metadata.get("pooling", "cls")).lower()
        if pooling not in ("cls", "mean"):
            raise RuntimeError(f"unsupported pooling mode in {metadata_path}: {pooling!r}")

        meta_max = metadata.get("max_length")
        if meta_max and not max_length:
            max_length = int(meta_max)
        self.max_length = min(int(max_length or DEFAULT_MAX_LENGTH), DEFAULT_MAX_LENGTH)

        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.tokenizer.enable_truncation(max_length=self.max_length)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = max(1, int(threads))
        opts.inter_op_num_threads = 1
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(
            model_path, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self.input_names = [i.name for i in self.session.get_inputs()]
        supported = {"input_ids", "attention_mask", "token_type_ids"}
        unknown = [name for name in self.input_names if name not in supported]
        if unknown:
            raise RuntimeError(f"unsupported ONNX model inputs: {unknown}")
        if "input_ids" not in self.input_names or "attention_mask" not in self.input_names:
            raise RuntimeError(f"ONNX model is missing required inputs; found {self.input_names}")

        self.pooling = pooling
        self.batch_size = max(1, int(batch_size))
        self._np = np
        self.info = {
            "pooling": pooling,
            "max_length": self.max_length,
            "onnx_dir": onnx_dir,
            "model_revision": metadata.get("revision"),
        }

    def encode(self, texts):
        if not texts:
            return self._np.zeros((0, 384), dtype=self._np.float32)
        np = self._np
        chunks = []
        for base in range(0, len(texts), self.batch_size):
            batch = texts[base:base + self.batch_size]
            encs = self.tokenizer.encode_batch(batch)
            ids = np.array([e.ids for e in encs], dtype=np.int64)
            mask = np.array([e.attention_mask for e in encs], dtype=np.int64)
            types = np.array([e.type_ids for e in encs], dtype=np.int64)
            feed = {}
            for name in self.input_names:
                if name == "input_ids":
                    feed[name] = ids
                elif name == "attention_mask":
                    feed[name] = mask
                elif name == "token_type_ids":
                    feed[name] = types
            outputs = self.session.run(None, feed)
            if not outputs:
                raise RuntimeError("ONNX model returned no outputs")
            hidden = np.asarray(outputs[0], dtype=np.float32)
            if hidden.ndim != 3 or hidden.shape[0] != len(batch):
                raise RuntimeError(f"unexpected ONNX output shape {hidden.shape} for batch of {len(batch)}")
            if self.pooling == "cls":
                pooled = hidden[:, 0, :]
            else:
                weights = mask[:, :, None].astype(np.float32)
                pooled = (hidden * weights).sum(axis=1) / np.maximum(weights.sum(axis=1), 1.0)
            norms = np.linalg.norm(pooled, axis=1, keepdims=True)
            chunks.append(pooled / np.maximum(norms, 1e-12))
        return np.concatenate(chunks, axis=0)


class TorchEmbedder:
    """Legacy sentence-transformers backend (imports torch on load)."""

    kind = "torch"

    def __init__(self, model_name, batch_size):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device="cpu")
        self.batch_size = max(1, int(batch_size))
        self.info = {"model": model_name}

    def encode(self, texts):
        import numpy as np
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        embeddings = self.model.encode(
            texts, normalize_embeddings=True, batch_size=self.batch_size, show_progress_bar=False
        )
        if isinstance(embeddings, list):
            embeddings = np.array(embeddings)
        return embeddings


def load_model_once(args):
    start = time.time()
    if args.backend == "onnx":
        log("loading ONNX embedding model", onnx_dir=args.onnx_dir, threads=args.onnx_threads)
        model = OnnxEmbedder(args.onnx_dir, args.onnx_threads, args.max_length, args.batch_size)
    else:
        log(f"loading model {args.model}", model=args.model, backend="torch")
        model = TorchEmbedder(args.model, args.batch_size)
    duration_ms = round((time.time() - start) * 1000, 2)
    log("model loaded", duration_ms=duration_ms, backend=model.kind, **model.info)
    return model, duration_ms


def encode_items(model, items, mode):
    import numpy as np
    texts = []
    for it in items:
        t = it["text"]
        if mode == "query":
            t = QUERY_PREFIX + t
        texts.append(t)
    embeddings = model.encode(texts)
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)
    if len(items) == 0:
        return []
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    if embeddings.shape[0] != len(items):
        raise RuntimeError(f"embedding count mismatch {embeddings.shape[0]} != {len(items)}")
    if embeddings.shape[1] != 384:
        raise RuntimeError(f"invalid dimension {embeddings.shape[1]} != 384")
    out_items = []
    for i, it in enumerate(items):
        vec = embeddings[i].tolist()
        if len(vec) != 384:
            raise RuntimeError(f"invalid vector length for {it['id']}")
        out_items.append({"id": it["id"], "embedding": vec})
    return out_items


def validate_items(items):
    for it in items:
        if not isinstance(it, dict):
            raise ValueError("invalid item: must be dict")
        if "id" not in it or "text" not in it:
            raise ValueError("invalid item: id and text required")
        if not isinstance(it["id"], str) or it["id"] == "":
            raise ValueError("invalid item: id must be non-empty string")
        if not isinstance(it["text"], str):
            raise ValueError("invalid item: text must be string")
        if it["text"].strip() == "":
            raise ValueError(f"empty text for id {it['id']}")


def main():
    args = parse_args()
    # Persistent mode must be dispatched BEFORE any one-shot input handling:
    # it is spawned without --input/--output and serves batches via stdin.
    if getattr(args, "persistent", False):
        return persistent_main(args)
    try:
        with open(args.input) as f:
            data = json.load(f)
    except Exception as e:
        err(f"failed to read input: {e}")
        sys.exit(2)

    items = data.get("items")
    if not isinstance(items, list):
        err("invalid input: items must be list")
        sys.exit(2)
    try:
        validate_items(items)
    except ValueError as e:
        err(str(e))
        sys.exit(2)

    if not args.input or not args.output:
        err("FATAL: --input and --output required in one-shot mode")
        sys.exit(2)

    # Mode detection: explicit --mode, or infer from input field "mode" or query prefix presence
    mode = args.mode
    if mode is None:
        mode = data.get("mode", "passage")

    try:
        model, _ = load_model_once(args)
    except Exception as e:
        err(f"failed to load model: {e}")
        sys.exit(3)

    log(f"embedding {len(items)} texts", count=len(items), mode=mode, backend=model.kind)
    emb_start = time.time()
    try:
        out_items = encode_items(model, items, mode)
    except Exception as e:
        err(f"embedding failed: {e}")
        sys.exit(3)

    log(f"embedding complete in {time.time()-emb_start:.2f}s", count=len(out_items), backend=model.kind)

    try:
        with open(args.output, "w") as f:
            json.dump({"items": out_items}, f)
    except Exception as e:
        err(f"failed to write output: {e}")
        sys.exit(2)

    log("embedding worker complete", count=len(out_items), total_duration_ms=round((time.time() - emb_start) * 1000))
    sys.exit(0)


def persistent_main(args):
    total_start = time.time()
    log("persistent_worker_start", pid=os.getpid(), backend=args.backend)
    try:
        model, load_ms = load_model_once(args)
    except Exception as e:
        err(f"failed to load model: {e}")
        sys.exit(3)
    log("persistent_ready", model_load_ms=load_ms, pid=os.getpid(), backend=model.kind, **model.info)
    try:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    except Exception:
        pass
    print(json.dumps({"status": "ready", "model_load_ms": load_ms, "backend": model.kind}), flush=True)
    batch_count = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "exit" or line == '{"command":"exit"}':
            log("persistent_exit_command", batch_count=batch_count)
            break
        try:
            req = json.loads(line)
        except Exception as e:
            log(f"persistent_bad_request: {e}")
            print(json.dumps({"status": "error", "error": f"bad request: {e}"}), flush=True)
            continue
        if req.get("command") == "exit":
            log("persistent_exit_command", batch_count=batch_count)
            break
        input_path = req.get("input")
        output_path = req.get("output")
        mode = req.get("mode", args.mode or "passage")
        if not input_path or not output_path:
            print(json.dumps({"status": "error", "error": "missing input/output paths"}), flush=True)
            continue
        batch_count += 1
        batch_start = time.time()
        try:
            with open(input_path) as f:
                data = json.load(f)
        except Exception as e:
            print(json.dumps({"status": "error", "error": f"failed to read input: {e}"}), flush=True)
            continue
        items = data.get("items", [])
        req_mode = data.get("mode", mode)
        if req_mode not in ("passage", "query"):
            req_mode = "passage"
        try:
            out_items = encode_items(model, items, req_mode)
        except Exception as e:
            log(f"persistent_inference_failed: {e}", batch=batch_count)
            print(json.dumps({"status": "error", "error": str(e)}), flush=True)
            continue
        try:
            with open(output_path, "w") as outf:
                json.dump({"items": out_items}, outf)
        except Exception as e:
            print(json.dumps({"status": "error", "error": f"failed to write output: {e}"}), flush=True)
            continue
        batch_ms = (time.time() - batch_start) * 1000
        log("persistent_batch_complete", batch=batch_count, count=len(out_items), duration_ms=round(batch_ms, 2), backend=model.kind)
        print(json.dumps({"status": "ok", "output": output_path, "count": len(out_items)}), flush=True)
    log("persistent_worker_complete", batches=batch_count, total_duration_ms=round((time.time() - total_start) * 1000))
    return 0


if __name__ == "__main__":
    main()
