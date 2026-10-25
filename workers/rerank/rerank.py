#!/usr/bin/env python3
"""Cross-encoder reranking worker.

Scores (query, document) pairs with mixedbread-ai/mxbai-rerank-xsmall-v1.
Higher score = more relevant. Two inference backends are supported:

* ``torch``: sentence-transformers CrossEncoder (PyTorch, CPU).
* ``onnx``:  ONNX Runtime CPUExecutionProvider against a prepared local
             artifact (see ``workers/rerank/prepare_onnx.py``). Optionally
             uses a dynamically INT8-quantized graph.

Both backends return raw cross-encoder logits (no sigmoid/softmax).

One-shot:
    python3 workers/rerank/rerank.py --input in.json --output out.json \
        [--backend torch|onnx] [--quantization none|int8] [--model ...]

    in.json:  {"query": "...", "items": [{"id": "...", "text": "..."}]}
    out.json: {"items": [{"id": "...", "score": float}]}

Persistent (model loaded once, batches via stdin/stdout):
    python3 workers/rerank/rerank.py --persistent [--backend ...] [--model ...]
    stdin:  {"input": "in.json", "output": "out.json"}\n
    stdout: {"status": "ok", "output": "out.json", "count": N}\n
    stdin:  {"command": "exit"}\n
"""
import argparse
import json
import os
import sys
import time

DEFAULT_MODEL = "mixedbread-ai/mxbai-rerank-xsmall-v1"
DEFAULT_BACKEND = "torch"
DEFAULT_QUANTIZATION = "none"
DEFAULT_ONNX_DIR = "workers/rerank/models/mxbai-rerank-xsmall-v1"
DEFAULT_BATCH_SIZE = 32
DEFAULT_ONNX_THREADS = 2
ARTIFACT_METADATA = "rerank_artifact.json"
SUPPORTED_ARTIFACT_FORMAT = 1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=False, default="", help="input JSON file")
    p.add_argument("--output", required=False, default="", help="output JSON file")
    p.add_argument("--model", required=False, default=DEFAULT_MODEL, help="cross-encoder model id")
    p.add_argument("--backend", required=False, default=DEFAULT_BACKEND, choices=["torch", "onnx"],
                   help="inference backend (default: torch)")
    p.add_argument("--quantization", required=False, default=DEFAULT_QUANTIZATION, choices=["none", "int8"],
                   help="ONNX precision; int8 requires a prepared INT8 artifact")
    p.add_argument("--onnx-dir", required=False, default=DEFAULT_ONNX_DIR,
                   help="directory containing the prepared ONNX artifact")
    p.add_argument("--onnx-threads", required=False, type=int, default=DEFAULT_ONNX_THREADS,
                   help="ONNX Runtime intra-op CPU threads (default: 2)")
    p.add_argument("--batch-size", required=False, type=int, default=DEFAULT_BATCH_SIZE,
                   help="pairs scored per inference batch (default: 32)")
    p.add_argument("--persistent", action="store_true", help="run in persistent mode")
    return p.parse_args()


def validate_args(args):
    if args.backend == "torch" and args.quantization != "none":
        raise ValueError("--quantization int8 requires --backend onnx")
    if args.onnx_threads < 1:
        raise ValueError("--onnx-threads must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")


def log(msg, **fields):
    rec = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": "INFO", "component": "rerank", "msg": msg}
    if fields:
        rec.update(fields)
    print(json.dumps(rec), file=sys.stderr, flush=True)


def err(msg):
    print(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": "ERROR", "component": "rerank", "msg": msg}), file=sys.stderr, flush=True)


def load_torch_model(args):
    start = time.time()
    log(f"loading torch model {args.model}", backend="torch", model=args.model)
    try:
        import torch
        from sentence_transformers import CrossEncoder
    except ImportError as e:
        raise RuntimeError(
            "torch backend requires PyTorch and sentence-transformers; "
            "install workers/rerank/requirements.txt"
        ) from e

    # Identity keeps the output as raw cross-encoder logits. The Go service
    # normalizes scores for display and relies on ordering only.
    model = CrossEncoder(args.model, device="cpu", activation_fn=torch.nn.Identity())
    duration_ms = round((time.time() - start) * 1000)
    log(f"torch model loaded in {duration_ms/1000:.2f}s", duration_ms=duration_ms)
    return {"kind": "torch", "model": model}, {"backend": "torch", "quantization": "none"}, duration_ms


def _artifact_path(onnx_dir, relative_path):
    # Never allow metadata to point outside the artifact directory.
    candidate = os.path.abspath(os.path.join(onnx_dir, relative_path))
    root = os.path.abspath(onnx_dir)
    if os.path.commonpath([candidate, root]) != root:
        raise RuntimeError(f"invalid artifact path outside {root}: {relative_path}")
    return candidate


def load_onnx_model(args):
    start = time.time()
    onnx_dir = os.path.abspath(args.onnx_dir)
    metadata_path = os.path.join(onnx_dir, ARTIFACT_METADATA)
    if not os.path.isfile(metadata_path):
        raise RuntimeError(
            f"ONNX artifact metadata not found: {metadata_path}. "
            f"Prepare it with: python3 workers/rerank/prepare_onnx.py "
            f"--model {args.model} --output-dir {args.onnx_dir}"
        )

    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
    except Exception as e:
        raise RuntimeError(f"failed to read ONNX artifact metadata {metadata_path}: {e}") from e

    if metadata.get("format_version") != SUPPORTED_ARTIFACT_FORMAT:
        raise RuntimeError(
            f"unsupported ONNX artifact format {metadata.get('format_version')!r} in {metadata_path}; "
            f"expected {SUPPORTED_ARTIFACT_FORMAT}"
        )
    source_model = metadata.get("source_model")
    if source_model != args.model:
        raise RuntimeError(
            f"ONNX artifact at {onnx_dir} was prepared for model {source_model!r}, "
            f"but RERANKER_MODEL is {args.model!r}. Re-run prepare_onnx.py or fix RERANKER_ONNX_DIR."
        )

    artifacts = metadata.get("artifacts") or {}
    artifact = artifacts.get(args.quantization)
    if not artifact or not artifact.get("file"):
        available = ", ".join(sorted(artifacts)) or "none"
        raise RuntimeError(
            f"ONNX artifact for quantization={args.quantization!r} is missing in {onnx_dir} "
            f"(available: {available}). Prepare it with: python3 workers/rerank/prepare_onnx.py "
            f"--model {args.model} --output-dir {args.onnx_dir} --quantization {args.quantization}"
        )

    model_path = _artifact_path(onnx_dir, artifact["file"])
    if not os.path.isfile(model_path):
        raise RuntimeError(
            f"ONNX graph not found: {model_path}. Re-run prepare_onnx.py to regenerate the artifact."
        )

    try:
        import onnxruntime as ort
    except ImportError as e:
        raise RuntimeError(
            "ONNX backend requires onnxruntime; install workers/rerank/requirements-onnx.txt"
        ) from e

    if "CPUExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError(
            f"ONNX Runtime CPUExecutionProvider is unavailable; providers={ort.get_available_providers()}"
        )

    log(
        "loading ONNX model",
        backend="onnx",
        model=args.model,
        quantization=args.quantization,
        artifact=model_path,
    )
    sess_options = ort.SessionOptions()
    sess_options.intra_op_num_threads = args.onnx_threads
    sess_options.inter_op_num_threads = 1
    sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_options.log_severity_level = 3
    try:
        session = ort.InferenceSession(model_path, sess_options=sess_options, providers=["CPUExecutionProvider"])
    except Exception as e:
        raise RuntimeError(f"failed to load ONNX graph {model_path}: {e}") from e

    inputs = session.get_inputs()
    input_names = {i.name for i in inputs}
    missing = {"input_ids", "attention_mask"} - input_names
    if missing:
        raise RuntimeError(
            f"ONNX graph {model_path} is incompatible: missing inputs {sorted(missing)}; "
            f"found {sorted(input_names)}"
        )
    input_dtypes = {}
    for i in inputs:
        if i.type == "tensor(int32)":
            input_dtypes[i.name] = "int32"
        else:
            input_dtypes[i.name] = "int64"
    output_name = session.get_outputs()[0].name if session.get_outputs() else ""
    if not output_name:
        raise RuntimeError(f"ONNX graph {model_path} has no outputs")

    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(onnx_dir, local_files_only=True)
    except Exception as e:
        raise RuntimeError(f"failed to load tokenizer from ONNX artifact {onnx_dir}: {e}") from e

    max_length = metadata.get("max_length") or tokenizer.model_max_length or 512
    try:
        max_length = int(max_length)
    except (TypeError, ValueError) as e:
        raise RuntimeError(f"invalid max_length in {metadata_path}: {max_length!r}") from e
    if max_length < 1:
        raise RuntimeError(f"invalid max_length in {metadata_path}: {max_length}")

    duration_ms = round((time.time() - start) * 1000)
    log(f"ONNX model loaded in {duration_ms/1000:.2f}s", duration_ms=duration_ms)
    return (
        {
            "kind": "onnx",
            "session": session,
            "tokenizer": tokenizer,
            "input_names": input_names,
            "input_dtypes": input_dtypes,
            "output_name": output_name,
            "max_length": max_length,
        },
        {
            "backend": "onnx",
            "quantization": args.quantization,
            "artifact": model_path,
            "max_length": max_length,
        },
        duration_ms,
    )


def load_model(args):
    if args.backend == "onnx":
        return load_onnx_model(args)
    return load_torch_model(args)


def validate_request(data):
    query = data.get("query")
    if not isinstance(query, str) or query.strip() == "":
        raise ValueError("invalid input: query must be non-empty string")
    items = data.get("items")
    if not isinstance(items, list):
        raise ValueError("invalid input: items must be list")
    for it in items:
        if not isinstance(it, dict):
            raise ValueError("invalid item: must be dict")
        if "id" not in it or "text" not in it:
            raise ValueError("invalid item: id and text required")
        if not isinstance(it["id"], str) or it["id"] == "":
            raise ValueError("invalid item: id must be non-empty string")
        if not isinstance(it["text"], str):
            raise ValueError("invalid item: text must be string")
    return query, items


def _item_text(item):
    # Cross-encoder needs text; empty descriptions score lowest rather than
    # failing the whole batch (the Go caller already substitutes " ").
    return item["text"] if item["text"].strip() != "" else " "


def score_items_torch(loaded, query, items, batch_size):
    if len(items) == 0:
        return []
    pairs = [(query, _item_text(it)) for it in items]
    scores = loaded["model"].predict(pairs, batch_size=batch_size, show_progress_bar=False)
    try:
        scores = scores.tolist()
    except AttributeError:
        scores = list(scores)
    return [{"id": it["id"], "score": float(s)} for it, s in zip(items, scores)]


def score_items_onnx(loaded, query, items, batch_size):
    if len(items) == 0:
        return []
    import numpy as np

    session = loaded["session"]
    tokenizer = loaded["tokenizer"]
    input_names = loaded["input_names"]
    input_dtypes = loaded["input_dtypes"]
    output_name = loaded["output_name"]
    max_length = loaded["max_length"]

    out = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        queries = [query] * len(batch)
        documents = [_item_text(it) for it in batch]
        # Mirror sentence-transformers: padding=True, truncation="longest_first"
        # and the model's configured max sequence length.
        encoded = tokenizer(
            queries,
            documents,
            padding=True,
            truncation="longest_first",
            max_length=max_length,
            return_tensors="np",
        )
        feed = {}
        for name in input_names:
            if name not in encoded:
                raise RuntimeError(f"tokenizer did not produce required ONNX input {name!r}")
            feed[name] = np.asarray(encoded[name], dtype=input_dtypes.get(name, "int64"))
        logits = np.asarray(session.run([output_name], feed)[0])
        if logits.ndim == 2 and logits.shape[1] == 1:
            batch_scores = logits[:, 0]
        elif logits.ndim == 1:
            batch_scores = logits
        else:
            raise RuntimeError(
                f"unexpected ONNX logits shape {logits.shape}; expected [batch, 1] for a single-label cross-encoder"
            )
        if len(batch_scores) != len(batch):
            raise RuntimeError(f"ONNX returned {len(batch_scores)} scores for {len(batch)} pairs")
        out.extend({"id": it["id"], "score": float(s)} for it, s in zip(batch, batch_scores))
    return out


def score_items(loaded, query, items, batch_size):
    if loaded["kind"] == "onnx":
        return score_items_onnx(loaded, query, items, batch_size)
    return score_items_torch(loaded, query, items, batch_size)


def one_shot(args):
    if not args.input or not args.output:
        err("FATAL: --input and --output required in one-shot mode")
        sys.exit(2)
    try:
        with open(args.input) as f:
            data = json.load(f)
    except Exception as e:
        err(f"failed to read input: {e}")
        sys.exit(2)
    try:
        query, items = validate_request(data)
    except ValueError as e:
        err(str(e))
        sys.exit(2)
    start = time.time()
    try:
        loaded, info, _ = load_model(args)
    except Exception as e:
        err(f"failed to load model: {e}")
        sys.exit(3)
    log(f"reranking {len(items)} pairs", count=len(items), model=args.model, **info)
    try:
        out_items = score_items(loaded, query, items, args.batch_size)
    except Exception as e:
        err(f"rerank failed: {e}")
        sys.exit(3)
    try:
        with open(args.output, "w") as f:
            json.dump({"items": out_items}, f)
    except Exception as e:
        err(f"failed to write output: {e}")
        sys.exit(2)
    log("rerank worker complete", count=len(out_items), total_duration_ms=round((time.time() - start) * 1000))


def persistent_main(args):
    total_start = time.time()
    log("persistent_worker_start", pid=os.getpid(), backend=args.backend, quantization=args.quantization)
    try:
        loaded, info, load_ms = load_model(args)
    except Exception as e:
        err(f"failed to load model: {e}")
        sys.exit(3)
    log("persistent_ready", model_load_ms=load_ms, pid=os.getpid(), **info)
    try:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    except Exception:
        pass
    print(json.dumps({"status": "ready", "model_load_ms": load_ms, "backend": args.backend, "quantization": args.quantization}), flush=True)
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
        try:
            query, items = validate_request(data)
        except ValueError as e:
            print(json.dumps({"status": "error", "error": str(e)}), flush=True)
            continue
        try:
            out_items = score_items(loaded, query, items, args.batch_size)
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
        log("persistent_batch_complete", batch=batch_count, count=len(out_items), duration_ms=round(batch_ms, 2))
        print(json.dumps({"status": "ok", "output": output_path, "count": len(out_items)}), flush=True)
    log("persistent_worker_complete", batches=batch_count, total_duration_ms=round((time.time() - total_start) * 1000))
    return 0


def main():
    args = parse_args()
    try:
        validate_args(args)
    except ValueError as e:
        err(f"FATAL: {e}")
        sys.exit(2)
    if getattr(args, "persistent", False):
        sys.exit(persistent_main(args))
    one_shot(args)


if __name__ == "__main__":
    main()
