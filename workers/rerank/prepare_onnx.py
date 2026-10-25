#!/usr/bin/env python3
"""Prepare ONNX artifacts for the ReCall cross-encoder reranker.

This script runs separately from the API and from search requests. It exports
the configured cross-encoder to ONNX, optionally produces a dynamically
INT8-quantized copy, saves the tokenizer alongside the graph, and records the
artifact configuration in ``rerank_artifact.json``.

Example:
    python3 workers/rerank/prepare_onnx.py \
        --model mixedbread-ai/mxbai-rerank-xsmall-v1 \
        --output-dir workers/rerank/models/mxbai-rerank-xsmall-v1 \
        --quantization int8

``--ensure`` makes the script idempotent for automation: it first checks an
existing artifact against the requested parameters (source model, opset,
quantization) and exits immediately when they already match; otherwise it
re-prepares. Flags left unset default to the RERANKER_* environment variables
(RERANKER_MODEL, RERANKER_QUANTIZATION, RERANKER_ONNX_DIR).

The resulting directory is consumed by ``workers/rerank/rerank.py
--backend onnx --onnx-dir <dir>`` and is intentionally kept out of git.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time

DEFAULT_MODEL = "mixedbread-ai/mxbai-rerank-xsmall-v1"
DEFAULT_ONNX_DIR = "workers/rerank/models/mxbai-rerank-xsmall-v1"
ARTIFACT_METADATA = "rerank_artifact.json"
ARTIFACT_FORMAT_VERSION = 1
DEFAULT_OPSET = 17
DEFAULT_THREADS = 2
FP32_FILE = "onnx/model.onnx"
INT8_FILE = "onnx/model_int8.onnx"


def _env_default(name, fallback):
    value = (os.environ.get(name) or "").strip()
    return value or fallback


def parse_args():
    p = argparse.ArgumentParser(description="Export the reranker to ONNX Runtime format.")
    p.add_argument("--model", default=_env_default("RERANKER_MODEL", DEFAULT_MODEL),
                   help="source cross-encoder model id (default: RERANKER_MODEL)")
    p.add_argument("--output-dir", default=_env_default("RERANKER_ONNX_DIR", DEFAULT_ONNX_DIR),
                   help="artifact directory to create (default: RERANKER_ONNX_DIR)")
    p.add_argument("--quantization", default=_env_default("RERANKER_QUANTIZATION", "none").lower(),
                   choices=["none", "int8"],
                   help="int8 also writes a separately selectable INT8 graph (default: RERANKER_QUANTIZATION)")
    p.add_argument("--opset", type=int, default=DEFAULT_OPSET, help="ONNX opset version")
    p.add_argument("--max-length", type=int, default=0,
                   help="tokenizer max length (0 = model/tokenizer default, capped at 512)")
    p.add_argument("--threads", type=int, default=DEFAULT_THREADS,
                   help="ONNX Runtime intra-op threads used for the validation pass")
    p.add_argument("--force", action="store_true", help="replace an existing output directory")
    p.add_argument("--ensure", action="store_true",
                   help="skip preparation when a valid artifact already exists for the "
                        "requested model/opset/quantization; otherwise prepare it")
    return p.parse_args()


def artifact_matches(args):
    """Fast check that an existing artifact satisfies the requested parameters.

    Intentionally lightweight (no graph parsing, no hashing) so `make run`
    stays fast on repeat invocations.
    """
    onnx_dir = os.path.abspath(args.output_dir)
    metadata_path = os.path.join(onnx_dir, ARTIFACT_METADATA)
    if not os.path.isfile(metadata_path):
        return False, f"artifact metadata missing: {metadata_path}"
    try:
        with open(metadata_path) as f:
            metadata = json.load(f)
    except Exception as e:
        return False, f"artifact metadata unreadable: {e}"
    if metadata.get("format_version") != ARTIFACT_FORMAT_VERSION:
        return False, f"format_version {metadata.get('format_version')!r} != {ARTIFACT_FORMAT_VERSION}"
    if metadata.get("source_model") != args.model:
        return False, f"source_model {metadata.get('source_model')!r} != requested {args.model!r}"
    if metadata.get("opset") != args.opset:
        return False, f"opset {metadata.get('opset')!r} != requested {args.opset}"
    artifacts = metadata.get("artifacts") or {}
    entry = artifacts.get(args.quantization)
    if not isinstance(entry, dict) or not entry.get("file"):
        available = ", ".join(sorted(artifacts)) or "none"
        return False, f"no {args.quantization!r} artifact present (available: {available})"
    graph_path = os.path.join(onnx_dir, entry["file"])
    if not os.path.isfile(graph_path) or os.path.getsize(graph_path) == 0:
        return False, f"graph file missing or empty: {graph_path}"
    return True, "ok"


def log(msg, **fields):
    rec = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": "INFO", "component": "rerank-prepare", "msg": msg}
    if fields:
        rec.update(fields)
    print(json.dumps(rec), file=sys.stderr, flush=True)


def err(msg):
    print(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": "ERROR", "component": "rerank-prepare", "msg": msg}), file=sys.stderr, flush=True)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def artifact_info(root, relative_path, quantization):
    path = os.path.join(root, relative_path)
    return {
        "file": relative_path,
        "size_bytes": os.path.getsize(path),
        "sha256": file_sha256(path),
        "quantization": quantization,
    }


def resolve_max_length(tokenizer, model, requested):
    if requested and requested > 0:
        return requested
    candidates = [
        getattr(tokenizer, "model_max_length", 0) or 0,
        getattr(getattr(model, "config", None), "max_position_embeddings", 0) or 0,
        512,
    ]
    usable = [int(v) for v in candidates if isinstance(v, int) or (isinstance(v, float) and v > 0)]
    usable = [v for v in usable if 0 < v < 1_000_000]
    return min(usable) if usable else 512


def export_fp32(args, tokenizer, model, out_root):
    import torch

    onnx_path = os.path.join(out_root, FP32_FILE)
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)

    class ExportWrapper(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask, token_type_ids):
            return self.inner(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            ).logits

    wrapper = ExportWrapper(model).eval()
    dummy = tokenizer(
        ["query"],
        ["document"],
        padding="max_length",
        truncation=True,
        max_length=16,
        return_tensors="pt",
    )
    args_tuple = (
        dummy["input_ids"],
        dummy["attention_mask"],
        dummy["token_type_ids"],
    )
    log("exporting ONNX graph", model=args.model, output=onnx_path, opset=args.opset)
    start = time.time()
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                args_tuple,
                onnx_path,
                input_names=["input_ids", "attention_mask", "token_type_ids"],
                output_names=["logits"],
                dynamic_axes={
                    "input_ids": {0: "batch", 1: "sequence"},
                    "attention_mask": {0: "batch", 1: "sequence"},
                    "token_type_ids": {0: "batch", 1: "sequence"},
                    "logits": {0: "batch"},
                },
                opset_version=args.opset,
                do_constant_folding=True,
                dynamo=False,
            )
    except Exception as e:
        raise RuntimeError(f"ONNX export failed: {e}") from e
    log("ONNX graph exported", duration_ms=round((time.time() - start) * 1000), output=onnx_path)

    # Correctness check: the exported graph must match the PyTorch logits.
    try:
        import numpy as np
        import onnxruntime as ort

        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = args.threads
        sess_options.inter_op_num_threads = 1
        sess_options.log_severity_level = 3
        session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=["CPUExecutionProvider"])
        feed = {i.name: dummy[i.name].numpy() for i in session.get_inputs()}
        onnx_logits = np.asarray(session.run(["logits"], feed)[0])
        with torch.no_grad():
            torch_logits = wrapper(*args_tuple).numpy()
        diff = float(np.max(np.abs(onnx_logits - torch_logits))) if onnx_logits.size else 0.0
        if diff > 1e-2:
            raise RuntimeError(f"exported ONNX logits differ from PyTorch (max abs diff {diff:.6g})")
        log("ONNX graph validated against PyTorch", max_abs_diff=diff)
    except Exception as e:
        raise RuntimeError(f"ONNX graph validation failed: {e}") from e


def export_int8(args, out_root):
    try:
        import numpy as np
        import onnxruntime as ort
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as e:
        raise RuntimeError(
            "INT8 quantization requires onnxruntime and onnx; "
            "install workers/rerank/requirements-onnx.txt"
        ) from e

    fp32_path = os.path.join(out_root, FP32_FILE)
    int8_path = os.path.join(out_root, INT8_FILE)
    log("quantizing ONNX graph (dynamic INT8)", input=fp32_path, output=int8_path)
    start = time.time()
    try:
        quantize_dynamic(
            fp32_path,
            int8_path,
            weight_type=QuantType.QInt8,
            per_channel=False,
            reduce_range=False,
        )
    except Exception as e:
        raise RuntimeError(f"INT8 quantization failed: {e}") from e

    try:
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = args.threads
        sess_options.inter_op_num_threads = 1
        sess_options.log_severity_level = 3
        session = ort.InferenceSession(int8_path, sess_options=sess_options, providers=["CPUExecutionProvider"])
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(out_root, local_files_only=True)
        encoded = tokenizer(["query"], ["document"], padding=True, truncation=True, max_length=16, return_tensors="np")
        feed = {i.name: encoded[i.name] for i in session.get_inputs()}
        scores = np.asarray(session.run(None, feed)[0])
        if scores.size == 0 or not np.all(np.isfinite(scores)):
            raise RuntimeError("quantized graph returned empty or non-finite scores")
    except Exception as e:
        raise RuntimeError(f"quantized ONNX graph failed validation: {e}") from e
    log("INT8 graph exported", duration_ms=round((time.time() - start) * 1000), output=int8_path)


def main():
    args = parse_args()
    if args.threads < 1:
        err("FATAL: --threads must be >= 1")
        return 2
    if args.opset < 11:
        err("FATAL: --opset must be >= 11")
        return 2

    out_dir = os.path.abspath(args.output_dir)
    parent = os.path.dirname(out_dir) or "."
    force = args.force or args.ensure
    if args.ensure:
        # Fast path: unchanged parameters and a valid artifact mean `make run`
        # pays only a metadata read, not an export/quantize.
        matches, reason = artifact_matches(args)
        if matches:
            log("artifact already prepared for requested parameters, skipping",
                model=args.model, quantization=args.quantization, opset=args.opset,
                output_dir=out_dir)
            print(json.dumps({
                "status": "ok",
                "skipped": True,
                "output_dir": out_dir,
                "model": args.model,
                "quantization": args.quantization,
                "reason": reason,
            }))
            return 0
        log("artifact needs preparation", reason=reason, model=args.model,
            quantization=args.quantization, output_dir=out_dir)

    if os.path.exists(out_dir) and not force:
        err(f"FATAL: output directory already exists: {out_dir} (use --force to replace)")
        return 2
    if os.path.exists(out_dir) and not os.path.isdir(out_dir):
        err(f"FATAL: output path exists and is not a directory: {out_dir}")
        return 2

    try:
        import torch
        import transformers
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError as e:
        err(f"FATAL: preparation requires torch and transformers: {e}")
        return 3
    try:
        import onnx
        import onnxruntime
    except ImportError as e:
        err(f"FATAL: preparation requires onnx and onnxruntime; install workers/rerank/requirements-onnx.txt: {e}")
        return 3

    try:
        os.makedirs(parent, exist_ok=True)
        temp_dir = tempfile.mkdtemp(prefix=".rerank-onnx-", dir=parent)
    except Exception as e:
        err(f"FATAL: cannot create temporary output directory in {parent}: {e}")
        return 3

    try:
        log("loading source model", model=args.model)
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model)
            # Force FP32: the checkpoint advertises float16, but dynamic INT8
            # quantization needs a float32 graph.
            try:
                model = AutoModelForSequenceClassification.from_pretrained(args.model, dtype=torch.float32).eval()
            except TypeError:
                model = AutoModelForSequenceClassification.from_pretrained(args.model, torch_dtype=torch.float32).eval()
        except Exception as e:
            raise RuntimeError(f"failed to load source model {args.model}: {e}") from e
        param_dtype = next(model.parameters()).dtype
        if param_dtype != torch.float32:
            raise RuntimeError(f"source model must be float32 for export/quantization, got {param_dtype}")

        max_length = resolve_max_length(tokenizer, model, args.max_length)
        if max_length > 512:
            log("capping max_length at 512 for mxbai/deberta compatibility", requested=max_length)
            max_length = 512

        try:
            tokenizer.save_pretrained(temp_dir)
        except Exception as e:
            raise RuntimeError(f"failed to save tokenizer: {e}") from e

        export_fp32(args, tokenizer, model, temp_dir)
        artifacts = {"none": artifact_info(temp_dir, FP32_FILE, "none")}
        if args.quantization == "int8":
            export_int8(args, temp_dir)
            artifacts["int8"] = artifact_info(temp_dir, INT8_FILE, "dynamic_qint8")

        try:
            import onnx
            import onnxruntime as ort

            onnx_version = onnx.__version__
            ort_version = ort.__version__
        except ImportError as e:
            raise RuntimeError(
                "preparation requires onnxruntime and onnx; "
                "install workers/rerank/requirements-onnx.txt"
            ) from e

        metadata = {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "source_model": args.model,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "backend": "onnx",
            "provider": "CPUExecutionProvider",
            "max_length": max_length,
            "truncation": "longest_first",
            "padding": True,
            "opset": args.opset,
            "artifacts": artifacts,
            "versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "onnx": onnx_version,
                "onnxruntime": ort_version,
            },
        }
        metadata_path = os.path.join(temp_dir, ARTIFACT_METADATA)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
            f.write("\n")

        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.replace(temp_dir, out_dir)
        temp_dir = None
        log("artifact published", output_dir=out_dir, artifacts=sorted(artifacts))
        print(json.dumps({
            "status": "ok",
            "output_dir": out_dir,
            "model": args.model,
            "max_length": max_length,
            "artifacts": {k: os.path.join(out_dir, v["file"]) for k, v in artifacts.items()},
        }))
        return 0
    except Exception as e:
        err(f"FATAL: {e}")
        return 3
    finally:
        if temp_dir and os.path.isdir(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
