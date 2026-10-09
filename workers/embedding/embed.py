#!/usr/bin/env python3
"""L2.9 BGE embedding worker.

Embeds passages or query with BAAI/bge-small-en-v1.5 (384 dim, normalized).
"""
import argparse
import json
import sys
import time

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=False, default="", help="input JSON file")
    p.add_argument("--output", required=False, default="", help="output JSON file")
    p.add_argument("--mode", choices=["passage", "query"], default=None, help="embedding mode")
    p.add_argument("--persistent", action="store_true", help="Run in persistent mode: load model once, handle batches via stdin/stdout")
    return p.parse_args()


def log(msg, **fields):
    rec = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": "INFO", "component": "embedding", "msg": msg}
    if fields:
        rec.update(fields)
    print(json.dumps(rec), file=sys.stderr, flush=True)


def main():
    args = parse_args()
    try:
        with open(args.input) as f:
            data = json.load(f)
    except Exception as e:
        print(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": "ERROR", "component": "embedding", "msg": f"failed to read input: {e}"}), file=sys.stderr, flush=True)
        sys.exit(2)

    items = data.get("items")
    if not isinstance(items, list):
        print(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": "ERROR", "component": "embedding", "msg": "invalid input: items must be list"}), file=sys.stderr, flush=True)
        sys.exit(2)

    # Mode detection: explicit --mode, or infer from input field "mode" or query prefix presence
    mode = args.mode
    if mode is None:
        # If items contain is_query flag, use it; else default to passage
        # Also support top-level "mode"
        mode = data.get("mode", "passage")

    for it in items:
        if not isinstance(it, dict):
            print(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": "ERROR", "component": "embedding", "msg": "invalid item: must be dict"}), file=sys.stderr, flush=True)
            sys.exit(2)
        if "id" not in it or "text" not in it:
            print(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": "ERROR", "component": "embedding", "msg": "invalid item: id and text required"}), file=sys.stderr, flush=True)
            sys.exit(2)
        if not isinstance(it["id"], str) or it["id"] == "":
            print(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": "ERROR", "component": "embedding", "msg": "invalid item: id must be non-empty string"}), file=sys.stderr, flush=True)
            sys.exit(2)
        if not isinstance(it["text"], str):
            print(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": "ERROR", "component": "embedding", "msg": "invalid item: text must be string"}), file=sys.stderr, flush=True)
            sys.exit(2)
        if it["text"].strip() == "":
            print(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": "ERROR", "component": "embedding", "msg": f"empty text for id {it['id']}"}), file=sys.stderr, flush=True)
            sys.exit(2)

    if getattr(args, "persistent", False):
        return persistent_main(args)

    if not args.input or not args.output:
        print(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": "ERROR", "component": "embedding", "msg": "FATAL: --input and --output required in one-shot mode"}), file=sys.stderr, flush=True)
        sys.exit(2)

    log(f"loading model {MODEL_NAME}", model=MODEL_NAME)
    start = time.time()
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(MODEL_NAME, device="cpu")
    except Exception as e:
        print(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": "ERROR", "component": "embedding", "msg": f"failed to load model: {e}"}), file=sys.stderr, flush=True)
        sys.exit(3)
    log(f"model loaded in {time.time()-start:.2f}s", duration_ms=round((time.time()-start)*1000))

    texts = []
    for it in items:
        t = it["text"]
        # Apply query prefix only in query mode
        if mode == "query":
            t = QUERY_PREFIX + t
        texts.append(t)

    log(f"embedding {len(texts)} texts", count=len(texts), mode=mode)
    emb_start = time.time()
    try:
        embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    except Exception as e:
        print(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": "ERROR", "component": "embedding", "msg": f"embedding failed: {e}"}), file=sys.stderr, flush=True)
        sys.exit(3)

    # Validate
    import numpy as np
    # embeddings is np array
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    if embeddings.shape[0] != len(items):
        print(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": "ERROR", "component": "embedding", "msg": f"embedding count mismatch {embeddings.shape[0]} != {len(items)}"}), file=sys.stderr, flush=True)
        sys.exit(3)
    if embeddings.shape[1] != 384:
        print(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": "ERROR", "component": "embedding", "msg": f"invalid dimension {embeddings.shape[1]} != 384"}), file=sys.stderr, flush=True)
        sys.exit(3)

    log(f"embedding complete in {time.time()-emb_start:.2f}s", duration_ms=round((time.time()-emb_start)*1000))

    out_items = []
    for i, it in enumerate(items):
        vec = embeddings[i].tolist()
        if len(vec) != 384:
            print(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": "ERROR", "component": "embedding", "msg": f"invalid vector length for {it['id']}"}), file=sys.stderr, flush=True)
            sys.exit(3)
        out_items.append({"id": it["id"], "embedding": vec})

    out = {"items": out_items}
    try:
        with open(args.output, "w") as f:
            json.dump(out, f)
    except Exception as e:
        print(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": "ERROR", "component": "embedding", "msg": f"failed to write output: {e}"}), file=sys.stderr, flush=True)
        sys.exit(2)

    log("embedding worker complete", count=len(out_items), total_duration_ms=round((time.time()-start)*1000))
    sys.exit(0)


def load_model_once():
    start = time.time()
    log(f"loading model {MODEL_NAME}", model=MODEL_NAME)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME, device="cpu")
    log(f"model loaded in {time.time()-start:.2f}s", duration_ms=round((time.time()-start)*1000))
    return model, round((time.time() - start) * 1000, 2)


def encode_items(model, items, mode):
    import numpy as np
    texts = []
    for it in items:
        t = it["text"]
        if mode == "query":
            t = QUERY_PREFIX + t
        texts.append(t)
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False)
    if isinstance(embeddings, list):
        embeddings = np.array(embeddings)
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


def persistent_main(args):
    import os
    total_start = time.time()
    log("persistent_worker_start", pid=os.getpid())
    try:
        model, load_ms = load_model_once()
    except Exception as e:
        print(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": "ERROR", "component": "embedding", "msg": f"failed to load model: {e}"}), file=sys.stderr, flush=True)
        sys.exit(3)
    log("persistent_ready", model_load_ms=load_ms, pid=os.getpid())
    try:
        sys.stdout.reconfigure(line_buffering=True, write_through=True)
    except Exception:
        pass
    print(json.dumps({"status": "ready", "model_load_ms": load_ms}), flush=True)
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
        log("persistent_batch_complete", batch=batch_count, count=len(out_items), duration_ms=round(batch_ms, 2))
        print(json.dumps({"status": "ok", "output": output_path, "count": len(out_items)}), flush=True)
    log("persistent_worker_complete", batches=batch_count, total_duration_ms=round((time.time() - total_start) * 1000))
    return 0


if __name__ == "__main__":
    main()
