#!/usr/bin/env python3
"""Cross-encoder reranking worker.

Scores (query, document) pairs with mixedbread-ai/mxbai-rerank-xsmall-v1
via sentence-transformers CrossEncoder. Higher score = more relevant.

One-shot:
    python3 workers/rerank/rerank.py --input in.json --output out.json [--model ...]

    in.json:  {"query": "...", "items": [{"id": "...", "text": "..."}]}
    out.json: {"items": [{"id": "...", "score": float}]}

Persistent (model loaded once, batches via stdin/stdout):
    python3 workers/rerank/rerank.py --persistent [--model ...]
    stdin:  {"input": "in.json", "output": "out.json"}\n
    stdout: {"status": "ok", "output": "out.json", "count": N}\n
    stdin:  {"command": "exit"}\n
"""
import argparse
import json
import sys
import time

DEFAULT_MODEL = "mixedbread-ai/mxbai-rerank-xsmall-v1"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=False, default="", help="input JSON file")
    p.add_argument("--output", required=False, default="", help="output JSON file")
    p.add_argument("--model", required=False, default=DEFAULT_MODEL, help="cross-encoder model id")
    p.add_argument("--persistent", action="store_true", help="Run in persistent mode")
    return p.parse_args()


def log(msg, **fields):
    rec = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": "INFO", "component": "rerank", "msg": msg}
    if fields:
        rec.update(fields)
    print(json.dumps(rec), file=sys.stderr, flush=True)


def err(msg):
    print(json.dumps({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "level": "ERROR", "component": "rerank", "msg": msg}), file=sys.stderr, flush=True)


def load_model_once(model_name):
    start = time.time()
    log(f"loading model {model_name}", model=model_name)
    from sentence_transformers import CrossEncoder
    model = CrossEncoder(model_name, device="cpu")
    log(f"model loaded in {time.time()-start:.2f}s", duration_ms=round((time.time() - start) * 1000))
    return model, round((time.time() - start) * 1000, 2)


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


def score_items(model, query, items):
    if len(items) == 0:
        return []
    pairs = [(query, it["text"] if it["text"].strip() != "" else " ") for it in items]
    scores = model.predict(pairs, show_progress_bar=False)
    try:
        scores = scores.tolist()
    except AttributeError:
        scores = list(scores)
    out = []
    for it, s in zip(items, scores):
        out.append({"id": it["id"], "score": float(s)})
    return out


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
        model, _ = load_model_once(args.model)
    except Exception as e:
        err(f"failed to load model: {e}")
        sys.exit(3)
    log(f"reranking {len(items)} pairs", count=len(items), model=args.model)
    try:
        out_items = score_items(model, query, items)
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
    import os
    total_start = time.time()
    log("persistent_worker_start", pid=os.getpid())
    try:
        model, load_ms = load_model_once(args.model)
    except Exception as e:
        err(f"failed to load model: {e}")
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
            out_items = score_items(model, query, items)
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
    if getattr(args, "persistent", False):
        sys.exit(persistent_main(args))
    one_shot(args)


if __name__ == "__main__":
    main()
