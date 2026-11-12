#!/usr/bin/env python3
"""Prepare the local ONNX artifact for the BGE embedding worker.

Downloads (or copies) the pinned BAAI/bge-small-en-v1.5 ONNX graph and
tokenizer into a local directory, verifies their SHA-256 checksums, and writes
``embedding_artifact.json`` describing the artifact. The API then loads only
local files at startup: no PyTorch import and no Hugging Face network calls.

Example:
    python3 workers/embedding/prepare_onnx.py --ensure
    python3 workers/embedding/prepare_onnx.py --source-dir /path/to/bge-small-en-v1.5

``--ensure`` is idempotent: when the artifact metadata matches the requested
model/revision and the files are present with the expected sizes, it exits
immediately. Everything is kept out of git (see .gitignore).
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_REVISION = "5c38ec7c405ec4b44b94cc5a9bb96e735b38267a"
DEFAULT_ONNX_DIR = "workers/embedding/models/bge-small-en-v1.5"
ARTIFACT_METADATA = "embedding_artifact.json"
ARTIFACT_FORMAT_VERSION = 1
DEFAULT_MAX_LENGTH = 512
DEFAULT_POOLING = "cls"

# Pinned upstream files (identical to the searcher's model manifest).
FILES = {
    "model.onnx": {
        "path": "onnx/model.onnx",
        "sha256": "828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35",
        "size": 133093490,
    },
    "tokenizer.json": {
        "path": "tokenizer.json",
        "sha256": "d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66",
        "size": 711396,
    },
}


def _env(name, fallback):
    value = (os.environ.get(name) or "").strip()
    return value or fallback


def parse_args():
    p = argparse.ArgumentParser(description="Prepare a local BGE ONNX embedding artifact.")
    p.add_argument("--model", default=_env("EMBEDDING_MODEL", DEFAULT_MODEL),
                   help="source embedding model id")
    p.add_argument("--revision", default=_env("EMBEDDING_MODEL_REVISION", DEFAULT_REVISION),
                   help="pinned upstream revision")
    p.add_argument("--output-dir", default=_env("EMBEDDING_ONNX_DIR", DEFAULT_ONNX_DIR),
                   help="artifact directory to create")
    p.add_argument("--source-dir", default="",
                   help="copy model.onnx/tokenizer.json from an existing local directory "
                        "instead of downloading")
    p.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH,
                   help="tokenizer truncation length (capped at 512)")
    p.add_argument("--timeout", type=int, default=60, help="per-request timeout in seconds")
    p.add_argument("--retries", type=int, default=5, help="download attempts")
    p.add_argument("--force", action="store_true", help="re-download/replace even when present")
    p.add_argument("--verify", action="store_true", help="re-hash existing files during --ensure")
    p.add_argument("--ensure", action="store_true",
                   help="skip when a valid artifact already exists; otherwise prepare it")
    return p.parse_args()


def log(msg, **fields):
    rec = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "level": "INFO", "component": "embedding-prepare", "msg": msg}
    if fields:
        rec.update(fields)
    print(json.dumps(rec), file=sys.stderr, flush=True)


def err(msg, **fields):
    rec = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "level": "ERROR", "component": "embedding-prepare", "msg": msg}
    if fields:
        rec.update(fields)
    print(json.dumps(rec), file=sys.stderr, flush=True)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_metadata(out_dir):
    path = os.path.join(out_dir, ARTIFACT_METADATA)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def artifact_matches(args, verify=False):
    out_dir = os.path.abspath(args.output_dir)
    metadata = read_metadata(out_dir)
    if not metadata:
        return False, f"artifact metadata missing: {os.path.join(out_dir, ARTIFACT_METADATA)}"
    if metadata.get("format_version") != ARTIFACT_FORMAT_VERSION:
        return False, f"artifact format version mismatch: {metadata.get('format_version')}"
    if metadata.get("source_model") != args.model:
        return False, f"model mismatch: artifact={metadata.get('source_model')} requested={args.model}"
    if metadata.get("revision") != args.revision:
        return False, f"revision mismatch: artifact={metadata.get('revision')} requested={args.revision}"
    files = metadata.get("files") or {}
    for name, spec in FILES.items():
        target = os.path.join(out_dir, name)
        if not os.path.isfile(target):
            return False, f"missing file: {target}"
        size = os.path.getsize(target)
        if size != spec["size"]:
            return False, f"size mismatch for {name}: {size} != {spec['size']}"
        entry = files.get(name) or {}
        if entry.get("sha256") != spec["sha256"]:
            return False, f"metadata checksum mismatch for {name}"
        if verify and file_sha256(target) != spec["sha256"]:
            return False, f"checksum mismatch for {name}"
    return True, "artifact already prepared"


def download_file(url, target, expected_sha, expected_size, timeout, retries, token):
    headers = {"User-Agent": "recall-embedding-prepare/1"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last_error = None
    for attempt in range(1, retries + 1):
        tmp_path = f"{target}.download.{os.getpid()}"
        try:
            log("downloading", url=url, attempt=attempt)
            req = urllib.request.Request(url, headers=headers)
            h = hashlib.sha256()
            size = 0
            with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp_path, "wb") as out:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    h.update(chunk)
                    size += len(chunk)
                    out.write(chunk)
            actual_sha = h.hexdigest()
            if size != expected_size:
                raise RuntimeError(f"size mismatch: expected {expected_size}, got {size}")
            if actual_sha != expected_sha:
                raise RuntimeError(f"sha256 mismatch: expected {expected_sha}, got {actual_sha}")
            os.replace(tmp_path, target)
            log("downloaded and verified", target=target, size=size)
            return
        except Exception as e:
            last_error = e
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except OSError:
                pass
            if attempt == retries:
                break
            delay = min(30, 2 ** attempt)
            log(f"download failed, retrying in {delay}s", error=str(e), attempt=attempt, retries=retries)
            time.sleep(delay)
    raise RuntimeError(f"download failed after {retries} attempts: {last_error}")


def copy_file(source, target, expected_sha, expected_size):
    if not os.path.isfile(source):
        raise RuntimeError(f"source file missing: {source}")
    size = os.path.getsize(source)
    if size != expected_size:
        raise RuntimeError(f"source size mismatch for {source}: {size} != {expected_size}")
    actual_sha = file_sha256(source)
    if actual_sha != expected_sha:
        raise RuntimeError(f"source checksum mismatch for {source}: {actual_sha}")
    os.makedirs(os.path.dirname(target), exist_ok=True)
    tmp_path = f"{target}.download.{os.getpid()}"
    shutil.copyfile(source, tmp_path)
    os.replace(tmp_path, target)
    log("copied and verified", source=source, target=target)


def main():
    args = parse_args()
    if args.max_length < 1:
        err("FATAL: --max-length must be >= 1")
        return 2
    max_length = min(args.max_length, DEFAULT_MAX_LENGTH)
    out_dir = os.path.abspath(args.output_dir)

    if args.ensure and not args.force:
        matches, reason = artifact_matches(args, verify=args.verify)
        if matches:
            log("artifact already prepared, skipping", reason=reason, output_dir=out_dir)
            print(json.dumps({"status": "ok", "skipped": True, "output_dir": out_dir,
                              "model": args.model, "reason": reason}))
            return 0
        log("artifact needs preparation", reason=reason, output_dir=out_dir)

    os.makedirs(out_dir, exist_ok=True)
    endpoint = _env("HF_ENDPOINT", "https://huggingface.co").rstrip("/")
    token = (os.environ.get("HF_TOKEN") or "").strip()
    source_dir = os.path.abspath(args.source_dir) if args.source_dir else ""

    try:
        for name, spec in FILES.items():
            target = os.path.join(out_dir, name)
            if not args.force and os.path.isfile(target):
                size = os.path.getsize(target)
                if size == spec["size"]:
                    if not args.verify or file_sha256(target) == spec["sha256"]:
                        log("file already present and verified", target=target)
                        continue
                log("existing file needs replacement", target=target, size=size)
            if source_dir:
                copy_file(os.path.join(source_dir, name), target, spec["sha256"], spec["size"])
            else:
                url = f"{endpoint}/{args.model}/resolve/{args.revision}/{spec['path']}"
                download_file(url, target, spec["sha256"], spec["size"],
                              args.timeout, max(1, args.retries), token)

        metadata = {
            "format_version": ARTIFACT_FORMAT_VERSION,
            "source_model": args.model,
            "revision": args.revision,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "backend": "onnx",
            "provider": "CPUExecutionProvider",
            "dimensions": 384,
            "pooling": DEFAULT_POOLING,
            "normalize": True,
            "max_length": max_length,
            "files": {
                name: {"sha256": spec["sha256"], "size": spec["size"], "path": spec["path"]}
                for name, spec in FILES.items()
            },
        }
        metadata_path = os.path.join(out_dir, ARTIFACT_METADATA)
        tmp_meta = f"{metadata_path}.tmp.{os.getpid()}"
        with open(tmp_meta, "w") as f:
            json.dump(metadata, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(tmp_meta, metadata_path)
        log("artifact published", output_dir=out_dir, model=args.model, revision=args.revision)
        print(json.dumps({"status": "ok", "output_dir": out_dir, "model": args.model,
                          "revision": args.revision, "max_length": max_length}))
        return 0
    except Exception as e:
        err(f"FATAL: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
