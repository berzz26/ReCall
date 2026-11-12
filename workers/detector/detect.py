#!/usr/bin/env python3
import argparse
import json
import os
import sys
import time

# Ensure unbuffered, line-buffered output
try:
    sys.stdout.reconfigure(line_buffering=True, write_through=True)
    sys.stderr.reconfigure(line_buffering=True, write_through=True)
except Exception:
    pass


def log(level, msg, **fields):
    """Production-grade JSON logger to stderr."""
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "level": level,
        "component": "detector",
        "msg": msg,
    }
    if fields:
        record.update(fields)
    try:
        print(json.dumps(record), file=sys.stderr, flush=True)
    except Exception:
        print(f"[{record['timestamp']}] [{level}] [detector] {msg} {fields}", file=sys.stderr, flush=True)


def clamp(v):
    return max(0.0, min(1.0, v))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=False, default="")
    p.add_argument("--output", required=False, default="")
    p.add_argument("--threshold", type=float, default=0.35)
    p.add_argument("--model", default="")
    p.add_argument("--persistent", action="store_true", help="Run in persistent mode: load model once and handle multiple batches via stdin/stdout")
    return p.parse_args()


# 13-class whitelist for B.2 - exact allowed YOLO classes (defined once, not user-configurable)
ALLOWED_CLASSES = {
    "person",
    "car",
    "motorcycle",
    "bus",
    "truck",
    "bicycle",
    "backpack",
    "handbag",
    "suitcase",
    "dog",
    "cat",
    "cell phone",
    "knife",
}


def load_model(model_path):
    """Load YOLO model with timed, structured logging."""
    load_start = time.time()
    log("INFO", "model_load_start", model_path=model_path if model_path else "yolov8n.pt")
    try:
        from ultralytics import YOLO

        mp = model_path if model_path else "yolov8n.pt"
        log("INFO", "model_loading", model_path=mp)
        model = YOLO(mp)
        duration_ms = (time.time() - load_start) * 1000
        log("INFO", "model_load_complete", model_path=mp, duration_ms=round(duration_ms, 2), model_type="ultralytics")
        return model, "ultralytics", duration_ms
    except Exception as e:
        duration_ms = (time.time() - load_start) * 1000
        log("ERROR", "model_load_failed", error=str(e), duration_ms=round(duration_ms, 2))
        print(f"ultralytics not available: {e}", file=sys.stderr)
        return None, None, duration_ms


def _inference_batch(model, frames, threshold):
    """Core batched inference that raises on failure instead of exiting - used by persistent mode."""
    total_frames = len(frames)
    if total_frames == 0:
        return {}

    results = {}
    for fr in frames:
        results[fr["frame_id"]] = []

    paths = [fr["path"] for fr in frames]

    batch_start = time.time()
    try:
        preds = model(source=paths, conf=threshold, verbose=False)
        if not isinstance(preds, list):
            preds = [preds]
    except Exception as e:
        duration_ms = (time.time() - batch_start) * 1000
        log("ERROR", "inference_failed", total_frames=total_frames, error=str(e), duration_ms=round(duration_ms, 2))
        raise RuntimeError(f"batched inference failed for {len(paths)} frames: {e}") from e

    batch_ms = (time.time() - batch_start) * 1000
    total_inference_ms = batch_ms

    if len(preds) != total_frames:
        log("ERROR", "inference_count_mismatch", expected=total_frames, got=len(preds))
        raise RuntimeError(f"batched inference returned {len(preds)} results for {total_frames} inputs")

    for idx, (fr, r) in enumerate(zip(frames, preds), start=1):
        fid = fr["frame_id"]
        w = fr.get("width", 640)
        h = fr.get("height", 480)

        dets = []
        det_count_before_filter = 0
        boxes = getattr(r, "boxes", None)
        if boxes is not None:
            for box in boxes:
                det_count_before_filter += 1
                cls_id = int(box.cls.item()) if hasattr(box.cls, "item") else int(box.cls[0])
                label = r.names.get(cls_id, str(cls_id)) if hasattr(r, "names") else str(cls_id)
                if label not in ALLOWED_CLASSES:
                    continue
                conf = float(box.conf.item()) if hasattr(box.conf, "item") else float(box.conf[0])
                if conf < threshold:
                    continue
                xyxy = box.xyxy[0].tolist() if hasattr(box.xyxy[0], "tolist") else list(box.xyxy[0])
                x1, y1, x2, y2 = xyxy
                fw = w if w > 0 else 640
                fh = h if h > 0 else 480
                bx = clamp(x1 / fw)
                by = clamp(y1 / fh)
                bw = clamp((x2 - x1) / fw)
                bh = clamp((y2 - y1) / fh)
                if bx + bw > 1:
                    bw = 1 - bx
                if by + bh > 1:
                    bh = 1 - by
                if bw <= 0 or bh <= 0:
                    continue
                dets.append(
                    {
                        "label": label,
                        "confidence": conf,
                        "bbox_x": bx,
                        "bbox_y": by,
                        "bbox_width": bw,
                        "bbox_height": bh,
                    }
                )

        avg_ms = total_inference_ms / total_frames if total_frames > 0 else 0
        log(
            "INFO",
            "frame_detection_complete",
            frame_id=fid,
            frame_index=idx,
            total_frames=total_frames,
            detections_kept=len(dets),
            detections_raw=det_count_before_filter,
            duration_ms=round(avg_ms, 2),
            avg_ms_per_frame=round(avg_ms, 2),
        )
        results[fid] = dets

    avg_ms = total_inference_ms / total_frames if total_frames > 0 else 0
    log(
        "INFO",
        "detection_complete",
        total_frames=total_frames,
        total_inference_ms=round(total_inference_ms, 2),
        avg_ms_per_frame=round(avg_ms, 2),
    )
    return results


def detect_with_yolo(model, frames, threshold):
    results = {}
    total_frames = len(frames)
    log("INFO", "detection_start", total_frames=total_frames, threshold=threshold)

    if total_frames == 0:
        log("INFO", "detection_complete", total_frames=0, total_inference_ms=0, avg_ms_per_frame=0)
        return results

    try:
        return _inference_batch(model, frames, threshold)
    except Exception as e:
        print(f"batched inference failed: {e}", file=sys.stderr)
        sys.exit(2)


def persistent_main(args):
    """Persistent mode: load model once, handle multiple batches via stdin/stdout."""
    total_start = time.time()
    log("INFO", "persistent_worker_start", threshold=args.threshold, model=args.model, pid=os.getpid())

    model, kind, model_load_ms = load_model(args.model)
    if model is None:
        log("ERROR", "model_unavailable_failing", model_path=args.model)
        print("FATAL: ultralytics not installed or model failed to load", file=sys.stderr)
        sys.exit(2)

    log("INFO", "persistent_ready", model_load_ms=round(model_load_ms, 2), pid=os.getpid())
    # Signal readiness to Go via stdout (JSON line)
    print(json.dumps({"status": "ready", "model_load_ms": round(model_load_ms, 2)}), flush=True)

    batch_count = 0
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        # Handle exit command
        if line == "exit" or line == '{"command":"exit"}':
            log("INFO", "persistent_exit_command", batch_count=batch_count)
            break
        try:
            req = json.loads(line)
        except Exception as e:
            log("ERROR", "persistent_bad_request", line=line[:500], error=str(e))
            print(json.dumps({"status": "error", "error": f"bad request: {e}"}), flush=True)
            continue

        # Allow explicit exit via JSON
        if req.get("command") == "exit":
            log("INFO", "persistent_exit_command", batch_count=batch_count)
            break

        input_path = req.get("input")
        output_path = req.get("output")
        threshold = req.get("threshold", args.threshold)
        if not input_path or not output_path:
            log("ERROR", "persistent_missing_paths", input=input_path, output=output_path)
            print(json.dumps({"status": "error", "error": "missing input/output paths"}), flush=True)
            continue

        batch_count += 1
        batch_start = time.time()
        try:
            with open(input_path) as f:
                data = json.load(f)
        except Exception as e:
            log("ERROR", "persistent_failed_to_read_input", input_path=input_path, error=str(e))
            print(json.dumps({"status": "error", "error": f"failed to read input: {e}"}), flush=True)
            continue

        frames = data.get("frames", [])
        try:
            out = _inference_batch(model, frames, threshold)
        except Exception as e:
            log("ERROR", "persistent_inference_failed", batch=batch_count, error=str(e))
            print(json.dumps({"status": "error", "error": str(e)}), flush=True)
            continue

        try:
            with open(output_path, "w") as outf:
                json.dump(out, outf)
        except Exception as e:
            log("ERROR", "persistent_failed_to_write_output", output_path=output_path, error=str(e))
            print(json.dumps({"status": "error", "error": f"failed to write output: {e}"}), flush=True)
            continue

        batch_ms = (time.time() - batch_start) * 1000
        total_dets = sum(len(v) for v in out.values())
        log("INFO", "persistent_batch_complete", batch=batch_count, frames=len(frames), detections=total_dets, duration_ms=round(batch_ms, 2))
        print(json.dumps({"status": "ok", "output": output_path, "frames": len(frames), "detections": total_dets}), flush=True)

    total_ms = (time.time() - total_start) * 1000
    log("INFO", "persistent_worker_complete", batches=batch_count, model_load_ms=round(model_load_ms, 2), total_duration_ms=round(total_ms, 2))
    return 0


def main():
    args = parse_args()

    # Persistent mode: load model once and handle multiple batches via stdin/stdout
    if args.persistent:
        return persistent_main(args)

    total_start = time.time()
    # One-shot mode requires input/output
    if not args.input or not args.output:
        log("ERROR", "missing_input_output", input=args.input, output=args.output)
        print("FATAL: --input and --output are required in one-shot mode", file=sys.stderr)
        sys.exit(2)

    log("INFO", "detector_worker_start", input=args.input, output=args.output, threshold=args.threshold, model=args.model, pid=os.getpid())

    # Read input JSON with timing
    read_start = time.time()
    try:
        with open(args.input) as f:
            data = json.load(f)
    except Exception as e:
        log("ERROR", "failed_to_read_input", input_path=args.input, error=str(e))
        sys.exit(2)

    read_ms = (time.time() - read_start) * 1000
    frames = data.get("frames", [])
    log("INFO", "input_loaded", frames=len(frames), duration_ms=round(read_ms, 2), input_path=args.input)

    threshold = args.threshold

    # Model loading (timed inside load_model)
    model, kind, model_load_ms = load_model(args.model)
    if model is None:
        log("ERROR", "model_unavailable_failing", model_path=args.model)
        print(
            "FATAL: ultralytics not installed or model failed to load; failing loud (no synthetic fallback)",
            file=sys.stderr,
        )
        try:
            with open(args.output, "w") as outf:
                json.dump({}, outf)
        except Exception:
            pass
        sys.exit(2)

    # Detection
    detect_start = time.time()
    out = detect_with_yolo(model, frames, threshold)
    detect_ms = (time.time() - detect_start) * 1000

    # Aggregate stats
    total_dets = sum(len(v) for v in out.values())
    log("INFO", "detection_summary", total_detections=total_dets, detection_duration_ms=round(detect_ms, 2))

    # Write output
    write_start = time.time()
    try:
        with open(args.output, "w") as outf:
            json.dump(out, outf)
    except Exception as e:
        log("ERROR", "failed_to_write_output", output_path=args.output, error=str(e))
        sys.exit(2)

    write_ms = (time.time() - write_start) * 1000
    total_ms = (time.time() - total_start) * 1000

    log(
        "INFO",
        "detector_worker_complete",
        total_frames=len(frames),
        total_detections=total_dets,
        model_load_ms=round(model_load_ms, 2),
        detection_ms=round(detect_ms, 2),
        write_ms=round(write_ms, 2),
        total_duration_ms=round(total_ms, 2),
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
