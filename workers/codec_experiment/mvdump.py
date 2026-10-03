#!/usr/bin/env python3
"""
Extracts codec-level temporal signals from an already-encoded video using
libavcodec directly (via PyAV), then prints one JSON document on stdout.

This is the only place where real motion vectors can be obtained: the decoder
must be opened with AV_CODEC_FLAG2_EXPORT_MVS and the resulting
AV_FRAME_DATA_MOTION_VECTORS side data is read per decoded picture.

The ffmpeg CLI cannot do this. `codecview` renders motion vectors as pixels and
exposes no numeric output, and no other filter serialises them.

Deliberately NOT implemented here (see FINDINGS.md):
  * residual / transform-coefficient statistics -- libavcodec never surfaces
    decoded residual blocks as frame side data. Not faked here.
  * optical flow or any pixel-difference substitute.

Motion vector units: AVMotionVector.motion_x/motion_y are in 1/4-pel units and
motion_scale is the divisor (4 for quarter-pel). Pixel displacement is
motion_x / motion_scale.
"""

import argparse
import json
import sys

PICT_TYPES = {0: "NONE", 1: "I", 2: "P", 3: "B", 4: "S", 5: "SI", 6: "SP", 7: "BI"}


def empty_motion(grid):
    return {
        "has_side_data": False,
        "count": 0,
        "mean_magnitude": 0.0,
        "median_magnitude": 0.0,
        "max_magnitude": 0.0,
        "mean_dx": 0.0,
        "mean_dy": 0.0,
        "mean_abs_dx": 0.0,
        "mean_abs_dy": 0.0,
        "zero_ratio": 0.0,
        "forward_count": 0,
        "backward_count": 0,
        "spatial_grid": [0.0] * (grid * grid),
    }


def motion_stats(mvs, grid, width, height):
    import numpy as np

    if mvs is None or len(mvs) == 0:
        return empty_motion(grid)

    scale = np.maximum(mvs["motion_scale"].astype(np.float64), 1e-9)
    dx = mvs["motion_x"] / scale
    dy = mvs["motion_y"] / scale
    mag = np.hypot(dx, dy)

    cell_w = max(width // grid, 1)
    cell_h = max(height // grid, 1)
    cx = np.clip(mvs["dst_x"] // cell_w, 0, grid - 1)
    cy = np.clip(mvs["dst_y"] // cell_h, 0, grid - 1)
    cell = cy * grid + cx

    sums = np.bincount(cell, weights=mag, minlength=grid * grid)
    counts = np.bincount(cell, minlength=grid * grid)
    cell_mean = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)

    return {
        "has_side_data": True,
        "count": int(len(mvs)),
        "mean_magnitude": round(float(mag.mean()), 6),
        "median_magnitude": round(float(np.median(mag)), 6),
        "max_magnitude": round(float(mag.max()), 6),
        "mean_dx": round(float(dx.mean()), 6),
        "mean_dy": round(float(dy.mean()), 6),
        "mean_abs_dx": round(float(np.abs(dx).mean()), 6),
        "mean_abs_dy": round(float(np.abs(dy).mean()), 6),
        "zero_ratio": round(float((mag < 1e-6).mean()), 6),
        "forward_count": int((mvs["source"] >= 0).sum()),
        "backward_count": int((mvs["source"] < 0).sum()),
        "spatial_grid": [round(float(v), 6) for v in cell_mean],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = no limit")
    ap.add_argument("--grid", type=int, default=4, help="spatial grid cells per axis")
    args = ap.parse_args()

    grid = max(1, min(args.grid, 32))

    try:
        import av
        from av.sidedata.sidedata import Type as SideDataType
    except Exception as exc:
        print(json.dumps({"extractor_error": "import failed: %s: %s" % (type(exc).__name__, exc)}))
        return 3

    try:
        container = av.open(args.input)
    except Exception as exc:
        print(json.dumps({"extractor_error": "av.open failed: %s" % exc}))
        return 3

    if not container.streams.video:
        print(json.dumps({"extractor_error": "no video stream in container"}))
        return 3

    stream = container.streams.video[0]
    codec_ctx = stream.codec_context
    width = int(codec_ctx.width or 0)
    height = int(codec_ctx.height or 0)

    container_format = ""
    try:
        container_format = container.format.name or ""
    except Exception:
        pass

    time_base = float(stream.time_base) if stream.time_base else 0.0
    try:
        fps = float(stream.average_rate) if stream.average_rate else 0.0
    except Exception:
        fps = 0.0
    try:
        has_b_frames = bool(codec_ctx.has_b_frames)
    except Exception:
        has_b_frames = False
    try:
        nb_frames = int(stream.frames or 0)
    except Exception:
        nb_frames = 0

    duration = 0.0
    try:
        if container.duration:
            import av as _av
            duration = float(container.duration / _av.time_base)
    except Exception:
        duration = 0.0

    try:
        profile = str(codec_ctx.profile or "")
    except Exception:
        profile = ""

    try:
        export_mvs = False
        codec_ctx.flags2 |= av.codec.context.Flags2.export_mvs
        export_mvs = True
        mv_mechanism = "libavcodec AV_CODEC_FLAG2_EXPORT_MVS -> AV_FRAME_DATA_MOTION_VECTORS"
    except Exception as exc:
        mv_mechanism = "AV_CODEC_FLAG2_EXPORT_MVS unavailable in this PyAV/libavcodec build: %s" % exc

    # Pass 1: decode. Pictures arrive in presentation order and carry the motion
    # vector side data that the encoder actually embedded in the bitstream.
    limit = args.max_frames if args.max_frames > 0 else 0
    frames = []
    truncated = False
    last_time = 0.0
    try:
        for picture in container.decode(stream):
            raw_mvs = None
            for sd in picture.side_data:
                if sd.type == SideDataType.MOTION_VECTORS:
                    try:
                        raw_mvs = sd.to_ndarray()
                    except Exception:
                        raw_mvs = None
                    break

            pts = int(picture.pts) if picture.pts is not None else 0
            ts = float(picture.pts) * time_base if picture.pts is not None else last_time
            last_time = ts

            frames.append({
                "pts": pts,
                "timestamp": round(ts, 6),
                "type": PICT_TYPES.get(int(picture.pict_type), "?"),
                "key_frame": bool(picture.key_frame),
                "motion": motion_stats(raw_mvs, grid, width, height),
            })

            if limit and len(frames) >= limit:
                truncated = True
                break
    except Exception as exc:
        print(json.dumps({"extractor_error": "decode failed: %s: %s" % (type(exc).__name__, exc)}))
        return 3
    finally:
        try:
            container.close()
        except Exception:
            pass

    # Pass 2: demux only (no decode) to obtain the bitstream size of each coded
    # picture. Packets arrive in decode order, so results are keyed by pts and
    # merged into the presentation-ordered frame list below.
    packet_bytes = {}
    try:
        container2 = av.open(args.input)
        stream2 = container2.streams.video[0]
        for pkt in container2.demux(stream2):
            if pkt.size and pkt.pts is not None:
                packet_bytes[int(pkt.pts)] = int(pkt.size)
        container2.close()
    except Exception as exc:
        print(json.dumps({"extractor_error": "demux failed: %s" % exc}))
        return 3

    for fr in frames:
        fr["packet_bytes"] = packet_bytes.get(fr["pts"], 0)

    if duration <= 0.0 and frames:
        duration = float(frames[-1]["timestamp"])
        if fps > 0.0:
            duration += 1.0 / fps

    print(json.dumps({
        "extractor": "pyav-libavcodec-export_mvs" if export_mvs else "pyav-frames-only",
        "libav_version": getattr(av, "__version__", ""),
        "container_format": container_format,
        "codec": codec_ctx.name or "",
        "profile": profile,
        "width": width,
        "height": height,
        "fps": round(fps, 6),
        "duration": round(duration, 6),
        "nb_frames_container": nb_frames,
        "has_b_frames": has_b_frames,
        "export_mvs": export_mvs,
        "motion_mechanism": mv_mechanism,
        "truncated": truncated,
        "grid": grid,
        "frames": frames,
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
