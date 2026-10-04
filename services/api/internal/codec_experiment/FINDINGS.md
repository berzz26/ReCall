# TEMPORARY experiment — CodecSight codec-signal inspection

**Status: throwaway. Delete `services/api/internal/codec_experiment/`,
`services/api/cmd/codec-experiment/`, `workers/codec_experiment/`, the
`CODEC_EXPERIMENT_*` config block, the processor hook, and this file when the
investigation is over.**

This experiment does exactly one thing: read codec-level temporal signals off an
already-encoded video and dump them for inspection. It is not a frame scheduler,
not a pipeline stage, and it does not change how ReCall processes video.

---

## 0. Correction to the premise: ReCall has no transcoding stage

The task was to insert the experiment "after transcoding, before downstream
frame processing". That stage does not exist in this codebase.

ReCall's actual media path is:

```
original uploaded file  (videos/{id}/original.mp4)   <- already encoded by whatever produced it
    |
    +-- ffprobe -show_format -show_streams          processing/ffprobe.go:270   metadata only
    |
    +-- ffmpeg -i <src> -vf fps=N -vcodec mjpeg -f image2pipe | pipe:1
                                                       video_frame/service.go:223  DECODE -> MJPEG
    |     |
    |     +-- storage/videos/{id}/frames/%06d.jpg   intra-only JPEGs
    |     +-- video_frames rows in Postgres
    |
    +-- YOLO -> ByteTrack -> events -> VLM -> BGE embeddings
```

There is no encoder producing a new video file. The only `-vcodec` in the whole
repository is `mjpeg` at `video_frame/service.go:229`, and its output is a pipe
of JPEGs, never a file. Grep for `transcode`, `-c:v`, `-preset`, `-crf`,
`-movflags`, `-g`, `-bf` returns nothing.

### What this changes about the experiment

- **The signal source is the original input file**, which ReCall passes to ffprobe
  and ffmpeg untouched. That is the file the experiment reads. It is a complete,
  valid compressed video with real GOP structure and real motion vectors.
- **ReCall does not destroy the information before the hook.** There is no
  re-encode, so there is no lossy transcode step that could have discarded it.
  Nothing is lost up to the point where the experiment runs.
- **ReCall destroys it immediately afterwards.** `frameService.GenerateForVideo`
  decodes the source and re-encodes every sampled frame as a standalone
  intra-only MJPEG. From that point on the JPEG files contain no picture types,
  no motion vectors and no residual structure — they are independent images. Any
  downstream consumer (YOLO, VLM) sees only pixels.

So the answer to question 5 ("do ReCall's current transcoding settings preserve or
destroy useful codec information") is: **the pipeline preserves it in the source
file and discards it at the frame-extraction boundary.** That boundary is
precisely where a cheap pre-inference selection stage would have to run, and the
hook is placed immediately before it.

The hook sits in `processing/ffprobe.go` right after media metadata is
persisted, which is the last point in `Process` where `videoPath` is the intact
source and nothing has been written yet.

---

## 1. Frame types — available

Every coded picture declares its type in the slice header, so this is readable
from any H.264/HEVC stream with no special flags.

- `decoder picture type` (`AVFrame.pict_type`) when PyAV is present
- `ffprobe -show_entries frame=pict_type` otherwise

Collected per frame: index, pts, timestamp, type, keyframe flag, plus the
bitstream size of that picture. From the keyframe positions the experiment
derives GOP length mean/median/min/max.

Cost: `-show_frames` requires a full decode. `ffprobe -show_packets` does not,
so bitstream sizes can be gathered cheaply and independently.

## 2. Motion vectors — available, but NOT through the ffmpeg CLI

Verified on this machine against `ffmpeg 6.1.1`:

```
$ ffmpeg -hide_banner -h filter=codecview
  mv   <flags>  pf|bf|bb   set motion vectors to visualize
  qp   <boolean>
  ...
```

`codecview` **renders** motion vectors into the picture. It exposes no metadata
option and no numeric output, and no other filter serialises
`AV_FRAME_DATA_MOTION_VECTORS`. There is no `ffprobe` field for them. So:

> The ffmpeg command line cannot extract motion vectors as numbers, no matter how
> it is configured. `-flags2 +export_mvs` only controls whether the decoder
> attaches the side data at all; `codecview` then draws it instead of reporting
> it.

What works is driving libavcodec directly with `AV_CODEC_FLAG2_EXPORT_MVS` and
reading `AV_FRAME_DATA_MOTION_VECTORS`. `workers/codec_experiment/mvdump.py` does
this via PyAV (`av>=12`, already installed here: 18.1.0). Per frame it reports:

| field | meaning |
|---|---|
| `count` | number of exported motion vectors |
| `mean_magnitude`, `median_magnitude`, `max_magnitude` | displacement magnitude in pixels |
| `mean_dx`, `mean_dy` | signed mean horizontal/vertical displacement |
| `mean_abs_dx`, `mean_abs_dy` | same, unsigned — more informative for B frames where forward and backward vectors cancel |
| `zero_ratio` | fraction of blocks the encoder predicted with zero displacement |
| `forward_count`, `backward_count` | MV_FIELD_FORWARD vs MV_FIELD_BACKWARD split |
| `spatial_grid` | 4x4 grid of mean magnitude by destination block position |

Unit note: `motion_x/motion_y` are quarter-pel, `motion_scale` is the divisor
(4). Pixel displacement is `motion_x / motion_scale`. Verified against a real
stream: `motion_x=5, motion_scale=4` yields a reported max magnitude of 1.25 px.

`has_side_data` is tracked separately from the values, so an intra picture or a
fully-skip inter picture is not silently reported as "static content".

**If PyAV is not installed**, the experiment falls back to `ffprobe` and records
motion vectors as `available: false` with the reason and the requirement. It
does not substitute optical flow, `mestimate`, or pixel differences.

## 3. Residuals — NOT available, and not obtainable through this API

> **Residual information: Not available.**

Why:

- In H.264/HEVC, residual data is the quantised transform-coefficient block
  produced after motion compensation, entropy-coded inside the slice. libavcodec
  decodes it into pixels and discards it. There is **no `AV_FRAME_DATA_*` entry
  for decoded residuals**, and no `ffprobe` field for them.
- It is a decoder-API limitation, not a missing flag. There is no configuration
  that turns it on.
- The nearest proxy that *is* exposed is per-block QP, and even that is only
  renderable: `codecview=qp=1` paints it, it does not report it.

What would be required: a bitstream-level NAL parser or an instrumented decoder
that intercepts inverse-transform output — for example a patched libavcodec,
`libde265`, or a custom decode loop that sums residual coefficients per block.

Would that require changing ReCall's encoding pipeline? **No** — the residual
information is already present in the source file and ReCall never re-encodes, so
nothing needs to change upstream. It only requires decoding the original stream,
which is exactly what the experiment already does for motion vectors.

The hard constraint is positional, not structural: residuals (like motion
vectors) only exist relative to a reference picture. They cannot survive
`video_frame.GenerateForVideo`'s intra-only MJPEG re-encode. Any capture point
after frame extraction is worthless.

### Encoded size is reported, but is not residual

`packet_bytes` per coded picture (bytes the encoder spent on that picture) is
included because it is free and genuinely codec-derived. It correlates with
residual energy — an I frame costs orders of magnitude more than a static P frame
(measured on a real clip: 300 273 bytes for an I frame vs 1 358 for a P frame).

It is labelled `packet_summary` and flagged in the report as
`NOT residual energy`. It is an entropy-coded byte count, not a coefficient
magnitude, and it must not be read as one.

---

## 4. Layout

```
codec_experiment/
    <video_name>.json   full per-frame record + availability verdicts + summaries
    <video_name>.csv    one row per frame, flat, with the 4x4 grid spread into columns
    <video_name>.png    timeline: frame types, motion magnitude/density/zero-ratio, encoded size
```

The PNG uses only `image/png` from the stdlib; the 5x7 label font is a small
table in `font.go`. No charting or font dependency was added to the repo.

Bar lanes are downsampled to one pixel column by **peak**, not mean, so a
one-frame spike stays visible; the zero-ratio lane uses mean because its meaning
is an aggregate. Each lane is annotated with its own maximum.

---

## 5. Running it

In-pipeline (default off; when off, `codecExperiment` is nil and the pipeline is
unchanged):

```
CODEC_EXPERIMENT_ENABLED=true
pip install -r workers/codec_experiment/requirements.txt
```

The report is printed to stdout and the artifacts land in
`CODEC_EXPERIMENT_OUT_DIR` (default `./codec_experiment`).

Standalone, against any file already on disk, without ingesting a video:

```
make codec-experiment INPUT=/home/berzz/recallTestVideo/video1.mp4
```

Note the experiment performs a full decode of the source video and runs
synchronously inside `Process`. `CODEC_EXPERIMENT_TIMEOUT` (default 10m) bounds
it, and `CODEC_EXPERIMENT_MAX_FRAMES` caps the frame count for very long files.
A truncated run is flagged as such in both the report and the JSON.

Side-effect freedom: `Run` only reads the source file and writes into its own
output directory. It performs no database writes, touches no storage key, and
every failure is logged and swallowed — `Process` cannot fail because of it.

---

## 6. Measured results

From the two videos already present in `~/recallTestVideo`, run via
`make codec-experiment INPUT=...`. Both extractors (PyAV and the ffprobe
fallback) agree exactly on picture type counts and per-picture byte totals,
which cross-validates both implementations against each other.

### video1.mp4 — 1270x720, 13.093 fps, 110.94 s, 1452 pictures

| | |
|---|---|
| picture types | B 1073, P 373, I 6 |
| GOP length | mean 250.0, median 250.0, min 250, max 250 |
| encoded size | total 23.8 MB, mean 16.4 kB, min 36 B, max 330 kB |
| mean bytes by type | I 305184, P 43124, B 5512 |
| motion available | yes, on 1446 of 1452 pictures |
| motion mean magnitude | mean 0.99 px, median 0.71 px, max 212.6 px |
| zero-vector ratio | 0.822 |

The 6 pictures without motion side data are exactly the 6 I pictures — the
`has_side_data` separation behaves as intended.

**Answering Q1 for this clip: no.** GOP length is 250 frames at *every* interval,
min = max = mean. That is the encoder's periodic keyframe insertion doing exactly
what the encoder always does, with zero relation to content. I/P/B structure
carried no usable temporal information here. The PNG timeline shows the six red
markers evenly spaced across the whole duration.

### video2.mp4 — 3840x2160, 15 fps, 11.91 s, 178 pictures

| | |
|---|---|
| picture types | B 127, P 50, I 1 |
| GOP length | n/a (single keyframe) |
| encoded size | total 5.2 MB, mean 29.3 kB, min 309 B, max 508 kB |
| mean bytes by type | I 508708, P 83814, B 4076 |
| motion mean magnitude | mean 0.21 px, median 0.11 px, max 532.6 px |
| zero-vector ratio | 0.944 |

### What the numbers say about usefulness so far

Still not a conclusion, but three things are now visible that were not before:

1. **Zero-vector ratio is doing the work, not vector count.** Both clips are
   overwhelmingly static (0.82 and 0.94 zero), yet they emit 10k–63k motion
   vectors per picture. A naive "count the vectors" activity signal would report
   both clips as extremely busy. `mv_count` on its own is actively misleading.
2. **Encoded size separates far more cleanly than motion.** I 305 kB / P 43 kB /
   B 5.5 kB is a two-order-of-magnitude spread. Mean motion magnitude spans
   0.21–0.99 px — a 5x range on clips whose visual difference is presumably far
   larger than that. Motion magnitude looks like a weak discriminator on this
   material.
3. **The two clips differ by ~10x in vectors/frame (10k vs 63k) purely as a
   function of resolution** (720p vs 4K). Any future signal has to be normalised
   for resolution or block count, or it is measuring the wrong thing.

None of this has been checked against what ReCall actually samples, which is the
comparison that decides whether these signals buy anything over the current
fixed 2 s sampling.

## 7. What this experiment does and does not answer

Answered by construction, from inspecting the stack:

- Frame types, timestamps and GOP structure are readable with no special flags,
  from the file ReCall already has.
- Motion vectors are obtainable, but only via libavcodec directly. The ffmpeg CLI
  cannot do it.
- Residuals are not obtainable through libavcodec or ffmpeg at all.
- ReCall's pipeline preserves codec information in the source file and destroys
  it at the MJPEG frame-extraction boundary.

Answered empirically, on the two available clips (section 6):

- **Q1 (I/P/B structure):** no useful signal. GOP length was 250 frames at every
  single interval on video1 — uniform, content-independent, encoder-imposed.
- **Q3 (residuals):** not accessible at all, so the question is moot until a
  bitstream parser or instrumented decoder exists.
- **Q2 and Q4 are still open.** Motion magnitude spread was narrow (0.2–1.0 px)
  while zero-vector ratio and encoded size both separated widely. Temporal
  resolution is per-picture and therefore higher than ReCall's 2 s sampling, but
  whether that extra resolution carries information these clips do not exhibit is
  not yet demonstrated.
- **Q5:** answered definitively — preserved in the source file, discarded at the
  MJPEG boundary. Nothing upstream destroys it.

**Not answered yet, and deliberately not claimed:** whether any of these signals
actually correlate with meaningful visual change. That is what the artifacts are
for. Extraction succeeding is not evidence of usefulness. The open questions:

1. Does I/P/B structure mark transitions, or is GOP placement just the encoder's
   periodic keyframe insertion with no relationship to content? **Answered: the
   latter, at least for video1 — see section 6. Confirm on more material.**
2. Does motion magnitude separate "nothing is happening" from "something is
   happening"? Watch the zero-ratio lane first: a static clip can still emit
   thousands of motion vectors that are almost all zero, which would make raw
   vector *count* actively misleading as an activity signal. **Partly answered:
   count is indeed misleading; magnitude spread was too narrow to judge.**
3. Is encoded size a better activity proxy than motion, given an I frame is large
   by construction and not because of the scene? **Open — this is the strongest
   candidate so far and deserves the next experiment.**
4. Is per-frame resolution enough to guide selection, given ReCall currently
   samples every 2 s (`FRAME_SAMPLE_INTERVAL`) while the codec exposes every
   picture? **Open.**

Compare the timeline against the frames ReCall actually sampled before drawing
any conclusion. Nothing in this directory turns these signals into a selection
policy, and no inference behaviour was changed.

---

## 8. Verification performed

- `make build` clean; `go vet` clean on both new packages.
- `make fmt` applied (it also reformats five pre-existing files that were not
  gofmt-clean; those were reverted to keep this diff focused).
- `make run` starts and shuts down cleanly with the experiment off, and logs no
  experiment lines — default behaviour is unchanged.
- `make run` with `CODEC_EXPERIMENT_ENABLED=true` logs the expected wiring.
  An invalid value (`maybe`) panics with the intended message.
- Both extractors exercised on both videos. The PyAV and ffprobe paths produce
  identical picture type counts and identical per-picture byte totals.
- `make test` passes except `pkg/database`, which fails identically on a clean
  checkout (that package does not load `.env`, so it cannot reach Postgres; the
  server does load it and connects fine). Pre-existing and unrelated.

---

## 9. Iteration 2 — dynamic codec-activity sampling (throwaway)

Opt-in via `CODECSIGHT_DYNAMIC_SAMPLING=true` (default `false`). When off, frame
extraction is the existing fixed-interval sampler, untouched. When on, the codec
analysis above produces an activity score per coded picture; pictures above the
enter threshold open a dense window (±1 s context, merged within a 1 s gap);
inside windows the existing single-process `fps` stream runs at the dense
interval (default 0.2 s) and only planned timestamps are retained. Dense outputs
are a subset of one deterministic grid, so the final sequence is sorted,
deduplicated, and chronological, and downstream stages see the same
`VideoFrame` representation.

Terminology used throughout: **high-activity frames / codec activity
candidates**. Never "high-value frames". The codec marks where encoded motion
changed; YOLO/VLM decide what it means.

Activity score (experimental weights, explicitly not optimal):

```
score = 0.40 * norm(motion mean) + 0.20 * norm(motion max)
      + 0.25 * (1 - norm(zero ratio)) + 0.15 * norm(packet bytes)
```

Normalization is percentile-based (p5–p95) for magnitudes and packet size, so a
few extreme values cannot dominate a video. Raw motion-vector count is excluded
on purpose: static content emits thousands of zero-displacement vectors. Enter
0.90 / exit 0.70 hysteresis prevents flapping; the first threshold pair tried
(0.55/0.35) marked a whole 12 s clip dense and was rejected for that reason.

Residuals remain unattempted, as before.

Measured, same two clips (`make codec-experiment INPUT=... ARGS="-dynamic"`):

| clip | baseline | dynamic | added | ratio | regions |
|---|---|---|---|---|---|
| video2 (12 s) | 6 | 29 | 23 | 4.8× | 1.1–3.1 s, 4.4–7.2 s |
| video1 (111 s) | 56 | 168 | 112 | 3.0× | 7 regions |

Without PyAV the planner degrades to `baseline_fallback` with a recorded
reason — it never fails the pipeline, and frame output is then identical to
baseline. With motion available, the 4–5 s region the human ground truth
describes (man opens car door) falls inside the 4.4–7.2 s dense window on
video2. Whether those added frames are *semantically* better is still
unevaluated, by design.

Artifacts per video gain `<name>_dynamic.json` (config, summary, regions, exact
baseline/selected/added timestamps, per-picture scores) and
`<name>_dynamic.png` (activity with enter/exit lines, baseline grid, dynamic
samples over shaded regions, region peaks).

