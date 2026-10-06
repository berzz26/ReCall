# ReCall Configuration Reference

All backend settings live in `.env` (see `.env.example` for a template).
The server reads them once at startup, so **restart the server after any change**.

Value formats:

| Format | Accepted values |
|---|---|
| Durations | Go syntax: `500ms`, `2s`, `5m`, `10m` |
| Booleans | `true`/`false`, `1`/`0`, `yes`/`no`, `on`/`off` (case-insensitive) |
| Sizes | Raw bytes, e.g. `1073741824` for 1 GiB |

Invalid values **fail fast**: the server refuses to start and names the bad key.
Anything not set falls back to the built-in default listed below.

## 1. Server & environment

| Variable | Default | What it does |
|---|---|---|
| `APP_ENV` | `development` | `development` enables request logging and dev helpers; use `production` for deployments. |
| `APP_PORT` | `8080` | HTTP port the API listens on. |
| `DATABASE_URL` | `postgres://recall:recall@localhost:5436/recall?sslmode=disable` | Postgres connection string. Tables are created by migrations. |
| `STORAGE_ROOT` | `./storage` | Local directory for uploaded videos and extracted frame JPEGs. |
| `MAX_UPLOAD_SIZE` | `1073741824` (1 GiB) | Largest accepted upload, in bytes. Larger files are rejected with a clean error. |

## 2. Ingest & processing pipeline

| Variable | Default | What it does |
|---|---|---|
| `LOCAL_FILE_STABILITY_SECONDS` | `5` | A watched local file is ingested only after its size stays unchanged for this many seconds — avoids importing half-copied files. |
| `PROCESSING_POLL_INTERVAL` | `2s` | How often the background worker looks for videos waiting to be processed. Lower = snappier pickup, more DB polling. |
| `VIDEO_SEGMENT_DURATION` | `30s` | Videos are split into fixed time windows of this length for per-segment analysis (tracks, events, descriptions). |
| `EVENT_MOVEMENT_THRESHOLD` | `0.05` | Normalized object-center movement (0–1) within a segment that counts as a movement event. Lower = more events. |
| `FFPROBE_PATH` / `FFMPEG_PATH` | `ffprobe` / `ffmpeg` | Binaries for metadata probing and frame extraction. Bare names resolve via `PATH`; set absolute paths to pin versions. |
| `FFPROBE_TIMEOUT` / `FFMPEG_TIMEOUT` | `60s` | Kill a hung probe/extract after this long; the video is marked failed instead of blocking the worker forever. |
| `FRAME_EXTRACT_WORKERS` | auto (`min(NumCPU, 8)`) | Max concurrent ffmpeg decodes during frame extraction. Extraction is one ffmpeg run per timestamp, so this parallelizes it near-linearly up to your core count. Set explicitly to cap CPU usage (e.g. `4`); higher than core count gives diminishing returns since each ffmpeg already multithreads. |

## 3. Frame sampler

ReCall does **not** run object detection on every frame. The sampler picks which
timestamps get extracted as JPEGs and sent to YOLO. Two modes:

- **Baseline**: one frame every `FRAME_SAMPLE_INTERVAL`. Simple, predictable.
- **Adaptive** (default): a cheap visual scan finds the busy moments, then YOLO
  frames are concentrated there — same total count, better coverage.

The budget invariant always holds: total YOLO frames ≤ `N_baseline`, where
`N_baseline = ceil(video_duration / FRAME_SAMPLE_INTERVAL)`.

| Variable | Default | What it does |
|---|---|---|
| `FRAME_SAMPLE_INTERVAL` | `2s` | Baseline grid spacing. Also defines the budget `N_baseline` for adaptive mode. Smaller = more frames = more GPU cost. |
| `SAMPLER_BETA` | `1.0` | Budget multiplier: budget `B = ceil(beta × N_baseline)`. `1.0` = same frame count as baseline. Below `1.0` = cheaper, sparser coverage. |
| `FRAME_JPEG_QUALITY` | `85` | JPEG quality (1–100) of extracted frames. Higher = bigger files on disk and slightly better detection, with diminishing returns past ~85. |
| `SAMPLER_ADAPTIVE` | `true` | Master switch. `false` = exact legacy fixed-interval plan (use in production if you are not ready for adaptive sampling). |

### 3a. Visual probe — the cheap motion scan

Before any YOLO work, each video is downscaled to tiny `PROBE_SIZE × PROBE_SIZE`
grayscale frames at `PROBE_FPS` and compared on a `PROBE_GRID × PROBE_GRID` grid.
Each probe point gets a `ChangedFraction` (0–1) = share of blocks that changed.
This scan is ~100× cheaper than YOLO and drives every decision below.

| Variable | Default | What it does |
|---|---|---|
| `PROBE_FPS` | `5` | Probe decode rate. Higher = finer motion detail, slower planning. |
| `PROBE_SIZE` | `64` | Probe frame is this many pixels square (grayscale). Must be divisible by `PROBE_GRID`. |
| `PROBE_GRID` | `8` | Grid is this many cells square (8×8 = 64 cells). |
| `PROBE_NOISE_K` | `3.0` | Noise floor = `median + K × MAD` of change scores across the video. Change below the floor counts as compression noise, not motion. Higher = only obvious motion registers (good for noisy/grainy footage). |

### 3b. Coarse planner — heartbeat + motion keeps

The planner walks the video in `COARSE_INTERVAL` steps ("heartbeat" grid) and keeps
a step when its probe change ≥ `COARSE_F_KEEP`. `COARSE_MAX_GAP` is a safety net:
no two kept frames may be farther apart than this, so quiet stretches still get
coverage. Leftover budget goes to refinement (§3c).

| Variable | Default | What it does |
|---|---|---|
| `COARSE_INTERVAL` | `4s` | Heartbeat spacing. Smaller = denser base coverage. |
| `COARSE_MAX_GAP` | `10s` | Maximum allowed gap between kept frames. Larger = fewer forced frames in static scenes. |
| `COARSE_F_KEEP` | `0.25` | Keep threshold on `ChangedFraction` (0–1). Higher = only clearly active moments are kept at this stage. |
| `SAMPLER_PLAN_TIMEOUT` | `5m` | Give up planning after this long and fall back to the baseline plan (protects very long videos from slow planning). |

### 3c. Refinement — best-first midpoint insertion

Remaining budget is spent by repeatedly splitting the highest-priority gap at its
midpoint. Gap priority = `max(tracker_disagreement, gamma × probe_rank) × gap`:

- **tracker_disagreement** = births/deaths/fragmentation when a throwaway tracker
  replays the gap — finds moments where tracks appear, vanish, or break.
- **probe_rank** = normalized probe motion in the gap — finds visually busy moments.

| Variable | Default | What it does |
|---|---|---|
| `SAMPLER_GAMMA` | `0.5` | Weight of probe motion vs tracker disagreement. Higher trusts the visual scan more; `0` = refine purely on tracking instability. |
| `SAMPLER_MIN_GAP` | `500ms` | Never insert a frame that would split a gap smaller than this (avoids near-duplicate frames). |
| `SAMPLER_EPSILON` | `0.1` | Stop refining when the best gap scores below this (avoids spending budget on negligible motion). |
| `SAMPLER_MAX_ROUNDS` | `4` | Cap on refinement passes (extra bound on planning time). |

### 3d. Robustness — busy fallback & shadow mode

| Variable | Default | What it does |
|---|---|---|
| `SAMPLER_SHADOW` | `false` | Evaluation-only mode. Computes the adaptive plan but keeps baseline frames authoritative (stats are logged, not used). The visual probe still runs, so this still costs planning time. |
| `SAMPLER_BUSY_THRESHOLD` | `0.25` | Per-point change level that counts as "active". |
| `SAMPLER_BUSY_FRACTION` | `0.6` | If the share of active probe points exceeds this, the video is declared "busy everywhere" and adaptive selection is skipped in favor of a plain uniform grid — when everything moves, motion-guided selection adds nothing. |

## 4. Object detection (YOLO)

| Variable | Default | What it does |
|---|---|---|
| `PYTHON_PATH` | `python3` | Interpreter that runs `workers/detector/detect.py`. Point at a venv/conda env with torch + ultralytics installed (system `python3` rarely has them). |
| `DETECTOR_NAME` | `yolov8n` | Single source of truth for the model. Weights auto-resolve to `workers/detector/{DETECTOR_NAME}.pt`. |
| `DETECTOR_VERSION` | `1` | Version tag stored with every detection row — bump when swapping models so old/new detections stay distinguishable. |
| `MODEL_PATH` | _(derived)_ | Optional absolute-path override of the weights file. |
| `DETECTION_CONFIDENCE_THRESHOLD` | `0.35` | YOLO boxes below this (0–1) are dropped before tracking. Higher = fewer false positives, more missed small/far objects. |
| `YOLO_BATCH_SIZE` | `16` | Frames per detector subprocess call. Higher = better GPU throughput, more VRAM/RAM per call. |

## 5. Tracking (detections → per-object tracks)

| Variable | Default | What it does |
|---|---|---|
| `TRACKER_TYPE` | `iou` | `iou` = simple overlap linker (cheap, fragile). `bytetrack` = Kalman motion + two-stage matching (keeps IDs through brief occlusions). |
| `TRACKER_HIGH_THRESHOLD` | `0.50` | Detections at/above this start new tracks and continue existing ones. Lower = tracks start easier, more false tracks. |
| `TRACKER_LOW_THRESHOLD` | `0.10` | Detections in `[low, high)` never start tracks; they can only extend existing ones (ByteTrack's recovery stage). Must be below high. |
| `TRACKER_MATCH_THRESHOLD` | `0.30` | Minimum box overlap (IoU, 0–1) to link a detection to a track. Lower tolerates fast motion/sparse frames but risks ID swaps. |
| `TRACKER_TRACK_BUFFER` | `5` | Frames a lost track is kept alive awaiting re-match. This is gap tolerance **in frames, not seconds** — sparse sampling eats through it quickly. |
| `TRACKER_FUSE_SCORE` | `true` | Multiply overlap by detection confidence when matching, so strong detections win ambiguous links. |
| `TRACKER_MIN_HITS` | `2` | Detections a candidate needs before it is reported as a track. Higher kills false positives but drops genuinely brief appearances. |

## 6. Video description (VLM)

| Variable | Default | What it does |
|---|---|---|
| `ENABLE_VIDEO_DESCRIPTION` | `true` | Master switch for per-segment text descriptions. Aliases: `ENABLE_VLM`, `VISION_ENABLED`, `ENABLE_DESCRIPTION`. **Embeddings (§7) are skipped too when this is off.** |
| `VISION_PROVIDER` | `local` | `local` = own GPU model, `gemini` = Google API. |
| `VISION_MODEL` / `VISION_MODEL_VERSION` | provider-dependent | Model id + version tag stored with rows. Defaults: local → `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` / `500M-Instruct`; gemini → `gemini-2.5-flash-lite` / `2.5-flash-lite`. |
| `VISION_MODEL_PATH` | _(provider default)_ | Local weights directory. **Required** for the local provider. |
| `VISION_PYTHON_PATH` | `python3` | Interpreter for `workers/vision/describe.py` (local provider only). |
| `GEMINI_API_KEY` | _(none — required for gemini)_ | API key. **Required** when `VISION_PROVIDER=gemini`. Keep secret. |
| `VISION_MAX_FRAMES` | `3` | Frames sent to the VLM per segment (≥ 1). More frames = richer descriptions, higher cost/latency. |
| `VISION_MAX_OUTPUT_TOKENS` | `256` | Cap on description length per segment. |
| `VISION_TIMEOUT` | `10m` | Kill a hung description job after this long. |
| `VISION_HISTORY_SEGMENTS` / `VISION_HISTORY_EVENTS` | `2` / `5` | How many previous segments/events are passed as context so descriptions stay consistent ("the same car"). |

## 7. Embeddings (semantic search index)

Segment descriptions are embedded for natural-language search. Generated only
when `ENABLE_VIDEO_DESCRIPTION=true`.

| Variable | Default | What it does |
|---|---|---|
| `EMBEDDING_PYTHON_PATH` | `python3` | Interpreter for `workers/embedding/embed.py`. |
| `EMBEDDING_MODEL` / `EMBEDDING_MODEL_VERSION` | `BAAI/bge-small-en-v1.5` / `v1.5` | Sentence-transformer id + version tag. |
| `EMBEDDING_TIMEOUT` | `5m` | Kill a hung embedding job after this long. |

## 8. Search API

| Variable | Default | What it does |
|---|---|---|
| `SEARCH_CANDIDATE_LIMIT` | `20` | Embedding recall pool size per query (retrieved, then re-ranked/filtered). Higher = better recall, slower queries. |
| `SEARCH_DEFAULT_LIMIT` | `10` | Results returned when the caller asks for no limit. |
| `SEARCH_MAX_LIMIT` | `50` | Hard cap on results per request (must be ≥ default limit). |
| `SEARCH_MIN_SIMILARITY` | `0.35` | Cosine-similarity floor (0–1); hits below this are cut. |

## 9. Frontend (`services/web/.env`)

| Variable | Default | What it does |
|---|---|---|
| `VITE_API_BASE_URL` | _(empty)_ | Leave empty to call the API same-origin through the Vite dev proxy (recommended). Set to e.g. `http://localhost:8081` for direct backend access. |

## 10. Keys you may see that currently do nothing

Some `.env` files contain `CODEC_EXPERIMENT_*` and `CODECSIGHT_DYNAMIC_*` keys
from an old codec-activity experiment. **No code reads them** — they have zero
effect on the server and can be left as-is or removed.
