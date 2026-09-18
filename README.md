# ReCall

### Search hours of video by what actually happened.

**ReCall turns hours long CCTV or camera footage into searchable events.**

Upload a video, or point ReCall at a local folder, and search for moments using natural language:

```text
"person in a white shirt near the counter"

"someone entered the restricted area"

"what happened before the machine stopped?"

"show me every vehicle after 10 PM"
```

ReCall analyzes the footage, detects and tracks objects, builds a time-aware understanding of what is happening, and lets you jump directly to the relevant moment in the original video.
Best part? it does everything locally.

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/a53b9213-7973-478c-91a9-15ff8e9e2f31" width="100%"></td>
    <td><img src="https://github.com/user-attachments/assets/8786562d-1269-4fc9-bb7e-ba3544cc4ae9" width="100%"></td>
  </tr>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/d91ed050-cd32-47d4-bd4c-cb047d7a1afc" width="100%"></td>
    <td><img src="https://github.com/user-attachments/assets/f91e8669-4d68-4a3d-b5de-f3629bcd5837" width="100%"></td>
  </tr>
</table>

---

### Demo:
https://github.com/user-attachments/assets/3778d3f5-c338-4298-a296-e29dbfb27227

---
## What is ReCall?

ReCall watches your footage for you and makes it searchable.

Instead of:

```text
Open 8-hour recording
        ↓
Scrub through the timeline
        ↓
Guess where the event happened
        ↓
Watch the surrounding footage
```

you can ask:

```text
"Show me when someone entered the restricted area."

"Find every time a person fell near the entrance."

"Show me every vehicle that entered after 10 PM."

"What happened right before the warning light?"
```

ReCall returns relevant moments from the footage with:

* the timestamp
* a description of what happened
* the matching video
* relevant frames
* detected objects and tracks
* bounding boxes where available

You can click a result and jump directly to that point in the original video.

### The basic idea

```text
Raw video
    ↓
Understand what is in each frame
    ↓
Track objects across time
    ↓
Understand what happened
    ↓
Store the resulting information
    ↓
Search it later
    ↓
Jump back to the original footage
```

ReCall is designed to run **locally by default**, so sensitive footage does not need to leave your machine. A cloud vision model can optionally be used when desired.

---

## The problem

Cameras are good at recording everything.

Finding one specific thing in that recording is a different problem.

A camera may record eight hours of footage containing thousands of frames. If something happened at 3:17 PM, finding it manually means searching through that recording yourself.

Object detection helps, but individual detections are not the same as understanding an event.

For example, a detector can tell you:

```text
14:31:04  Person
14:31:08  Person
14:31:12  Person
14:31:16  Person
```

That is useful, but it does not directly tell you:

```text
14:31:04
A person approaches the machine.

14:31:18
The person opens the machine panel.

14:31:32
A warning light appears.

14:31:41
The machine stops.
```

The useful information is not just **what appeared in a frame**.

It is **what happened over time**.

That is what ReCall is built around.

## How ReCall is different

There are already good systems for analyzing and searching video. ReCall is not trying to replace all of them.

The difference is in **what ReCall treats as the fundamental unit of information**.

Traditional video analytics often starts with labels:

```text
Person
Car
Truck
Face
License plate
```

ReCall is being built around the idea that the useful unit is an **event over time**:

```text
Person
  ↓
Track
  ↓
Movement
  ↓
Context
  ↓
Event
  ↓
Description
  ↓
Searchable evidence
```

| Approach                       | Typical strength                                                             | ReCall's focus                                                                                           |
| ------------------------------ | ---------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| **Traditional NVR / VMS**      | Recording, playback, timelines, motion search                                | Search by what happened, not just when it happened                                                       |
| **Cloud video APIs**           | Large set of managed detection and media-analysis capabilities               | Local-first processing and an application-specific event representation                                  |
| **Video understanding APIs**   | Natural-language search and high-level video understanding                   | Open, inspectable pipeline from detection to track to event to evidence                                  |
| **Enterprise video analytics** | Multi-camera investigation, appearance search, alerts and security workflows | A developer-oriented system that can be run and modified on your own infrastructure                      |
| **Local NVR / CV systems**     | Local detection, tracking, recording and increasingly semantic search        | Move from tracked objects toward temporal descriptions and searchable events                             |
| **DIY CV pipelines**           | Maximum flexibility                                                          | One pipeline that connects video processing, tracking, event generation, embeddings, search and evidence |

### ReCall's focus

**Local**

The core pipeline can run on your own hardware. Sensitive footage does not have to be sent to a cloud AI provider.

**Temporal**

ReCall does not treat every frame as an independent observation. Objects are tracked across time and used to build events and descriptions.

**Searchable**

The output of processing is not just a collection of detection labels. Video segments receive semantic descriptions and embeddings that can be searched using natural language.

**Inspectable**

Search results remain connected to their source data:

```text
Query
  ↓
Segment
  ↓
Description
  ↓
Events
  ↓
Tracks
  ↓
Frames
  ↓
Original video
```

**Composable**

The pipeline is built from replaceable components rather than being tied to a single proprietary video analytics stack.

Today that means:

```text
YOLO
  +
ByteTrack / IoU
  +
SmolVLM2 or Gemini
  +
BGE
  +
PostgreSQL / pgvector
```

The individual components can evolve independently as better models and processing techniques become available.

### Where ReCall fits

ReCall is best thought of as the layer between **computer vision and video search**.

```text
                    Video
                      │
                      ▼
              Computer Vision
            detection + tracking
                      │
                      ▼
                  ReCall
          events + descriptions
             + semantic search
                      │
                      ▼
                  Evidence
             frames + video
```

The goal is not simply to detect more objects.

The goal is to make recorded video **queryable**.


## The processing pipeline

The current pipeline is intentionally straightforward.

```text
                    Video
                      │
                      ▼
               Video Ingestion
                      │
                      ▼
              Segment + Sample
                      │
                      │
                frames every 2s
                      │
                      ▼
              Object Detection
                      │
                      │
                   YOLO
                      │
                      ▼
                 Tracking
                      │
                ByteTrack / IoU
                      │
                      ▼
              Event Generation
                      │
                      ▼
           Temporal Description
                      │
              SmolVLM2 / Gemini
                      │
                      ▼
                 Embedding
                      │
                    BGE
                      │
                      ▼
               PostgreSQL
                + pgvector
                      │
                      ▼
                  Search
                      │
                      ▼
             Relevant Segments
                      │
                      ▼
                  Evidence
                      │
                      ▼
             Video + Frames
```

The current implementation processes videos sequentially through this pipeline.

There is no Kafka, Redis, or distributed processing queue in the current system.

Processing is driven by a Go polling worker that claims uploaded videos from PostgreSQL and records progress using processing checkpoints.

This keeps the current system simple while the core pipeline is being developed.

---

## Why sample instead of processing every frame?

A 30 FPS camera produces:

```text
30 × 60 × 60 = 108,000 frames per hour
```

Running every expensive model on all 108,000 frames would be wasteful.

ReCall currently samples footage at a lower rate for the initial processing stage:

```text
30 FPS video
    ↓
1 frame every 2 seconds
    ↓
30 frames / minute
```

This makes the pipeline substantially cheaper.

However, sampling introduces a fundamental tradeoff.

An event can happen between two sampled frames:

```text
Frame A
   │
   │
   │   Person falls
   │
   │
Frame B
```

If the event is never captured, it cannot be recovered later.

This is why ReCall is designed around **high recall**.

The current implementation uses fixed sampling. An adaptive multi-stage pipeline that increases processing around potentially interesting activity is planned for a future stage.

---

## Search

Once the video has been processed, the descriptions of its segments are embedded using:

```text
BAAI/bge-small-en-v1.5
```

and stored in PostgreSQL using pgvector.

A search works roughly like this:

```text
User query
    ↓
Generate query embedding
    ↓
pgvector similarity search
    ↓
Find relevant segments
    ↓
Enrich with detections, tracks and events
    ↓
Return matching moments
    ↓
Jump to timestamp in video
```

For example:

```text
Query:

"person in white shirt near the counter"
```

can retrieve a segment whose description contains:

```text
A person wearing a white shirt approaches
and stands near the counter.
```

The result still points back to the actual footage.

This is important because ReCall is not intended to produce an answer and leave the user to trust it.

The result should be **verifiable against the video**.

---

## Evidence

Every search result is connected to the underlying video data.

```text
Search result
     │
     ▼
Segment
     │
     ├── Description
     ├── Timestamp
     ├── Detections
     ├── Tracks
     └── Events
             │
             ▼
           Frames
             │
             ▼
        Original video
```

From the result, the user can:

* jump to the relevant timestamp
* play the surrounding video
* inspect matching frames
* inspect detected objects
* inspect tracks and events
* view bounding boxes

The goal is simple:

**If ReCall tells you something happened, you should be able to see the footage that supports it.**

---

## Architecture

ReCall currently consists of a Go application, a web application, Python-based ML workers, and PostgreSQL.

```text
                         ┌───────────────┐
                         │   React Web   │
                         │  Vite + TS    │
                         └───────┬───────┘
                                 │
                                 ▼
                         ┌───────────────┐
                         │   Go API      │
                         │    Fiber      │
                         └───────┬───────┘
                                 │
                    ┌────────────┼────────────┐
                    │            │            │
                    ▼            ▼            ▼
                Processing    Search       Video
                  Worker       API         Storage
                    │            │
                    │            ▼
                    │       PostgreSQL
                    │        + pgvector
                    │
                    ▼
             Python ML Workers
                    │
          ┌─────────┼─────────┐
          ▼         ▼         ▼
        YOLO     SmolVLM2     BGE
```

The current processing architecture is deliberately simple.

A Go worker polls PostgreSQL for uploaded videos, claims work using PostgreSQL row locking, and runs the processing pipeline.

Progress is stored in:

```text
video_processing_checkpoints
```

This allows processing to resume from completed segments rather than starting an entire video from scratch.

The ML components are currently invoked as Python processes from the Go processing pipeline rather than operating as independently scaled services.

---

## Tech stack

| Layer                 | Technology                   |
| --------------------- | ---------------------------- |
| API                   | Go + Fiber                   |
| Web                   | React + Vite + TypeScript    |
| Database              | PostgreSQL 16                |
| Vector search         | pgvector                     |
| Video processing      | FFmpeg + ffprobe             |
| Object detection      | Ultralytics YOLO             |
| Tracking              | ByteTrack / IoU              |
| Vision-language model | SmolVLM2-500M-Video-Instruct |
| Optional cloud vision | Gemini                       |
| Embeddings            | BAAI/bge-small-en-v1.5       |
| ML workers            | Python                       |
| Video storage         | Local filesystem             |
| Deployment            | Docker + Docker Compose      |

### Current models

```text
Detection
    YOLOv8n / YOLO11n

Tracking
    ByteTrack / IoU

Temporal description
    SmolVLM2-500M-Video-Instruct
    or Gemini

Embeddings
    BAAI/bge-small-en-v1.5

Vector storage
    PostgreSQL + pgvector
```

The local pipeline does not require a cloud vision API.

Gemini is an optional provider for users who want to use a cloud VLM instead of the local vision model.

---

## Repository

```text
recall/
├── migrations/
│   └── PostgreSQL + pgvector migrations
│
├── pkg/
│   └── database/
│       └── PostgreSQL connection and queries
│
├── services/
│   ├── api/
│   │   ├── API
│   │   ├── processing
│   │   ├── search
│   │   ├── tracker
│   │   └── vision
│   │
│   └── web/
│       └── React + Vite application
│
├── workers/
│   ├── detector/
│   │   └── YOLO inference
│   │
│   ├── vision/
│   │   └── temporal descriptions
│   │
│   └── embedding/
│       └── BGE embeddings
│
├── storage/
│   └── local video and frame storage
│
├── docker-compose.yml
├── Makefile
├── go.mod
└── go.sum
```

---

## Run locally

### Requirements

* Go
* Docker
* Docker Compose
* Node.js
* Python
* FFmpeg

Clone the repository:

```bash
git clone https://github.com/berzz26/recall.git
cd recall
```

Configure the environment:

```bash
cp .env.example .env
```

For fully local vision processing:

```text
VISION_PROVIDER=local
```

For Gemini:

```text
VISION_PROVIDER=gemini
GEMINI_API_KEY=...
```

Start the backend:

```bash
docker compose up -d
```

The API runs on:

```text
http://localhost:8081
```

Start the web application:

```bash
cd services/web
npm install
npm run dev
```

The web application runs on:

```text
http://localhost:5173
```

### Add video

Videos can be added in two ways:

**Upload**

```text
Videos → + Add Video
```

**Local folder**

```text
Settings → Add Local Source
```

Provide an absolute path to the directory containing your videos.

ReCall will ingest the video and begin processing it through the pipeline.

---

## Current status

ReCall is under active development.

The current end-to-end pipeline is working for local video files:

```text
Video
  ↓
Detection
  ↓
Tracking
  ↓
Events
  ↓
Temporal description
  ↓
Embeddings
  ↓
pgvector
  ↓
Semantic search
  ↓
Video evidence
```

The current system uses fixed frame sampling and sequential processing.

### Currently implemented

* Local video ingestion
* Video segmentation
* Frame sampling
* YOLO object detection
* Batched detection
* ByteTrack tracking
* IoU tracking
* Track persistence
* Event generation
* Temporal video descriptions
* Local SmolVLM2 inference
* Optional Gemini vision
* BGE embeddings
* PostgreSQL
* pgvector semantic search
* Search result enrichment
* Timestamp-based video playback
* Frame and bounding-box inspection
* Processing checkpoints

### Not implemented yet

The following are planned rather than current capabilities:

* Adaptive frame sampling
* Automatic cheap-to-expensive processing
* Audio and speech understanding
* OCR
* Cross-camera re-identification
* Distributed processing queues
* Complex temporal queries such as `before`, `after`, and `stayed > 30s`
* Large-scale distributed video processing

---

## Roadmap

The immediate focus is reliability and better temporal understanding.

### Processing

* Improve processing checkpoints
* Improve track persistence
* Improve temporal context
* Reduce unnecessary model inference
* Adaptive frame sampling

### Search

* Better semantic retrieval
* Temporal queries
* Event-aware ranking
* Combined object and event queries
* Search across multiple videos

### Understanding

* Better temporal event detection
* Audio and speech
* OCR
* More object and action types
* Cross-camera identity tracking

### Scale

* Distributed processing
* Queue-based workers
* GPU worker pools
* Large video collections
* Multi-camera deployments

---

## Why build ReCall?

There is an enormous amount of information trapped inside recorded video. The cameras already captured it.

The problem is finding it.

ReCall is an attempt to build a system where video can be treated less like a recording that you manually inspect and more like a **dataset that you can query**.

The project combines:

* Computer vision
* Object detection
* Object tracking
* Vision-language models
* Temporal reasoning
* Vector search
* Video processing
* Backend systems
* Distributed systems

The long-term goal is simple:

> **Ask what happened. Find the moment. See the evidence.**

---

## Contributing

ReCall is actively being developed.

Issues, discussions, experiments, and pull requests are welcome.
