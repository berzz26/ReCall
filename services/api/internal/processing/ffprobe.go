package processing

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/berzz26/recall/services/api/internal/segment_description"
	"github.com/berzz26/recall/services/api/internal/segment_embedding"
	"github.com/berzz26/recall/services/api/internal/storage"
	"github.com/berzz26/recall/services/api/internal/vision"
	"github.com/berzz26/recall/services/api/internal/video"
	"github.com/berzz26/recall/services/api/internal/video_event"
	"github.com/berzz26/recall/services/api/internal/video_frame"
	"github.com/berzz26/recall/services/api/internal/video_media"
	"github.com/berzz26/recall/services/api/internal/video_processing_checkpoint"
	"github.com/berzz26/recall/services/api/internal/video_segment"
	"github.com/berzz26/recall/services/api/internal/video_track"
	"github.com/berzz26/recall/services/api/internal/visual"
)

type FFprobeProcessor struct {
	ffprobePath    string
	timeout        time.Duration
	storage        storage.Storage
	mediaService   *video_media.Service
	segmentService *video_segment.Service
	frameService   *video_frame.Service
	visualService  *visual.Service
	trackService   *video_track.Service
	eventService   *video_event.Service
	descService    *segment_description.Service
	embedService   *segment_embedding.Service
	checkpointRepo *video_processing_checkpoint.Repository
	// adaptiveSampler, when set, owns the entire frame-planning stage
	// (probe → coarse/busy/refine → validated plan → extraction). Nil
	// preserves the legacy baseline frame path exactly.
	adaptiveSampler *AdaptiveSampler
}

// WithAdaptiveSampler installs the Phase 5 production adaptive boundary.
// Nil (default) keeps the pre-adaptive frame path untouched.
func (p *FFprobeProcessor) WithAdaptiveSampler(a *AdaptiveSampler) *FFprobeProcessor {
	p.adaptiveSampler = a
	return p
}

func NewFFprobeProcessor(ffprobePath string, timeout time.Duration, store storage.Storage, mediaService *video_media.Service) *FFprobeProcessor {
	if ffprobePath == "" {
		ffprobePath = "ffprobe"
	}
	if timeout <= 0 {
		timeout = 60 * time.Second
	}
	return &FFprobeProcessor{
		ffprobePath:  ffprobePath,
		timeout:      timeout,
		storage:      store,
		mediaService: mediaService,
	}
}

func NewFFprobeProcessorWithSegments(ffprobePath string, timeout time.Duration, store storage.Storage, mediaService *video_media.Service, segmentService *video_segment.Service) *FFprobeProcessor {
	p := NewFFprobeProcessor(ffprobePath, timeout, store, mediaService)
	p.segmentService = segmentService
	return p
}

func NewFFprobeProcessorWithFrames(ffprobePath string, timeout time.Duration, store storage.Storage, mediaService *video_media.Service, segmentService *video_segment.Service, frameService *video_frame.Service) *FFprobeProcessor {
	p := NewFFprobeProcessor(ffprobePath, timeout, store, mediaService)
	p.segmentService = segmentService
	p.frameService = frameService
	return p
}

func NewFFprobeProcessorWithVisual(ffprobePath string, timeout time.Duration, store storage.Storage, mediaService *video_media.Service, segmentService *video_segment.Service, frameService *video_frame.Service, visualService *visual.Service) *FFprobeProcessor {
	p := NewFFprobeProcessor(ffprobePath, timeout, store, mediaService)
	p.segmentService = segmentService
	p.frameService = frameService
	p.visualService = visualService
	return p
}

func NewFFprobeProcessorWithTracking(ffprobePath string, timeout time.Duration, store storage.Storage, mediaService *video_media.Service, segmentService *video_segment.Service, frameService *video_frame.Service, visualService *visual.Service, trackService *video_track.Service) *FFprobeProcessor {
	p := NewFFprobeProcessor(ffprobePath, timeout, store, mediaService)
	p.segmentService = segmentService
	p.frameService = frameService
	p.visualService = visualService
	p.trackService = trackService
	return p
}

func NewFFprobeProcessorWithEvents(ffprobePath string, timeout time.Duration, store storage.Storage, mediaService *video_media.Service, segmentService *video_segment.Service, frameService *video_frame.Service, visualService *visual.Service, trackService *video_track.Service, eventService *video_event.Service) *FFprobeProcessor {
	p := NewFFprobeProcessor(ffprobePath, timeout, store, mediaService)
	p.segmentService = segmentService
	p.frameService = frameService
	p.visualService = visualService
	p.trackService = trackService
	p.eventService = eventService
	return p
}

func NewFFprobeProcessorWithDescriptions(ffprobePath string, timeout time.Duration, store storage.Storage, mediaService *video_media.Service, segmentService *video_segment.Service, frameService *video_frame.Service, visualService *visual.Service, trackService *video_track.Service, eventService *video_event.Service, descService *segment_description.Service) *FFprobeProcessor {
	p := NewFFprobeProcessor(ffprobePath, timeout, store, mediaService)
	p.segmentService = segmentService
	p.frameService = frameService
	p.visualService = visualService
	p.trackService = trackService
	p.eventService = eventService
	p.descService = descService
	return p
}

func NewFFprobeProcessorWithEmbeddings(ffprobePath string, timeout time.Duration, store storage.Storage, mediaService *video_media.Service, segmentService *video_segment.Service, frameService *video_frame.Service, visualService *visual.Service, trackService *video_track.Service, eventService *video_event.Service, descService *segment_description.Service, embedService *segment_embedding.Service) *FFprobeProcessor {
	p := NewFFprobeProcessorWithDescriptions(ffprobePath, timeout, store, mediaService, segmentService, frameService, visualService, trackService, eventService, descService)
	p.embedService = embedService
	return p
}

func NewFFprobeProcessorWithCheckpoints(ffprobePath string, timeout time.Duration, store storage.Storage, mediaService *video_media.Service, segmentService *video_segment.Service, frameService *video_frame.Service, visualService *visual.Service, trackService *video_track.Service, eventService *video_event.Service, descService *segment_description.Service, embedService *segment_embedding.Service, checkpointRepo *video_processing_checkpoint.Repository) *FFprobeProcessor {
	p := NewFFprobeProcessorWithEmbeddings(ffprobePath, timeout, store, mediaService, segmentService, frameService, visualService, trackService, eventService, descService, embedService)
	p.checkpointRepo = checkpointRepo
	return p
}

type probeResult struct {
	Streams []probeStream `json:"streams"`
	Format  probeFormat   `json:"format"`
}

type probeStream struct {
	Index         int     `json:"index"`
	CodecType     string  `json:"codec_type"`
	CodecName     *string `json:"codec_name"`
	CodecLongName *string `json:"codec_long_name"`
	Width         *int    `json:"width"`
	Height        *int    `json:"height"`
	RFrameRate    *string `json:"r_frame_rate"`
	AvgFrameRate  *string `json:"avg_frame_rate"`
	BitRate       *string `json:"bit_rate"`
	SampleRate    *string `json:"sample_rate"`
	Channels      *int    `json:"channels"`
	ChannelLayout *string `json:"channel_layout"`
	Duration      *string `json:"duration"`
}

type probeFormat struct {
	FormatName     *string `json:"format_name"`
	FormatLongName *string `json:"format_long_name"`
	Duration       *string `json:"duration"`
	Size           *string `json:"size"`
	BitRate        *string `json:"bit_rate"`
	StartTime      *string `json:"start_time"`
}

func parseFps(s *string) (*float64, *string) {
	if s == nil || *s == "" || *s == "0/0" {
		return nil, s
	}
	raw := *s
	parts := strings.Split(raw, "/")
	if len(parts) == 2 {
		num, err1 := strconv.ParseFloat(parts[0], 64)
		den, err2 := strconv.ParseFloat(parts[1], 64)
		if err1 == nil && err2 == nil && den != 0 {
			fps := num / den
			return &fps, &raw
		}
	}
	if v, err := strconv.ParseFloat(raw, 64); err == nil {
		return &v, &raw
	}
	return nil, &raw
}

func parseFloatPtr(s *string) *float64 {
	if s == nil || *s == "" {
		return nil
	}
	v, err := strconv.ParseFloat(*s, 64)
	if err != nil {
		return nil
	}
	return &v
}

func parseIntPtr(s *string) *int64 {
	if s == nil || *s == "" {
		return nil
	}
	v, err := strconv.ParseInt(*s, 10, 64)
	if err != nil {
		return nil
	}
	return &v
}

func parseInt(s *string) *int {
	if s == nil || *s == "" {
		return nil
	}
	v, err := strconv.Atoi(*s)
	if err != nil {
		return nil
	}
	return &v
}

func getExt(key string) string {
	idx := strings.LastIndex(key, ".")
	if idx == -1 {
		return ".mp4"
	}
	return key[idx:]
}

func formatMs(ms int64) string {
	if ms <= 0 {
		return "0s"
	}
	return (time.Duration(ms) * time.Millisecond).String()
}

func (p *FFprobeProcessor) Process(ctx context.Context, v *video.Video) error {
	pipelineStart := time.Now()
	slog.Info("pipeline: start", "video_id", v.ID.String(), "source_type", v.SourceType)
	var ffprobeMs, parseMs, mediaMs, segMs, frameMs, visualMs, trackMs, eventMs, vlmMs, embedMs int64
	var videoPath string
	var tempFile string
	var cleanup func()

	if v.SourceType == video.SourceTypeLocal {
		if v.SourcePath == nil || *v.SourcePath == "" {
			return fmt.Errorf("missing source_path for LOCAL video")
		}
		videoPath = *v.SourcePath
		if _, err := os.Stat(videoPath); err != nil {
			return fmt.Errorf("source file not found: %w", err)
		}
	} else {
		if v.StorageKey == nil || *v.StorageKey == "" {
			return fmt.Errorf("missing storage_key for UPLOAD video")
		}
		rc, err := p.storage.Open(ctx, *v.StorageKey)
		if err != nil {
			return fmt.Errorf("failed to open storage: %w", err)
		}
		defer rc.Close()

		tmp, err := os.CreateTemp("", "ffprobe-*"+getExt(*v.StorageKey))
		if err != nil {
			return fmt.Errorf("failed to create temp file: %w", err)
		}
		tempFile = tmp.Name()
		cleanup = func() { os.Remove(tempFile) }
		if _, err := io.Copy(tmp, rc); err != nil {
			tmp.Close()
			cleanup()
			return fmt.Errorf("failed to copy to temp file: %w", err)
		}
		tmp.Close()
		videoPath = tempFile
	}
	if cleanup != nil {
		defer cleanup()
	}

	probeCtx, cancel := context.WithTimeout(ctx, p.timeout)
	defer cancel()

	cmd := exec.CommandContext(probeCtx, p.ffprobePath,
		"-v", "error",
		"-print_format", "json",
		"-show_format",
		"-show_streams",
		videoPath,
	)

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	ffprobeStart := time.Now()
	slog.Info("ffprobe: start", "video_id", v.ID.String(), "path", videoPath)
	err := cmd.Run()
	ffprobeMs = time.Since(ffprobeStart).Milliseconds()
	if err != nil {
		if probeCtx.Err() == context.DeadlineExceeded {
			slog.Error("ffprobe: timeout", "video_id", v.ID.String(), "duration_ms", ffprobeMs, "error", probeCtx.Err())
			return fmt.Errorf("ffprobe timeout")
		}
		msg := strings.TrimSpace(stderr.String())
		if len(msg) > 500 {
			msg = msg[:500]
		}
		if msg == "" {
			msg = err.Error()
		}
		slog.Error("ffprobe: failed", "video_id", v.ID.String(), "duration_ms", ffprobeMs, "error", msg)
		return fmt.Errorf("ffprobe failed: %s", msg)
	}
	slog.Info("ffprobe: complete", "video_id", v.ID.String(), "duration_ms", ffprobeMs, "output_bytes", stdout.Len())

	parseStart := time.Now()
	var result probeResult
	if err := json.Unmarshal(stdout.Bytes(), &result); err != nil {
		return fmt.Errorf("failed to parse ffprobe json: %w", err)
	}
	parseMs = time.Since(parseStart).Milliseconds()
	slog.Debug("ffprobe: parsed", "video_id", v.ID.String(), "duration_ms", parseMs)

	meta := &video_media.MediaMetadata{
		VideoID: v.ID,
	}

	if result.Format.FormatName != nil {
		meta.FormatName = result.Format.FormatName
	}
	if result.Format.FormatLongName != nil {
		meta.FormatLongName = result.Format.FormatLongName
	}
	meta.DurationSeconds = parseFloatPtr(result.Format.Duration)
	meta.SizeBytes = parseIntPtr(result.Format.Size)
	meta.BitRate = parseIntPtr(result.Format.BitRate)
	meta.StartTime = parseFloatPtr(result.Format.StartTime)

	var videoStream *probeStream
	var audioStream *probeStream
	for i := range result.Streams {
		s := &result.Streams[i]
		if s.CodecType == "video" && videoStream == nil {
			videoStream = s
		} else if s.CodecType == "audio" && audioStream == nil {
			audioStream = s
		}
	}

	if videoStream != nil {
		meta.VideoCodec = videoStream.CodecName
		meta.VideoWidth = videoStream.Width
		meta.VideoHeight = videoStream.Height
		if videoStream.RFrameRate != nil {
			fps, raw := parseFps(videoStream.RFrameRate)
			meta.VideoFps = fps
			meta.VideoFpsRaw = raw
		} else if videoStream.AvgFrameRate != nil {
			fps, raw := parseFps(videoStream.AvgFrameRate)
			meta.VideoFps = fps
			meta.VideoFpsRaw = raw
		}
	}

	if audioStream != nil {
		meta.AudioCodec = audioStream.CodecName
		meta.AudioSampleRate = parseInt(audioStream.SampleRate)
		meta.AudioChannels = audioStream.Channels
		meta.AudioChannelLayout = audioStream.ChannelLayout
	}

	mediaStart := time.Now()
	if _, err := p.mediaService.Upsert(ctx, meta); err != nil {
		return fmt.Errorf("failed to persist media metadata: %w", err)
	}
	mediaMs = time.Since(mediaStart).Milliseconds()
	slog.Info("pipeline: media persisted", "video_id", v.ID.String(), "duration_ms", mediaMs)

	var segments []video_segment.VideoSegment
	if p.segmentService != nil {
		if meta.DurationSeconds == nil || *meta.DurationSeconds <= 0 {
			return fmt.Errorf("video duration unavailable; cannot generate segments")
		}
		segStart := time.Now()
		var err error
		if p.checkpointRepo != nil {
			// Reuse existing segments if already present to preserve checkpoint state and avoid cascade deletion of frames/detections.
			existing, getErr := p.segmentService.GetByVideoID(ctx, v.ID)
			if getErr != nil {
				return fmt.Errorf("failed to get existing segments: %w", getErr)
			}
			if len(existing) > 0 {
				segments = existing
				segMs = time.Since(segStart).Milliseconds()
				slog.Info("pipeline: segments reused (checkpoint)", "video_id", v.ID.String(), "segments", len(segments), "duration_ms", segMs)
			} else {
				segments, err = p.segmentService.GenerateForVideo(ctx, v.ID, *meta.DurationSeconds)
				segMs = time.Since(segStart).Milliseconds()
				if err != nil {
					slog.Error("pipeline: segment generation failed", "video_id", v.ID.String(), "duration_ms", segMs, "error", err)
					return fmt.Errorf("failed to generate segments: %w", err)
				}
				slog.Info("pipeline: segments generated", "video_id", v.ID.String(), "segments", len(segments), "duration_ms", segMs, "duration_seconds", *meta.DurationSeconds)
			}
			segmentIDs := make([]uuid.UUID, 0, len(segments))
			for _, s := range segments {
				segmentIDs = append(segmentIDs, s.ID)
			}
			if len(segmentIDs) > 0 {
				if err := p.checkpointRepo.EnsureForSegments(ctx, v.ID, segmentIDs); err != nil {
					slog.Error("pipeline: checkpoint ensure failed", "video_id", v.ID.String(), "error", err)
					return fmt.Errorf("failed to ensure checkpoints: %w", err)
				}
				slog.Info("pipeline: checkpoints ensured", "video_id", v.ID.String(), "segments", len(segments))
			}
		} else {
			segments, err = p.segmentService.GenerateForVideo(ctx, v.ID, *meta.DurationSeconds)
			segMs = time.Since(segStart).Milliseconds()
			if err != nil {
				slog.Error("pipeline: segment generation failed", "video_id", v.ID.String(), "duration_ms", segMs, "error", err)
				return fmt.Errorf("failed to generate segments: %w", err)
			}
			slog.Info("pipeline: segments generated", "video_id", v.ID.String(), "segments", len(segments), "duration_ms", segMs, "duration_seconds", *meta.DurationSeconds)
		}
	} else if p.frameService != nil {
		return fmt.Errorf("frame extraction requires segments")
	}

	if p.frameService != nil {
		if meta.DurationSeconds == nil || *meta.DurationSeconds <= 0 {
			return fmt.Errorf("video duration unavailable; cannot extract frames")
		}
		w := 0
		h := 0
		if meta.VideoWidth != nil {
			w = *meta.VideoWidth
		}
		if meta.VideoHeight != nil {
			h = *meta.VideoHeight
		}
		if w <= 0 || h <= 0 {
			w = 1280
			h = 720
		}
		if p.adaptiveSampler != nil {
			// Production adaptive boundary: probe → coarse/busy/refine →
			// validated plan → extraction, with baseline fallbacks. The
			// orchestrator owns resume-skip semantics for this path.
			frameStart := time.Now()
			_, stats, err := p.adaptiveSampler.PlanAndExtract(ctx, v, segments, *meta.DurationSeconds, w, h, videoPath, p.checkpointRepo != nil)
			frameMs = time.Since(frameStart).Milliseconds()
			if err != nil {
				slog.Error("pipeline: adaptive sampling failed", "video_id", v.ID.String(), "duration_ms", frameMs, "error", err)
				return fmt.Errorf("failed adaptive sampling: %w", err)
			}
			slog.Info("pipeline: frame extraction complete (adaptive)",
				"video_id", v.ID.String(), "duration_ms", frameMs,
				"sampler_mode", stats.Mode, "sampler_version", stats.Version,
				"final_frames", stats.FinalFrames, "baseline_frames", stats.BaselineFrames,
				"estimated_savings", stats.EstimatedSavings,
				"fallback_reason", stats.FallbackReason)
			// Downstream stages consume frames/detections from storage as
			// usual; they never see the plan object or the sampler mode.
		} else if p.checkpointRepo != nil {
			// Checkpoint-aware frame extraction: skip if frames already exist for video (preserve completed segment frames)
			if existingFrames, err := p.frameService.GetByVideoID(ctx, v.ID); err == nil && len(existingFrames) > 0 {
				slog.Info("pipeline: frame extraction skipped (frames already exist, checkpoint)", "video_id", v.ID.String(), "existing_frames", len(existingFrames))
				frameMs = 0
			} else {
				frameStart := time.Now()
				if _, err := p.frameService.GenerateForVideo(ctx, v, segments, *meta.DurationSeconds, w, h); err != nil {
					frameMs = time.Since(frameStart).Milliseconds()
					slog.Error("pipeline: frame extraction failed", "video_id", v.ID.String(), "duration_ms", frameMs, "error", err)
					return fmt.Errorf("failed to extract frames: %w", err)
				}
				frameMs = time.Since(frameStart).Milliseconds()
				slog.Info("pipeline: frame extraction complete", "video_id", v.ID.String(), "duration_ms", frameMs)
			}
		} else {
			frameStart := time.Now()
			if _, err := p.frameService.GenerateForVideo(ctx, v, segments, *meta.DurationSeconds, w, h); err != nil {
				frameMs = time.Since(frameStart).Milliseconds()
				slog.Error("pipeline: frame extraction failed", "video_id", v.ID.String(), "duration_ms", frameMs, "error", err)
				return fmt.Errorf("failed to extract frames: %w", err)
			}
			frameMs = time.Since(frameStart).Milliseconds()
			slog.Info("pipeline: frame extraction complete", "video_id", v.ID.String(), "duration_ms", frameMs)
		}
	}

	if p.visualService != nil {
		if p.checkpointRepo != nil {
			// Segment-level checkpointed detection: one YOLO session per video, bounded batches per segment
			visualStart := time.Now()
			slog.Info("pipeline: detection start (checkpointed)", "video_id", v.ID.String(), "segments", len(segments))
			sessObj, sessErr := p.visualService.NewDetectionSession(ctx)
			var useSess bool
			if sessErr == nil && sessObj != nil {
				useSess = true
				defer func() {
					if cerr := sessObj.Close(); cerr != nil {
						slog.Warn("pipeline: failed to close YOLO session", "video_id", v.ID.String(), "error", cerr)
					}
				}()
				slog.Info("pipeline: YOLO session created for checkpointed detection", "video_id", v.ID.String())
			} else {
				slog.Info("pipeline: YOLO session not available, using per-batch detection", "video_id", v.ID.String(), "reason", sessErr)
			}
			// Iterate segments in deterministic segment_index order
			for _, seg := range segments {
				if ctx.Err() != nil {
					return ctx.Err()
				}
				cp, err := p.checkpointRepo.GetByVideoAndSegment(ctx, v.ID, seg.ID)
				if err != nil {
					// If checkpoint not found, ensure it
					if err := p.checkpointRepo.EnsureForSegments(ctx, v.ID, []uuid.UUID{seg.ID}); err != nil {
						return fmt.Errorf("failed to ensure checkpoint for segment %d: %w", seg.SegmentIndex, err)
					}
					cp, err = p.checkpointRepo.GetByVideoAndSegment(ctx, v.ID, seg.ID)
					if err != nil {
						return fmt.Errorf("failed to get checkpoint for segment %d: %w", seg.SegmentIndex, err)
					}
				}
				if cp.Status == video_processing_checkpoint.StatusComplete {
					slog.Info("pipeline: segment skip COMPLETE", "video_id", v.ID.String(), "segment_index", seg.SegmentIndex, "segment_id", seg.ID.String())
					continue
				}
				if cp.Status == video_processing_checkpoint.StatusFailed {
					slog.Info("pipeline: segment skip FAILED (no auto retry)", "video_id", v.ID.String(), "segment_index", seg.SegmentIndex, "segment_id", seg.ID.String())
					continue
				}
				if cp.Status == video_processing_checkpoint.StatusProcessing {
					slog.Info("pipeline: segment was PROCESSING, reprocessing", "video_id", v.ID.String(), "segment_index", seg.SegmentIndex, "segment_id", seg.ID.String())
				}
				if _, err := p.checkpointRepo.MarkProcessing(ctx, v.ID, seg.ID); err != nil {
					return fmt.Errorf("failed to mark segment %d PROCESSING: %w", seg.SegmentIndex, err)
				}
				slog.Info("pipeline: segment PROCESSING", "video_id", v.ID.String(), "segment_index", seg.SegmentIndex, "segment_id", seg.ID.String(), "start_time", seg.StartTime, "end_time", seg.EndTime)
				// Process segment: detection (and future per-segment stages)
				// Use checkpoint-aware per segment detection; whole-video tracking etc. remains after loop.
				var segErr error
				if useSess && sessObj != nil {
					_, segErr = p.visualService.AnalyzeSegmentWithSession(ctx, v.ID, seg.ID, sessObj)
				} else {
					_, segErr = p.visualService.AnalyzeSegment(ctx, v.ID, seg.ID)
				}
				if segErr != nil {
					errStr := segErr.Error()
					if len(errStr) > 1000 {
						errStr = errStr[:1000]
					}
					if _, markErr := p.checkpointRepo.MarkFailed(ctx, v.ID, seg.ID, errStr); markErr != nil {
						slog.Error("pipeline: failed to mark segment FAILED", "video_id", v.ID.String(), "segment_index", seg.SegmentIndex, "error", markErr)
					}
					slog.Error("pipeline: segment processing failed", "video_id", v.ID.String(), "segment_index", seg.SegmentIndex, "segment_id", seg.ID.String(), "error", segErr)
					visualMs = time.Since(visualStart).Milliseconds()
					return fmt.Errorf("segment %d failed: %w", seg.SegmentIndex, segErr)
				}
				if _, err := p.checkpointRepo.MarkComplete(ctx, v.ID, seg.ID); err != nil {
					return fmt.Errorf("failed to mark segment %d COMPLETE: %w", seg.SegmentIndex, err)
				}
				slog.Info("pipeline: segment COMPLETE", "video_id", v.ID.String(), "segment_index", seg.SegmentIndex, "segment_id", seg.ID.String())
			}
			visualMs = time.Since(visualStart).Milliseconds()
			slog.Info("pipeline: detection complete (checkpointed)", "video_id", v.ID.String(), "segments", len(segments), "duration_ms", visualMs)
			// After per-segment loop, check for any FAILED checkpoints - must not proceed to final stages, video should be FAILED
			allCps, err := p.checkpointRepo.GetByVideoID(ctx, v.ID)
			if err == nil {
				for _, c := range allCps {
					if c.Status == video_processing_checkpoint.StatusFailed {
						slog.Error("pipeline: segment FAILED, aborting pipeline", "video_id", v.ID.String(), "segment_id", c.SegmentID.String(), "error", c.Error)
						return fmt.Errorf("segment %s failed: %s", c.SegmentID.String(), func() string {
							if c.Error != nil {
								return *c.Error
							}
							return "unknown"
						}())
					}
				}
				// Also check for any PENDING/PROCESSING that were not completed (should not happen after loop)
				for _, c := range allCps {
					if c.Status != video_processing_checkpoint.StatusComplete {
						slog.Warn("pipeline: segment not COMPLETE after loop", "video_id", v.ID.String(), "segment_id", c.SegmentID.String(), "status", c.Status)
					}
				}
			}
		} else {
			visualStart := time.Now()
			slog.Info("pipeline: detection start", "video_id", v.ID.String())
			if _, err := p.visualService.AnalyzeVideo(ctx, v.ID); err != nil {
				visualMs = time.Since(visualStart).Milliseconds()
				slog.Error("pipeline: detection failed", "video_id", v.ID.String(), "duration_ms", visualMs, "error", err)
				return fmt.Errorf("failed to analyze visuals: %w", err)
			}
			visualMs = time.Since(visualStart).Milliseconds()
			slog.Info("pipeline: detection complete", "video_id", v.ID.String(), "duration_ms", visualMs)
		}
	}

	if p.trackService != nil {
		trackStart := time.Now()
		slog.Info("pipeline: tracking start", "video_id", v.ID.String())
		if _, err := p.trackService.GenerateForVideoID(ctx, v.ID); err != nil {
			trackMs = time.Since(trackStart).Milliseconds()
			slog.Error("pipeline: tracking failed", "video_id", v.ID.String(), "duration_ms", trackMs, "error", err)
			return fmt.Errorf("failed to generate tracks: %w", err)
		}
		trackMs = time.Since(trackStart).Milliseconds()
		slog.Info("pipeline: tracking complete", "video_id", v.ID.String(), "duration_ms", trackMs)
	}

	if p.eventService != nil {
		eventStart := time.Now()
		slog.Info("pipeline: event generation start", "video_id", v.ID.String())
		if _, err := p.eventService.GenerateForVideo(ctx, v.ID); err != nil {
			eventMs = time.Since(eventStart).Milliseconds()
			slog.Error("pipeline: event generation failed", "video_id", v.ID.String(), "duration_ms", eventMs, "error", err)
			return fmt.Errorf("failed to generate events: %w", err)
		}
		eventMs = time.Since(eventStart).Milliseconds()
		slog.Info("pipeline: events complete", "video_id", v.ID.String(), "duration_ms", eventMs)
	}

	if p.descService != nil {
		vlmStart := time.Now()
		slog.Info("pipeline: VLM generation start", "video_id", v.ID.String())
		if _, err := p.descService.GenerateForVideo(ctx, v.ID); err != nil {
			vlmMs = time.Since(vlmStart).Milliseconds()
			// Description is optional: rate-limited errors must not fail the pipeline.
			// Preserve partial descriptions (if any) and continue to embedding.
			if vision.IsRateLimited(err) {
				slog.Warn("pipeline: VLM rate limited, continuing pipeline (description optional)", "video_id", v.ID.String(), "duration_ms", vlmMs, "error", err)
			} else {
				slog.Error("pipeline: VLM generation failed", "video_id", v.ID.String(), "duration_ms", vlmMs, "error", err)
				return fmt.Errorf("failed to generate descriptions: %w", err)
			}
		} else {
			vlmMs = time.Since(vlmStart).Milliseconds()
			slog.Info("pipeline: VLM generation complete", "video_id", v.ID.String(), "duration_ms", vlmMs)
		}
	} else {
		slog.Info("pipeline: VLM generation skipped (disabled)", "video_id", v.ID.String())
	}

	if p.embedService != nil {
		embedStart := time.Now()
		slog.Info("pipeline: embedding generation start", "video_id", v.ID.String())
		if _, err := p.embedService.GenerateForVideo(ctx, v.ID); err != nil {
			embedMs = time.Since(embedStart).Milliseconds()
			slog.Error("pipeline: embedding generation failed", "video_id", v.ID.String(), "duration_ms", embedMs, "error", err)
			return fmt.Errorf("failed to generate embeddings: %w", err)
		}
		embedMs = time.Since(embedStart).Milliseconds()
		slog.Info("pipeline: embedding generation complete", "video_id", v.ID.String(), "duration_ms", embedMs)
	} else {
		slog.Info("pipeline: embedding generation skipped", "video_id", v.ID.String())
	}

	totalMs := time.Since(pipelineStart).Milliseconds()
	// Detailed breakdown for READY: mirrors requested format ffprobe/frame extraction/YOLO/tracking/events/Gemini/embeddings/DB writes
	dbWritesMs := mediaMs // primary DB write (media metadata); other DB writes are included in stage timings (segments/frames/detections/tracks/events)
	overheadMs := totalMs - (ffprobeMs + parseMs + mediaMs + segMs + frameMs + visualMs + trackMs + eventMs + vlmMs + embedMs)
	if overheadMs < 0 {
		overheadMs = 0
	}
	slog.Info("pipeline: breakdown",
		"video_id", v.ID.String(),
		"ffprobe_ms", ffprobeMs, "ffprobe", formatMs(ffprobeMs),
		"ffprobe_parse_ms", parseMs, "ffprobe_parse", formatMs(parseMs),
		"db_writes_ms", dbWritesMs, "db_writes", formatMs(dbWritesMs),
		"segments_ms", segMs, "segments", formatMs(segMs),
		"frame_extraction_ms", frameMs, "frame_extraction", formatMs(frameMs),
		"yolo_detection_ms", visualMs, "yolo", formatMs(visualMs),
		"tracking_ms", trackMs, "tracking", formatMs(trackMs),
		"events_ms", eventMs, "events", formatMs(eventMs),
		"vlm_gemini_ms", vlmMs, "vlm_gemini", formatMs(vlmMs),
		"embeddings_ms", embedMs, "embeddings", formatMs(embedMs),
		"overhead_ms", overheadMs, "overhead", formatMs(overheadMs),
		"total_ms", totalMs, "total", formatMs(totalMs),
	)
	// Human-readable table (single log line) for `kubectl logs` / console readability
	slog.Info(fmt.Sprintf(
		"pipeline: timing breakdown video=%s\n  %-18s %8s (%6d ms)\n  %-18s %8s (%6d ms)\n  %-18s %8s (%6d ms)\n  %-18s %8s (%6d ms)\n  %-18s %8s (%6d ms)\n  %-18s %8s (%6d ms)\n  %-18s %8s (%6d ms)\n  %-18s %8s (%6d ms)\n  %-18s %8s (%6d ms)\n  %-18s %8s (%6d ms)\n  %-18s %8s (%6d ms)\n  %-18s %8s (%6d ms)",
		v.ID.String(),
		"ffprobe", formatMs(ffprobeMs), ffprobeMs,
		"db_writes", formatMs(dbWritesMs), dbWritesMs,
		"segments", formatMs(segMs), segMs,
		"frame extraction", formatMs(frameMs), frameMs,
		"YOLO", formatMs(visualMs), visualMs,
		"tracking", formatMs(trackMs), trackMs,
		"events", formatMs(eventMs), eventMs,
		"Gemini/VLM", formatMs(vlmMs), vlmMs,
		"embeddings", formatMs(embedMs), embedMs,
		"ffprobe_parse", formatMs(parseMs), parseMs,
		"overhead", formatMs(overheadMs), overheadMs,
		"TOTAL", formatMs(totalMs), totalMs,
	))
	slog.Info("pipeline: success", "video_id", v.ID.String(), "total_duration_ms", totalMs)
	return nil
}
