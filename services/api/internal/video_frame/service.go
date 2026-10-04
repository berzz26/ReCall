package video_frame

import (
	"bytes"
	"context"
	"fmt"
	"image"
	_ "image/jpeg"
	"io"
	"log/slog"
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/berzz26/recall/services/api/internal/codec_experiment"
	"github.com/berzz26/recall/services/api/internal/storage"
	"github.com/berzz26/recall/services/api/internal/video"
	"github.com/berzz26/recall/services/api/internal/video_segment"
)

type Service struct {
	repo           *Repository
	storage        storage.Storage
	sampleInterval time.Duration
	ffmpegPath     string
	// ffmpegTimeout is retained for compatibility (FFMPEG_TIMEOUT env).
	// A.2 semantics: FFMPEG_TIMEOUT is the maximum allowed duration of an
	// individual FFmpeg subprocess before cancellation, NOT a whole-video
	// wall-clock limit. The processing job lifetime is controlled by the
	// parent processing context (worker/application) and may run arbitrarily
	// long for long videos. To avoid killing legitimate long extractions,
	// GenerateForVideo does not impose a fixed 60s wall-clock timeout over
	// the entire extraction.
	ffmpegTimeout time.Duration
	jpegQuality   int
	// TEMPORARY EXPERIMENT: nil unless CODECSIGHT_DYNAMIC_SAMPLING is set.
	// When nil, GenerateForVideo follows the existing 2-second path exactly.
	dynamicSampler *codec_experiment.DynamicSampler
}

func NewService(repo *Repository, store storage.Storage, sampleInterval time.Duration, ffmpegPath string, ffmpegTimeout time.Duration, jpegQuality int) *Service {
	if sampleInterval <= 0 {
		sampleInterval = 2 * time.Second
	}
	if ffmpegPath == "" {
		ffmpegPath = "ffmpeg"
	}
	if ffmpegTimeout <= 0 {
		ffmpegTimeout = 60 * time.Second
	}
	if jpegQuality < 1 || jpegQuality > 100 {
		jpegQuality = 85
	}
	return &Service{repo: repo, storage: store, sampleInterval: sampleInterval, ffmpegPath: ffmpegPath, ffmpegTimeout: ffmpegTimeout, jpegQuality: jpegQuality}
}

// NewServiceWithDynamicSampler is TEMPORARY experiment wiring. A nil sampler
// preserves the existing constructor behavior exactly.
func NewServiceWithDynamicSampler(repo *Repository, store storage.Storage, sampleInterval time.Duration, ffmpegPath string, ffmpegTimeout time.Duration, jpegQuality int, dynamicSampler *codec_experiment.DynamicSampler) *Service {
	s := NewService(repo, store, sampleInterval, ffmpegPath, ffmpegTimeout, jpegQuality)
	s.dynamicSampler = dynamicSampler
	return s
}

// DynamicEnabled reports whether the experimental dynamic sampling path may run.
func (s *Service) DynamicEnabled() bool {
	return s != nil && s.dynamicSampler != nil && s.dynamicSampler.Enabled()
}

func SampleTimestamps(durationSeconds float64, interval time.Duration) ([]float64, error) {
	if interval <= 0 {
		return nil, fmt.Errorf("frame sample interval must be > 0")
	}
	if math.IsNaN(durationSeconds) || math.IsInf(durationSeconds, 0) || durationSeconds <= 0 {
		return nil, fmt.Errorf("video duration unavailable; cannot sample frames")
	}
	iv := interval.Seconds()
	if iv <= 0 {
		return nil, fmt.Errorf("frame sample interval must be > 0")
	}
	var ts []float64
	for t := 0.0; t < durationSeconds-1e-9; t += iv {
		ts = append(ts, t)
	}
	if len(ts) == 0 {
		return nil, fmt.Errorf("video duration unavailable; cannot sample frames")
	}
	return ts, nil
}

func FindSegment(segments []video_segment.VideoSegment, timestamp float64) *video_segment.VideoSegment {
	for i := range segments {
		s := &segments[i]
		if timestamp >= s.StartTime && timestamp < s.EndTime {
			return s
		}
	}
	if len(segments) > 0 {
		last := &segments[len(segments)-1]
		if timestamp >= last.StartTime && timestamp <= last.EndTime {
			return last
		}
	}
	return nil
}

func frameStorageKey(videoID uuid.UUID, frameIndex int) string {
	return fmt.Sprintf("videos/%s/frames/%06d.jpg", videoID.String(), frameIndex)
}

func roundMicroTimestamp(ts float64) int64 {
	return int64(math.Round(ts * 1e6))
}

// limitedWriter caps stderr capture to avoid unbounded memory.
type limitedWriter struct {
	buf   *bytes.Buffer
	limit int
}

func (w *limitedWriter) Write(p []byte) (int, error) {
	if w.buf.Len() >= w.limit {
		return len(p), nil
	}
	remaining := w.limit - w.buf.Len()
	if len(p) > remaining {
		p = p[:remaining]
	}
	return w.buf.Write(p)
}

func (s *Service) GenerateForVideo(ctx context.Context, v *video.Video, segments []video_segment.VideoSegment, durationSeconds float64, width, height int) ([]VideoFrame, error) {
	frameStart := time.Now()
	if s.storage == nil {
		return nil, fmt.Errorf("storage not configured")
	}
	timestamps, err := SampleTimestamps(durationSeconds, s.sampleInterval)
	if err != nil {
		return nil, err
	}
	if len(segments) == 0 {
		return nil, fmt.Errorf("no segments available for frame association")
	}
	if width <= 0 || height <= 0 {
		return nil, fmt.Errorf("invalid dimensions")
	}
	slog.Info("frame: start",
		"video_id", v.ID.String(),
		"duration_seconds", durationSeconds,
		"frame_count", len(timestamps),
		"interval", s.sampleInterval.String(),
		"width", width, "height", height,
		"ffmpeg", s.ffmpegPath, "jpeg_quality", s.jpegQuality,
	)

	var videoPath string
	var tempVideo string
	var cleanupVideo func()

	if v.SourceType == video.SourceTypeLocal {
		if v.SourcePath == nil || *v.SourcePath == "" {
			return nil, fmt.Errorf("missing source_path for LOCAL video")
		}
		videoPath = *v.SourcePath
		if _, err := os.Stat(videoPath); err != nil {
			return nil, fmt.Errorf("source file not found: %w", err)
		}
	} else {
		if v.StorageKey == nil || *v.StorageKey == "" {
			return nil, fmt.Errorf("missing storage_key for UPLOAD video")
		}
		// Prefer direct filesystem path if storage is LocalStorage to avoid extra copy.
		// Fallback to single copy via temp file for other storage implementations.
		useDirect := false
		if ls, ok := s.storage.(*storage.LocalStorage); ok {
			candidate := filepath.Join(ls.Root(), *v.StorageKey)
			if _, err := os.Stat(candidate); err == nil {
				videoPath = candidate
				useDirect = true
			}
		}
		if !useDirect {
			rc, err := s.storage.Open(ctx, *v.StorageKey)
			if err != nil {
				return nil, fmt.Errorf("failed to open storage: %w", err)
			}
			ext := filepath.Ext(*v.StorageKey)
			if ext == "" {
				ext = ".mp4"
			}
			tmp, err := os.CreateTemp("", "frame-src-*"+ext)
			if err != nil {
				rc.Close()
				return nil, fmt.Errorf("failed to create temp video: %w", err)
			}
			tempVideo = tmp.Name()
			cleanupVideo = func() { os.Remove(tempVideo) }
			if _, err := io.Copy(tmp, rc); err != nil {
				tmp.Close()
				rc.Close()
				cleanupVideo()
				return nil, fmt.Errorf("failed to copy to temp video: %w", err)
			}
			tmp.Close()
			rc.Close()
			videoPath = tempVideo
		}
	}
	if cleanupVideo != nil {
		defer cleanupVideo()
	}

	existing, err := s.repo.GetByVideoID(ctx, v.ID)
	if err != nil {
		return nil, err
	}
	var oldKeys []string
	for _, f := range existing {
		oldKeys = append(oldKeys, f.StorageKey)
	}
	if len(existing) > 0 {
		if err := s.repo.DeleteByVideoID(ctx, v.ID); err != nil {
			return nil, fmt.Errorf("failed to delete old frames: %w", err)
		}
		for _, k := range oldKeys {
			_ = s.storage.Delete(ctx, k)
		}
	}

	// Single-process streaming FFmpeg extraction: continuous decode via fps filter.
	// The dynamic experiment keeps this mechanism and only changes which dense
	// outputs are retained. Baseline mode always retains every output.
	intervalSec := s.sampleInterval.Seconds()
	expectedTimestamps := timestamps
	keepSelected := map[int64]bool(nil)
	dynamicActive := false
	if s.DynamicEnabled() {
		dynRes, dynErr := s.dynamicSampler.RunDynamic(ctx, videoPath, v.ID.String(), durationSeconds, s.sampleInterval)
		if dynErr != nil {
			slog.Warn("frame: dynamic sampling unavailable, using baseline sampling",
				"video_id", v.ID.String(), "error", dynErr)
		} else {
			fmt.Fprintln(os.Stderr, dynRes.Report)
			if dynRes.Plan.Summary.Mode == "dynamic" && len(dynRes.Plan.SelectedTimestamps) > 0 {
				intervalSec = dynRes.Plan.Config.DenseInterval.Seconds()
				expectedTimestamps = dynRes.Plan.SelectedTimestamps
				keepSelected = dynRes.Plan.KeepKeys()
				dynamicActive = true
				slog.Info("frame: dynamic sampling plan selected",
					"video_id", v.ID.String(),
					"baseline_frames", dynRes.Plan.Summary.BaselineCount,
					"dynamic_frames", dynRes.Plan.Summary.DynamicCount,
					"regions", dynRes.Plan.Summary.RegionCount)
			} else {
				slog.Info("frame: dynamic sampler active but no additional regions selected",
					"video_id", v.ID.String(),
					"mode", dynRes.Plan.Summary.Mode,
					"reason", dynRes.Plan.Summary.Reason)
			}
		}
	}
	fpsVal := 1.0 / intervalSec
	fpsFilter := fmt.Sprintf("fps=%.6f:round=up", fpsVal)

	q := fmt.Sprintf("%d", s.jpegQuality)

	// Prepare ffmpeg args: continuous decode, fps sampling, JPEG via pipe
	// Do NOT use -ss per-frame seeks. Decode sequentially once.
	args := []string{
		"-loglevel", "error",
		"-i", videoPath,
		"-vf", fpsFilter,
		"-q:v", q,
		"-f", "image2pipe",
		"-vcodec", "mjpeg",
		"pipe:1",
	}

	// FFmpeg subprocess is supervised by the parent processing context.
	// FFMPEG_TIMEOUT is no longer used as a whole-video wall-clock timeout.
	// The extraction may run arbitrarily long for long videos; cancellation
	// propagates from ctx (worker shutdown / application cancellation).
	cmd := exec.CommandContext(ctx, s.ffmpegPath, args...)
	// Separate stderr, stdout is JPEG stream
	var stderrBuf bytes.Buffer
	cmd.Stderr = &limitedWriter{buf: &stderrBuf, limit: 8192}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("failed to create stdout pipe: %w", err)
	}

	ffmpegStart := time.Now()
	if err := cmd.Start(); err != nil {
		msg := strings.TrimSpace(stderrBuf.String())
		if msg == "" {
			msg = err.Error()
		}
		return nil, fmt.Errorf("ffmpeg failed to start: %s", msg)
	}

	// Ensure process cleanup on early return
	defer func() {
		if cmd.Process != nil {
			_ = cmd.Process.Kill()
		}
	}()

	var createdKeys []string
	var batch []VideoFrame
	var allSaved []VideoFrame
	var totalSaveMs int64
	var persistMs int64

	cleanupOnFail := func() {
		for _, k := range createdKeys {
			_ = s.storage.Delete(ctx, k)
		}
		// Remove any DB rows already persisted in batches
		_ = s.repo.DeleteByVideoID(ctx, v.ID)
	}

	// Streaming JPEG boundary parsing
	// JPEG SOI = FF D8, EOI = FF D9
	buffer := make([]byte, 0, 128*1024)
	tmpRead := make([]byte, 32*1024)
	jpegIndex := 0
	lastStoredTs := 0.0
	ffmpegProcessCount := 1

	// Helper to persist batch in bounded manner
	persistBatch := func() error {
		if len(batch) == 0 {
			return nil
		}
		start := time.Now()
		saved, err := s.repo.CreateBatch(ctx, batch)
		persistMs += time.Since(start).Milliseconds()
		if err != nil {
			return err
		}
		allSaved = append(allSaved, saved...)
		batch = batch[:0]
		return nil
	}

	readErrOuter := error(nil)
loopRead:
	for {
		n, readErr := stdout.Read(tmpRead)
		if n > 0 {
			buffer = append(buffer, tmpRead[:n]...)
			// Extract as many JPEGs as possible from buffer
			for {
				soi := bytes.Index(buffer, []byte{0xFF, 0xD8})
				if soi == -1 {
					if len(buffer) > 0 && buffer[len(buffer)-1] == 0xFF {
						// Keep trailing FF for split marker
						buffer = buffer[len(buffer)-1:]
					} else {
						buffer = buffer[:0]
					}
					break
				}
				if soi > 0 {
					buffer = buffer[soi:]
				}
				if len(buffer) < 2 {
					break
				}
				// Search EOI after SOI
				eoi := bytes.Index(buffer[2:], []byte{0xFF, 0xD9})
				if eoi == -1 {
					if len(buffer) > 10*1024*1024 {
						readErrOuter = fmt.Errorf("jpeg frame too large or missing EOI")
						break loopRead
					}
					break
				}
				eoiPos := 2 + eoi + 2
				jpegBytes := make([]byte, eoiPos)
				copy(jpegBytes, buffer[:eoiPos])
				buffer = buffer[eoiPos:]

				ts := float64(jpegIndex) * intervalSec
				if ts >= durationSeconds-1e-9 {
					// Beyond duration, skip but still count index to preserve alignment?
					// Spec says do not create frames beyond duration, so skip persisting.
					jpegIndex++
					// If we skip, we should still continue to next JPEG but not store.
					// However fps should not produce beyond; if it does, ignore.
					continue
				}
				if keepSelected != nil && !keepSelected[roundMicroTimestamp(ts)] {
					// Dynamic mode decodes the dense grid once, then retains only the
					// planned baseline and high-activity timestamps.
					jpegIndex++
					continue
				}
				seg := FindSegment(segments, ts)
				if seg == nil {
					readErrOuter = fmt.Errorf("no segment for timestamp %f", ts)
					break loopRead
				}
				// Determine dimensions without full decode
				cfg, _, cfgErr := image.DecodeConfig(bytes.NewReader(jpegBytes))
				fw, fh := width, height
				if cfgErr == nil && cfg.Width > 0 && cfg.Height > 0 {
					fw = cfg.Width
					fh = cfg.Height
				}
				storedIndex := len(allSaved) + len(batch)
				key := frameStorageKey(v.ID, storedIndex)
				saveStart := time.Now()
				if err := s.storage.Save(ctx, key, bytes.NewReader(jpegBytes)); err != nil {
					readErrOuter = fmt.Errorf("failed to store frame %d: %w", storedIndex, err)
					break loopRead
				}
				totalSaveMs += time.Since(saveStart).Milliseconds()
				createdKeys = append(createdKeys, key)
				batch = append(batch, VideoFrame{
					VideoID:          v.ID,
					SegmentID:        seg.ID,
					FrameIndex:       storedIndex,
					TimestampSeconds: ts,
					StorageKey:       key,
					Width:            fw,
					Height:           fh,
				})
				lastStoredTs = ts
				if len(batch) >= 500 {
					if err := persistBatch(); err != nil {
						readErrOuter = fmt.Errorf("failed to persist frames: %w", err)
						break loopRead
					}
				}
				if (jpegIndex+1)%1000 == 0 {
					slog.Info("frame: progress", "video_id", v.ID.String(), "frames", jpegIndex+1, "timestamp", ts, "elapsed_ms", time.Since(frameStart).Milliseconds())
				}
				jpegIndex++
			}
		}
		if readErr != nil {
			if readErr == io.EOF {
				break
			}
			// Check parent context cancellation (A.2: processing lifetime controls FFmpeg)
			if ctx.Err() != nil {
				readErrOuter = ctx.Err()
				break
			}
			readErrOuter = readErr
			break
		}
	}

	// Wait for ffmpeg to exit
	waitErr := cmd.Wait()
	ffmpegMs := time.Since(ffmpegStart).Milliseconds()

	// Handle read outer error before checking ffmpeg exit
	if readErrOuter != nil {
		cleanupOnFail()
		if ctx.Err() == context.DeadlineExceeded {
			slog.Error("frame: ffmpeg timeout", "video_id", v.ID.String(), "duration_ms", ffmpegMs, "error", ctx.Err())
			return nil, fmt.Errorf("ffmpeg timeout: %w", ctx.Err())
		}
		if ctx.Err() == context.Canceled {
			slog.Error("frame: context canceled", "video_id", v.ID.String(), "duration_ms", ffmpegMs, "error", ctx.Err())
			return nil, fmt.Errorf("context canceled: %w", ctx.Err())
		}
		// If ffmpeg also failed, include stderr
		if waitErr != nil {
			msg := strings.TrimSpace(stderrBuf.String())
			if len(msg) > 500 {
				msg = msg[:500]
			}
			if msg == "" {
				msg = readErrOuter.Error()
			}
			slog.Error("frame: ffmpeg failed", "video_id", v.ID.String(), "duration_ms", ffmpegMs, "error", msg)
			return nil, fmt.Errorf("ffmpeg failed: %s", msg)
		}
		slog.Error("frame: failed", "video_id", v.ID.String(), "error", readErrOuter)
		return nil, readErrOuter
	}

	if waitErr != nil {
		cleanupOnFail()
		msg := strings.TrimSpace(stderrBuf.String())
		if len(msg) > 500 {
			msg = msg[:500]
		}
		if msg == "" {
			msg = waitErr.Error()
		}
		if ctx.Err() == context.DeadlineExceeded {
			slog.Error("frame: ffmpeg timeout", "video_id", v.ID.String(), "duration_ms", ffmpegMs, "error", ctx.Err())
			return nil, fmt.Errorf("ffmpeg timeout: %w", ctx.Err())
		}
		if ctx.Err() == context.Canceled {
			slog.Error("frame: context canceled", "video_id", v.ID.String(), "duration_ms", ffmpegMs, "error", ctx.Err())
			return nil, fmt.Errorf("context canceled: %w", ctx.Err())
		}
		slog.Error("frame: ffmpeg failed", "video_id", v.ID.String(), "duration_ms", ffmpegMs, "error", msg)
		return nil, fmt.Errorf("ffmpeg failed: %s", msg)
	}

	// Flush remaining batch
	if err := persistBatch(); err != nil {
		cleanupOnFail()
		slog.Error("frame: persist failed", "video_id", v.ID.String(), "duration_ms", persistMs, "error", err)
		return nil, fmt.Errorf("failed to persist frames: %w", err)
	}

	// If ffmpeg produced zero frames but expected some, treat as failure
	if jpegIndex == 0 && len(expectedTimestamps) > 0 {
		cleanupOnFail()
		msg := strings.TrimSpace(stderrBuf.String())
		if len(msg) > 500 {
			msg = msg[:500]
		}
		if msg == "" {
			msg = "no frames extracted"
		}
		slog.Error("frame: no frames extracted", "video_id", v.ID.String(), "expected", len(expectedTimestamps), "got", jpegIndex, "duration_ms", ffmpegMs, "error", msg)
		return nil, fmt.Errorf("ffmpeg failed: %s", msg)
	}

	// Note: jpegIndex may be slightly less/more than len(timestamps) due to fps rounding.
	// We use deterministic timestamps, so we accept jpegIndex count. If mismatch, log warning.
	if !dynamicActive && jpegIndex != len(timestamps) {
		slog.Warn("frame: frame count mismatch", "video_id", v.ID.String(), "expected", len(timestamps), "got", jpegIndex, "duration_seconds", durationSeconds, "interval", s.sampleInterval.String())
	}

	totalMs := time.Since(frameStart).Milliseconds()
	// allSaved already contains persisted frames; if we used batch persisting, allSaved length == jpegIndex (skipping beyond-duration)
	// Ensure we return in order
	if len(allSaved) != jpegIndex {
		// In case batch persist already done, allSaved holds all; verify
	}
	if dynamicActive {
		storedCount := len(allSaved)
		if storedCount == 0 && len(expectedTimestamps) > 0 {
			cleanupOnFail()
			slog.Error("frame: dynamic extraction stored no frames", "video_id", v.ID.String(), "expected", len(expectedTimestamps), "duration_ms", ffmpegMs)
			return nil, fmt.Errorf("ffmpeg failed: dynamic extraction stored no frames")
		}
		if storedCount != len(expectedTimestamps) {
			slog.Warn("frame: dynamic frame count mismatch", "video_id", v.ID.String(), "expected", len(expectedTimestamps), "got", storedCount, "duration_seconds", durationSeconds)
		}
	}

	var finalTimestamp float64
	if dynamicActive {
		finalTimestamp = lastStoredTs
	} else if jpegIndex > 0 {
		finalTimestamp = float64(jpegIndex-1) * intervalSec
	}

	slog.Info("frame: complete",
		"video_id", v.ID.String(),
		"frame_count", len(allSaved),
		"total_duration_ms", totalMs,
		"ffmpeg_total_ms", ffmpegMs,
		"ffmpeg_process_count", ffmpegProcessCount,
		"save_total_ms", totalSaveMs,
		"persist_ms", persistMs,
		"avg_ms_per_frame", float64(totalMs)/float64(len(allSaved)),
		"final_timestamp", finalTimestamp,
	)
	return allSaved, nil
}

func (s *Service) GetByVideoID(ctx context.Context, videoID uuid.UUID) ([]VideoFrame, error) {
	return s.repo.GetByVideoID(ctx, videoID)
}

func (s *Service) DeleteByVideoID(ctx context.Context, videoID uuid.UUID) error {
	frames, err := s.repo.GetByVideoID(ctx, videoID)
	if err != nil {
		return err
	}
	if err := s.repo.DeleteByVideoID(ctx, videoID); err != nil {
		return err
	}
	for _, f := range frames {
		_ = s.storage.Delete(ctx, f.StorageKey)
	}
	return nil
}
