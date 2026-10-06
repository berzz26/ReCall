package video_frame

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"image"
	_ "image/jpeg"
	"io"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/google/uuid"

	"github.com/berzz26/recall/services/api/internal/sampler"
	"github.com/berzz26/recall/services/api/internal/storage"
	"github.com/berzz26/recall/services/api/internal/video"
	"github.com/berzz26/recall/services/api/internal/video_segment"
)

// frameRepository is the persistence seam used by Service. *Repository
// implements it; tests inject fakes. The constructor keeps taking
// *Repository so existing call sites are unaffected.
type frameRepository interface {
	GetByVideoID(ctx context.Context, videoID uuid.UUID) ([]VideoFrame, error)
	DeleteByVideoID(ctx context.Context, videoID uuid.UUID) error
	CreateBatch(ctx context.Context, frames []VideoFrame) ([]VideoFrame, error)
}

// emptyFrameError marks a decode that ran fine but yielded no frame bytes:
// the requested timestamp has no decodable frame (phantom tail past the last
// real frame, or a local gap). The message matches the legacy string so
// existing log searches keep working. The extractor retries these with
// step-back and skips them at the EOF edge instead of failing the video.
type emptyFrameError struct{ Timestamp float64 }

func (e *emptyFrameError) Error() string {
	return fmt.Sprintf("ffmpeg produced empty frame at %f", e.Timestamp)
}

// Retry policy for empty decodes: step back in extractStepBack increments up
// to extractMaxStepBack (one baseline interval) looking for the nearest
// decodable frame.
const (
	extractStepBack    = 0.5
	extractMaxStepBack = 2.0
)

// extractFunc extracts one frame at the given timestamp and returns the
// JPEG bytes. It is a field (not a hard ffmpeg call) so tests can stub the
// decoder while exercising the planner -> extractor contract.
type extractFunc func(ctx context.Context, ffmpegPath, videoPath string, timestamp float64, jpegQuality int) ([]byte, error)

type Service struct {
	repo           frameRepository
	storage        storage.Storage
	sampleInterval time.Duration
	ffmpegPath     string
	// ffmpegTimeout is retained for compatibility (FFMPEG_TIMEOUT env).
	// A.2 semantics: FFMPEG_TIMEOUT is the maximum allowed duration of an
	// individual FFmpeg subprocess before cancellation, NOT a whole-video
	// wall-clock limit. The processing job lifetime is controlled by the
	// parent processing context (worker/application) and may run arbitrarily
	// long for long videos.
	ffmpegTimeout time.Duration
	jpegQuality   int
	// samplerBeta scales the baseline budget (B = ceil(beta * N_baseline)).
	// Phase 1 uses 1.0. Configurable via SAMPLER_BETA.
	samplerBeta float64
	// extractOne performs single-timestamp decoding. Defaults to
	// extractSingleFrame (ffmpeg seeking); tests override it.
	extractOne extractFunc
	// videoPlanner, when set, replaces the fixed 2-second baseline grid
	// with an adaptive plan (e.g. the visual-probe coarse planner). The
	// extractor still only consumes the resulting timestamp list.
	videoPlanner sampler.VideoPlanner
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
	return &Service{repo: repo, storage: store, sampleInterval: sampleInterval, ffmpegPath: ffmpegPath, ffmpegTimeout: ffmpegTimeout, jpegQuality: jpegQuality, samplerBeta: sampler.DefaultBeta, extractOne: extractSingleFrame}
}

// WithSamplerBeta sets the budget scale factor (B = ceil(beta * N_baseline))
// and returns the service for chaining. Non-positive values keep the
// current setting.
func (s *Service) WithSamplerBeta(beta float64) *Service {
	if beta > 0 {
		s.samplerBeta = beta
	}
	return s
}

// SampleTimestamps delegates to the sampler package so the uniform grid is
// defined in exactly one place. Kept for backward compatibility.
func SampleTimestamps(durationSeconds float64, interval time.Duration) ([]float64, error) {
	return sampler.BaselineTimestamps(durationSeconds, interval)
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

// WithVideoPlanner installs an adaptive planner consulted by
// GenerateForVideo before extraction. Nil (default) preserves the Phase 1
// fixed 2-second baseline behavior.
func (s *Service) WithVideoPlanner(p sampler.VideoPlanner) *Service {
	s.videoPlanner = p
	return s
}

// GenerateForVideo is the legacy entry point. It now builds an explicit
// baseline sampling plan (fixed 2-second grid, budget-capped) and delegates
// to GenerateForVideoWithPlan, so effective frames are unchanged while the
// planner/extractor separation is in force.
func (s *Service) GenerateForVideo(ctx context.Context, v *video.Video, segments []video_segment.VideoSegment, durationSeconds float64, width, height int) ([]VideoFrame, error) {
	if s.videoPlanner != nil {
		// Adaptive path: plan from the concrete video file, then extract
		// exactly the planned timestamps. The planner falls back to the
		// baseline grid internally on any probe failure.
		videoPath, cleanup, err := s.resolveVideoPath(ctx, v)
		if err != nil {
			return nil, err
		}
		if cleanup != nil {
			defer cleanup()
		}
		res, err := s.videoPlanner.PlanVideo(ctx, v.ID, videoPath, durationSeconds)
		if err != nil {
			return nil, err
		}
		return s.GenerateForVideoWithPlan(ctx, v, segments, res.Plan, width, height)
	}
	planner := sampler.NewBaselinePlanner(s.sampleInterval, s.samplerBeta)
	plan, err := planner.Plan(v.ID, durationSeconds)
	if err != nil {
		return nil, err
	}
	return s.GenerateForVideoWithPlan(ctx, v, segments, plan, width, height)
}

// GenerateForVideoWithPlan materializes an explicit sampling plan: it
// extracts exactly the plan's (normalized, budget-capped) timestamps as
// VideoFrame objects. It never decides which frames are important; that is
// the planner's job. The returned VideoFrame shape is unchanged, so YOLO,
// ByteTrack, events, VLM, and embeddings are unaffected.
func (s *Service) GenerateForVideoWithPlan(ctx context.Context, v *video.Video, segments []video_segment.VideoSegment, plan sampler.SamplingPlan, width, height int) ([]VideoFrame, error) {
	frameStart := time.Now()
	if s.storage == nil {
		return nil, fmt.Errorf("storage not configured")
	}
	if len(segments) == 0 {
		return nil, fmt.Errorf("no segments available for frame association")
	}
	if width <= 0 || height <= 0 {
		return nil, fmt.Errorf("invalid dimensions")
	}

	// Defensive normalization + hard budget cap. Plans built by a Planner
	// already satisfy these; re-applying keeps the extractor contract
	// ("never more frames than the plan / budget allows") even for
	// hand-built plans. Selection metadata stays on the plan side: the
	// extractor only consumes the float list.
	timestamps := sampler.NormalizeTimestamps(plan.Timestamps(), plan.DurationSeconds)
	if plan.Budget.Cap > 0 {
		timestamps = sampler.CapTimestamps(timestamps, plan.Budget.Cap)
	}
	if len(timestamps) > len(plan.Entries) {
		timestamps = timestamps[:len(plan.Entries)]
	}

	slog.Info("frame: plan",
		"video_id", v.ID.String(),
		"mode", plan.Mode,
		"duration_seconds", plan.DurationSeconds,
		"baseline_count", plan.Budget.BaselineCount,
		"budget_cap", plan.Budget.Cap,
		"requested", len(plan.Entries),
		"extracting", len(timestamps),
		"width", width, "height", height,
		"ffmpeg", s.ffmpegPath, "jpeg_quality", s.jpegQuality,
	)

	// Replace previously extracted frames (idempotent regeneration),
	// mirroring legacy behavior.
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

	if len(timestamps) == 0 {
		slog.Info("frame: complete (empty plan)", "video_id", v.ID.String(), "frame_count", 0)
		return []VideoFrame{}, nil
	}

	// Batch extraction: the source video path is resolved ONCE for the whole
	// timestamp batch, then each timestamp is seek-extracted. Seeking (rather
	// than decoding the entire video and discarding frames) keeps sparse
	// future plans cheap; dense baseline plans decode only what they need.
	videoPath, cleanupVideo, err := s.resolveVideoPath(ctx, v)
	if err != nil {
		return nil, err
	}
	if cleanupVideo != nil {
		defer cleanupVideo()
	}

	var createdKeys []string
	var batch []VideoFrame
	var allSaved []VideoFrame
	var totalExtractMs int64
	var totalSaveMs int64
	var persistMs int64

	cleanupOnFail := func() {
		for _, k := range createdKeys {
			_ = s.storage.Delete(ctx, k)
		}
		_ = s.repo.DeleteByVideoID(ctx, v.ID)
	}

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

	for i, ts := range timestamps {
		if err := ctx.Err(); err != nil {
			cleanupOnFail()
			return nil, err
		}
		seg := FindSegment(segments, ts)
		if seg == nil {
			cleanupOnFail()
			return nil, fmt.Errorf("no segment for timestamp %f", ts)
		}
		extractStart := time.Now()
		jpegBytes, skipped, err := s.extractResilient(ctx, videoPath, ts, plan.DurationSeconds)
		totalExtractMs += time.Since(extractStart).Milliseconds()
		if err != nil {
			cleanupOnFail()
			if ctx.Err() == context.DeadlineExceeded {
				slog.Error("frame: ffmpeg timeout", "video_id", v.ID.String(), "timestamp", ts, "error", ctx.Err())
				return nil, fmt.Errorf("ffmpeg timeout: %w", ctx.Err())
			}
			if ctx.Err() == context.Canceled {
				slog.Error("frame: context canceled", "video_id", v.ID.String(), "timestamp", ts, "error", ctx.Err())
				return nil, fmt.Errorf("context canceled: %w", ctx.Err())
			}
			slog.Error("frame: ffmpeg failed", "video_id", v.ID.String(), "timestamp", ts, "error", err)
			return nil, fmt.Errorf("ffmpeg failed at timestamp %f: %w", ts, err)
		}
		if skipped {
			// Phantom tail timestamp with no decodable frame: keep the
			// other frames instead of failing the video.
			slog.Warn("frame: skipped undecodable tail timestamp", "video_id", v.ID.String(), "timestamp", ts)
			continue
		}
		fw, fh := width, height
		if cfg, _, cfgErr := image.DecodeConfig(bytes.NewReader(jpegBytes)); cfgErr == nil && cfg.Width > 0 && cfg.Height > 0 {
			fw = cfg.Width
			fh = cfg.Height
		}
		frameIdx := len(allSaved) + len(batch)
		key := frameStorageKey(v.ID, frameIdx)
		saveStart := time.Now()
		if err := s.storage.Save(ctx, key, bytes.NewReader(jpegBytes)); err != nil {
			cleanupOnFail()
			return nil, fmt.Errorf("failed to store frame %d: %w", i, err)
		}
		totalSaveMs += time.Since(saveStart).Milliseconds()
		createdKeys = append(createdKeys, key)
		batch = append(batch, VideoFrame{
			VideoID:          v.ID,
			SegmentID:        seg.ID,
			FrameIndex:       frameIdx,
			TimestampSeconds: ts,
			StorageKey:       key,
			Width:            fw,
			Height:           fh,
		})
		if len(batch) >= 500 {
			if err := persistBatch(); err != nil {
				cleanupOnFail()
				return nil, fmt.Errorf("failed to persist frames: %w", err)
			}
		}
		if (i+1)%1000 == 0 {
			slog.Info("frame: progress", "video_id", v.ID.String(), "frames", i+1, "timestamp", ts, "elapsed_ms", time.Since(frameStart).Milliseconds())
		}
	}

	if err := persistBatch(); err != nil {
		cleanupOnFail()
		slog.Error("frame: persist failed", "video_id", v.ID.String(), "duration_ms", persistMs, "error", err)
		return nil, fmt.Errorf("failed to persist frames: %w", err)
	}

	// Contract guarantee: never return more frames than the plan requested.
	if len(allSaved) > len(plan.Entries) {
		allSaved = allSaved[:len(plan.Entries)]
	}

	totalMs := time.Since(frameStart).Milliseconds()
	var finalTimestamp float64
	if len(allSaved) > 0 {
		finalTimestamp = allSaved[len(allSaved)-1].TimestampSeconds
	}
	var avgMs float64
	if len(allSaved) > 0 {
		avgMs = float64(totalMs) / float64(len(allSaved))
	}
	slog.Info("frame: complete",
		"video_id", v.ID.String(),
		"frame_count", len(allSaved),
		"total_duration_ms", totalMs,
		"extract_total_ms", totalExtractMs,
		"save_total_ms", totalSaveMs,
		"persist_ms", persistMs,
		"avg_ms_per_frame", avgMs,
		"final_timestamp", finalTimestamp,
	)
	return allSaved, nil
}

// resolveVideoPath maps a video to a local file path, downloading uploaded
// videos to a temp file once per batch. The caller runs cleanup when done.
//
// ResolveVideoPath exposes resolveVideoPath for the adaptive-sampling
// orchestrator, which needs the concrete file path for the visual probe.
// Caller runs the returned cleanup when done (nil when no temp file).
func (s *Service) ResolveVideoPath(ctx context.Context, v *video.Video) (string, func(), error) {
	return s.resolveVideoPath(ctx, v)
}

func (s *Service) resolveVideoPath(ctx context.Context, v *video.Video) (string, func(), error) {
	if v.SourceType == video.SourceTypeLocal {
		if v.SourcePath == nil || *v.SourcePath == "" {
			return "", nil, fmt.Errorf("missing source_path for LOCAL video")
		}
		if _, err := os.Stat(*v.SourcePath); err != nil {
			return "", nil, fmt.Errorf("source file not found: %w", err)
		}
		return *v.SourcePath, nil, nil
	}
	if v.StorageKey == nil || *v.StorageKey == "" {
		return "", nil, fmt.Errorf("missing storage_key for UPLOAD video")
	}
	// Prefer direct filesystem path if storage is LocalStorage to avoid extra copy.
	if ls, ok := s.storage.(*storage.LocalStorage); ok {
		candidate := filepath.Join(ls.Root(), *v.StorageKey)
		if _, err := os.Stat(candidate); err == nil {
			return candidate, nil, nil
		}
	}
	rc, err := s.storage.Open(ctx, *v.StorageKey)
	if err != nil {
		return "", nil, fmt.Errorf("failed to open storage: %w", err)
	}
	ext := filepath.Ext(*v.StorageKey)
	if ext == "" {
		ext = ".mp4"
	}
	tmp, err := os.CreateTemp("", "frame-src-*"+ext)
	if err != nil {
		rc.Close()
		return "", nil, fmt.Errorf("failed to create temp video: %w", err)
	}
	tempVideo := tmp.Name()
	cleanup := func() { os.Remove(tempVideo) }
	if _, err := io.Copy(tmp, rc); err != nil {
		tmp.Close()
		rc.Close()
		cleanup()
		return "", nil, fmt.Errorf("failed to copy to temp video: %w", err)
	}
	tmp.Close()
	rc.Close()
	return tempVideo, cleanup, nil
}

// extractSingleFrame decodes exactly one frame at the requested timestamp
// using ffmpeg seeking (-ss before -i: fast seek without decoding the whole
// file). Accuracy is subject to normal decoder limitations (seek lands on or
// near the requested timestamp); the extractor records the requested
// timestamp on the VideoFrame.
func extractSingleFrame(ctx context.Context, ffmpegPath, videoPath string, timestamp float64, jpegQuality int) ([]byte, error) {
	tmp, err := os.CreateTemp("", "frame-seek-*.jpg")
	if err != nil {
		return nil, fmt.Errorf("failed to create temp frame: %w", err)
	}
	tmpPath := tmp.Name()
	tmp.Close()
	defer os.Remove(tmpPath)

	tsArg := strconv.FormatFloat(timestamp, 'f', 6, 64)
	args := []string{
		"-loglevel", "error",
		"-y",
		"-ss", tsArg,
		"-i", videoPath,
		"-frames:v", "1",
		"-q:v", strconv.Itoa(jpegQuality),
		"-f", "image2",
		"-vcodec", "mjpeg",
		tmpPath,
	}
	cmd := exec.CommandContext(ctx, ffmpegPath, args...)
	var stderrBuf bytes.Buffer
	cmd.Stderr = &limitedWriter{buf: &stderrBuf, limit: 8192}
	if err := cmd.Run(); err != nil {
		msg := strings.TrimSpace(stderrBuf.String())
		if len(msg) > 500 {
			msg = msg[:500]
		}
		if msg == "" {
			msg = err.Error()
		}
		return nil, fmt.Errorf("ffmpeg seek to %f failed: %s", timestamp, msg)
	}
	data, err := os.ReadFile(tmpPath)
	if err != nil {
		return nil, fmt.Errorf("failed to read extracted frame: %w", err)
	}
	if len(data) == 0 {
		return nil, &emptyFrameError{Timestamp: timestamp}
	}
	return data, nil
}

// extractResilient decodes the requested timestamp with tolerance for
// timestamps that have no decodable frame. It returns (data, skipped, err):
//   - data non-nil: frame bytes. On step-back success the bytes come from a
//     slightly earlier timestamp, but the VideoFrame still records the
//     requested timestamp (seek lands on-or-near is the standing contract,
//     and segment association already used the requested timestamp).
//   - skipped true: the timestamp is a phantom tail — within one sample
//     interval of durationSeconds with no decodable frame. The caller should
//     skip it with a warning, not fail the video.
//   - err non-nil: genuine failure (decoder error, timeout, cancel, or a
//     persistent gap away from EOF); the caller fails as before.
//
// Only empty-output decodes are retried. Hard decoder errors and context
// cancellation fail immediately without retry.
func (s *Service) extractResilient(ctx context.Context, videoPath string, timestamp, durationSeconds float64) ([]byte, bool, error) {
	var empty *emptyFrameError
	data, err := s.extractOne(ctx, s.ffmpegPath, videoPath, timestamp, s.jpegQuality)
	if err == nil {
		return data, false, nil
	}
	if ctx.Err() != nil {
		return nil, false, err
	}
	if !errors.As(err, &empty) {
		return nil, false, err
	}
	for back := extractStepBack; back <= extractMaxStepBack+1e-9; back += extractStepBack {
		t2 := timestamp - back
		if t2 < 0 {
			t2 = 0
		}
		retry, retryErr := s.extractOne(ctx, s.ffmpegPath, videoPath, t2, s.jpegQuality)
		if retryErr == nil {
			slog.Warn("frame: stepped back to decodable timestamp",
				"requested", timestamp, "extracted", t2)
			return retry, false, nil
		}
		if ctx.Err() != nil {
			return nil, false, retryErr
		}
		if !errors.As(retryErr, &empty) {
			return nil, false, retryErr
		}
		if t2 == 0 {
			break
		}
	}
	// Nothing decodable from timestamp back to timestamp-maxStepBack. Near
	// EOF this is the phantom tail (container duration overshoots the last
	// real frame): skip it. Anywhere else it is a real gap: fail.
	if durationSeconds-timestamp <= s.sampleInterval.Seconds() {
		return nil, true, nil
	}
	return nil, false, err
}

// ExtractAdditional extracts extra timestamps WITHOUT deleting existing
// frames, for adaptive refinement rounds. Timestamps are normalized
// (sorted, deduplicated, range-checked); already-extracted timestamps are
// skipped so no frame is ever processed twice. FrameIndex and storage keys
// continue after the existing frames. Unlike GenerateForVideoWithPlan it
// applies no budget cap: the refinement planner owns budget accounting.
func (s *Service) ExtractAdditional(ctx context.Context, v *video.Video, segments []video_segment.VideoSegment, timestamps []float64, durationSeconds float64, width, height int) ([]VideoFrame, error) {
	if s.storage == nil {
		return nil, fmt.Errorf("storage not configured")
	}
	if len(segments) == 0 {
		return nil, fmt.Errorf("no segments available for frame association")
	}
	if width <= 0 || height <= 0 {
		return nil, fmt.Errorf("invalid dimensions")
	}
	want := sampler.NormalizeTimestamps(timestamps, durationSeconds)
	if len(want) == 0 {
		return []VideoFrame{}, nil
	}

	existing, err := s.repo.GetByVideoID(ctx, v.ID)
	if err != nil {
		return nil, err
	}
	done := make(map[float64]bool, len(existing))
	base := 0
	for _, f := range existing {
		done[f.TimestampSeconds] = true
		if f.FrameIndex >= base {
			base = f.FrameIndex + 1
		}
	}
	var fresh []float64
	for _, t := range want {
		dup := false
		for dt := range done {
			if dt-t < 1e-6 && t-dt < 1e-6 {
				dup = true
				break
			}
		}
		if !dup {
			fresh = append(fresh, t)
			done[t] = true
		}
	}
	if len(fresh) == 0 {
		return []VideoFrame{}, nil
	}

	videoPath, cleanupVideo, err := s.resolveVideoPath(ctx, v)
	if err != nil {
		return nil, err
	}
	if cleanupVideo != nil {
		defer cleanupVideo()
	}

	var createdKeys []string
	var batch []VideoFrame
	cleanupOnFail := func() {
		for _, k := range createdKeys {
			_ = s.storage.Delete(ctx, k)
		}
	}
	for _, ts := range fresh {
		if err := ctx.Err(); err != nil {
			cleanupOnFail()
			return nil, err
		}
		seg := FindSegment(segments, ts)
		if seg == nil {
			cleanupOnFail()
			return nil, fmt.Errorf("no segment for timestamp %f", ts)
		}
		jpegBytes, skipped, err := s.extractResilient(ctx, videoPath, ts, durationSeconds)
		if err != nil {
			cleanupOnFail()
			return nil, fmt.Errorf("ffmpeg failed at timestamp %f: %w", ts, err)
		}
		if skipped {
			slog.Warn("frame: skipped undecodable tail timestamp", "video_id", v.ID.String(), "timestamp", ts)
			continue
		}
		fw, fh := width, height
		if cfg, _, cfgErr := image.DecodeConfig(bytes.NewReader(jpegBytes)); cfgErr == nil && cfg.Width > 0 && cfg.Height > 0 {
			fw = cfg.Width
			fh = cfg.Height
		}
		frameIdx := base + len(batch)
		key := frameStorageKey(v.ID, frameIdx)
		if err := s.storage.Save(ctx, key, bytes.NewReader(jpegBytes)); err != nil {
			cleanupOnFail()
			return nil, fmt.Errorf("failed to store frame at %f: %w", ts, err)
		}
		createdKeys = append(createdKeys, key)
		batch = append(batch, VideoFrame{
			VideoID:          v.ID,
			SegmentID:        seg.ID,
			FrameIndex:       frameIdx,
			TimestampSeconds: ts,
			StorageKey:       key,
			Width:            fw,
			Height:           fh,
		})
	}
	saved, err := s.repo.CreateBatch(ctx, batch)
	if err != nil {
		cleanupOnFail()
		return nil, fmt.Errorf("failed to persist frames: %w", err)
	}
	slog.Info("frame: additional extraction complete",
		"video_id", v.ID.String(),
		"requested", len(want),
		"extracted", len(saved),
		"skipped_existing", len(want)-len(fresh),
		"skipped_undecodable", len(fresh)-len(saved),
	)
	return saved, nil
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
