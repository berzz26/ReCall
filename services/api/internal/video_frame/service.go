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
	"math"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"sync"
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
	GetByVideoIDSet(ctx context.Context, videoID uuid.UUID, set Set) ([]VideoFrame, error)
	GetBySegmentIDSet(ctx context.Context, segmentID uuid.UUID, set Set) ([]VideoFrame, error)
	DeleteByVideoID(ctx context.Context, videoID uuid.UUID) error
	UpdatePurpose(ctx context.Context, frameID uuid.UUID, purpose Purpose) error
	DeleteByIDs(ctx context.Context, ids []uuid.UUID) error
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

// frameTSEpsilon is the timestamp-matching tolerance for membership walks: an
// existing frame whose timestamp is within this epsilon of a requested
// timestamp reuses its stored asset instead of being re-extracted. It matches
// the sampler's dedupe epsilon and the orchestration matching tolerance.
const frameTSEpsilon = 1e-6

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
	// extractWorkers bounds concurrent ffmpeg decodes. <=0 means auto:
	// min(NumCPU, 8). Configurable via FRAME_EXTRACT_WORKERS.
	extractWorkers int
	// videoPlanner, when set, replaces the fixed 2-second baseline grid
	// with an adaptive plan (e.g. the visual-probe coarse planner). The
	// extractor still only consumes the resulting timestamp list.
	videoPlanner sampler.VideoPlanner
	// perceptionInterval is the fixed perception-sampling interval
	// (PERCEPTION_SAMPLE_FPS). It is fully independent of the semantic
	// sampler's sampleInterval: a stationary scene still gets perception
	// observations at this rate.
	perceptionInterval time.Duration
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
	return &Service{
		repo:               repo,
		storage:            store,
		sampleInterval:     sampleInterval,
		perceptionInterval: 2 * time.Second,
		ffmpegPath:         ffmpegPath,
		ffmpegTimeout:      ffmpegTimeout,
		jpegQuality:        jpegQuality,
		samplerBeta:        sampler.DefaultBeta,
		extractOne:         extractSingleFrame,
	}
}

// WithExtractWorkers sets the max concurrent ffmpeg decodes (<=0 = auto)
// and returns the service for chaining.
func (s *Service) WithExtractWorkers(n int) *Service {
	s.extractWorkers = n
	return s
}

// effectiveWorkers resolves the worker count: explicit value when positive,
// else min(NumCPU, 8).
func (s *Service) effectiveWorkers() int {
	if s.extractWorkers > 0 {
		return s.extractWorkers
	}
	return defaultExtractWorkers()
}

func defaultExtractWorkers() int {
	n := runtime.NumCPU()
	if n < 1 {
		n = 1
	}
	if n > 8 {
		n = 8
	}
	return n
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

// WithPerceptionSampleFPS sets the fixed perception sampling rate and returns
// the service for chaining. fps <= 0 keeps the current setting (invalid
// values are rejected by config validation). The default is 2s (0.5 fps).
func (s *Service) WithPerceptionSampleFPS(fps float64) *Service {
	if fps > 0 {
		if interval := time.Duration(float64(time.Second) / fps); interval > 0 {
			s.perceptionInterval = interval
		}
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
// to GenerateForVideoWithPlan, so the semantic set it produces is unchanged
// while the planner/extractor separation is in force.
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

// GenerateForVideoWithPlan materializes the SEMANTIC frame set from an
// explicit adaptive-sampling plan: it replaces the semantic set's membership
// with exactly the plan's (normalized, budget-capped) timestamps. It never
// decides which frames are important; that is the planner's job.
//
// Replacement is membership-aware: frames of the perception set (or shared
// 'both' frames outside the plan) are never deleted or detached. A plan
// timestamp whose row already exists is re-used without re-extraction
// (promoted to 'both' when it was perception-only). Set-owning rows the plan
// drops are deleted; shared 'both' rows the plan drops are demoted to
// perception-only. The returned shape is unchanged, so YOLO, ByteTrack,
// events, VLM, and embeddings are unaffected.
func (s *Service) GenerateForVideoWithPlan(ctx context.Context, v *video.Video, segments []video_segment.VideoSegment, plan sampler.SamplingPlan, width, height int) ([]VideoFrame, error) {
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
		"set", PurposeSemantic,
		"mode", plan.Mode,
		"duration_seconds", plan.DurationSeconds,
		"baseline_count", plan.Budget.BaselineCount,
		"budget_cap", plan.Budget.Cap,
		"requested", len(plan.Entries),
		"extracting", len(timestamps),
		"width", width, "height", height,
		"ffmpeg", s.ffmpegPath, "jpeg_quality", s.jpegQuality,
	)

	outcome, err := s.replaceFrameSet(ctx, v, segments, timestamps, plan.DurationSeconds, PurposeSemantic, PurposePerception, width, height)
	if err != nil {
		return nil, err
	}
	return outcome.Frames, nil
}

// PerceptionResult is the outcome of a perception-set replacement. Frames is
// the full perception membership after the walk; DroppedFrameIDs lists rows
// demoted out of the perception set (former 'both' rows now semantic-only):
// their previously persisted production detections are stale for the new
// cadence and the orchestrator deletes them before running production
// detection.
type PerceptionResult struct {
	Frames          []VideoFrame
	DroppedFrameIDs []uuid.UUID
}

// GeneratePerceptionForVideo materializes the PERCEPTION frame set: a
// regular, fixed-rate grid at the configured perception interval
// (PERCEPTION_SAMPLE_FPS, default 0.5 fps = 2s). It never consults the
// adaptive sampler — a stationary scene still gets observations at the
// configured rate — and it never deletes or detaches the semantic set's
// membership (timestamps shared with the semantic set reuse a single row as
// 'both').
func (s *Service) GeneratePerceptionForVideo(ctx context.Context, v *video.Video, segments []video_segment.VideoSegment, durationSeconds float64, width, height int) (*PerceptionResult, error) {
	if s.storage == nil {
		return nil, fmt.Errorf("storage not configured")
	}
	if len(segments) == 0 {
		return nil, fmt.Errorf("no segments available for frame association")
	}
	if width <= 0 || height <= 0 {
		return nil, fmt.Errorf("invalid dimensions")
	}

	targets, err := sampler.BaselineTimestamps(durationSeconds, s.perceptionInterval)
	if err != nil {
		return nil, err
	}
	targets = sampler.NormalizeTimestamps(targets, durationSeconds)

	slog.Info("frame: perception plan",
		"video_id", v.ID.String(),
		"set", PurposePerception,
		"duration_seconds", durationSeconds,
		"interval_seconds", s.perceptionInterval.Seconds(),
		"requested", len(targets),
		"width", width, "height", height,
		"ffmpeg", s.ffmpegPath, "jpeg_quality", s.jpegQuality,
	)

	outcome, err := s.replaceFrameSet(ctx, v, segments, targets, durationSeconds, PurposePerception, PurposeSemantic, width, height)
	if err != nil {
		return nil, err
	}
	return &PerceptionResult{Frames: outcome.Frames, DroppedFrameIDs: outcome.DroppedFrameIDs}, nil
}

// setWalkOutcome reports the result of replacing one frame set's membership.
type setWalkOutcome struct {
	// Frames is the full membership of the regenerated set after the walk,
	// in target-timestamp order (reused rows first-class with freshly
	// extracted ones; skipped phantom-tail timestamps absent).
	Frames []VideoFrame
	// DroppedFrameIDs are rows demoted OUT of the regenerated set (former
	// 'both' rows now owned by the other set only). Deleted rows are not
	// listed: their dependent detections leave via FK cascade with the row.
	DroppedFrameIDs []uuid.UUID
}

// replaceFrameSet replaces the membership of one frame set (target) with
// exactly the given timestamps, leaving the other set (other) untouched.
//
// The walk, evaluated entirely before any mutation:
//
//	matched row (|ts - target| < eps):     member; reuse the stored asset,
//	                                       never re-extract (promoted to
//	                                       'both' when owned by the other
//	                                       set only).
//	unmatched, purpose == target:          delete row + asset (attached
//	                                       production detections leave via
//	                                       FK cascade).
//	unmatched, purpose == both:            demote to other-only; asset and
//	                                       the other set's membership
//	                                       survive; its production
//	                                       detections may become stale
//	                                       (surfaced as DroppedFrameIDs).
//	unmatched, purpose == other:           untouched (other set's row).
//	timestamp with no matching row:        freshly extracted as target.
//
// Mutations are applied in a fixed order — deletes (rows then assets),
// demotions, promotions, extraction+insertion — so a database failure aborts
// before extraction allocates work. On later failure, cleanup removes ONLY
// rows/assets created by this call: the other set and pre-existing rows
// always survive a failed regeneration.
func (s *Service) replaceFrameSet(ctx context.Context, v *video.Video, segments []video_segment.VideoSegment, targets []float64, durationSeconds float64, target, other Purpose, width, height int) (*setWalkOutcome, error) {
	runStart := time.Now()
	existing, err := s.repo.GetByVideoID(ctx, v.ID)
	if err != nil {
		return nil, err
	}

	out := &setWalkOutcome{}
	rowByTarget := make([]*VideoFrame, len(targets))
	var toDelete []VideoFrame
	var toDemote []VideoFrame
	var toPromote []VideoFrame
	// First ownership pass: classify every existing row. `claim` prevents
	// two rows claiming the same target; ties resolve deterministically by
	// the repository's chronological order.
	claim := make([]bool, len(targets))
	for i := range existing {
		f := existing[i] // loop-local copy; pointer stored when matched
		match := -1
		for j, want := range targets {
			if claim[j] {
				continue
			}
			if math.Abs(f.TimestampSeconds-want) < frameTSEpsilon {
				match = j
				break
			}
		}
		if match < 0 {
			switch f.Purpose.resolve() {
			case target:
				toDelete = append(toDelete, f)
			case PurposeBoth:
				f.Purpose = other
				toDemote = append(toDemote, f)
			default:
				// other set's row: untouched
			}
			continue
		}
		claim[match] = true
		if f.Purpose.resolve() == other {
			f.Purpose = PurposeBoth
			toPromote = append(toPromote, f)
		}
		rowByTarget[match] = &f
	}

	// FrameIndex base = max remaining index + 1 (deleted rows do not count).
	delIDs := make([]uuid.UUID, 0, len(toDelete))
	delSet := make(map[uuid.UUID]bool, len(toDelete))
	for _, f := range toDelete {
		delIDs = append(delIDs, f.ID)
		delSet[f.ID] = true
	}
	frameIdxBase := 0
	for _, f := range existing {
		if delSet[f.ID] {
			continue
		}
		if f.FrameIndex+1 > frameIdxBase {
			frameIdxBase = f.FrameIndex + 1
		}
	}

	// Mutation phase 1: deletes (detections cascade), then asset removal,
	// then demotions and promotions.
	if len(delIDs) > 0 {
		if err := s.repo.DeleteByIDs(ctx, delIDs); err != nil {
			return nil, fmt.Errorf("failed to delete replaced %s frames: %w", target, err)
		}
		for _, f := range toDelete {
			_ = s.storage.Delete(ctx, f.StorageKey)
		}
	}
	for _, f := range toDemote {
		if err := s.repo.UpdatePurpose(ctx, f.ID, other); err != nil {
			return nil, fmt.Errorf("failed to demote shared frame: %w", err)
		}
		out.DroppedFrameIDs = append(out.DroppedFrameIDs, f.ID)
	}
	for _, f := range toPromote {
		if err := s.repo.UpdatePurpose(ctx, f.ID, PurposeBoth); err != nil {
			return nil, fmt.Errorf("failed to adopt frame into %s set: %w", target, err)
		}
	}

	var fresh []float64
	for j := range targets {
		if rowByTarget[j] == nil {
			fresh = append(fresh, targets[j])
		}
	}

	if len(fresh) == 0 {
		out.Frames = membershipInTargetOrder(rowByTarget)
		slog.Info("frame: complete (no new extraction needed)",
			"video_id", v.ID.String(), "set", target, "frame_count", len(out.Frames))
		return out, nil
	}

	// Extraction phase: the source video path is resolved ONCE for the whole
	// timestamp batch, then each timestamp is seek-extracted (see the
	// semantic extraction notes above; semantics unchanged).
	videoPath, cleanupVideo, err := s.resolveVideoPath(ctx, v)
	if err != nil {
		return nil, err
	}
	if cleanupVideo != nil {
		defer cleanupVideo()
	}

	var createdKeys []string
	var createdIDs []uuid.UUID
	var batch []VideoFrame
	var allSaved []VideoFrame
	var totalExtractMs int64
	var totalSaveMs int64
	var persistMs int64

	cleanupOnFail := func() {
		// Remove ONLY what this call created: the other frame set and the
		// pre-existing rows already mutated above must never be destroyed by
		// a failed regeneration.
		for _, k := range createdKeys {
			_ = s.storage.Delete(ctx, k)
		}
		if len(createdIDs) > 0 {
			_ = s.repo.DeleteByIDs(ctx, createdIDs)
		}
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
		for i := range saved {
			createdIDs = append(createdIDs, saved[i].ID)
		}
		allSaved = append(allSaved, saved...)
		batch = batch[:0]
		return nil
	}

	results, err := s.extractAllOrdered(ctx, videoPath, fresh, durationSeconds)
	if err != nil {
		var xerr *extractError
		ts := 0.0
		if errors.As(err, &xerr) {
			ts = xerr.Timestamp
		}
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

	createdByTS := make(map[float64]VideoFrame, len(fresh))
	for i := range results {
		ts := fresh[i]
		if err := ctx.Err(); err != nil {
			cleanupOnFail()
			return nil, err
		}
		seg := FindSegment(segments, ts)
		if seg == nil {
			cleanupOnFail()
			return nil, fmt.Errorf("no segment for timestamp %f", ts)
		}
		totalExtractMs += results[i].extractMs
		if results[i].skipped {
			// Phantom tail timestamp with no decodable frame: keep the
			// other frames instead of failing the video.
			slog.Warn("frame: skipped undecodable tail timestamp", "video_id", v.ID.String(), "timestamp", ts)
			continue
		}
		jpegBytes := results[i].jpeg
		fw, fh := width, height
		if cfg, _, cfgErr := image.DecodeConfig(bytes.NewReader(jpegBytes)); cfgErr == nil && cfg.Width > 0 && cfg.Height > 0 {
			fw = cfg.Width
			fh = cfg.Height
		}
		frameIdx := frameIdxBase + len(allSaved) + len(batch)
		key := frameStorageKey(v.ID, frameIdx)
		saveStart := time.Now()
		if err := s.storage.Save(ctx, key, bytes.NewReader(jpegBytes)); err != nil {
			cleanupOnFail()
			return nil, fmt.Errorf("failed to store frame %d: %w", i, err)
		}
		totalSaveMs += time.Since(saveStart).Milliseconds()
		createdKeys = append(createdKeys, key)
		frame := VideoFrame{
			VideoID:          v.ID,
			SegmentID:        seg.ID,
			FrameIndex:       frameIdx,
			TimestampSeconds: ts,
			StorageKey:       key,
			Width:            fw,
			Height:           fh,
			Purpose:          target,
		}
		batch = append(batch, frame)
		if len(batch) >= 500 {
			if err := persistBatch(); err != nil {
				cleanupOnFail()
				return nil, fmt.Errorf("failed to persist frames: %w", err)
			}
		}
		if (i+1)%1000 == 0 {
			slog.Info("frame: progress", "video_id", v.ID.String(), "set", target, "frames", i+1, "timestamp", ts, "elapsed_ms", time.Since(runStart).Milliseconds())
		}
	}

	if err := persistBatch(); err != nil {
		cleanupOnFail()
		slog.Error("frame: persist failed", "video_id", v.ID.String(), "set", target, "duration_ms", persistMs, "error", err)
		return nil, fmt.Errorf("failed to persist frames: %w", err)
	}

	for _, f := range allSaved {
		createdByTS[f.TimestampSeconds] = f
	}
	out.Frames = mergeMembership(rowByTarget, createdByTS, targets)

	totalMs := time.Since(runStart).Milliseconds()
	var finalTimestamp float64
	if len(out.Frames) > 0 {
		finalTimestamp = out.Frames[len(out.Frames)-1].TimestampSeconds
	}
	var avgMs float64
	if len(out.Frames) > 0 {
		avgMs = float64(totalMs) / float64(len(out.Frames))
	}
	slog.Info("frame: complete",
		"video_id", v.ID.String(),
		"set", target,
		"frame_count", len(out.Frames),
		"reused", len(out.Frames)-len(allSaved),
		"extracted", len(allSaved),
		"demoted_out", len(toDemote),
		"deleted", len(toDelete),
		"total_duration_ms", totalMs,
		"extract_total_ms", totalExtractMs,
		"save_total_ms", totalSaveMs,
		"persist_ms", persistMs,
		"avg_ms_per_frame", avgMs,
		"final_timestamp", finalTimestamp,
	)
	return out, nil
}

// membershipInTargetOrder assembles the final membership list in target
// timestamp order from the matched rows (reused only; extracted frames are
// merged in later via mergeMembership). Called only when every target
// matched an existing row (len(fresh) == 0).
func membershipInTargetOrder(rowByTarget []*VideoFrame) []VideoFrame {
	out := make([]VideoFrame, 0, len(rowByTarget))
	for _, row := range rowByTarget {
		if row != nil {
			out = append(out, *row)
		}
	}
	return out
}

// mergeMembership assembles the final membership list in target timestamp
// order, combining reused rows and freshly extracted frames.
func mergeMembership(rowByTarget []*VideoFrame, createdByTS map[float64]VideoFrame, targets []float64) []VideoFrame {
	out := make([]VideoFrame, 0, len(targets))
	for j, row := range rowByTarget {
		if row != nil {
			out = append(out, *row)
			continue
		}
		if f, ok := createdByTS[targets[j]]; ok {
			out = append(out, f)
		}
	}
	return out
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

// extractError carries a decode failure with its timestamp so callers can
// reproduce the sequential error messages.
type extractError struct {
	Timestamp float64
	Err       error
}

func (e *extractError) Error() string { return e.Err.Error() }
func (e *extractError) Unwrap() error { return e.Err }

// orderedFrame is one decoded timestamp in plan order. jpeg is nil when skipped.
type orderedFrame struct {
	jpeg      []byte
	skipped   bool
	extractMs int64
}

// frameResult is one worker's output for timestamps[idx].
type frameResult struct {
	idx       int
	jpeg      []byte
	skipped   bool
	extractMs int64
	err       *extractError
}

// extractChunkSize bounds transient memory: one chunk's JPEGs are held while
// its workers run (~256 * ~200KB worst case), no matter how long the video.
const extractChunkSize = 256

// extractAllOrdered decodes every timestamp with up to effectiveWorkers()
// concurrent ffmpeg processes and returns results in timestamp order.
// Skipped (phantom tail) entries keep their position with skipped=true.
// The first failure in timestamp order aborts remaining work and is returned
// as *extractError, matching sequential semantics (fail at the lowest failing
// timestamp, never a partial frame set).
func (s *Service) extractAllOrdered(ctx context.Context, videoPath string, timestamps []float64, durationSeconds float64) ([]orderedFrame, error) {
	out := make([]orderedFrame, 0, len(timestamps))
	for start := 0; start < len(timestamps); start += extractChunkSize {
		end := start + extractChunkSize
		if end > len(timestamps) {
			end = len(timestamps)
		}
		chunk, err := s.extractChunk(ctx, videoPath, timestamps[start:end], durationSeconds)
		if err != nil {
			return nil, err
		}
		out = append(out, chunk...)
	}
	return out, nil
}

func (s *Service) extractChunk(ctx context.Context, videoPath string, timestamps []float64, durationSeconds float64) ([]orderedFrame, error) {
	n := len(timestamps)
	out := make([]orderedFrame, n)
	if n == 0 {
		return out, nil
	}
	workers := s.effectiveWorkers()
	if workers < 1 {
		workers = 1
	}
	if workers > n {
		workers = n
	}
	cctx, cancel := context.WithCancel(ctx)
	defer cancel()

	jobs := make(chan int)
	resCh := make(chan frameResult, workers*2)
	var wg sync.WaitGroup

	wg.Add(1)
	go func() {
		defer wg.Done()
		defer close(jobs)
		for i := range timestamps {
			select {
			case jobs <- i:
			case <-cctx.Done():
				return
			}
		}
	}()

	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range jobs {
				start := time.Now()
				jpeg, skipped, err := s.extractResilient(cctx, videoPath, timestamps[i], durationSeconds)
				r := frameResult{idx: i, jpeg: jpeg, skipped: skipped, extractMs: time.Since(start).Milliseconds()}
				if err != nil {
					r.err = &extractError{Timestamp: timestamps[i], Err: err}
				}
				select {
				case resCh <- r:
				case <-cctx.Done():
					return
				}
			}
		}()
	}

	go func() {
		wg.Wait()
		close(resCh)
	}()

	pending := make(map[int]frameResult, workers*2)
	next := 0
	var firstErr *extractError
	for r := range resCh {
		pending[r.idx] = r
		for {
			p, ok := pending[next]
			if !ok {
				break
			}
			delete(pending, next)
			if p.err != nil && firstErr == nil {
				firstErr = p.err
			}
			out[next] = orderedFrame{jpeg: p.jpeg, skipped: p.skipped, extractMs: p.extractMs}
			next++
		}
		if firstErr != nil {
			break
		}
	}
	if firstErr != nil {
		cancel()
		wg.Wait()
		return nil, firstErr
	}
	return out, nil
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
// frames, for adaptive semantic refinement rounds. Timestamps are normalized
// (sorted, deduplicated, range-checked). Membership-aware classification:
//
//   - a row already in the semantic set ('semantic' or 'both') is skipped:
//     never re-extracted and NOT returned (the refiner only sees what it
//     still needs);
//   - a perception-only row is adopted into the semantic set (promoted to
//     'both', asset reused, no re-extraction) and IS returned, so the
//     refiner still gets the midpoint observations it asked about;
//   - anything else is freshly extracted as a semantic frame.
//
// FrameIndex and storage keys continue after the remaining frames. Unlike
// GenerateForVideoWithPlan it applies no budget cap: the refinement planner
// owns budget accounting.
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
	base := 0
	for _, f := range existing {
		if f.FrameIndex+1 > base {
			base = f.FrameIndex + 1
		}
	}

	var fresh []float64
	var reused []VideoFrame
	var toPromote []VideoFrame
	rowClaim := make([]bool, len(existing))
	for _, t := range want {
		match := -1
		for i := range existing {
			if rowClaim[i] {
				continue
			}
			if math.Abs(existing[i].TimestampSeconds-t) < frameTSEpsilon {
				match = i
				break
			}
		}
		if match < 0 {
			fresh = append(fresh, t)
			continue
		}
		rowClaim[match] = true
		f := existing[match]
		if f.Purpose.resolve() == PurposePerception {
			f.Purpose = PurposeBoth
			reused = append(reused, f)
			toPromote = append(toPromote, f)
		}
		// 'semantic' or 'both': already in the semantic set; skip silently.
	}

	if len(fresh) == 0 {
		if err := s.applySemanticAdoptions(ctx, toPromote); err != nil {
			return nil, err
		}
		slog.Info("frame: additional extraction complete",
			"video_id", v.ID.String(),
			"requested", len(want),
			"extracted", 0,
			"adopted_from_perception", len(reused),
			"skipped_existing", len(want)-len(reused),
			"skipped_undecodable", 0,
		)
		return reused, nil
	}

	videoPath, cleanupVideo, err := s.resolveVideoPath(ctx, v)
	if err != nil {
		return nil, err
	}
	if cleanupVideo != nil {
		defer cleanupVideo()
	}

	var createdKeys []string
	var createdIDs []uuid.UUID
	cleanupOnFail := func() {
		for _, k := range createdKeys {
			_ = s.storage.Delete(ctx, k)
		}
		if len(createdIDs) > 0 {
			_ = s.repo.DeleteByIDs(ctx, createdIDs)
		}
	}
	results, err := s.extractAllOrdered(ctx, videoPath, fresh, durationSeconds)
	if err != nil {
		var xerr *extractError
		ts := 0.0
		if errors.As(err, &xerr) {
			ts = xerr.Timestamp
		}
		cleanupOnFail()
		return nil, fmt.Errorf("ffmpeg failed at timestamp %f: %w", ts, err)
	}
	var saved []VideoFrame
	var batch []VideoFrame
	for i := range results {
		ts := fresh[i]
		if err := ctx.Err(); err != nil {
			cleanupOnFail()
			return nil, err
		}
		seg := FindSegment(segments, ts)
		if seg == nil {
			cleanupOnFail()
			return nil, fmt.Errorf("no segment for timestamp %f", ts)
		}
		if results[i].skipped {
			slog.Warn("frame: skipped undecodable tail timestamp", "video_id", v.ID.String(), "timestamp", ts)
			continue
		}
		jpegBytes := results[i].jpeg
		fw, fh := width, height
		if cfg, _, cfgErr := image.DecodeConfig(bytes.NewReader(jpegBytes)); cfgErr == nil && cfg.Width > 0 && cfg.Height > 0 {
			fw = cfg.Width
			fh = cfg.Height
		}
		frameIdx := base + len(saved) + len(batch)
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
			Purpose:          PurposeSemantic,
		})
	}
	saved, err = s.repo.CreateBatch(ctx, batch)
	if err != nil {
		cleanupOnFail()
		return nil, fmt.Errorf("failed to persist frames: %w", err)
	}
	for i := range saved {
		createdIDs = append(createdIDs, saved[i].ID)
	}
	if err := s.applySemanticAdoptions(ctx, toPromote); err != nil {
		cleanupOnFail()
		return nil, err
	}

	// Assemble the result in requested timestamp order: adopted rows and
	// freshly extracted frames share one list so the refiner sees exactly
	// one entry per requested midpoint.
	byTS := make(map[float64]VideoFrame, len(saved))
	for _, f := range saved {
		byTS[f.TimestampSeconds] = f
	}
	reusedByTS := make(map[float64]VideoFrame, len(reused))
	for _, f := range reused {
		reusedByTS[f.TimestampSeconds] = f
	}
	out := make([]VideoFrame, 0, len(want))
	for _, t := range want {
		if f, ok := reusedByTS[t]; ok {
			out = append(out, f)
			continue
		}
		if f, ok := byTS[t]; ok {
			out = append(out, f)
		}
	}
	slog.Info("frame: additional extraction complete",
		"video_id", v.ID.String(),
		"requested", len(want),
		"extracted", len(saved),
		"adopted_from_perception", len(reused),
		"skipped_existing", len(want)-len(reused)-len(saved),
		"skipped_undecodable", len(fresh)-len(saved),
	)
	return out, nil
}

// applySemanticAdoptions persists perception→'both' promotions for timestamps
// the semantic refinement adopted. Adoptions are applied only after the fresh
// frames persisted successfully so a failure never leaves the semantic set
// holding frames the refinement did not fully materialize.
func (s *Service) applySemanticAdoptions(ctx context.Context, toPromote []VideoFrame) error {
	for _, f := range toPromote {
		if err := s.repo.UpdatePurpose(ctx, f.ID, PurposeBoth); err != nil {
			return fmt.Errorf("failed to adopt perception frame into semantic set: %w", err)
		}
	}
	return nil
}

func (s *Service) GetByVideoID(ctx context.Context, videoID uuid.UUID) ([]VideoFrame, error) {
	return s.repo.GetByVideoID(ctx, videoID)
}

// GetByVideoIDSet returns one frame set (the perception set consumed by
// production detection/tracking, or the semantic set consumed by the VLM)
// for checkpoint/resume checks and membership queries.
func (s *Service) GetByVideoIDSet(ctx context.Context, videoID uuid.UUID, set Set) ([]VideoFrame, error) {
	return s.repo.GetByVideoIDSet(ctx, videoID, set)
}

// GetBySegmentIDSet returns one frame set within a segment.
func (s *Service) GetBySegmentIDSet(ctx context.Context, segmentID uuid.UUID, set Set) ([]VideoFrame, error) {
	return s.repo.GetBySegmentIDSet(ctx, segmentID, set)
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
