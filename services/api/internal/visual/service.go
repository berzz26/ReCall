package visual

import (
	"context"
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"time"

	"github.com/berzz26/recall/services/api/internal/detection"
	"github.com/berzz26/recall/services/api/internal/detector"
	"github.com/berzz26/recall/services/api/internal/storage"
	"github.com/berzz26/recall/services/api/internal/tracker"
	"github.com/berzz26/recall/services/api/internal/video_frame"
	"github.com/google/uuid"
)

// 13-class whitelist for B.2 - exact allowed YOLO classes (defined once, not user-configurable)
var allowedClasses = map[string]struct{}{
	"person":     {},
	"car":        {},
	"motorcycle": {},
	"bus":        {},
	"truck":      {},
	"bicycle":    {},
	"backpack":   {},
	"handbag":    {},
	"suitcase":   {},
	"dog":        {},
	"cat":        {},
	"cell phone": {},
	"knife":      {},
}

type Service struct {
	detectionRepo   *detection.Repository
	frameRepo       *video_frame.Repository
	storage         storage.Storage
	analyzer        detector.VisualAnalyzer
	threshold       float64
	detectorName    string
	detectorVersion string
	batchSize       int
}

func NewService(detRepo *detection.Repository, frameRepo *video_frame.Repository, store storage.Storage, analyzer detector.VisualAnalyzer, threshold float64, name, version string) *Service {
	return NewServiceWithBatchSize(detRepo, frameRepo, store, analyzer, threshold, name, version, 16)
}

func NewServiceWithBatchSize(detRepo *detection.Repository, frameRepo *video_frame.Repository, store storage.Storage, analyzer detector.VisualAnalyzer, threshold float64, name, version string, batchSize int) *Service {
	if threshold < 0 || threshold > 1 {
		threshold = 0.35
	}
	if name == "" {
		name = "yolov8n"
	}
	if version == "" {
		version = "1"
	}
	if batchSize <= 0 {
		batchSize = 16
	}
	return &Service{detectionRepo: detRepo, frameRepo: frameRepo, storage: store, analyzer: analyzer, threshold: threshold, detectorName: name, detectorVersion: version, batchSize: batchSize}
}

func (s *Service) AnalyzeVideo(ctx context.Context, videoID uuid.UUID) ([]detection.Detection, error) {
	visualStart := time.Now()
	frames, err := s.frameRepo.GetByVideoID(ctx, videoID)
	if err != nil {
		return nil, fmt.Errorf("get frames: %w", err)
	}
	if err := s.detectionRepo.DeleteByVideoID(ctx, videoID); err != nil {
		return nil, fmt.Errorf("delete old detections: %w", err)
	}
	if len(frames) == 0 {
		slog.Info("visual: no frames, skipped", "video_id", videoID.String(), "duration_ms", time.Since(visualStart).Milliseconds())
		return []detection.Detection{}, nil
	}
	if s.analyzer == nil {
		return nil, fmt.Errorf("analyzer not configured")
	}
	batchSize := s.batchSize
	if batchSize <= 0 {
		batchSize = 16
	}
	totalBatches := (len(frames) + batchSize - 1) / batchSize
	slog.Info("visual: start", "video_id", videoID.String(), "frames", len(frames), "threshold", s.threshold, "detector", s.detectorName, "batch_size", batchSize, "batches", totalBatches)

	// Load YOLO once per detection (per video) when using real detector.
	// Persistent session keeps model loaded across bounded batches, preserving bounded memory
	// while eliminating per-batch model reload overhead.
	var sess *detector.Session
	var useSession bool
	if yolo, ok := s.analyzer.(*detector.YoloDetector); ok {
		var err error
		sess, err = yolo.NewSession(ctx)
		if err != nil {
			return nil, fmt.Errorf("create detector session: %w", err)
		}
		useSession = true
		defer func() {
			if cerr := sess.Close(); cerr != nil {
				slog.Warn("visual: failed to close detector session", "video_id", videoID.String(), "error", cerr)
			}
		}()
		slog.Info("visual: persistent detector session started", "video_id", videoID.String(), "batches", totalBatches)
	}

	var allSaved []detection.Detection
	var totalAnalyzeMs int64
	var totalPersistMs int64

	for start := 0; start < len(frames); start += batchSize {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		end := start + batchSize
		if end > len(frames) {
			end = len(frames)
		}
		batchFrames := frames[start:end]
		batchNum := start/batchSize + 1

		type tmpFile struct {
			path  string
			frame video_frame.VideoFrame
		}
		var tmps []tmpFile
		var inputs []detector.FrameInput

		cleanup := func() {
			for _, t := range tmps {
				_ = os.Remove(t.path)
			}
		}

		// Materialize bounded batch of frames from storage.
		batchPrepStart := time.Now()
		materializeFailed := false
		var materializeErr error
		for _, f := range batchFrames {
			rc, err := s.storage.Open(ctx, f.StorageKey)
			if err != nil {
				materializeErr = fmt.Errorf("open frame %s: %w", f.ID, err)
				materializeFailed = true
				break
			}
			ext := filepath.Ext(f.StorageKey)
			if ext == "" {
				ext = ".jpg"
			}
			tmp, err := os.CreateTemp("", "visual-*"+ext)
			if err != nil {
				rc.Close()
				materializeErr = err
				materializeFailed = true
				break
			}
			if _, err := io.Copy(tmp, rc); err != nil {
				tmp.Close()
				rc.Close()
				_ = os.Remove(tmp.Name())
				materializeErr = fmt.Errorf("copy frame %s: %w", f.ID, err)
				materializeFailed = true
				break
			}
			tmp.Close()
			rc.Close()
			p := tmp.Name()
			tmps = append(tmps, tmpFile{path: p, frame: f})
			inputs = append(inputs, detector.FrameInput{
				FrameID: f.ID, VideoID: f.VideoID, SegmentID: f.SegmentID, Timestamp: f.TimestampSeconds, Width: f.Width, Height: f.Height, StorageKey: f.StorageKey, LocalPath: p,
			})
		}
		if materializeFailed {
			cleanup()
			return nil, materializeErr
		}
		prepMs := time.Since(batchPrepStart).Milliseconds()
		slog.Info("visual: batch frames prepared", "video_id", videoID.String(), "batch", batchNum, "batches", totalBatches, "frames", len(inputs), "prep_duration_ms", prepMs)

		analyzeStart := time.Now()
		var results map[uuid.UUID][]detector.DetectionResult
		if useSession {
			results, err = sess.AnalyzeBatch(ctx, inputs)
		} else {
			results, err = s.analyzer.AnalyzeBatch(ctx, inputs)
		}
		analyzeMs := time.Since(analyzeStart).Milliseconds()
		totalAnalyzeMs += analyzeMs
		if err != nil {
			cleanup()
			slog.Error("visual: batch detection failed", "video_id", videoID.String(), "batch", batchNum, "batches", totalBatches, "duration_ms", analyzeMs, "error", err)
			return nil, fmt.Errorf("analyze batch %d: %w", batchNum, err)
		}
		slog.Info("visual: batch detection complete", "video_id", videoID.String(), "batch", batchNum, "batches", totalBatches, "frames", len(inputs), "duration_ms", analyzeMs)

		// Release temporary JPEGs and batch memory before persistence; detector has finished reading them.
		cleanup()
		// Clear tmps to avoid double-remove and allow GC; inputs still needed for mapping but will be dropped after this batch.
		tmps = nil

		var toInsert []detection.Detection
		for _, f := range batchFrames {
			dets := results[f.ID]
			for _, r := range dets {
				if r.Label == "" {
					continue
				}
				if _, ok := allowedClasses[r.Label]; !ok {
					continue
				}
				if r.Confidence < s.threshold {
					continue
				}
				x, y, w, h := detection.ClampBBox(r.BBoxX, r.BBoxY, r.BBoxWidth, r.BBoxHeight)
				if w <= 0 || h <= 0 {
					continue
				}
				if r.Confidence < 0 || r.Confidence > 1 {
					continue
				}
				toInsert = append(toInsert, detection.Detection{
					VideoID: f.VideoID, SegmentID: f.SegmentID, FrameID: f.ID, Label: r.Label, Confidence: r.Confidence,
					BBoxX: x, BBoxY: y, BBoxWidth: w, BBoxHeight: h,
					DetectorName: s.detectorName, DetectorVersion: s.detectorVersion,
				})
			}
		}

		persistStart := time.Now()
		saved, err := s.detectionRepo.CreateBatch(ctx, toInsert)
		persistMs := time.Since(persistStart).Milliseconds()
		totalPersistMs += persistMs
		if err != nil {
			slog.Error("visual: batch persist failed", "video_id", videoID.String(), "batch", batchNum, "batches", totalBatches, "duration_ms", persistMs, "error", err)
			return nil, fmt.Errorf("persist detections batch %d: %w", batchNum, err)
		}
		// Do not hold previous batch's temporary frames/inputs; release explicitly.
		inputs = nil
		toInsert = nil
		allSaved = append(allSaved, saved...)
		_ = prepMs
	}

	totalMs := time.Since(visualStart).Milliseconds()
	slog.Info("visual: complete",
		"video_id", videoID.String(),
		"frames", len(frames),
		"detections", len(allSaved),
		"batches", totalBatches,
		"batch_size", batchSize,
		"analyze_ms", totalAnalyzeMs,
		"persist_ms", totalPersistMs,
		"total_duration_ms", totalMs,
	)
	return allSaved, nil
}

func (s *Service) NewDetectionSession(ctx context.Context) (*detector.Session, error) {
	if yolo, ok := s.analyzer.(*detector.YoloDetector); ok {
		return yolo.NewSession(ctx)
	}
	return nil, fmt.Errorf("analyzer is not YoloDetector")
}

func (s *Service) AnalyzeSegment(ctx context.Context, videoID, segmentID uuid.UUID) ([]detection.Detection, error) {
	frames, err := s.frameRepo.GetBySegmentID(ctx, segmentID)
	if err != nil {
		return nil, fmt.Errorf("get frames for segment %s: %w", segmentID, err)
	}
	if len(frames) == 0 {
		slog.Info("visual: no frames for segment, skipped", "video_id", videoID.String(), "segment_id", segmentID.String())
		// Still delete old detections for this segment to keep idempotent
		if err := s.detectionRepo.DeleteBySegmentID(ctx, segmentID); err != nil {
			return nil, fmt.Errorf("delete old detections for segment: %w", err)
		}
		return []detection.Detection{}, nil
	}
	if err := s.detectionRepo.DeleteBySegmentID(ctx, segmentID); err != nil {
		return nil, fmt.Errorf("delete old detections for segment: %w", err)
	}
	if s.analyzer == nil {
		return nil, fmt.Errorf("analyzer not configured")
	}
	return s.analyzeFrames(ctx, videoID, frames)
}

func (s *Service) AnalyzeSegmentWithSession(ctx context.Context, videoID, segmentID uuid.UUID, sess *detector.Session) ([]detection.Detection, error) {
	frames, err := s.frameRepo.GetBySegmentID(ctx, segmentID)
	if err != nil {
		return nil, fmt.Errorf("get frames for segment %s: %w", segmentID, err)
	}
	if len(frames) == 0 {
		slog.Info("visual: no frames for segment, skipped", "video_id", videoID.String(), "segment_id", segmentID.String())
		if err := s.detectionRepo.DeleteBySegmentID(ctx, segmentID); err != nil {
			return nil, fmt.Errorf("delete old detections for segment: %w", err)
		}
		return []detection.Detection{}, nil
	}
	if err := s.detectionRepo.DeleteBySegmentID(ctx, segmentID); err != nil {
		return nil, fmt.Errorf("delete old detections for segment: %w", err)
	}
	if s.analyzer == nil {
		return nil, fmt.Errorf("analyzer not configured")
	}
	return s.analyzeFramesWithSession(ctx, videoID, frames, sess)
}

// DetectFrames runs planning-only detection over exactly the given frames
// and returns tracker-ready inputs WITHOUT persisting anything. Used by the
// adaptive sampler's refinement loop: planning detections are ephemeral
// (in-memory) and must never be confused with production detections, which
// the pipeline's visual stage produces separately over the final frame set.
// Filtering (13-class whitelist, confidence threshold, bbox clamp) mirrors
// analyzeFramesWithSessionInternal exactly so planning judgments match what
// production YOLO will report for the same frames.
func (s *Service) DetectFrames(ctx context.Context, videoID uuid.UUID, frames []video_frame.VideoFrame) ([]tracker.DetectionInput, error) {
	detectStart := time.Now()
	if len(frames) == 0 {
		return []tracker.DetectionInput{}, nil
	}
	if s.analyzer == nil {
		return nil, fmt.Errorf("analyzer not configured")
	}
	batchSize := s.batchSize
	if batchSize <= 0 {
		batchSize = 16
	}

	var sess *detector.Session
	var useSession bool
	if yolo, ok := s.analyzer.(*detector.YoloDetector); ok {
		var err error
		sess, err = yolo.NewSession(ctx)
		if err != nil {
			return nil, fmt.Errorf("create detector session: %w", err)
		}
		useSession = true
		defer func() {
			if cerr := sess.Close(); cerr != nil {
				slog.Warn("visual: failed to close planning detector session", "video_id", videoID.String(), "error", cerr)
			}
		}()
	}

	var out []tracker.DetectionInput
	for start := 0; start < len(frames); start += batchSize {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		end := start + batchSize
		if end > len(frames) {
			end = len(frames)
		}
		batchFrames := frames[start:end]

		var tmps []string
		var inputs []detector.FrameInput
		cleanup := func() {
			for _, p := range tmps {
				_ = os.Remove(p)
			}
		}
		failed := false
		var matErr error
		for _, f := range batchFrames {
			rc, err := s.storage.Open(ctx, f.StorageKey)
			if err != nil {
				matErr = fmt.Errorf("open frame %s: %w", f.ID, err)
				failed = true
				break
			}
			ext := filepath.Ext(f.StorageKey)
			if ext == "" {
				ext = ".jpg"
			}
			tmp, err := os.CreateTemp("", "visual-plan-*"+ext)
			if err != nil {
				rc.Close()
				matErr = err
				failed = true
				break
			}
			if _, err := io.Copy(tmp, rc); err != nil {
				tmp.Close()
				rc.Close()
				_ = os.Remove(tmp.Name())
				matErr = fmt.Errorf("copy frame %s: %w", f.ID, err)
				failed = true
				break
			}
			tmp.Close()
			rc.Close()
			tmps = append(tmps, tmp.Name())
			inputs = append(inputs, detector.FrameInput{
				FrameID: f.ID, VideoID: f.VideoID, SegmentID: f.SegmentID, Timestamp: f.TimestampSeconds, Width: f.Width, Height: f.Height, StorageKey: f.StorageKey, LocalPath: tmp.Name(),
			})
		}
		if failed {
			cleanup()
			return nil, matErr
		}
		var results map[uuid.UUID][]detector.DetectionResult
		var err error
		if useSession {
			results, err = sess.AnalyzeBatch(ctx, inputs)
		} else {
			results, err = s.analyzer.AnalyzeBatch(ctx, inputs)
		}
		cleanup()
		if err != nil {
			return nil, fmt.Errorf("planning analyze batch: %w", err)
		}
		for _, f := range batchFrames {
			for _, r := range results[f.ID] {
				if r.Label == "" {
					continue
				}
				if _, ok := allowedClasses[r.Label]; !ok {
					continue
				}
				if r.Confidence < s.threshold {
					continue
				}
				x, y, w, h := detection.ClampBBox(r.BBoxX, r.BBoxY, r.BBoxWidth, r.BBoxHeight)
				if w <= 0 || h <= 0 {
					continue
				}
				if r.Confidence < 0 || r.Confidence > 1 {
					continue
				}
				out = append(out, tracker.DetectionInput{
					ID: uuid.New(), Label: r.Label, Confidence: r.Confidence,
					BBoxX: x, BBoxY: y, BBoxWidth: w, BBoxHeight: h,
					FrameID: f.ID, Timestamp: f.TimestampSeconds,
				})
			}
		}
	}
	slog.Info("visual: planning detection complete (ephemeral, not persisted)",
		"video_id", videoID.String(),
		"frames", len(frames),
		"detections", len(out),
		"duration_ms", time.Since(detectStart).Milliseconds(),
	)
	return out, nil
}

func (s *Service) analyzeFrames(ctx context.Context, videoID uuid.UUID, frames []video_frame.VideoFrame) ([]detection.Detection, error) {
	visualStart := time.Now()
	if len(frames) == 0 {
		return []detection.Detection{}, nil
	}
	batchSize := s.batchSize
	if batchSize <= 0 {
		batchSize = 16
	}
	totalBatches := (len(frames) + batchSize - 1) / batchSize
	slog.Info("visual: segment start", "video_id", videoID.String(), "frames", len(frames), "batches", totalBatches)

	var sess *detector.Session
	var useSession bool
	if yolo, ok := s.analyzer.(*detector.YoloDetector); ok {
		var err error
		sess, err = yolo.NewSession(ctx)
		if err != nil {
			return nil, fmt.Errorf("create detector session: %w", err)
		}
		useSession = true
		defer func() {
			if cerr := sess.Close(); cerr != nil {
				slog.Warn("visual: failed to close detector session", "video_id", videoID.String(), "error", cerr)
			}
		}()
	}
	return s.analyzeFramesWithSessionInternal(ctx, videoID, frames, sess, useSession, visualStart)
}

func (s *Service) analyzeFramesWithSession(ctx context.Context, videoID uuid.UUID, frames []video_frame.VideoFrame, sess *detector.Session) ([]detection.Detection, error) {
	visualStart := time.Now()
	if len(frames) == 0 {
		return []detection.Detection{}, nil
	}
	useSession := sess != nil
	return s.analyzeFramesWithSessionInternal(ctx, videoID, frames, sess, useSession, visualStart)
}

func (s *Service) analyzeFramesWithSessionInternal(ctx context.Context, videoID uuid.UUID, frames []video_frame.VideoFrame, sess *detector.Session, useSession bool, visualStart time.Time) ([]detection.Detection, error) {
	batchSize := s.batchSize
	if batchSize <= 0 {
		batchSize = 16
	}
	totalBatches := (len(frames) + batchSize - 1) / batchSize
	var allSaved []detection.Detection
	var totalAnalyzeMs int64
	var totalPersistMs int64

	for start := 0; start < len(frames); start += batchSize {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		end := start + batchSize
		if end > len(frames) {
			end = len(frames)
		}
		batchFrames := frames[start:end]
		batchNum := start/batchSize + 1

		type tmpFile struct {
			path  string
			frame video_frame.VideoFrame
		}
		var tmps []tmpFile
		var inputs []detector.FrameInput

		cleanup := func() {
			for _, t := range tmps {
				_ = os.Remove(t.path)
			}
		}

		batchPrepStart := time.Now()
		materializeFailed := false
		var materializeErr error
		for _, f := range batchFrames {
			rc, err := s.storage.Open(ctx, f.StorageKey)
			if err != nil {
				materializeErr = fmt.Errorf("open frame %s: %w", f.ID, err)
				materializeFailed = true
				break
			}
			ext := filepath.Ext(f.StorageKey)
			if ext == "" {
				ext = ".jpg"
			}
			tmp, err := os.CreateTemp("", "visual-*"+ext)
			if err != nil {
				rc.Close()
				materializeErr = err
				materializeFailed = true
				break
			}
			if _, err := io.Copy(tmp, rc); err != nil {
				tmp.Close()
				rc.Close()
				_ = os.Remove(tmp.Name())
				materializeErr = fmt.Errorf("copy frame %s: %w", f.ID, err)
				materializeFailed = true
				break
			}
			tmp.Close()
			rc.Close()
			p := tmp.Name()
			tmps = append(tmps, tmpFile{path: p, frame: f})
			inputs = append(inputs, detector.FrameInput{
				FrameID: f.ID, VideoID: f.VideoID, SegmentID: f.SegmentID, Timestamp: f.TimestampSeconds, Width: f.Width, Height: f.Height, StorageKey: f.StorageKey, LocalPath: p,
			})
		}
		if materializeFailed {
			cleanup()
			return nil, materializeErr
		}
		prepMs := time.Since(batchPrepStart).Milliseconds()
		_ = prepMs

		analyzeStart := time.Now()
		var results map[uuid.UUID][]detector.DetectionResult
		var err error
		if useSession {
			results, err = sess.AnalyzeBatch(ctx, inputs)
		} else {
			results, err = s.analyzer.AnalyzeBatch(ctx, inputs)
		}
		analyzeMs := time.Since(analyzeStart).Milliseconds()
		totalAnalyzeMs += analyzeMs
		if err != nil {
			cleanup()
			slog.Error("visual: segment batch detection failed", "video_id", videoID.String(), "batch", batchNum, "batches", totalBatches, "duration_ms", analyzeMs, "error", err)
			return nil, fmt.Errorf("analyze batch %d: %w", batchNum, err)
		}

		cleanup()
		tmps = nil

		var toInsert []detection.Detection
		for _, f := range batchFrames {
			dets := results[f.ID]
			for _, r := range dets {
				if r.Label == "" {
					continue
				}
				if _, ok := allowedClasses[r.Label]; !ok {
					continue
				}
				if r.Confidence < s.threshold {
					continue
				}
				x, y, w, h := detection.ClampBBox(r.BBoxX, r.BBoxY, r.BBoxWidth, r.BBoxHeight)
				if w <= 0 || h <= 0 {
					continue
				}
				if r.Confidence < 0 || r.Confidence > 1 {
					continue
				}
				toInsert = append(toInsert, detection.Detection{
					VideoID: f.VideoID, SegmentID: f.SegmentID, FrameID: f.ID, Label: r.Label, Confidence: r.Confidence,
					BBoxX: x, BBoxY: y, BBoxWidth: w, BBoxHeight: h,
					DetectorName: s.detectorName, DetectorVersion: s.detectorVersion,
				})
			}
		}

		persistStart := time.Now()
		saved, err := s.detectionRepo.CreateBatch(ctx, toInsert)
		persistMs := time.Since(persistStart).Milliseconds()
		totalPersistMs += persistMs
		if err != nil {
			slog.Error("visual: segment batch persist failed", "video_id", videoID.String(), "batch", batchNum, "batches", totalBatches, "duration_ms", persistMs, "error", err)
			return nil, fmt.Errorf("persist detections batch %d: %w", batchNum, err)
		}
		inputs = nil
		toInsert = nil
		allSaved = append(allSaved, saved...)
	}
	totalMs := time.Since(visualStart).Milliseconds()
	slog.Info("visual: segment complete",
		"video_id", videoID.String(),
		"frames", len(frames),
		"detections", len(allSaved),
		"batches", totalBatches,
		"analyze_ms", totalAnalyzeMs,
		"persist_ms", totalPersistMs,
		"total_duration_ms", totalMs,
	)
	return allSaved, nil
}

func (s *Service) GetByVideoID(ctx context.Context, videoID uuid.UUID) ([]detection.Detection, error) {
	return s.detectionRepo.GetByVideoID(ctx, videoID)
}

func (s *Service) DeleteByVideoID(ctx context.Context, videoID uuid.UUID) error {
	return s.detectionRepo.DeleteByVideoID(ctx, videoID)
}
