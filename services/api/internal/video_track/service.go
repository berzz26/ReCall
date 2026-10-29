package video_track

import (
	"context"
	"fmt"
	"log/slog"
	"sort"
	"time"

	"github.com/berzz26/recall/services/api/internal/detection"
	"github.com/berzz26/recall/services/api/internal/tracker"
	"github.com/berzz26/recall/services/api/internal/video_frame"
	"github.com/google/uuid"
)

type Service struct {
	repo          *Repository
	frameRepo     *video_frame.Repository
	detectionRepo *detection.Repository
	tracker       tracker.Tracker
}

func NewService(repo *Repository, tr tracker.Tracker) *Service {
	if tr == nil {
		tr = tracker.NewIoUTracker()
	}
	return &Service{repo: repo, tracker: tr}
}

func NewServiceWithDeps(repo *Repository, frameRepo *video_frame.Repository, detectionRepo *detection.Repository, tr tracker.Tracker) *Service {
	if tr == nil {
		tr = tracker.NewIoUTracker()
	}
	return &Service{repo: repo, frameRepo: frameRepo, detectionRepo: detectionRepo, tracker: tr}
}

func (s *Service) GenerateForVideo(ctx context.Context, videoID uuid.UUID, frames []video_frame.VideoFrame, detections []detection.Detection) ([]Track, error) {
	trackStart := time.Now()
	if len(frames) == 0 {
		// delete existing and return empty
		if err := s.repo.DeleteByVideoID(ctx, videoID); err != nil {
			return nil, err
		}
		slog.Info("tracking: no frames, skipped", "video_id", videoID.String(), "duration_ms", time.Since(trackStart).Milliseconds())
		return []Track{}, nil
	}
	slog.Info("tracking: start", "video_id", videoID.String(), "frames", len(frames), "detections", len(detections), "tracker", func() string {
		if s.tracker != nil {
			return s.tracker.Name()
		}
		return "iou"
	}())
	// order frames by timestamp / frame_index
	sort.Slice(frames, func(i, j int) bool {
		if frames[i].TimestampSeconds == frames[j].TimestampSeconds {
			return frames[i].FrameIndex < frames[j].FrameIndex
		}
		return frames[i].TimestampSeconds < frames[j].TimestampSeconds
	})
	sort.Slice(detections, func(i, j int) bool {
		if detections[i].FrameID == detections[j].FrameID {
			return detections[i].Confidence > detections[j].Confidence
		}
		// find frame timestamp for ordering - we need map
		return detections[i].CreatedAt.Before(detections[j].CreatedAt)
	})

	// Build tracker inputs
	var frameInputs []tracker.FrameInput
	detsByFrame := make(map[uuid.UUID][]tracker.DetectionInput)
	frameMap := make(map[uuid.UUID]video_frame.VideoFrame)
	for _, f := range frames {
		frameInputs = append(frameInputs, tracker.FrameInput{FrameID: f.ID, Timestamp: f.TimestampSeconds})
		frameMap[f.ID] = f
	}
	for _, d := range detections {
		di := tracker.DetectionInput{
			ID: d.ID, Label: d.Label, Confidence: d.Confidence,
			BBoxX: d.BBoxX, BBoxY: d.BBoxY, BBoxWidth: d.BBoxWidth, BBoxHeight: d.BBoxHeight,
			FrameID: d.FrameID, Timestamp: 0,
		}
		if f, ok := frameMap[d.FrameID]; ok {
			di.Timestamp = f.TimestampSeconds
		}
		detsByFrame[d.FrameID] = append(detsByFrame[d.FrameID], di)
	}

	trackerStart := time.Now()
	assignments, err := s.tracker.Track(ctx, frameInputs, detsByFrame)
	trackerMs := time.Since(trackerStart).Milliseconds()
	if err != nil {
		slog.Error("tracking: tracker failed", "video_id", videoID.String(), "duration_ms", trackerMs, "error", err)
		return nil, fmt.Errorf("tracker failed: %w", err)
	}
	slog.Info("tracking: association complete",
		"video_id", videoID.String(),
		"frames", len(frameInputs),
		"detections", len(detections),
		"assignments", len(assignments),
		"tracker", trackerNameForLog(s.tracker),
		"duration_ms", trackerMs,
	)
	if len(detections) == 0 {
		if err := s.repo.DeleteByVideoID(ctx, videoID); err != nil {
			return nil, err
		}
		slog.Info("tracking: no detections, cleared tracks", "video_id", videoID.String(), "duration_ms", time.Since(trackStart).Milliseconds())
		return []Track{}, nil
	}
	// Group detections by trackIndex
	type trackInfo struct {
		Label     string
		Start     float64
		End       float64
		SegmentID *uuid.UUID
		Dets      []detection.Detection
	}
	trackGroups := make(map[int]*trackInfo)
	detectionByID := make(map[uuid.UUID]detection.Detection)
	for _, d := range detections {
		detectionByID[d.ID] = d
	}
	for detID, idx := range assignments {
		d, ok := detectionByID[detID]
		if !ok {
			continue
		}
		ti, ok := trackGroups[idx]
		if !ok {
			// find first frame's segment for this detection
			var segID *uuid.UUID
			if f, ok := frameMap[d.FrameID]; ok {
				sid := f.SegmentID
				segID = &sid
			}
			ti = &trackInfo{Label: d.Label, Start: 1e9, End: -1e9, SegmentID: segID}
			trackGroups[idx] = ti
		}
		if f, ok := frameMap[d.FrameID]; ok {
			if f.TimestampSeconds < ti.Start {
				ti.Start = f.TimestampSeconds
			}
			if f.TimestampSeconds > ti.End {
				ti.End = f.TimestampSeconds
			}
		}
		ti.Dets = append(ti.Dets, d)
	}

	// Build tracks and links
	trackerName := "iou"
	trackerVersion := "1"
	if s.tracker != nil {
		trackerName = s.tracker.Name()
		trackerVersion = s.tracker.Version()
	}
	var tracks []Track
	linksByIndex := make(map[int][]TrackDetection)
	for idx, info := range trackGroups {
		if len(info.Dets) == 0 {
			continue
		}
		// sort dets by timestamp for start/end already computed
		t := Track{
			VideoID:        videoID,
			SegmentID:      info.SegmentID,
			Label:          info.Label,
			TrackIndex:     idx,
			StartTimestamp: info.Start,
			EndTimestamp:   info.End,
			TrackerName:    trackerName,
			TrackerVersion: trackerVersion,
		}
		if t.StartTimestamp > t.EndTimestamp {
			t.StartTimestamp = info.Dets[0].CreatedAt.Sub(info.Dets[0].CreatedAt).Seconds()
		}
		tracks = append(tracks, t)
		var links []TrackDetection
		for _, d := range info.Dets {
			fid := d.FrameID
			var ts float64
			if f, ok := frameMap[fid]; ok {
				ts = f.TimestampSeconds
			}
			links = append(links, TrackDetection{
				DetectionID:      d.ID,
				FrameID:          fid,
				TimestampSeconds: ts,
			})
		}
		linksByIndex[idx] = links
	}
	// Sort tracks by index for deterministic
	sort.Slice(tracks, func(i, j int) bool { return tracks[i].TrackIndex < tracks[j].TrackIndex })

	persistStart := time.Now()
	saved, err := s.repo.ReplaceForVideo(ctx, videoID, tracks, linksByIndex)
	persistMs := time.Since(persistStart).Milliseconds()
	if err != nil {
		slog.Error("tracking: persist failed", "video_id", videoID.String(), "duration_ms", persistMs, "error", err)
		return nil, err
	}
	totalMs := time.Since(trackStart).Milliseconds()
	slog.Info("tracking: complete",
		"video_id", videoID.String(),
		"tracks", len(saved),
		"tracker", trackerNameForLog(s.tracker),
		"tracker_ms", trackerMs,
		"persist_ms", persistMs,
		"total_duration_ms", totalMs,
	)
	return saved, nil
}

func (s *Service) GenerateForVideoID(ctx context.Context, videoID uuid.UUID) ([]Track, error) {
	if s.frameRepo == nil || s.detectionRepo == nil {
		return nil, fmt.Errorf("tracking deps not configured")
	}
	// Tracking reads the full perception timeline: every frame of the
	// perception set (regular fixed-rate cadence) plus the production
	// detections attached to those frames. Semantic-only rows are not
	// tracked.
	frames, err := s.frameRepo.GetByVideoIDSet(ctx, videoID, video_frame.SetPerception)
	if err != nil {
		return nil, err
	}
	dets, err := s.detectionRepo.GetByVideoID(ctx, videoID)
	if err != nil {
		return nil, err
	}
	return s.GenerateForVideo(ctx, videoID, frames, dets)
}

func trackerNameForLog(tr tracker.Tracker) string {
	if tr == nil {
		return "iou"
	}
	return tr.Name()
}

func (s *Service) GetByVideoID(ctx context.Context, videoID uuid.UUID) ([]Track, error) {
	return s.repo.GetByVideoID(ctx, videoID)
}

func (s *Service) GetWithCounts(ctx context.Context, videoID uuid.UUID) ([]TrackWithCount, error) {
	return s.repo.GetTracksWithCounts(ctx, videoID)
}

func (s *Service) GetDetectionsByTrackID(ctx context.Context, trackID uuid.UUID) ([]TrackDetection, error) {
	return s.repo.GetDetectionsByTrackID(ctx, trackID)
}

func (s *Service) DeleteByVideoID(ctx context.Context, videoID uuid.UUID) error {
	return s.repo.DeleteByVideoID(ctx, videoID)
}
