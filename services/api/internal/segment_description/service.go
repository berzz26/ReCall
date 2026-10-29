package segment_description

import (
	"context"
	"fmt"
	"log/slog"
	"sort"
	"strings"
	"time"

	"github.com/berzz26/recall/services/api/internal/detection"
	"github.com/berzz26/recall/services/api/internal/video_event"
	"github.com/berzz26/recall/services/api/internal/video_frame"
	"github.com/berzz26/recall/services/api/internal/video_segment"
	"github.com/berzz26/recall/services/api/internal/video_track"
	"github.com/berzz26/recall/services/api/internal/vision"
	"github.com/google/uuid"
)

type Service struct {
	repo                 *Repository
	segmentRepo          *video_segment.Repository
	frameRepo            *video_frame.Repository
	detectionRepo        *detection.Repository
	trackRepo            *video_track.Repository
	eventRepo            *video_event.Repository
	describer            vision.VisionDescriber
	expectedModelName    string
	expectedModelVersion string
	historySegments      int
	historyEvents        int
}

func NewService(repo *Repository, segRepo *video_segment.Repository, frameRepo *video_frame.Repository, detRepo *detection.Repository, trackRepo *video_track.Repository, eventRepo *video_event.Repository, describer vision.VisionDescriber, modelName, modelVersion string) *Service {
	return NewServiceWithHistory(repo, segRepo, frameRepo, detRepo, trackRepo, eventRepo, describer, modelName, modelVersion, 2, 5)
}

func NewServiceWithHistory(repo *Repository, segRepo *video_segment.Repository, frameRepo *video_frame.Repository, detRepo *detection.Repository, trackRepo *video_track.Repository, eventRepo *video_event.Repository, describer vision.VisionDescriber, modelName, modelVersion string, historySegments, historyEvents int) *Service {
	if historySegments < 0 {
		historySegments = 0
	}
	if historyEvents < 0 {
		historyEvents = 0
	}
	return &Service{repo: repo, segmentRepo: segRepo, frameRepo: frameRepo, detectionRepo: detRepo, trackRepo: trackRepo, eventRepo: eventRepo, describer: describer, expectedModelName: modelName, expectedModelVersion: modelVersion, historySegments: historySegments, historyEvents: historyEvents}
}

func (s *Service) GenerateForVideo(ctx context.Context, videoID uuid.UUID) ([]Description, error) {
	svcStart := time.Now()
	if s.describer == nil {
		return nil, fmt.Errorf("vision describer not configured")
	}
	segments, err := s.segmentRepo.GetByVideoID(ctx, videoID)
	if err != nil {
		return nil, err
	}
	if len(segments) == 0 {
		slog.Info("segment_description: no segments, skipping VLM", "video_id", videoID.String(), "duration_ms", time.Since(svcStart).Milliseconds())
		return s.repo.ReplaceForVideo(ctx, videoID, nil)
	}
	slog.Info("segment_description: start", "video_id", videoID.String(), "segments", len(segments))
	// VLM inputs are assembled from the SEMANTIC frame set only — the
	// adaptive sampler decides which timestamps deserve description.
	frames, err := s.frameRepo.GetByVideoIDSet(ctx, videoID, video_frame.SetSemantic)
	if err != nil {
		return nil, err
	}
	perceptionFrames, err := s.frameRepo.GetByVideoIDSet(ctx, videoID, video_frame.SetPerception)
	if err != nil {
		return nil, err
	}
	dets, err := s.detectionRepo.GetByVideoID(ctx, videoID)
	if err != nil {
		return nil, err
	}
	tracksWithCounts, err := s.trackRepo.GetTracksWithCounts(ctx, videoID)
	if err != nil {
		return nil, err
	}
	events, err := s.eventRepo.GetByVideoID(ctx, videoID, "", "", nil)
	if err != nil {
		return nil, err
	}

	// frameByID resolves the timestamps of production detections and
	// persistence-line history labels. Detections live on the perception
	// timeline, which may contain frames the semantic set never selected;
	// the map is therefore the union of both sets, and no detection
	// timestamp is assumed from semantic frames it is not attached to.
	frameByID := map[uuid.UUID]video_frame.VideoFrame{}
	for _, f := range frames {
		frameByID[f.ID] = f
	}
	for _, f := range perceptionFrames {
		frameByID[f.ID] = f
	}

	var segInputs []vision.SegmentInput
	for idx, seg := range segments {
		var segFrames []vision.FrameInput
		for _, f := range frames {
			if f.SegmentID == seg.ID || (f.TimestampSeconds >= seg.StartTime && f.TimestampSeconds < seg.EndTime) {
				segFrames = append(segFrames, vision.FrameInput{ID: f.ID, Timestamp: f.TimestampSeconds, StorageKey: f.StorageKey})
			}
		}
		if len(segFrames) == 0 {
			return nil, fmt.Errorf("segment %d has no frames; cannot describe without visual evidence", seg.SegmentIndex)
		}
		var segDets []vision.DetectionInput
		for _, d := range dets {
			f, ok := frameByID[d.FrameID]
			if !ok {
				continue
			}
			if f.TimestampSeconds >= seg.StartTime && f.TimestampSeconds < seg.EndTime {
				segDets = append(segDets, vision.DetectionInput{Label: d.Label, FrameID: d.FrameID, Timestamp: f.TimestampSeconds})
			}
		}
		var segTracks []vision.TrackInput
		for _, tr := range tracksWithCounts {
			if tr.EndTimestamp < seg.StartTime || tr.StartTimestamp >= seg.EndTime {
				continue
			}
			segTracks = append(segTracks, vision.TrackInput{Label: tr.Label, TrackIndex: tr.TrackIndex, Start: tr.StartTimestamp, End: tr.EndTimestamp})
		}
		sort.Slice(segTracks, func(i, j int) bool {
			if segTracks[i].Start == segTracks[j].Start {
				return segTracks[i].TrackIndex < segTracks[j].TrackIndex
			}
			return segTracks[i].Start < segTracks[j].Start
		})
		var segEvents []vision.EventInput
		for _, e := range events {
			eEnd := e.StartTimestamp
			if e.EndTimestamp != nil {
				eEnd = *e.EndTimestamp
			}
			if eEnd < seg.StartTime || e.StartTimestamp >= seg.EndTime {
				continue
			}
			segEvents = append(segEvents, vision.EventInput{EventType: e.EventType, Label: e.Label, Start: e.StartTimestamp, End: e.EndTimestamp})
		}
		sort.Slice(segEvents, func(i, j int) bool {
			if segEvents[i].Start == segEvents[j].Start {
				return segEvents[i].EventType < segEvents[j].EventType
			}
			return segEvents[i].Start < segEvents[j].Start
		})

		// Build temporal history: bounded recent prior segments + persistent tracks/events for VLM context
		// Bound prior segments to most recent historySegments (configurable, default 2) to avoid unbounded prompt growth.
		// Persistent tracks remain available even if started many segments earlier (required for continuity).
		var priorSegments []vision.HistorySegment
		startIdx := 0
		if s.historySegments >= 0 && idx > s.historySegments {
			startIdx = idx - s.historySegments
		} else if s.historySegments == 0 {
			startIdx = idx // no prior segments when limit 0
		}
		for j := startIdx; j < idx; j++ {
			priorSeg := segments[j]
			// distinct labels in prior segment
			labelSet := map[string]struct{}{}
			for _, d := range dets {
				f, ok := frameByID[d.FrameID]
				if !ok {
					continue
				}
				if f.TimestampSeconds >= priorSeg.StartTime && f.TimestampSeconds < priorSeg.EndTime {
					if d.Label != "" {
						labelSet[d.Label] = struct{}{}
					}
				}
			}
			var pLabels []string
			for l := range labelSet {
				pLabels = append(pLabels, l)
			}
			sort.Strings(pLabels)
			// tracks overlapping prior segment
			var pTracks []vision.TrackInput
			for _, tr := range tracksWithCounts {
				if tr.EndTimestamp < priorSeg.StartTime || tr.StartTimestamp >= priorSeg.EndTime {
					continue
				}
				pTracks = append(pTracks, vision.TrackInput{Label: tr.Label, TrackIndex: tr.TrackIndex, Start: tr.StartTimestamp, End: tr.EndTimestamp})
			}
			sort.Slice(pTracks, func(a, b int) bool {
				if pTracks[a].Start == pTracks[b].Start {
					return pTracks[a].TrackIndex < pTracks[b].TrackIndex
				}
				return pTracks[a].Start < pTracks[b].Start
			})
			// events overlapping prior segment
			var pEvents []vision.EventInput
			for _, e := range events {
				eEnd := e.StartTimestamp
				if e.EndTimestamp != nil {
					eEnd = *e.EndTimestamp
				}
				if eEnd < priorSeg.StartTime || e.StartTimestamp >= priorSeg.EndTime {
					continue
				}
				pEvents = append(pEvents, vision.EventInput{EventType: e.EventType, Label: e.Label, Start: e.StartTimestamp, End: e.EndTimestamp})
			}
			sort.Slice(pEvents, func(a, b int) bool {
				if pEvents[a].Start == pEvents[b].Start {
					return pEvents[a].EventType < pEvents[b].EventType
				}
				return pEvents[a].Start < pEvents[b].Start
			})
			if pTracks == nil {
				pTracks = []vision.TrackInput{}
			}
			if pEvents == nil {
				pEvents = []vision.EventInput{}
			}
			if pLabels == nil {
				pLabels = []string{}
			}
			priorSegments = append(priorSegments, vision.HistorySegment{
				SegmentID: priorSeg.ID, SegmentIndex: priorSeg.SegmentIndex,
				StartTime: priorSeg.StartTime, EndTime: priorSeg.EndTime,
				Labels: pLabels, Tracks: pTracks, Events: pEvents,
			})
		}
		if priorSegments == nil {
			priorSegments = []vision.HistorySegment{}
		}
		// Persistent tracks: those that started before this segment and are still relevant for continuity.
		// Remains available even if started many segments earlier (required for "the same person continues..." language).
		var persistentTracks []vision.TrackInput
		for _, tr := range tracksWithCounts {
			if tr.StartTimestamp < seg.StartTime && tr.EndTimestamp >= seg.StartTime {
				persistentTracks = append(persistentTracks, vision.TrackInput{Label: tr.Label, TrackIndex: tr.TrackIndex, Start: tr.StartTimestamp, End: tr.EndTimestamp})
			}
		}
		sort.Slice(persistentTracks, func(a, b int) bool {
			if persistentTracks[a].Start == persistentTracks[b].Start {
				return persistentTracks[a].TrackIndex < persistentTracks[b].TrackIndex
			}
			return persistentTracks[a].Start < persistentTracks[b].Start
		})
		if persistentTracks == nil {
			persistentTracks = []vision.TrackInput{}
		}
		// Prior events: most recent historyEvents before this segment (bounded)
		var priorEvents []vision.EventInput
		for _, e := range events {
			if e.StartTimestamp < seg.StartTime {
				priorEvents = append(priorEvents, vision.EventInput{EventType: e.EventType, Label: e.Label, Start: e.StartTimestamp, End: e.EndTimestamp})
			}
		}
		sort.Slice(priorEvents, func(a, b int) bool {
			if priorEvents[a].Start == priorEvents[b].Start {
				return priorEvents[a].EventType < priorEvents[b].EventType
			}
			return priorEvents[a].Start < priorEvents[b].Start
		})
		if s.historyEvents >= 0 && len(priorEvents) > s.historyEvents {
			priorEvents = priorEvents[len(priorEvents)-s.historyEvents:]
		}
		if priorEvents == nil {
			priorEvents = []vision.EventInput{}
		}
		history := &vision.SegmentHistory{
			PriorSegments:    priorSegments,
			PersistentTracks: persistentTracks,
			PriorEvents:      priorEvents,
		}
		segInputs = append(segInputs, vision.SegmentInput{
			SegmentID: seg.ID, StartTime: seg.StartTime, EndTime: seg.EndTime,
			Frames: segFrames, Detections: segDets, Tracks: segTracks, Events: segEvents,
			History: history,
		})
	}

	// Invoke the VLM exactly once for the video. Descriptions must come
	// from visual inspection; never synthesize from metadata here.
	vlmStart := time.Now()
	slog.Info("segment_description: VLM invoke start", "video_id", videoID.String(), "segments", len(segInputs))
	results, err := s.describer.DescribeVideo(ctx, vision.VideoDescriptionInput{VideoID: videoID, Segments: segInputs})
	vlmMs := time.Since(vlmStart).Milliseconds()
	isRateLimited := err != nil && vision.IsRateLimited(err)
	if err != nil && !isRateLimited {
		slog.Error("segment_description: VLM failed", "video_id", videoID.String(), "duration_ms", vlmMs, "error", err)
		return nil, err
	}
	if isRateLimited {
		if results == nil {
			results = []vision.DescriptionResult{}
		}
		slog.Warn("segment_description: VLM rate limited, persisting partial descriptions", "video_id", videoID.String(), "duration_ms", vlmMs, "descriptions", len(results), "segments", len(segments), "error", err)
		if len(results) == 0 {
			slog.Info("segment_description: no descriptions due to rate limit, skipping (optional)", "video_id", videoID.String(), "duration_ms", vlmMs)
			// Persisting empty clears any stale descriptions but does not fail pipeline since description is optional.
			empty, repErr := s.repo.ReplaceForVideo(ctx, videoID, nil)
			if repErr != nil {
				slog.Error("segment_description: persist failed after rate limit", "video_id", videoID.String(), "error", repErr)
				return nil, repErr
			}
			return empty, nil
		}
	} else {
		slog.Info("segment_description: VLM complete", "video_id", videoID.String(), "segments", len(segInputs), "descriptions", len(results), "duration_ms", vlmMs)
	}
	if !isRateLimited && len(results) != len(segments) {
		return nil, fmt.Errorf("vision returned %d descriptions for %d segments", len(results), len(segments))
	}
	if isRateLimited && len(results) > len(segments) {
		return nil, fmt.Errorf("vision returned %d descriptions for %d segments", len(results), len(segments))
	}
	bySegment := map[uuid.UUID]vision.DescriptionResult{}
	for _, r := range results {
		if r.Description == "" {
			return nil, fmt.Errorf("empty vision description for segment %s", r.SegmentID)
		}
		if _, dup := bySegment[r.SegmentID]; dup {
			return nil, fmt.Errorf("duplicate vision description for segment %s", r.SegmentID)
		}
		bySegment[r.SegmentID] = r
	}

	var descs []Description
	for _, seg := range segments {
		r, ok := bySegment[seg.ID]
		if !ok {
			if isRateLimited {
				slog.Warn("segment_description: missing description for segment due to rate limit, skipping", "video_id", videoID.String(), "segment_index", seg.SegmentIndex, "segment_id", seg.ID.String())
				continue
			}
			return nil, fmt.Errorf("missing vision description for segment %d", seg.SegmentIndex)
		}
		desc := strings.TrimSpace(r.Description)
		if desc == "" {
			return nil, fmt.Errorf("empty vision description for segment %d", seg.SegmentIndex)
		}
		if len(desc) > 2000 {
			return nil, fmt.Errorf("vision description for segment %d exceeds 2000 characters", seg.SegmentIndex)
		}
		if r.SegmentID != seg.ID {
			return nil, fmt.Errorf("vision description segment ID mismatch for segment %d", seg.SegmentIndex)
		}
		if r.ModelName != s.expectedModelName {
			return nil, fmt.Errorf("vision description model name mismatch for segment %d", seg.SegmentIndex)
		}
		if r.ModelVersion != s.expectedModelVersion {
			return nil, fmt.Errorf("vision description model version mismatch for segment %d", seg.SegmentIndex)
		}
		descs = append(descs, Description{
			VideoID: videoID, SegmentID: seg.ID,
			Description: desc,
			ModelName:   r.ModelName, ModelVersion: r.ModelVersion,
		})
	}

	persistStart := time.Now()
	result, err := s.repo.ReplaceForVideo(ctx, videoID, descs)
	persistMs := time.Since(persistStart).Milliseconds()
	totalMs := time.Since(svcStart).Milliseconds()
	if err != nil {
		slog.Error("segment_description: persist failed", "video_id", videoID.String(), "duration_ms", persistMs, "error", err)
		return nil, err
	}
	slog.Info("segment_description: complete", "video_id", videoID.String(), "segments", len(segments), "descriptions", len(result), "vlm_ms", vlmMs, "persist_ms", persistMs, "total_duration_ms", totalMs)
	return result, nil
}

func (s *Service) GetByVideoID(ctx context.Context, videoID uuid.UUID) ([]Description, error) {
	return s.repo.GetByVideoID(ctx, videoID)
}

func (s *Service) GetBySegmentID(ctx context.Context, segmentID uuid.UUID) (*Description, error) {
	return s.repo.GetBySegmentID(ctx, segmentID)
}
