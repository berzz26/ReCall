package handlers

import (
	"context"
	"io"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"

	"github.com/berzz26/recall/services/api/internal/detection"
	"github.com/berzz26/recall/services/api/internal/segment_description"
	"github.com/berzz26/recall/services/api/internal/storage"
	"github.com/berzz26/recall/services/api/internal/video_event"
	"github.com/berzz26/recall/services/api/internal/video_frame"
	"github.com/berzz26/recall/services/api/internal/video_media"
	"github.com/berzz26/recall/services/api/internal/video_segment"
	"github.com/berzz26/recall/services/api/internal/video_track"
)

type VideoDetailHandler struct {
	mediaRepo     *video_media.Repository
	segmentRepo   *video_segment.Repository
	frameRepo     *video_frame.Repository
	detectionRepo *detection.Repository
	trackRepo     *video_track.Repository
	eventRepo     *video_event.Repository
	descRepo      *segment_description.Repository
	storage       storage.Storage
}

func NewVideoDetailHandler(mr *video_media.Repository, sr *video_segment.Repository, fr *video_frame.Repository, dr *detection.Repository, st storage.Storage) *VideoDetailHandler {
	return &VideoDetailHandler{mediaRepo: mr, segmentRepo: sr, frameRepo: fr, detectionRepo: dr, storage: st}
}

func NewVideoDetailHandlerWithTracks(mr *video_media.Repository, sr *video_segment.Repository, fr *video_frame.Repository, dr *detection.Repository, tr *video_track.Repository, st storage.Storage) *VideoDetailHandler {
	return &VideoDetailHandler{mediaRepo: mr, segmentRepo: sr, frameRepo: fr, detectionRepo: dr, trackRepo: tr, storage: st}
}

func NewVideoDetailHandlerWithEvents(mr *video_media.Repository, sr *video_segment.Repository, fr *video_frame.Repository, dr *detection.Repository, tr *video_track.Repository, er *video_event.Repository, st storage.Storage) *VideoDetailHandler {
	return &VideoDetailHandler{mediaRepo: mr, segmentRepo: sr, frameRepo: fr, detectionRepo: dr, trackRepo: tr, eventRepo: er, storage: st}
}

func NewVideoDetailHandlerWithDescriptions(mr *video_media.Repository, sr *video_segment.Repository, fr *video_frame.Repository, dr *detection.Repository, tr *video_track.Repository, er *video_event.Repository, dr2 *segment_description.Repository, st storage.Storage) *VideoDetailHandler {
	return &VideoDetailHandler{mediaRepo: mr, segmentRepo: sr, frameRepo: fr, detectionRepo: dr, trackRepo: tr, eventRepo: er, descRepo: dr2, storage: st}
}

func (h *VideoDetailHandler) GetMedia(c *fiber.Ctx) error {
	id, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid id"})
	}
	ctx, cancel := context.WithTimeout(c.UserContext(), 5*time.Second)
	defer cancel()
	m, err := h.mediaRepo.GetByVideoID(ctx, id)
	if err != nil {
		return c.Status(404).JSON(fiber.Map{"error": "media not found"})
	}
	return c.JSON(m)
}

func (h *VideoDetailHandler) GetSegments(c *fiber.Ctx) error {
	id, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid id"})
	}
	ctx, cancel := context.WithTimeout(c.UserContext(), 5*time.Second)
	defer cancel()
	segs, err := h.segmentRepo.GetByVideoID(ctx, id)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": "failed"})
	}
	return c.JSON(segs)
}

func (h *VideoDetailHandler) GetFrames(c *fiber.Ctx) error {
	id, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid id"})
	}
	ctx, cancel := context.WithTimeout(c.UserContext(), 5*time.Second)
	defer cancel()
	// UI frame strip pairs detections with frames by frame_id; both live on
	// the perception timeline, so serve the perception set.
	frames, err := h.frameRepo.GetByVideoIDSet(ctx, id, video_frame.SetPerception)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": "failed"})
	}
	return c.JSON(frames)
}

func (h *VideoDetailHandler) GetDetections(c *fiber.Ctx) error {
	id, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid id"})
	}
	ctx, cancel := context.WithTimeout(c.UserContext(), 5*time.Second)
	defer cancel()
	dets, err := h.detectionRepo.GetByVideoID(ctx, id)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": "failed"})
	}
	return c.JSON(dets)
}

func (h *VideoDetailHandler) GetFrameImage(c *fiber.Ctx) error {
	vid, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid id"})
	}
	fid, err := uuid.Parse(c.Params("frameId"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid frameId"})
	}
	ctx, cancel := context.WithTimeout(c.UserContext(), 5*time.Second)
	defer cancel()
	frames, err := h.frameRepo.GetByVideoID(ctx, vid)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": "failed"})
	}
	var target *video_frame.VideoFrame
	for _, f := range frames {
		if f.ID == fid {
			t := f
			target = &t
			break
		}
	}
	if target == nil {
		return c.Status(404).JSON(fiber.Map{"error": "frame not found"})
	}
	rc, err := h.storage.Open(ctx, target.StorageKey)
	if err != nil {
		return c.Status(404).JSON(fiber.Map{"error": "frame file not found"})
	}
	defer rc.Close()
	data, err := io.ReadAll(rc)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": "failed to read frame"})
	}
	if len(data) == 0 {
		return c.Status(500).JSON(fiber.Map{"error": "empty frame"})
	}
	c.Set("Content-Type", "image/jpeg")
	c.Set("Cache-Control", "public, max-age=86400")
	return c.Send(data)
}

func (h *VideoDetailHandler) GetTracks(c *fiber.Ctx) error {
	if h.trackRepo == nil {
		return c.Status(500).JSON(fiber.Map{"error": "tracking not configured"})
	}
	id, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid id"})
	}
	ctx, cancel := context.WithTimeout(c.UserContext(), 5*time.Second)
	defer cancel()
	tracks, err := h.trackRepo.GetTracksWithCounts(ctx, id)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": "failed"})
	}
	return c.JSON(tracks)
}

func (h *VideoDetailHandler) GetTrackDetections(c *fiber.Ctx) error {
	if h.trackRepo == nil {
		return c.Status(500).JSON(fiber.Map{"error": "tracking not configured"})
	}
	tid, err := uuid.Parse(c.Params("trackId"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid trackId"})
	}
	ctx, cancel := context.WithTimeout(c.UserContext(), 5*time.Second)
	defer cancel()
	dets, err := h.trackRepo.GetDetectionsByTrackID(ctx, tid)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": "failed"})
	}
	return c.JSON(dets)
}

func (h *VideoDetailHandler) GetEvents(c *fiber.Ctx) error {
	if h.eventRepo == nil {
		return c.Status(500).JSON(fiber.Map{"error": "events not configured"})
	}
	id, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid id"})
	}
	eventType := c.Query("event_type")
	label := c.Query("label")
	trackIDStr := c.Query("track_id")
	var trackID *uuid.UUID
	if trackIDStr != "" {
		tid, err := uuid.Parse(trackIDStr)
		if err != nil {
			return c.Status(400).JSON(fiber.Map{"error": "invalid track_id"})
		}
		trackID = &tid
	}
	ctx, cancel := context.WithTimeout(c.UserContext(), 5*time.Second)
	defer cancel()
	events, err := h.eventRepo.GetByVideoID(ctx, id, eventType, label, trackID)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": "failed"})
	}
	return c.JSON(events)
}

func (h *VideoDetailHandler) GetTrackEvents(c *fiber.Ctx) error {
	if h.eventRepo == nil {
		return c.Status(500).JSON(fiber.Map{"error": "events not configured"})
	}
	tid, err := uuid.Parse(c.Params("trackId"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid trackId"})
	}
	ctx, cancel := context.WithTimeout(c.UserContext(), 5*time.Second)
	defer cancel()
	events, err := h.eventRepo.GetByTrackID(ctx, tid)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": "failed"})
	}
	return c.JSON(events)
}

func (h *VideoDetailHandler) GetDescriptions(c *fiber.Ctx) error {
	if h.descRepo == nil {
		return c.Status(500).JSON(fiber.Map{"error": "descriptions not configured"})
	}
	id, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid id"})
	}
	ctx, cancel := context.WithTimeout(c.UserContext(), 5*time.Second)
	defer cancel()
	descs, err := h.descRepo.GetByVideoIDWithSegments(ctx, id)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": "failed"})
	}
	return c.JSON(descs)
}

func (h *VideoDetailHandler) GetSegmentDescription(c *fiber.Ctx) error {
	if h.descRepo == nil {
		return c.Status(500).JSON(fiber.Map{"error": "descriptions not configured"})
	}
	sid, err := uuid.Parse(c.Params("segmentId"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid segmentId"})
	}
	ctx, cancel := context.WithTimeout(c.UserContext(), 5*time.Second)
	defer cancel()
	desc, err := h.descRepo.GetBySegmentID(ctx, sid)
	if err != nil {
		return c.Status(404).JSON(fiber.Map{"error": "description not found"})
	}
	return c.JSON(desc)
}
