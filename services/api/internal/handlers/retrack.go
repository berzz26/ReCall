package handlers

import (
	"context"
	"log/slog"
	"time"

	"github.com/berzz26/recall/services/api/internal/video_event"
	"github.com/berzz26/recall/services/api/internal/video_track"
	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
)

// RetrackHandler re-runs tracking (and event generation) for an already
// processed video without redoing detection. Useful after changing tracker
// behavior or config: detections and frames stay untouched.
type RetrackHandler struct {
	trackService *video_track.Service
	eventService *video_event.Service
}

func NewRetrackHandler(trackService *video_track.Service, eventService *video_event.Service) *RetrackHandler {
	return &RetrackHandler{trackService: trackService, eventService: eventService}
}

// Retrack regenerates tracks for a video from its stored detections, then
// rebuilds events so they point at the new track IDs.
func (h *RetrackHandler) Retrack(c *fiber.Ctx) error {
	id, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid id"})
	}
	ctx, cancel := context.WithTimeout(c.UserContext(), 5*time.Minute)
	defer cancel()

	tracks, err := h.trackService.GenerateForVideoID(ctx, id)
	if err != nil {
		slog.Error("retrack: tracking failed", "video_id", id.String(), "error", err)
		return c.Status(500).JSON(fiber.Map{"error": "retrack failed: " + err.Error()})
	}
	// Events reference track IDs; rebuild them so they stay consistent.
	if h.eventService != nil {
		if _, err := h.eventService.GenerateForVideo(ctx, id); err != nil {
			slog.Error("retrack: event regeneration failed", "video_id", id.String(), "error", err)
			return c.Status(500).JSON(fiber.Map{"error": "events regenerate failed: " + err.Error()})
		}
	}
	return c.JSON(fiber.Map{
		"video_id":    id,
		"track_count": len(tracks),
	})
}
