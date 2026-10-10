package handlers

import (
	"context"
	"strconv"
	"time"

	"github.com/berzz26/recall/services/api/internal/detection"
	"github.com/berzz26/recall/services/api/internal/segment_description"
	"github.com/berzz26/recall/services/api/internal/video_event"
	"github.com/berzz26/recall/services/api/internal/video_frame"
	"github.com/berzz26/recall/services/api/internal/video_segment"
	"github.com/berzz26/recall/services/api/internal/video_track"
	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
)

// SegmentEvidenceHandler serves scoped evidence for one segment so the
// detail page can avoid full-video refetch when arriving via search deep-link.
type SegmentEvidenceHandler struct {
	db          *pgxpool.Pool
	segmentRepo *video_segment.Repository
	frameRepo   *video_frame.Repository
	detRepo     *detection.Repository
	trackRepo   *video_track.Repository
	eventRepo   *video_event.Repository
	descRepo    *segment_description.Repository
}

func NewSegmentEvidenceHandler(db *pgxpool.Pool, sr *video_segment.Repository, fr *video_frame.Repository, dr *detection.Repository, tr *video_track.Repository, er *video_event.Repository, desc *segment_description.Repository) *SegmentEvidenceHandler {
	return &SegmentEvidenceHandler{db: db, segmentRepo: sr, frameRepo: fr, detRepo: dr, trackRepo: tr, eventRepo: er, descRepo: desc}
}

type segmentEvidenceResponse struct {
	Segment     *video_segment.VideoSegment              `json:"segment"`
	Description *segment_description.Description         `json:"description,omitempty"`
	Frames      []video_frame.VideoFrame                 `json:"frames"`
	Detections  []detection.Detection                    `json:"detections"`
	Tracks      []video_track.TrackWithCount             `json:"tracks"`
	Events      []video_event.Event                      `json:"events"`
	Truncated   map[string]bool                          `json:"truncated,omitempty"`
}

// GetEvidence: GET /api/v1/videos/:id/segments/:segmentId/evidence
func (h *SegmentEvidenceHandler) GetEvidence(c *fiber.Ctx) error {
	vid, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid id"})
	}
	sid, err := uuid.Parse(c.Params("segmentId"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid segmentId"})
	}
	trackLimit := parseLimitQuery(c.Query("track_limit"), 20, 100)
	eventLimit := parseLimitQuery(c.Query("event_limit"), 50, 200)

	ctx, cancel := context.WithTimeout(c.UserContext(), 10*time.Second)
	defer cancel()

	segs, err := h.segmentRepo.GetByVideoID(ctx, vid)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": "failed"})
	}
	var seg *video_segment.VideoSegment
	for _, s := range segs {
		if s.ID == sid {
			t := s
			seg = &t
			break
		}
	}
	if seg == nil {
		return c.Status(404).JSON(fiber.Map{"error": "segment not found for video"})
	}

	frames, err := h.frameRepo.GetBySegmentID(ctx, sid)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": "failed"})
	}
	dets, err := h.detRepo.GetBySegmentID(ctx, sid)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": "failed"})
	}
	var desc *segment_description.Description
	if h.descRepo != nil {
		if d, err := h.descRepo.GetBySegmentID(ctx, sid); err == nil {
			desc = d
		}
	}
	tracks, tracksTruncated, err := h.scopedTracks(ctx, vid, seg, trackLimit)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": "failed"})
	}
	events, eventsTruncated, err := h.scopedEvents(ctx, vid, seg, eventLimit)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": "failed"})
	}
	return c.JSON(segmentEvidenceResponse{
		Segment: seg, Description: desc,
		Frames: frames, Detections: dets, Tracks: tracks, Events: events,
		Truncated: map[string]bool{"tracks": tracksTruncated, "events": eventsTruncated},
	})
}

func parseLimitQuery(raw string, def, max int) int {
	if raw == "" {
		return def
	}
	n, err := strconv.Atoi(raw)
	if err != nil || n <= 0 {
		return def
	}
	if n > max {
		return max
	}
	return n
}

func (h *SegmentEvidenceHandler) scopedTracks(ctx context.Context, vid uuid.UUID, seg *video_segment.VideoSegment, limit int) ([]video_track.TrackWithCount, bool, error) {
	if h.trackRepo == nil {
		return []video_track.TrackWithCount{}, false, nil
	}
	// Overlap-ranked, bounded. Uses composite index from 000016.
	rows, err := h.db.Query(ctx, `
		SELECT vt.id, vt.video_id, vt.segment_id, vt.label, vt.track_index,
		       vt.start_timestamp, vt.end_timestamp, vt.tracker_name, vt.tracker_version,
		       vt.created_at, vt.updated_at, COUNT(vtd.id) AS cnt,
		       LEAST(vt.end_timestamp, $3) - GREATEST(vt.start_timestamp, $2) AS overlap
		FROM video_tracks vt LEFT JOIN video_track_detections vtd ON vtd.track_id = vt.id
		WHERE vt.video_id = $1 AND vt.start_timestamp < $3 AND vt.end_timestamp > $2
		GROUP BY vt.id
		ORDER BY overlap DESC, vt.start_timestamp ASC
		LIMIT $4
	`, vid, seg.StartTime, seg.EndTime, limit+1)
	if err != nil {
		return nil, false, err
	}
	defer rows.Close()
	var out []video_track.TrackWithCount
	for rows.Next() {
		var t video_track.TrackWithCount
		var overlap float64
		if err := rows.Scan(&t.ID, &t.VideoID, &t.SegmentID, &t.Label, &t.TrackIndex, &t.StartTimestamp, &t.EndTimestamp, &t.TrackerName, &t.TrackerVersion, &t.CreatedAt, &t.UpdatedAt, &t.DetectionCount, &overlap); err != nil {
			return nil, false, err
		}
		out = append(out, t)
	}
	if err := rows.Err(); err != nil {
		return nil, false, err
	}
	truncated := len(out) > limit
	if truncated {
		out = out[:limit]
	}
	if out == nil {
		out = []video_track.TrackWithCount{}
	}
	return out, truncated, nil
}

func (h *SegmentEvidenceHandler) scopedEvents(ctx context.Context, vid uuid.UUID, seg *video_segment.VideoSegment, limit int) ([]video_event.Event, bool, error) {
	if h.eventRepo == nil {
		return []video_event.Event{}, false, nil
	}
	// Fixed predicate: point events must start inside segment; ranged use overlap.
	rows, err := h.db.Query(ctx, `
		SELECT id, video_id, track_id, segment_id, event_type, label,
		       start_timestamp, end_timestamp, confidence, metadata, created_at, updated_at
		FROM video_events
		WHERE video_id = $1 AND (
		  (end_timestamp IS NULL AND start_timestamp >= $2 AND start_timestamp < $3)
		  OR (end_timestamp IS NOT NULL AND start_timestamp < $3 AND end_timestamp > $2)
		)
		ORDER BY start_timestamp ASC
		LIMIT $4
	`, vid, seg.StartTime, seg.EndTime, limit+1)
	if err != nil {
		return nil, false, err
	}
	defer rows.Close()
	var out []video_event.Event
	for rows.Next() {
		var e video_event.Event
		if err := rows.Scan(&e.ID, &e.VideoID, &e.TrackID, &e.SegmentID, &e.EventType, &e.Label, &e.StartTimestamp, &e.EndTimestamp, &e.Confidence, &e.Metadata, &e.CreatedAt, &e.UpdatedAt); err != nil {
			return nil, false, err
		}
		out = append(out, e)
	}
	if err := rows.Err(); err != nil {
		return nil, false, err
	}
	truncated := len(out) > limit
	if truncated {
		out = out[:limit]
	}
	if out == nil {
		out = []video_event.Event{}
	}
	return out, truncated, nil
}
