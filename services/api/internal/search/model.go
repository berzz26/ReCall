package search

import (
	"github.com/google/uuid"
)

// DetailLevel controls how much evidence rides on each search hit.
// "summary" (default) returns counts only — cheap list rendering.
// "full" returns detections/tracks/events arrays (legacy, heavier).
type DetailLevel string

const (
	DetailSummary DetailLevel = "summary"
	DetailFull    DetailLevel = "full"
)

type SearchRequest struct {
	Query   string     `json:"query"`
	Limit   int        `json:"limit,omitempty"`
	Offset  int        `json:"offset,omitempty"`
	VideoID *uuid.UUID `json:"video_id,omitempty"`
	Detail  DetailLevel `json:"detail,omitempty"`
}

type SearchResponse struct {
	Query   string         `json:"query"`
	Results []SearchResult `json:"results"`
	Total   int            `json:"total,omitempty"`
}

type SearchResult struct {
	VideoID     uuid.UUID       `json:"video_id"`
	VideoName   string          `json:"video_filename,omitempty"`
	Filename    string          `json:"filename,omitempty"`
	SegmentID   uuid.UUID       `json:"segment_id"`
	StartTime   float64         `json:"start_time"`
	EndTime     float64         `json:"end_time"`
	Description string          `json:"description"`
	MatchedText string          `json:"matched_text,omitempty"`
	Similarity  float64         `json:"similarity"`
	Detections  []DetectionInfo `json:"detections"`
	Tracks      []TrackInfo     `json:"tracks"`
	Events      []EventInfo     `json:"events"`
	// Summary-only fields (populated in both modes, cheap to compute).
	DetectionCounts map[string]int `json:"detection_counts,omitempty"`
	TrackCount      int            `json:"track_count,omitempty"`
	EventCount      int            `json:"event_count,omitempty"`
	ThumbnailFrameID *uuid.UUID    `json:"thumbnail_frame_id,omitempty"`
	// Truncation flags for full mode.
	TracksTruncated bool `json:"tracks_truncated,omitempty"`
	EventsTruncated bool `json:"events_truncated,omitempty"`
}

type DetectionInfo struct {
	Label      string  `json:"label"`
	Confidence float64 `json:"confidence"`
}

type TrackInfo struct {
	TrackID   uuid.UUID `json:"track_id"`
	Label     string    `json:"label"`
	StartTime float64   `json:"start_time"`
	EndTime   float64   `json:"end_time"`
}

type EventInfo struct {
	EventID    uuid.UUID `json:"event_id"`
	EventType  string    `json:"event_type"`
	Label      string    `json:"label"`
	StartTime  float64   `json:"start_time"`
	EndTime    *float64  `json:"end_time,omitempty"`
	Confidence *float64  `json:"confidence,omitempty"`
}
