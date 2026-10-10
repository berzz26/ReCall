package handlers

import (
	"context"
	"strings"
	"time"

	"github.com/berzz26/recall/services/api/internal/search"
	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
)

type UnifiedSearchHandler struct {
	svc *search.Service
}

func NewUnifiedSearchHandler(svc *search.Service) *UnifiedSearchHandler {
	return &UnifiedSearchHandler{svc: svc}
}

type unifiedSearchRequest struct {
	Query   string  `json:"query"`
	Limit   *int    `json:"limit,omitempty"`
	Offset  *int    `json:"offset,omitempty"`
	VideoID *string `json:"video_id,omitempty"`
	Detail  *string `json:"detail,omitempty"`
}

func (h *UnifiedSearchHandler) Search(c *fiber.Ctx) error {
	var req unifiedSearchRequest
	if err := c.BodyParser(&req); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid request body"})
	}
	// Trim and validate query
	query := strings.TrimSpace(req.Query)
	if query == "" {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "query is required"})
	}
	// Normalize limit
	// Use search model conventions via service, but validate early for 400
	var limit int
	if req.Limit == nil {
		limit = 0 // will default to SearchDefaultLimit in service
	} else {
		limit = *req.Limit
		if limit <= 0 {
			limit = 0 // spec: <=0 defaults to SEARCH_DEFAULT_LIMIT
		} else if limit > 50 {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "limit must be <= 50"})
		}
	}
	var offset int
	if req.Offset != nil {
		if *req.Offset < 0 {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "offset must be >= 0"})
		}
		offset = *req.Offset
	}
	var vid *uuid.UUID
	if req.VideoID != nil && strings.TrimSpace(*req.VideoID) != "" {
		parsed, err := uuid.Parse(strings.TrimSpace(*req.VideoID))
		if err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid video_id"})
		}
		vid = &parsed
	}
	detail := search.DetailSummary
	if req.Detail != nil && strings.TrimSpace(*req.Detail) != "" {
		switch strings.ToLower(strings.TrimSpace(*req.Detail)) {
		case "summary":
			detail = search.DetailSummary
		case "full":
			detail = search.DetailFull
		default:
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "detail must be summary or full"})
		}
	}
	// Query-string overrides for deep-links / GET-style callers.
	if q := strings.TrimSpace(c.Query("detail")); q != "" {
		switch strings.ToLower(q) {
		case "summary":
			detail = search.DetailSummary
		case "full":
			detail = search.DetailFull
		default:
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "detail must be summary or full"})
		}
	}
	searchReq := search.SearchRequest{
		Query:   query,
		Limit:   limit,
		Offset:  offset,
		VideoID: vid,
		Detail:  detail,
	}
	ctx, cancel := context.WithTimeout(c.UserContext(), 90*time.Second)
	defer cancel()
	results, err := h.svc.Search(ctx, searchReq)
	if err != nil {
		// Map validation errors to 400, others to 500
		msg := err.Error()
		if strings.Contains(msg, "query is required") || strings.Contains(msg, "limit must be") || strings.Contains(msg, "invalid video_id") || strings.Contains(msg, "offset must be") || strings.Contains(msg, "detail must be") {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": msg})
		}
		if strings.Contains(msg, "failed to embed") || strings.Contains(msg, "invalid query embedding") || strings.Contains(msg, "invalid query vector") {
			return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": "failed to embed query"})
		}
		// enrichment / retrieval DB failures -> 500 without exposing internals
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": "search failed"})
	}
	if results == nil {
		results = []search.SearchResult{}
	}
	// Ensure empty arrays not null
	for i := range results {
		if results[i].Detections == nil {
			results[i].Detections = []search.DetectionInfo{}
		}
		if results[i].Tracks == nil {
			results[i].Tracks = []search.TrackInfo{}
		}
		if results[i].Events == nil {
			results[i].Events = []search.EventInfo{}
		}
		if results[i].DetectionCounts == nil {
			results[i].DetectionCounts = map[string]int{}
		}
	}
	return c.JSON(search.SearchResponse{Query: query, Results: results, Total: len(results)})
}
