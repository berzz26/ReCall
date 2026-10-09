package search

import (
	"context"
	"fmt"
	"strings"
	"sync"

	"github.com/berzz26/recall/services/api/internal/embedding"
	"github.com/berzz26/recall/services/api/internal/segment_embedding"
	"github.com/berzz26/recall/services/api/internal/video"
	"github.com/google/uuid"
	"github.com/jackc/pgx/v5/pgxpool"
)

const (
	// Bounds keep per-hit payloads small. List views use summary mode and
	// never pay for full arrays; full mode is capped for the evidence view.
	maxTracksPerResult = 20
	maxEventsPerResult = 50
	// maxConcurrentEnrich bounds DB fan-out per search.
	maxConcurrentEnrich = 6
)

type Service struct {
	embedder             embedding.TextEmbedder
	segmentEmbeddingRepo *segment_embedding.Repository
	db                   *pgxpool.Pool
	videoRepo            *video.Repository
	candidateLimit       int
	defaultLimit         int
	maxLimit             int
	minSimilarity        float64
}

func NewService(
	embedder embedding.TextEmbedder,
	segmentEmbeddingRepo *segment_embedding.Repository,
	db *pgxpool.Pool,
	videoRepo *video.Repository,
	candidateLimit, defaultLimit, maxLimit int,
	minSimilarity float64,
) *Service {
	return &Service{
		embedder:             embedder,
		segmentEmbeddingRepo: segmentEmbeddingRepo,
		db:                   db,
		videoRepo:            videoRepo,
		candidateLimit:       candidateLimit,
		defaultLimit:         defaultLimit,
		maxLimit:             maxLimit,
		minSimilarity:        minSimilarity,
	}
}

func (s *Service) Search(ctx context.Context, req SearchRequest) ([]SearchResult, error) {
	query := strings.TrimSpace(req.Query)
	if query == "" {
		return nil, fmt.Errorf("query is required")
	}
	limit := req.Limit
	if limit <= 0 {
		limit = s.defaultLimit
	}
	if limit < 1 || limit > s.maxLimit {
		return nil, fmt.Errorf("limit must be between 1 and %d", s.maxLimit)
	}
	offset := req.Offset
	if offset < 0 {
		return nil, fmt.Errorf("offset must be >= 0")
	}
	if req.VideoID != nil {
		if *req.VideoID == uuid.Nil {
			return nil, fmt.Errorf("invalid video_id")
		}
	}
	detail := req.Detail
	if detail == "" {
		detail = DetailSummary
	}
	if detail != DetailSummary && detail != DetailFull {
		return nil, fmt.Errorf("detail must be summary or full")
	}
	// 1. Generate query embedding (persistent worker when available).
	vec, err := s.embedder.EmbedQuery(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("failed to embed query: %w", err)
	}
	if len(vec) != 384 {
		return nil, fmt.Errorf("invalid query embedding dimension %d", len(vec))
	}
	// 2. Retrieve candidates (bounded). Fetch enough to cover offset+limit
	// after similarity filtering; candidateLimit is the recall pool cap.
	fetchN := s.candidateLimit
	if fetchN < offset+limit {
		fetchN = offset + limit
	}
	candidates, err := s.segmentEmbeddingRepo.SearchSimilar(ctx, vec, fetchN, req.VideoID)
	if err != nil {
		return nil, fmt.Errorf("semantic retrieval failed: %w", err)
	}
	// 3. Apply threshold, preserve ranking (already descending similarity).
	var filtered []segment_embedding.SearchResult
	for _, c := range candidates {
		if c.Similarity >= s.minSimilarity {
			filtered = append(filtered, c)
		}
	}
	// 4. Paginate, then enrich concurrently (bounded).
	if offset >= len(filtered) {
		return []SearchResult{}, nil
	}
	end := offset + limit
	if end > len(filtered) {
		end = len(filtered)
	}
	page := filtered[offset:end]

	// Batch: filenames + summary counts in a fixed number of queries,
	// independent of page size (no N+1).
	names, err := s.batchFilenames(ctx, distinctVideoIDs(page))
	if err != nil {
		return nil, err
	}
	summaries, err := s.batchSummaries(ctx, page)
	if err != nil {
		return nil, err
	}

	results := make([]SearchResult, len(page))
	for i, c := range page {
		sum := summaries[c.Embedding.SegmentID]
		if sum == nil {
			sum = &segSummary{counts: map[string]int{}}
		}
		r := SearchResult{
			VideoID:     c.Embedding.VideoID,
			SegmentID:   c.Embedding.SegmentID,
			StartTime:   c.StartTime,
			EndTime:     c.EndTime,
			Description: c.Description,
			MatchedText: extractMatchedText(query, c.Description),
			Similarity:  c.Similarity,
			Detections:  []DetectionInfo{},
			Tracks:      []TrackInfo{},
			Events:      []EventInfo{},
		}
		if fn, ok := names[c.Embedding.VideoID]; ok {
			r.Filename = fn
			r.VideoName = fn
		} else if s.videoRepo != nil {
			// Fallback for any missing name (should not happen).
			if v, verr := s.videoRepo.GetByID(ctx, c.Embedding.VideoID); verr == nil {
				r.Filename = v.Filename
				r.VideoName = v.Filename
			}
		}
		r.DetectionCounts = sum.counts
		r.TrackCount = sum.trackCount
		r.EventCount = sum.eventCount
		r.ThumbnailFrameID = sum.thumb
		results[i] = r
	}
	if detail == DetailFull {
		// Full arrays are inherently per-segment; bounded + concurrent.
		// Counts/thumbnails above are already batched.
		sem := make(chan struct{}, maxConcurrentEnrich)
		var wg sync.WaitGroup
		errs := make([]error, len(page))
		for i, c := range page {
			i, c := i, c
			wg.Add(1)
			go func() {
				defer wg.Done()
				select {
				case sem <- struct{}{}:
					defer func() { <-sem }()
				case <-ctx.Done():
					errs[i] = ctx.Err()
					return
				}
				dets, derr := s.enrichDetections(ctx, c.Embedding.VideoID, c.StartTime, c.EndTime)
				if derr != nil {
					errs[i] = derr
					return
				}
				tracks, ttrunc, terr := s.enrichTracksBounded(ctx, c.Embedding.VideoID, c.StartTime, c.EndTime, maxTracksPerResult)
				if terr != nil {
					errs[i] = terr
					return
				}
				events, etrunc, eerr := s.enrichEventsBounded(ctx, c.Embedding.VideoID, c.StartTime, c.EndTime, maxEventsPerResult)
				if eerr != nil {
					errs[i] = eerr
					return
				}
				results[i].Detections = dets
				results[i].Tracks = tracks
				results[i].Events = events
				results[i].TracksTruncated = ttrunc
				results[i].EventsTruncated = etrunc
			}()
		}
		wg.Wait()
		for _, e := range errs {
			if e != nil {
				return nil, e
			}
		}
	}
	if results == nil {
		results = []SearchResult{}
	}
	return results, nil
}

// segSummary is the batched per-segment summary (fixed query count).
type segSummary struct {
	counts     map[string]int
	trackCount int
	eventCount int
	thumb      *uuid.UUID
}

func distinctVideoIDs(page []segment_embedding.SearchResult) []uuid.UUID {
	seen := make(map[uuid.UUID]struct{}, len(page))
	out := make([]uuid.UUID, 0, len(page))
	for _, c := range page {
		if _, ok := seen[c.Embedding.VideoID]; !ok {
			seen[c.Embedding.VideoID] = struct{}{}
			out = append(out, c.Embedding.VideoID)
		}
	}
	return out
}

// batchFilenames resolves all video filenames in one query.
func (s *Service) batchFilenames(ctx context.Context, vids []uuid.UUID) (map[uuid.UUID]string, error) {
	out := make(map[uuid.UUID]string, len(vids))
	if len(vids) == 0 {
		return out, nil
	}
	rows, err := s.db.Query(ctx, `SELECT id, filename FROM videos WHERE id = ANY($1)`, vids)
	if err != nil {
		return nil, fmt.Errorf("video enrichment failed: %w", err)
	}
	defer rows.Close()
	for rows.Next() {
		var id uuid.UUID
		var fn string
		if err := rows.Scan(&id, &fn); err != nil {
			return nil, fmt.Errorf("video enrichment scan failed: %w", err)
		}
		out[id] = fn
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("video enrichment failed: %w", err)
	}
	return out, nil
}

// batchSummaries computes detection/track/event counts + thumbnails for the
// whole page in 4 queries total, regardless of page size.
func (s *Service) batchSummaries(ctx context.Context, page []segment_embedding.SearchResult) (map[uuid.UUID]*segSummary, error) {
	out := make(map[uuid.UUID]*segSummary, len(page))
	for _, c := range page {
		out[c.Embedding.SegmentID] = &segSummary{counts: map[string]int{}}
	}
	if len(page) == 0 {
		return out, nil
	}
	segIDs := make([]uuid.UUID, len(page))
	vids := make([]uuid.UUID, len(page))
	starts := make([]float64, len(page))
	ends := make([]float64, len(page))
	for i, c := range page {
		segIDs[i] = c.Embedding.SegmentID
		vids[i] = c.Embedding.VideoID
		starts[i] = c.StartTime
		ends[i] = c.EndTime
	}
	// 1. Detection counts per segment window.
	rows, err := s.db.Query(ctx, `
		SELECT w.seg_id, d.label, COUNT(*)::int
		FROM (
			SELECT unnest($1::uuid[]) AS seg_id, unnest($2::uuid[]) AS vid,
			       unnest($3::float8[]) AS s, unnest($4::float8[]) AS e
		) w
		JOIN video_frames f ON f.video_id = w.vid
		  AND f.timestamp_seconds >= w.s AND f.timestamp_seconds < w.e
		JOIN video_frame_detections d ON d.frame_id = f.id AND d.video_id = w.vid
		GROUP BY w.seg_id, d.label
	`, segIDs, vids, starts, ends)
	if err != nil {
		return nil, fmt.Errorf("detection summary failed: %w", err)
	}
	for rows.Next() {
		var segID uuid.UUID
		var label string
		var n int
		if err := rows.Scan(&segID, &label, &n); err != nil {
			rows.Close()
			return nil, fmt.Errorf("detection summary scan failed: %w", err)
		}
		if sum, ok := out[segID]; ok {
			sum.counts[label] = n
		}
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("detection summary failed: %w", err)
	}
	// 2. Track counts per segment window (overlap).
	rows, err = s.db.Query(ctx, `
		SELECT w.seg_id, COUNT(vt.id)::int
		FROM (
			SELECT unnest($1::uuid[]) AS seg_id, unnest($2::uuid[]) AS vid,
			       unnest($3::float8[]) AS s, unnest($4::float8[]) AS e
		) w
		LEFT JOIN video_tracks vt ON vt.video_id = w.vid
		  AND vt.start_timestamp < w.e AND vt.end_timestamp > w.s
		GROUP BY w.seg_id
	`, segIDs, vids, starts, ends)
	if err != nil {
		return nil, fmt.Errorf("track summary failed: %w", err)
	}
	for rows.Next() {
		var segID uuid.UUID
		var n int
		if err := rows.Scan(&segID, &n); err != nil {
			rows.Close()
			return nil, fmt.Errorf("track summary scan failed: %w", err)
		}
		if sum, ok := out[segID]; ok {
			sum.trackCount = n
		}
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("track summary failed: %w", err)
	}
	// 3. Event counts per segment window (fixed predicate: point events
	// must start inside; ranged use overlap).
	rows, err = s.db.Query(ctx, `
		SELECT w.seg_id, COUNT(ve.id)::int
		FROM (
			SELECT unnest($1::uuid[]) AS seg_id, unnest($2::uuid[]) AS vid,
			       unnest($3::float8[]) AS s, unnest($4::float8[]) AS e
		) w
		LEFT JOIN video_events ve ON ve.video_id = w.vid AND (
		  (ve.end_timestamp IS NULL AND ve.start_timestamp >= w.s AND ve.start_timestamp < w.e)
		  OR (ve.end_timestamp IS NOT NULL AND ve.start_timestamp < w.e AND ve.end_timestamp > w.s)
		)
		GROUP BY w.seg_id
	`, segIDs, vids, starts, ends)
	if err != nil {
		return nil, fmt.Errorf("event summary failed: %w", err)
	}
	for rows.Next() {
		var segID uuid.UUID
		var n int
		if err := rows.Scan(&segID, &n); err != nil {
			rows.Close()
			return nil, fmt.Errorf("event summary scan failed: %w", err)
		}
		if sum, ok := out[segID]; ok {
			sum.eventCount = n
		}
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("event summary failed: %w", err)
	}
	// 4. Thumbnails: earliest frame per segment.
	rows, err = s.db.Query(ctx, `
		SELECT DISTINCT ON (segment_id) segment_id, id
		FROM video_frames WHERE segment_id = ANY($1)
		ORDER BY segment_id, timestamp_seconds ASC
	`, segIDs)
	if err != nil {
		return nil, fmt.Errorf("thumbnail lookup failed: %w", err)
	}
	for rows.Next() {
		var segID, fid uuid.UUID
		if err := rows.Scan(&segID, &fid); err != nil {
			rows.Close()
			return nil, fmt.Errorf("thumbnail scan failed: %w", err)
		}
		if sum, ok := out[segID]; ok {
			c := fid
			sum.thumb = &c
		}
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("thumbnail lookup failed: %w", err)
	}
	return out, nil
}

// enrichSummary returns cheap counts + a thumbnail frame without full arrays.
func (s *Service) enrichSummary(ctx context.Context, videoID, segmentID uuid.UUID, segStart, segEnd float64) (map[string]int, int, int, *uuid.UUID, error) {
	counts := map[string]int{}
	// Detection counts per label within segment time range.
	rows, err := s.db.Query(ctx, `
		SELECT label, COUNT(*)::int
		FROM video_frame_detections d
		JOIN video_frames f ON f.id = d.frame_id
		WHERE d.video_id = $1
		  AND f.timestamp_seconds >= $2
		  AND f.timestamp_seconds < $3
		GROUP BY label
	`, videoID, segStart, segEnd)
	if err != nil {
		return nil, 0, 0, nil, fmt.Errorf("detection summary failed: %w", err)
	}
	for rows.Next() {
		var label string
		var n int
		if err := rows.Scan(&label, &n); err != nil {
			rows.Close()
			return nil, 0, 0, nil, fmt.Errorf("detection summary scan failed: %w", err)
		}
		counts[label] = n
	}
	rows.Close()
	if err := rows.Err(); err != nil {
		return nil, 0, 0, nil, fmt.Errorf("detection summary failed: %w", err)
	}
	var trackCount int
	if err := s.db.QueryRow(ctx, `
		SELECT COUNT(*)::int FROM video_tracks
		WHERE video_id = $1 AND start_timestamp < $3 AND end_timestamp > $2
	`, videoID, segStart, segEnd).Scan(&trackCount); err != nil {
		return nil, 0, 0, nil, fmt.Errorf("track summary failed: %w", err)
	}
	var eventCount int
	if err := s.db.QueryRow(ctx, `
		SELECT COUNT(*)::int FROM video_events
		WHERE video_id = $1 AND (
		  (end_timestamp IS NULL AND start_timestamp >= $2 AND start_timestamp < $3)
		  OR (end_timestamp IS NOT NULL AND start_timestamp < $3 AND end_timestamp > $2)
		)
	`, videoID, segStart, segEnd).Scan(&eventCount); err != nil {
		return nil, 0, 0, nil, fmt.Errorf("event summary failed: %w", err)
	}
	// Thumbnail: earliest frame in segment (by segment_id when available).
	var thumb *uuid.UUID
	var thumbID uuid.UUID
	err = s.db.QueryRow(ctx, `
		SELECT id FROM video_frames WHERE segment_id = $1
		ORDER BY timestamp_seconds ASC LIMIT 1
	`, segmentID).Scan(&thumbID)
	if err == nil {
		thumb = &thumbID
	}
	return counts, trackCount, eventCount, thumb, nil
}

func extractMatchedText(query, description string) string {
	desc := strings.TrimSpace(description)
	if desc == "" || strings.TrimSpace(query) == "" {
		return ""
	}
	lowerDesc := strings.ToLower(desc)
	lowerQuery := strings.ToLower(strings.TrimSpace(query))
	// tokenise query, keep tokens >=2 chars
	tokens := strings.Fields(lowerQuery)
	var filteredTokens []string
	for _, t := range tokens {
		t = strings.Trim(t, ".,!?;:\"'()[]{}")
		if len(t) >= 2 {
			filteredTokens = append(filteredTokens, t)
		}
	}
	if len(filteredTokens) == 0 {
		filteredTokens = strings.Fields(lowerQuery)
	}
	// find earliest token occurrence
	earliestPos := -1
	earliestLen := 0
	for _, tok := range filteredTokens {
		pos := strings.Index(lowerDesc, tok)
		if pos >= 0 && (earliestPos == -1 || pos < earliestPos) {
			earliestPos = pos
			earliestLen = len(tok)
		}
	}
	// also try full query phrase
	if phrasePos := strings.Index(lowerDesc, lowerQuery); phrasePos >= 0 {
		if earliestPos == -1 || phrasePos < earliestPos {
			earliestPos = phrasePos
			earliestLen = len(lowerQuery)
		} else if phrasePos == earliestPos && len(lowerQuery) > earliestLen {
			earliestLen = len(lowerQuery)
		}
	}
	if earliestPos == -1 {
		// no lexical match – fallback to first ~110 chars at word boundary (still exact substring for highlighting)
		if len(desc) <= 110 {
			return desc
		}
		cut := 110
		// avoid cutting in middle of word
		if idx := strings.LastIndex(desc[:cut], " "); idx > 60 {
			cut = idx
		}
		return strings.TrimSpace(desc[:cut])
	}
	// expand window around match: ~30 chars before, ~70 after
	start := earliestPos - 30
	if start < 0 {
		start = 0
	} else {
		// snap to previous word boundary
		if sp := strings.LastIndex(desc[:start], " "); sp >= 0 && start-sp < 20 {
			start = sp + 1
		}
	}
	end := earliestPos + earliestLen + 70
	if end > len(desc) {
		end = len(desc)
	} else {
		if sp := strings.Index(desc[end:], " "); sp >= 0 && sp < 20 {
			end = end + sp
		}
	}
	snippet := strings.TrimSpace(desc[start:end])
	// ensure snippet is exact substring of description (it is)
	if len(snippet) < 10 {
		return desc
	}
	return snippet
}

func (s *Service) enrichDetections(ctx context.Context, videoID uuid.UUID, segStart, segEnd float64) ([]DetectionInfo, error) {
	// Join frames to restrict to segment time range, then dedup by label max confidence
	query := `
		SELECT label, MAX(confidence) as max_conf
		FROM video_frame_detections d
		JOIN video_frames f ON f.id = d.frame_id
		WHERE d.video_id = $1
		  AND f.timestamp_seconds >= $2
		  AND f.timestamp_seconds < $3
		GROUP BY label
		ORDER BY label ASC
	`
	rows, err := s.db.Query(ctx, query, videoID, segStart, segEnd)
	if err != nil {
		return nil, fmt.Errorf("detection enrichment failed: %w", err)
	}
	defer rows.Close()
	var out []DetectionInfo
	for rows.Next() {
		var label string
		var conf float64
		if err := rows.Scan(&label, &conf); err != nil {
			return nil, fmt.Errorf("detection enrichment scan failed: %w", err)
		}
		out = append(out, DetectionInfo{Label: label, Confidence: conf})
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("detection enrichment failed: %w", err)
	}
	if out == nil {
		out = []DetectionInfo{}
	}
	return out, nil
}

// enrichTracksBounded returns overlapping tracks ranked by overlap duration,
// capped at maxN. truncated=true when more exist than returned.
func (s *Service) enrichTracksBounded(ctx context.Context, videoID uuid.UUID, segStart, segEnd float64, maxN int) ([]TrackInfo, bool, error) {
	if maxN <= 0 {
		maxN = maxTracksPerResult
	}
	query := `
		SELECT id, label, start_timestamp, end_timestamp,
		       LEAST(end_timestamp, $3) - GREATEST(start_timestamp, $2) AS overlap
		FROM video_tracks
		WHERE video_id = $1
		  AND start_timestamp < $3
		  AND end_timestamp > $2
		ORDER BY overlap DESC, start_timestamp ASC
		LIMIT $4
	`
	rows, err := s.db.Query(ctx, query, videoID, segStart, segEnd, maxN+1)
	if err != nil {
		return nil, false, fmt.Errorf("track enrichment failed: %w", err)
	}
	defer rows.Close()
	var out []TrackInfo
	for rows.Next() {
		var id uuid.UUID
		var label string
		var st, et, overlap float64
		if err := rows.Scan(&id, &label, &st, &et, &overlap); err != nil {
			return nil, false, fmt.Errorf("track enrichment scan failed: %w", err)
		}
		out = append(out, TrackInfo{TrackID: id, Label: label, StartTime: st, EndTime: et})
	}
	if err := rows.Err(); err != nil {
		return nil, false, fmt.Errorf("track enrichment failed: %w", err)
	}
	truncated := len(out) > maxN
	if truncated {
		out = out[:maxN]
	}
	if out == nil {
		out = []TrackInfo{}
	}
	return out, truncated, nil
}

func (s *Service) enrichTracks(ctx context.Context, videoID uuid.UUID, segStart, segEnd float64) ([]TrackInfo, error) {
	out, _, err := s.enrichTracksBounded(ctx, videoID, segStart, segEnd, maxTracksPerResult)
	return out, err
}

// enrichEventsBounded fixes the point-event leak: APPEARED/DISAPPEARED with
// NULL end must fall inside [segStart, segEnd); only ranged PRESENT/MOVED
// use overlap. Capped at maxN, truncated flag when more exist.
func (s *Service) enrichEventsBounded(ctx context.Context, videoID uuid.UUID, segStart, segEnd float64, maxN int) ([]EventInfo, bool, error) {
	if maxN <= 0 {
		maxN = maxEventsPerResult
	}
	query := `
		SELECT id, event_type, label, start_timestamp, end_timestamp, confidence
		FROM video_events
		WHERE video_id = $1
		  AND (
		    (end_timestamp IS NULL AND start_timestamp >= $2 AND start_timestamp < $3)
		    OR (end_timestamp IS NOT NULL AND start_timestamp < $3 AND end_timestamp > $2)
		  )
		ORDER BY start_timestamp ASC
		LIMIT $4
	`
	rows, err := s.db.Query(ctx, query, videoID, segStart, segEnd, maxN+1)
	if err != nil {
		return nil, false, fmt.Errorf("event enrichment failed: %w", err)
	}
	defer rows.Close()
	seen := make(map[uuid.UUID]bool)
	var out []EventInfo
	for rows.Next() {
		var id uuid.UUID
		var eventType, label string
		var st float64
		var et *float64
		var conf *float64
		if err := rows.Scan(&id, &eventType, &label, &st, &et, &conf); err != nil {
			return nil, false, fmt.Errorf("event enrichment scan failed: %w", err)
		}
		if seen[id] {
			continue
		}
		seen[id] = true
		out = append(out, EventInfo{EventID: id, EventType: eventType, Label: label, StartTime: st, EndTime: et, Confidence: conf})
	}
	if err := rows.Err(); err != nil {
		return nil, false, fmt.Errorf("event enrichment failed: %w", err)
	}
	truncated := len(out) > maxN
	if truncated {
		out = out[:maxN]
	}
	if out == nil {
		out = []EventInfo{}
	}
	return out, truncated, nil
}

func (s *Service) enrichEvents(ctx context.Context, videoID uuid.UUID, segStart, segEnd float64) ([]EventInfo, error) {
	out, _, err := s.enrichEventsBounded(ctx, videoID, segStart, segEnd, maxEventsPerResult)
	return out, err
}
