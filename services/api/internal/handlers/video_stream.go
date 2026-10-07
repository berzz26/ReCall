package handlers

import (
	"context"
	"os"
	"path/filepath"
	"time"

	"github.com/berzz26/recall/services/api/internal/playable"
	"github.com/berzz26/recall/services/api/internal/storage"
	"github.com/berzz26/recall/services/api/internal/video"
	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
)

type VideoStreamHandler struct {
	videoRepo *video.Repository
	storage   storage.Storage
}

func NewVideoStreamHandler(videoRepo *video.Repository, storage storage.Storage) *VideoStreamHandler {
	return &VideoStreamHandler{videoRepo: videoRepo, storage: storage}
}

func (h *VideoStreamHandler) Stream(c *fiber.Ctx) error {
	id, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid id"})
	}
	ctx, cancel := context.WithTimeout(c.UserContext(), 10*time.Second)
	defer cancel()
	v, err := h.videoRepo.GetByID(ctx, id)
	if err != nil {
		return c.Status(404).JSON(fiber.Map{"error": "video not found"})
	}
	mime := v.MimeType
	if mime == "" {
		mime = "video/mp4"
	}
	// Prefer the browser-playable proxy when one was generated: the original
	// may use a codec browsers cannot decode (e.g. MPEG-4 Part 2), in which
	// case serving it yields "no video with supported format and MIME type".
	if v.PlayableKey != nil && *v.PlayableKey != "" {
		if p, err := playable.LocalProxyPath(h.storage, *v.PlayableKey); err == nil {
			if _, serr := os.Stat(p); serr == nil {
				c.Set("Content-Type", "video/mp4")
				c.Set("Accept-Ranges", "bytes")
				return c.SendFile(p)
			}
		}
	}
	// Prefer storage key for UPLOAD
	if v.StorageKey != nil && *v.StorageKey != "" {
		// Try SendFile for LocalStorage to support Range requests and avoid loading whole file
		if ls, ok := h.storage.(*storage.LocalStorage); ok {
			p := filepath.Join(ls.Root(), *v.StorageKey)
			if _, err := os.Stat(p); err == nil {
				c.Set("Content-Type", mime)
				c.Set("Accept-Ranges", "bytes")
				return c.SendFile(p)
			}
		}
		// Fallback: stream via storage Open with SendStream (for non-local storage)
		// Use Fiber's SendStream if file exists but not local
		if rc, err := h.storage.Open(ctx, *v.StorageKey); err == nil {
			defer rc.Close()
			c.Set("Content-Type", mime)
			c.Set("Accept-Ranges", "bytes")
			// For fallback we still need to avoid ReadAll; stream directly
			return c.SendStream(rc)
		}
	}
	if v.SourcePath != nil && *v.SourcePath != "" {
		p := *v.SourcePath
		if _, err := os.Stat(p); err == nil {
			c.Set("Content-Type", mime)
			c.Set("Accept-Ranges", "bytes")
			return c.SendFile(p)
		}
	}
	return c.Status(404).JSON(fiber.Map{"error": "video file not found"})
}
