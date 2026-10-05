package video

import (
	"context"
	"io"
	"mime"
	"mime/multipart"
	"strings"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
)

type Handler struct {
	service *Service
}

func NewHandler(service *Service) *Handler {
	return &Handler{service: service}
}

func (h *Handler) Create(c *fiber.Ctx) error {
	contentType := c.Get("Content-Type")
	if strings.Contains(contentType, "multipart/form-data") {
		// Stream the upload: with StreamRequestBody the body is not buffered
		// in RAM, so read the file part via multipart.Reader and pipe it
		// straight through UploadVideo -> storage (disk). Peak extra memory
		// is a few KB regardless of file size.
		_, params, err := mime.ParseMediaType(contentType)
		if err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid multipart request"})
		}
		boundary := params["boundary"]
		if boundary == "" {
			if b := c.Request().Header.MultipartFormBoundary(); len(b) > 0 {
				boundary = string(b)
			} else {
				return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "missing multipart boundary"})
			}
		}
		mr := multipart.NewReader(c.Request().BodyStream(), boundary)
		var filename, partMime string
		var part io.Reader
		for {
			p, err := mr.NextPart()
			if err == io.EOF {
				break
			}
			if err != nil {
				return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "failed to read upload"})
			}
			if p.FormName() != "file" {
				_, _ = io.Copy(io.Discard, p)
				continue
			}
			filename = p.FileName()
			partMime = p.Header.Get("Content-Type")
			part = p
			break
		}
		if part == nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "file is required"})
		}
		if filename == "" {
			filename = "upload.mp4"
		}
		if partMime == "" {
			partMime = "application/octet-stream"
		}

		// Request-scoped context: no artificial timeout (network transfer
		// time now counts, unlike the old buffered path), and a client
		// disconnect cancels the copy to disk.
		v, err := h.service.UploadVideo(c.UserContext(), filename, part, partMime)
		if err != nil {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": err.Error()})
		}
		// Drain the trailing boundary bytes so the connection can be reused
		// (success path only: the file part was fully consumed).
		_, _ = io.Copy(io.Discard, c.Request().BodyStream())
		return c.Status(fiber.StatusCreated).JSON(v)
	}

	var req CreateVideoRequest
	if err := c.BodyParser(&req); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid request body"})
	}
	if req.Filename == "" {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "filename is required"})
	}
	if req.SourceType != nil && *req.SourceType != SourceTypeLocal && *req.SourceType != SourceTypeUpload {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid source_type"})
	}
	if req.SourceType != nil && *req.SourceType == SourceTypeLocal && (req.SourcePath == nil || *req.SourcePath == "") {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "source_path is required for LOCAL source_type"})
	}

	ctx, cancel := context.WithTimeout(c.UserContext(), 5*time.Second)
	defer cancel()

	v, err := h.service.CreateVideo(ctx, req.Filename, req.ContentHash, req.MimeType, req.SizeBytes, req.SourceType, req.SourcePath)
	if err != nil {
		if strings.Contains(err.Error(), "does not exist") {
			return c.Status(fiber.StatusNotFound).JSON(fiber.Map{"error": err.Error()})
		}
		if strings.Contains(err.Error(), "not a regular file") || strings.Contains(err.Error(), "unsupported file type") || strings.Contains(err.Error(), "source_path is required") {
			return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": err.Error()})
		}
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": err.Error()})
	}
	return c.Status(fiber.StatusCreated).JSON(v)
}

func (h *Handler) List(c *fiber.Ctx) error {
	ctx, cancel := context.WithTimeout(c.UserContext(), 5*time.Second)
	defer cancel()

	videos, err := h.service.ListVideos(ctx, 20, 0)
	if err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": "failed to list videos"})
	}
	return c.JSON(videos)
}

func (h *Handler) Get(c *fiber.Ctx) error {
	idStr := c.Params("id")
	id, err := uuid.Parse(idStr)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid id"})
	}

	ctx, cancel := context.WithTimeout(c.UserContext(), 5*time.Second)
	defer cancel()

	v, err := h.service.GetVideo(ctx, id)
	if err != nil {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{"error": "video not found"})
	}
	return c.JSON(v)
}

func (h *Handler) Delete(c *fiber.Ctx) error {
	idStr := c.Params("id")
	id, err := uuid.Parse(idStr)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid id"})
	}

	ctx, cancel := context.WithTimeout(c.UserContext(), 5*time.Second)
	defer cancel()

	if _, err := h.service.GetVideo(ctx, id); err != nil {
		return c.Status(fiber.StatusNotFound).JSON(fiber.Map{"error": "video not found"})
	}

	if err := h.service.DeleteVideo(ctx, id); err != nil {
		return c.Status(fiber.StatusInternalServerError).JSON(fiber.Map{"error": "failed to delete video"})
	}
	return c.SendStatus(fiber.StatusNoContent)
}

func (h *Handler) IngestLocal(c *fiber.Ctx) error {
	var req struct {
		Directory string `json:"directory"`
		Path      string `json:"path"`
	}
	if err := c.BodyParser(&req); err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "invalid request body"})
	}
	dir := req.Directory
	if dir == "" {
		dir = req.Path
	}
	if dir == "" {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": "directory is required"})
	}

	ctx, cancel := context.WithTimeout(c.UserContext(), 30*time.Second)
	defer cancel()

	count, err := h.service.IngestLocalDirectory(ctx, dir)
	if err != nil {
		return c.Status(fiber.StatusBadRequest).JSON(fiber.Map{"error": err.Error()})
	}
	return c.JSON(fiber.Map{"ingested": count, "directory": dir})
}
