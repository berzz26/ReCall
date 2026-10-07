package playable

import (
	"context"

	"github.com/gofiber/fiber/v2"
	"github.com/google/uuid"
)

// Handler exposes proxy management over HTTP.
type Handler struct {
	svc *Service
}

// NewHandler wires the proxy endpoint.
func NewHandler(svc *Service) *Handler {
	return &Handler{svc: svc}
}

// EnsureProxy triggers proxy generation for one video in the background and
// returns immediately: {"status":"ready"} (already playable/proxied),
// {"status":"native"} (original plays directly, nothing to do), or
// {"status":"generating"} (transcode running; poll GET detail / retry
// /stream until the proxy appears). A 500 means the probe itself failed.
func (h *Handler) EnsureProxy(c *fiber.Ctx) error {
	id, err := uuid.Parse(c.Params("id"))
	if err != nil {
		return c.Status(400).JSON(fiber.Map{"error": "invalid id"})
	}
	state, _, err := h.svc.Status(c.UserContext(), id)
	if err != nil {
		return c.Status(500).JSON(fiber.Map{"error": err.Error()})
	}
	switch state {
	case "ready":
		return c.JSON(fiber.Map{"status": "ready"})
	case "native":
		return c.JSON(fiber.Map{"status": "native"})
	}
	// Detached context: the transcode outlives the request (minutes for
	// large files). Timeouts are enforced by the service itself.
	go func() {
		_, _, _ = h.svc.Ensure(context.Background(), id)
	}()
	return c.Status(202).JSON(fiber.Map{"status": "generating"})
}
