package video

import "github.com/gofiber/fiber/v2"

func (h *Handler) SetupRoutes(bodyLimit int) *fiber.App {
	// bodyLimit must cover the whole multipart body: file bytes plus framing
	// overhead. The caller derives it from the configured max upload size.
	// Fiber's default is 4 MiB, which rejects virtually any real video with
	// 413 (surfaced through the web dev proxy as ECONNRESET).
	if bodyLimit <= 0 {
		bodyLimit = 1 << 30
	}
	router := fiber.New(fiber.Config{BodyLimit: bodyLimit})

	router.Post("/", h.Create)
	router.Post("", h.Create)
	router.Get("/", h.List)
	router.Get("", h.List)
	router.Get("/:id", h.Get)
	router.Delete("/:id", h.Delete)

	return router
}
