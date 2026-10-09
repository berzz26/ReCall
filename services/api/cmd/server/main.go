package main

import (
	"context"
	"log/slog"
	"os"
	"os/exec"
	"os/signal"
	"path/filepath"
	"syscall"

	"github.com/berzz26/recall/pkg/database"
	"github.com/berzz26/recall/services/api/internal/config"
	"github.com/berzz26/recall/services/api/internal/detection"
	"github.com/berzz26/recall/services/api/internal/detector"
	"github.com/berzz26/recall/services/api/internal/embedding"
	"github.com/berzz26/recall/services/api/internal/handlers"
	"github.com/berzz26/recall/services/api/internal/health"
	local_source "github.com/berzz26/recall/services/api/internal/local_source"
	"github.com/berzz26/recall/services/api/internal/playable"
	"github.com/berzz26/recall/services/api/internal/processing"
	"github.com/berzz26/recall/services/api/internal/sampler"
	"github.com/berzz26/recall/services/api/internal/search"
	"github.com/berzz26/recall/services/api/internal/segment_description"
	"github.com/berzz26/recall/services/api/internal/segment_embedding"
	"github.com/berzz26/recall/services/api/internal/storage"
	"github.com/berzz26/recall/services/api/internal/tracker"
	"github.com/berzz26/recall/services/api/internal/video"
	"github.com/berzz26/recall/services/api/internal/video_event"
	"github.com/berzz26/recall/services/api/internal/video_frame"
	"github.com/berzz26/recall/services/api/internal/video_media"
	"github.com/berzz26/recall/services/api/internal/video_processing_checkpoint"
	"github.com/berzz26/recall/services/api/internal/video_segment"
	"github.com/berzz26/recall/services/api/internal/video_track"
	"github.com/berzz26/recall/services/api/internal/vision"
	"github.com/berzz26/recall/services/api/internal/visual"
	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/gofiber/fiber/v2/middleware/logger"
	"github.com/gofiber/fiber/v2/middleware/recover"
)

func main() {
	slog.SetDefault(slog.New(slog.NewJSONHandler(os.Stdout, nil)))

	cfg := config.Load()

	db, err := database.New(cfg.DatabaseURL)
	if err != nil {
		slog.Error("failed to connect to database", "error", err)
		os.Exit(1)
	}
	defer db.Close()

	slog.Info("database connected", "env", cfg.Env)

	store, err := storage.NewLocalStorage(cfg.StorageRoot)
	if err != nil {
		slog.Error("failed to create storage", "error", err)
		os.Exit(1)
	}
	slog.Info("storage initialized", "root", store.Root())

	videoRepo := video.NewRepository(db.DB)
	videoService := video.NewServiceWithConfig(videoRepo, store, cfg.MaxUploadSize)
	videoHandler := video.NewHandler(videoService)

	videoMediaRepo := video_media.NewRepository(db.DB)
	videoMediaService := video_media.NewService(videoMediaRepo)

	videoSegmentRepo := video_segment.NewRepository(db.DB)
	videoSegmentService := video_segment.NewService(videoSegmentRepo, cfg.SegmentDuration)

	videoFrameRepo := video_frame.NewRepository(db.DB)
	videoFrameService := video_frame.NewService(videoFrameRepo, store, cfg.FrameSampleInterval, cfg.FFmpegPath, cfg.FFmpegTimeout, cfg.FrameJPEGQuality).WithSamplerBeta(cfg.SamplerBeta).WithExtractWorkers(cfg.ExtractWorkers)
	slog.Info("frame sampler configured", "sample_interval", cfg.FrameSampleInterval.String(), "sampler_beta", cfg.SamplerBeta)

	detectionRepo := detection.NewRepository(db.DB)
	scriptPath := filepath.Join("workers", "detector", "detect.py")
	if _, err := os.Stat(scriptPath); err != nil {
		if abs, err2 := filepath.Abs(scriptPath); err2 == nil {
			if _, err3 := os.Stat(abs); err3 == nil {
				scriptPath = abs
			}
		}
		if _, err := os.Stat(scriptPath); err != nil {
			alt := "/home/berzz/recall/workers/detector/detect.py"
			if _, err2 := os.Stat(alt); err2 == nil {
				scriptPath = alt
			}
		}
	} else {
		if abs, err := filepath.Abs(scriptPath); err == nil {
			scriptPath = abs
		}
	}
	yolo := detector.NewYoloDetector(cfg.PythonPath, scriptPath, cfg.ModelPath, cfg.DetectionThreshold)
	visualService := visual.NewServiceWithBatchSize(detectionRepo, videoFrameRepo, store, yolo, cfg.DetectionThreshold, cfg.DetectorName, cfg.DetectorVersion, cfg.YOLOBatchSize)

	trackRepo := video_track.NewRepository(db.DB)
	selectedTracker, err := tracker.New(cfg.TrackerType, cfg.TrackerHighThreshold, cfg.TrackerLowThreshold, cfg.TrackerMatchThreshold, cfg.TrackerTrackBuffer, cfg.TrackerFuseScore, cfg.TrackerMinHits)
	if err != nil {
		slog.Error("failed to create tracker", "error", err, "tracker_type", cfg.TrackerType)
		os.Exit(1)
	}
	slog.Info("tracker selected", "tracker_type", selectedTracker.Name(), "tracker_version", selectedTracker.Version())
	trackService := video_track.NewServiceWithDeps(trackRepo, videoFrameRepo, detectionRepo, selectedTracker)

	eventRepo := video_event.NewRepository(db.DB)
	eventService := video_event.NewServiceWithThreshold(eventRepo, trackRepo, videoSegmentRepo, detectionRepo, cfg.EventMovementThreshold)

	segmentDescRepo := segment_description.NewRepository(db.DB)
	var segmentDescService *segment_description.Service
	var describer vision.VisionDescriber
	if cfg.EnableVideoDescription {
		switch cfg.VisionProvider {
		case "gemini":
			describer = vision.NewGeminiDescriber(cfg.VisionModel, cfg.VisionModelVersion, cfg.VisionMaxFrames, cfg.VisionMaxOutputTokens, cfg.VisionTimeout, cfg.GeminiAPIKey, store)
			slog.Info("vision provider selected", "provider", "gemini", "model", cfg.VisionModel, "version", cfg.VisionModelVersion, "max_frames", cfg.VisionMaxFrames, "max_output_tokens", cfg.VisionMaxOutputTokens)
		case "local":
			visionScriptPath := filepath.Join("workers", "vision", "describe.py")
			if _, err := os.Stat(visionScriptPath); err != nil {
				if abs, err2 := filepath.Abs(visionScriptPath); err2 == nil {
					if _, err3 := os.Stat(abs); err3 == nil {
						visionScriptPath = abs
					}
				}
				if _, err := os.Stat(visionScriptPath); err != nil {
					alt := "/home/berzz/recall/workers/vision/describe.py"
					if _, err2 := os.Stat(alt); err2 == nil {
						visionScriptPath = alt
					}
				}
			} else {
				if abs, err := filepath.Abs(visionScriptPath); err == nil {
					visionScriptPath = abs
				}
			}
			describer = vision.NewSmolVLMDescriberWithConfig(cfg.VisionPythonPath, visionScriptPath, cfg.VisionModel, cfg.VisionModelPath, cfg.VisionModelVersion, cfg.VisionMaxFrames, cfg.VisionMaxOutputTokens, store, cfg.VisionTimeout)
			slog.Info("vision provider selected", "provider", "local", "model", cfg.VisionModel, "model_path", cfg.VisionModelPath, "version", cfg.VisionModelVersion, "max_frames", cfg.VisionMaxFrames, "max_output_tokens", cfg.VisionMaxOutputTokens)
		default:
			slog.Error("unsupported vision provider", "provider", cfg.VisionProvider)
			os.Exit(1)
		}
		segmentDescService = segment_description.NewServiceWithHistory(segmentDescRepo, videoSegmentRepo, videoFrameRepo, detectionRepo, trackRepo, eventRepo, describer, cfg.VisionModel, cfg.VisionModelVersion, cfg.VisionHistorySegments, cfg.VisionHistoryEvents)
		slog.Info("video description pipeline enabled", "provider", cfg.VisionProvider, "model", cfg.VisionModel, "version", cfg.VisionModelVersion, "history_segments", cfg.VisionHistorySegments, "history_events", cfg.VisionHistoryEvents)
	} else {
		slog.Info("video description pipeline disabled via ENABLE_VIDEO_DESCRIPTION=false — VLM generation will be skipped")
	}

	localSourceRepo := local_source.NewRepository(db.DB)
	localSourceService := local_source.NewService(localSourceRepo, videoService, cfg.StabilityDuration)
	localSourceHandler := local_source.NewHandler(localSourceService)

	healthHandler := health.NewHandler(db.DB)

	if err := localSourceService.StartAllWatchers(context.Background()); err != nil {
		slog.Error("failed to start watchers", "error", err)
	}

	if _, err := os.Stat(cfg.FFprobePath); err != nil {
		if _, err2 := exec.LookPath(cfg.FFprobePath); err2 != nil {
			slog.Warn("ffprobe not found, processing will fail", "path", cfg.FFprobePath, "error", err2)
		}
	}

	if _, err := exec.LookPath(cfg.FFmpegPath); err != nil {
		slog.Warn("ffmpeg not found, frame extraction will fail", "path", cfg.FFmpegPath, "error", err)
	}

	// Embedding setup (B3: persistent worker keeps model resident across
	// ingest + search queries instead of reloading per call).
	embedRepo := segment_embedding.NewRepository(db.DB)
	embedder := embedding.NewBGEEmbedder(cfg.EmbeddingPythonPath, "workers/embedding/embed.py", cfg.EmbeddingTimeout)
	embedder.StartPersistent(context.Background())
	defer embedder.Close()
	embedService := segment_embedding.NewService(embedRepo, segmentDescRepo, embedder, cfg.EmbeddingModel, cfg.EmbeddingModelVersion)
	if cfg.EnableVideoDescription {
		slog.Info("embedding provider selected", "model", cfg.EmbeddingModel, "version", cfg.EmbeddingModelVersion)
	} else {
		slog.Info("embedding generation will be skipped when descriptions disabled")
	}
	// Wire embedding into pipeline; if descriptions disabled, embedding will be skipped via nil check
	var embedServiceForPipeline *segment_embedding.Service
	if cfg.EnableVideoDescription {
		embedServiceForPipeline = embedService
	}

	checkpointRepo := video_processing_checkpoint.NewRepository(db.DB)
	processor := processing.NewFFprobeProcessorWithCheckpoints(cfg.FFprobePath, cfg.FFprobeTimeout, store, videoMediaService, videoSegmentService, videoFrameService, visualService, trackService, eventService, segmentDescService, embedServiceForPipeline, checkpointRepo)
	if cfg.SamplerAdaptive {
		// Phase 5 production boundary: probe → coarse/busy/refine →
		// validated plan → extraction, with baseline fallbacks. Planning
		// detections stay ephemeral; production YOLO runs once downstream
		// over the final frame set. SAMPLER_ADAPTIVE=false (or shadow)
		// keeps baseline frames authoritative.
		adaptiveCfg := sampler.AdaptiveConfig{
			BaselineInterval: cfg.FrameSampleInterval,
			Beta:             cfg.SamplerBeta,
			ProbeFPS:         cfg.ProbeFPS,
			ProbeSize:        cfg.ProbeSize,
			ProbeGrid:        cfg.ProbeGrid,
			NoiseK:           cfg.ProbeNoiseK,
			CoarseInterval:   cfg.CoarseInterval,
			MaxGap:           cfg.CoarseMaxGap,
			FKeep:            cfg.CoarseFKeep,
			FFmpegPath:       cfg.FFmpegPath,
			PlanTimeout:      cfg.SamplerPlanTimeout,
		}
		refineCfg := sampler.RefinementConfig{
			Gamma:         cfg.SamplerGamma,
			MinGapSeconds: cfg.SamplerMinGap.Seconds(),
			Epsilon:       cfg.SamplerEpsilon,
			MaxRounds:     cfg.SamplerMaxRounds,
			ActivityFloor: sampler.DefaultProbeActivityFloor,
			Disagreement: sampler.DisagreementConfig{
				HighThreshold:  cfg.TrackerHighThreshold,
				LowThreshold:   cfg.TrackerLowThreshold,
				MatchThreshold: cfg.TrackerMatchThreshold,
				FuseScore:      cfg.TrackerFuseScore,
			},
		}
		orchestrator, err := processing.NewAdaptiveSampler(
			videoFrameService,
			visualService,
			sampler.NewAdaptiveCoarsePlanner(adaptiveCfg),
			sampler.NewRefiner(refineCfg),
			processing.AdaptiveSamplerConfig{
				Busy:             sampler.BusyConfig{Threshold: cfg.SamplerBusyThreshold, Fraction: cfg.SamplerBusyFraction},
				Timeout:          cfg.SamplerPlanTimeout,
				Shadow:           cfg.SamplerShadow,
				BaselineInterval: cfg.FrameSampleInterval,
				Beta:             cfg.SamplerBeta,
			},
		)
		if err != nil {
			slog.Error("failed to configure adaptive sampler", "error", err)
			os.Exit(1)
		}
		processor.WithAdaptiveSampler(orchestrator)
		slog.Info("adaptive sampler enabled",
			"sampler_version", sampler.SamplerVersion,
			"probe_fps", cfg.ProbeFPS, "probe_size", cfg.ProbeSize, "probe_grid", cfg.ProbeGrid,
			"noise_k", cfg.ProbeNoiseK, "f_keep", cfg.CoarseFKeep,
			"g", cfg.CoarseInterval.String(), "max_gap", cfg.CoarseMaxGap.String(),
			"gamma", cfg.SamplerGamma, "min_gap", cfg.SamplerMinGap.String(),
			"epsilon", cfg.SamplerEpsilon, "max_rounds", cfg.SamplerMaxRounds,
			"busy_threshold", cfg.SamplerBusyThreshold, "busy_fraction", cfg.SamplerBusyFraction,
			"plan_timeout", cfg.SamplerPlanTimeout.String(), "shadow", cfg.SamplerShadow)
	} else {
		slog.Info("adaptive sampler disabled via SAMPLER_ADAPTIVE=false; using fixed baseline plan")
	}
	searchHandler := handlers.NewSearchHandler(embedder, embedRepo)
	searchService := search.NewService(embedder, embedRepo, db.DB, videoRepo, cfg.SearchCandidateLimit, cfg.SearchDefaultLimit, cfg.SearchMaxLimit, cfg.SearchMinSimilarity)
	unifiedSearchHandler := handlers.NewUnifiedSearchHandler(searchService)
	worker := processing.NewWorker(videoService, processor, cfg.PollInterval)
	// Browser-playable proxy (H.264 sidecar) for codecs browsers cannot
	// decode (e.g. MPEG-4 Part 2). Runs before each processing job; never
	// fails the video on transcode errors. Also serves existing videos via
	// POST /api/v1/videos/:id/playable below.
	playableService := playable.NewService(videoRepo, store, cfg.FFprobePath, cfg.FFmpegPath, cfg.PlayableTimeout, nil)
	playableHandler := playable.NewHandler(playableService)
	worker.WithPlayable(playableService)
	workerCtx, workerCancel := context.WithCancel(context.Background())
	go worker.Start(workerCtx)

	// Fiber's default BodyLimit is 4 MiB, which rejects any real video upload
	// with 413 (the web dev proxy surfaces the reset connection as
	// ECONNRESET). Derive the HTTP body cap from the configured max upload
	// size plus headroom for multipart framing; the service layer still
	// enforces MaxUploadSize on the file bytes themselves.
	bodyLimit := int(cfg.MaxUploadSize) + (32 << 20)
	app := fiber.New(fiber.Config{
		BodyLimit: bodyLimit,
		// Stream request bodies instead of buffering them in RAM: the upload
		// handler reads the multipart file part as a stream straight to disk.
		// DisablePreParseMultipartForm is required so fasthttp does not parse
		// multipart itself (which would leave BodyStream() nil); the handler
		// parses the stream with multipart.Reader instead. Other endpoints
		// use small JSON bodies, which Body()/BodyParser still read on demand.
		// (BodyLimit is still enforced by fasthttp while streaming.)
		StreamRequestBody:            true,
		DisablePreParseMultipartForm: true,
		ErrorHandler: func(c *fiber.Ctx, err error) error {
			code := fiber.StatusInternalServerError
			if e, ok := err.(*fiber.Error); ok {
				code = e.Code
			}
			return c.Status(code).JSON(fiber.Map{"error": err.Error()})
		},
	})

	app.Use(recover.New())
	app.Use(cors.New(cors.Config{
		AllowOrigins:  "*",
		AllowHeaders:  "*",
		ExposeHeaders: "Content-Range, Accept-Ranges, Content-Length, Content-Type",
		AllowMethods:  "GET,POST,PUT,PATCH,DELETE,OPTIONS",
	}))
	if cfg.Env == "development" {
		app.Use(logger.New(logger.Config{
			Format: "[${time}] ${status} - ${latency} ${method} ${path}\n",
		}))
	}

	app.Get("/health", healthHandler.Check)

	detailHandler := handlers.NewVideoDetailHandlerWithDescriptions(videoMediaRepo, videoSegmentRepo, videoFrameRepo, detectionRepo, trackRepo, eventRepo, segmentDescRepo, store)
	videoStreamHandler := handlers.NewVideoStreamHandler(videoRepo, store)

	api := app.Group("/api")
	v1 := api.Group("/v1")
	// The /videos sub-app owns its route space (a parent-level
	// /videos/:id/* route registered after Mount would be shadowed), so the
	// proxy endpoint is attached to the sub-app directly.
	videoRoutes := videoHandler.SetupRoutes(bodyLimit)
	videoRoutes.Post("/:id/playable", playableHandler.EnsureProxy)
	v1.Mount("/videos", videoRoutes)
	v1.Get("/videos/:id/media", detailHandler.GetMedia)
	v1.Get("/videos/:id/stream", videoStreamHandler.Stream)
	v1.Get("/videos/:id/segments", detailHandler.GetSegments)
	v1.Get("/videos/:id/frames", detailHandler.GetFrames)
	v1.Get("/videos/:id/detections", detailHandler.GetDetections)
	v1.Get("/videos/:id/frames/:frameId/image", detailHandler.GetFrameImage)
	v1.Get("/videos/:id/tracks", detailHandler.GetTracks)
	v1.Get("/tracks/:trackId/detections", detailHandler.GetTrackDetections)
	v1.Get("/videos/:id/events", detailHandler.GetEvents)
	v1.Get("/tracks/:trackId/events", detailHandler.GetTrackEvents)
	v1.Get("/videos/:id/descriptions", detailHandler.GetDescriptions)
	v1.Get("/videos/:id/segments/:segmentId/description", detailHandler.GetSegmentDescription)
	// B4: scoped evidence for search deep-links (avoids full-video refetch).
	evidenceHandler := handlers.NewSegmentEvidenceHandler(db.DB, videoSegmentRepo, videoFrameRepo, detectionRepo, trackRepo, eventRepo, segmentDescRepo)
	v1.Get("/videos/:id/segments/:segmentId/evidence", evidenceHandler.GetEvidence)
	v1.Post("/ingest/local", videoHandler.IngestLocal)
	v1.Mount("/local-sources", localSourceHandler.SetupRoutes())
	v1.Post("/search/semantic", searchHandler.Search)
	v1.Post("/search", unifiedSearchHandler.Search)

	sigCtx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	go func() {
		slog.Info("starting server", "addr", cfg.Addr, "env", cfg.Env)
		if err := app.Listen(cfg.Addr); err != nil {
			slog.Error("server listen error", "error", err)
			os.Exit(1)
		}
	}()

	<-sigCtx.Done()
	slog.Info("shutdown signal received, shutting down")

	workerCancel()
	localSourceService.StopAllWatchers()

	if err := app.Shutdown(); err != nil {
		slog.Error("graceful shutdown failed", "error", err)
		os.Exit(1)
	}

	db.Close()
	slog.Info("server stopped")
}
