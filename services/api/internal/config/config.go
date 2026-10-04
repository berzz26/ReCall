package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"

	"github.com/joho/godotenv"
)

type Config struct {
	Env                    string
	Port                   string
	Addr                   string
	DatabaseURL            string
	StorageRoot            string
	MaxUploadSize          int64
	StabilitySeconds       int
	StabilityDuration      time.Duration
	PollInterval           time.Duration
	FFprobePath            string
	FFprobeTimeout         time.Duration
	SegmentDuration        time.Duration
	FrameSampleInterval    time.Duration
	SamplerBeta            float64
	SamplerAdaptive        bool
	ProbeFPS               float64
	ProbeSize              int
	ProbeGrid              int
	ProbeNoiseK            float64
	CoarseInterval         time.Duration
	CoarseMaxGap           time.Duration
	CoarseFKeep            float64
	SamplerPlanTimeout     time.Duration
	FFmpegPath             string
	FFmpegTimeout          time.Duration
	FrameJPEGQuality       int
	DetectionThreshold     float64
	DetectorName           string
	DetectorVersion        string
	ModelPath              string
	PythonPath             string
	YOLOBatchSize          int
	EventMovementThreshold float64
	VisionProvider         string
	VisionPythonPath       string
	VisionModel            string
	VisionModelPath        string
	VisionModelVersion     string
	VisionMaxFrames        int
	VisionMaxOutputTokens  int
	VisionTimeout          time.Duration
	VisionHistorySegments  int
	VisionHistoryEvents    int
	GeminiAPIKey           string
	EmbeddingPythonPath    string
	EmbeddingModel         string
	EmbeddingModelVersion  string
	EmbeddingTimeout       time.Duration
	SearchCandidateLimit   int
	SearchDefaultLimit     int
	SearchMaxLimit         int
	SearchMinSimilarity    float64
	EnableVideoDescription bool
	TrackerType            string
	TrackerHighThreshold   float64
	TrackerLowThreshold    float64
	TrackerMatchThreshold  float64
	TrackerTrackBuffer     int
	TrackerFuseScore       bool
	TrackerMinHits         int
}

func Load() Config {
	_ = godotenv.Load()

	env := os.Getenv("APP_ENV")
	if env == "" {
		env = "development"
	}

	port := os.Getenv("APP_PORT")
	if port == "" {
		port = "8080"
	}

	dbURL := os.Getenv("DATABASE_URL")
	if dbURL == "" {
		dbURL = "postgres://recall:recall@localhost:5436/recall?sslmode=disable"
	}

	storageRoot := os.Getenv("STORAGE_ROOT")
	if storageRoot == "" {
		storageRoot = "./storage"
	}

	maxUploadSize := int64(1 << 30)
	if v := os.Getenv("MAX_UPLOAD_SIZE"); v != "" {
		var parsed int64
		if _, err := fmt.Sscanf(v, "%d", &parsed); err == nil && parsed > 0 {
			maxUploadSize = parsed
		}
	}

	stabilitySeconds := 5
	if v := os.Getenv("LOCAL_FILE_STABILITY_SECONDS"); v != "" {
		if parsed, err := strconv.Atoi(v); err == nil && parsed > 0 {
			stabilitySeconds = parsed
		}
	}

	pollInterval := 2 * time.Second
	if v := os.Getenv("PROCESSING_POLL_INTERVAL"); v != "" {
		if d, err := time.ParseDuration(v); err == nil && d > 0 {
			pollInterval = d
		}
	}

	ffprobePath := os.Getenv("FFPROBE_PATH")
	if ffprobePath == "" {
		ffprobePath = "ffprobe"
	}

	ffprobeTimeout := 60 * time.Second
	if v := os.Getenv("FFPROBE_TIMEOUT"); v != "" {
		if d, err := time.ParseDuration(v); err == nil && d > 0 {
			ffprobeTimeout = d
		}
	}

	segmentDuration := 30 * time.Second
	if v := os.Getenv("VIDEO_SEGMENT_DURATION"); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			if d <= 0 {
				panic(fmt.Sprintf("VIDEO_SEGMENT_DURATION must be > 0, got %s", v))
			}
			segmentDuration = d
		} else {
			panic(fmt.Sprintf("invalid VIDEO_SEGMENT_DURATION %q: %v", v, err))
		}
	}

	frameSampleInterval := 2 * time.Second
	if v := os.Getenv("FRAME_SAMPLE_INTERVAL"); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			if d <= 0 {
				panic(fmt.Sprintf("FRAME_SAMPLE_INTERVAL must be > 0, got %s", v))
			}
			frameSampleInterval = d
		} else {
			panic(fmt.Sprintf("invalid FRAME_SAMPLE_INTERVAL %q: %v", v, err))
		}
	}

	ffmpegPath := os.Getenv("FFMPEG_PATH")
	if ffmpegPath == "" {
		ffmpegPath = "ffmpeg"
	}

	// Sampler budget scale: B = ceil(beta * N_baseline). Phase 1 uses 1.0,
	// so the adaptive sampler never sends more than N_baseline frames to YOLO.
	samplerBeta := 1.0
	if v := os.Getenv("SAMPLER_BETA"); v != "" {
		parsed, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
		if err != nil {
			panic(fmt.Sprintf("invalid SAMPLER_BETA %q: %v", v, err))
		}
		if parsed <= 0 {
			panic(fmt.Sprintf("SAMPLER_BETA must be > 0, got %s", v))
		}
		samplerBeta = parsed
	}

	// Phase 2 adaptive coarse sampler: visual probe (5 FPS, 64x64 gray, 8x8
	// grid) driving coarse/heartbeat selection within the baseline budget.
	// SAMPLER_ADAPTIVE=false restores the fixed 2-second baseline plan.
	samplerAdaptive := true
	if v := os.Getenv("SAMPLER_ADAPTIVE"); v != "" {
		switch v2 := strings.ToLower(strings.TrimSpace(v)); v2 {
		case "1", "true", "yes", "y", "on", "enable", "enabled":
			samplerAdaptive = true
		case "0", "false", "no", "n", "off", "disable", "disabled":
			samplerAdaptive = false
		default:
			panic(fmt.Sprintf("invalid SAMPLER_ADAPTIVE %q: must be boolean (true/false, 1/0, yes/no, on/off)", v))
		}
	}

	probeFPS := 5.0
	if v := os.Getenv("PROBE_FPS"); v != "" {
		parsed, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
		if err != nil || parsed <= 0 {
			panic(fmt.Sprintf("invalid PROBE_FPS %q: must be > 0", v))
		}
		probeFPS = parsed
	}

	probeSize := 64
	if v := os.Getenv("PROBE_SIZE"); v != "" {
		parsed, err := strconv.Atoi(strings.TrimSpace(v))
		if err != nil || parsed <= 0 {
			panic(fmt.Sprintf("invalid PROBE_SIZE %q: must be > 0", v))
		}
		probeSize = parsed
	}

	probeGrid := 8
	if v := os.Getenv("PROBE_GRID"); v != "" {
		parsed, err := strconv.Atoi(strings.TrimSpace(v))
		if err != nil || parsed <= 0 {
			panic(fmt.Sprintf("invalid PROBE_GRID %q: must be > 0", v))
		}
		probeGrid = parsed
	}
	if probeSize%probeGrid != 0 {
		panic(fmt.Sprintf("PROBE_SIZE (%d) must be divisible by PROBE_GRID (%d)", probeSize, probeGrid))
	}

	probeNoiseK := 3.0
	if v := os.Getenv("PROBE_NOISE_K"); v != "" {
		parsed, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
		if err != nil || parsed < 0 {
			panic(fmt.Sprintf("invalid PROBE_NOISE_K %q: must be >= 0", v))
		}
		probeNoiseK = parsed
	}

	coarseInterval := 4 * time.Second
	if v := os.Getenv("COARSE_INTERVAL"); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			if d <= 0 {
				panic(fmt.Sprintf("COARSE_INTERVAL must be > 0, got %s", v))
			}
			coarseInterval = d
		} else {
			panic(fmt.Sprintf("invalid COARSE_INTERVAL %q: %v", v, err))
		}
	}

	coarseMaxGap := 10 * time.Second
	if v := os.Getenv("COARSE_MAX_GAP"); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			if d <= 0 {
				panic(fmt.Sprintf("COARSE_MAX_GAP must be > 0, got %s", v))
			}
			coarseMaxGap = d
		} else {
			panic(fmt.Sprintf("invalid COARSE_MAX_GAP %q: %v", v, err))
		}
	}

	coarseFKeep := 0.25
	if v := os.Getenv("COARSE_F_KEEP"); v != "" {
		parsed, err := strconv.ParseFloat(strings.TrimSpace(v), 64)
		if err != nil || parsed < 0 || parsed > 1 {
			panic(fmt.Sprintf("invalid COARSE_F_KEEP %q: must be 0..1", v))
		}
		coarseFKeep = parsed
	}

	samplerPlanTimeout := 5 * time.Minute
	if v := os.Getenv("SAMPLER_PLAN_TIMEOUT"); v != "" {
		if d, err := time.ParseDuration(v); err == nil && d > 0 {
			samplerPlanTimeout = d
		} else {
			panic(fmt.Sprintf("invalid SAMPLER_PLAN_TIMEOUT %q", v))
		}
	}

	ffmpegTimeout := 60 * time.Second
	if v := os.Getenv("FFMPEG_TIMEOUT"); v != "" {
		if d, err := time.ParseDuration(v); err == nil && d > 0 {
			ffmpegTimeout = d
		}
	}

	frameJPEGQuality := 85
	if v := os.Getenv("FRAME_JPEG_QUALITY"); v != "" {
		if parsed, err := strconv.Atoi(v); err == nil {
			if parsed < 1 || parsed > 100 {
				panic(fmt.Sprintf("FRAME_JPEG_QUALITY must be 1-100, got %s", v))
			}
			frameJPEGQuality = parsed
		} else {
			panic(fmt.Sprintf("invalid FRAME_JPEG_QUALITY %q: %v", v, err))
		}
	}

	detectionThreshold := 0.35
	if v := os.Getenv("DETECTION_CONFIDENCE_THRESHOLD"); v != "" {
		if parsed, err := strconv.ParseFloat(v, 64); err == nil {
			if parsed < 0 || parsed > 1 {
				panic(fmt.Sprintf("DETECTION_CONFIDENCE_THRESHOLD must be 0..1, got %s", v))
			}
			detectionThreshold = parsed
		} else {
			panic(fmt.Sprintf("invalid DETECTION_CONFIDENCE_THRESHOLD %q: %v", v, err))
		}
	}

	detectorNameRaw := os.Getenv("DETECTOR_NAME")
	detectorVersion := os.Getenv("DETECTOR_VERSION")
	if detectorVersion == "" {
		detectorVersion = "1"
	}
	modelPathRaw := os.Getenv("MODEL_PATH")
	pythonPath := os.Getenv("PYTHON_PATH")
	if pythonPath == "" {
		pythonPath = "python3"
	}

	// Fix redundancy: MODEL_PATH and DETECTOR_NAME were duplicated.
	// Now DETECTOR_NAME is the single source of truth; MODEL_PATH is derived if not set.
	// If MODEL_PATH is explicitly set and DETECTOR_NAME is not, derive DETECTOR_NAME from MODEL_PATH basename.
	// If both are explicitly set, validate consistency (basename without ext should match DETECTOR_NAME).
	var detectorName string
	var modelPath string
	switch {
	case detectorNameRaw != "" && modelPathRaw == "":
		detectorName = detectorNameRaw
		modelPath = filepath.Join("workers", "detector", detectorName+".pt")
	case detectorNameRaw == "" && modelPathRaw != "":
		modelPath = modelPathRaw
		base := filepath.Base(modelPath)
		ext := filepath.Ext(base)
		derived := strings.TrimSuffix(base, ext)
		if derived == "" {
			derived = "yolov8n"
		}
		detectorName = derived
	case detectorNameRaw != "" && modelPathRaw != "":
		detectorName = detectorNameRaw
		modelPath = modelPathRaw
		// Validate consistency but don't fail - just ensure derived name matches if user set both
		base := filepath.Base(modelPath)
		derived := strings.TrimSuffix(base, filepath.Ext(base))
		if derived != "" && derived != detectorName {
			// Keep MODEL_PATH as explicit override, but keep DETECTOR_NAME as set.
			// This allows custom paths like /home/berzz/.../yolo11n.pt with DETECTOR_NAME=yolo11n (consistent case)
			// Mismatch is tolerated to allow versioned file names without renaming detector.
		}
	default:
		detectorName = "yolov8n"
		modelPath = filepath.Join("workers", "detector", "yolov8n.pt")
	}
	yoloBatchSize := 16
	if v := os.Getenv("YOLO_BATCH_SIZE"); v != "" {
		parsed, err := strconv.Atoi(strings.TrimSpace(v))
		if err != nil {
			panic(fmt.Sprintf("invalid YOLO_BATCH_SIZE %q: %v", v, err))
		}
		if parsed <= 0 {
			panic(fmt.Sprintf("YOLO_BATCH_SIZE must be > 0, got %s", v))
		}
		yoloBatchSize = parsed
	}

	movementThreshold := 0.05
	if v := os.Getenv("EVENT_MOVEMENT_THRESHOLD"); v != "" {
		if parsed, err := strconv.ParseFloat(v, 64); err == nil {
			if parsed <= 0 || parsed > 1 {
				panic(fmt.Sprintf("EVENT_MOVEMENT_THRESHOLD must be 0..1, got %s", v))
			}
			movementThreshold = parsed
		} else {
			panic(fmt.Sprintf("invalid EVENT_MOVEMENT_THRESHOLD %q: %v", v, err))
		}
	}

	visionProvider := os.Getenv("VISION_PROVIDER")
	if visionProvider == "" {
		visionProvider = "local"
	}
	visionProvider = strings.ToLower(strings.TrimSpace(visionProvider))
	if visionProvider != "local" && visionProvider != "gemini" {
		panic(fmt.Sprintf("invalid VISION_PROVIDER %q: must be one of [local, gemini]", visionProvider))
	}
	visionPythonPath := os.Getenv("VISION_PYTHON_PATH")
	if visionPythonPath == "" {
		visionPythonPath = "python3"
	}
	// Default vision model depends on provider for better DX, but all values remain configurable
	defaultVisionModel := "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
	defaultVisionVersion := "500M-Instruct"
	if visionProvider == "gemini" {
		defaultVisionModel = "gemini-2.5-flash-lite"
		defaultVisionVersion = "2.5-flash-lite"
	}
	visionModel := os.Getenv("VISION_MODEL")
	if visionModel == "" {
		visionModel = defaultVisionModel
	}
	visionModelPath := os.Getenv("VISION_MODEL_PATH")
	if visionModelPath == "" {
		// For local, default path mirrors model name; for gemini the path is irrelevant
		if visionProvider == "local" {
			visionModelPath = "/models/SmolVLM2-500M-Video-Instruct"
		}
	}
	visionVersion := os.Getenv("VISION_MODEL_VERSION")
	if visionVersion == "" {
		visionVersion = defaultVisionVersion
	}
	visionMaxFrames := 3
	if v := os.Getenv("VISION_MAX_FRAMES"); v != "" {
		parsed, err := strconv.Atoi(v)
		if err != nil || parsed < 1 {
			panic(fmt.Sprintf("invalid VISION_MAX_FRAMES %q: must be >= 1", v))
		}
		visionMaxFrames = parsed
	}
	visionMaxOutputTokens := 256
	if v := os.Getenv("VISION_MAX_OUTPUT_TOKENS"); v != "" {
		parsed, err := strconv.Atoi(v)
		if err != nil || parsed < 1 {
			panic(fmt.Sprintf("invalid VISION_MAX_OUTPUT_TOKENS %q: must be >= 1", v))
		}
		visionMaxOutputTokens = parsed
	}
	visionTimeout := 10 * time.Minute
	if v := os.Getenv("VISION_TIMEOUT"); v != "" {
		if d, err := time.ParseDuration(v); err == nil && d > 0 {
			visionTimeout = d
		} else {
			panic(fmt.Sprintf("invalid VISION_TIMEOUT %q", v))
		}
	}
	visionHistorySegments := 2
	if v := os.Getenv("VISION_HISTORY_SEGMENTS"); v != "" {
		parsed, err := strconv.Atoi(strings.TrimSpace(v))
		if err != nil {
			panic(fmt.Sprintf("invalid VISION_HISTORY_SEGMENTS %q: %v", v, err))
		}
		if parsed < 0 {
			panic(fmt.Sprintf("VISION_HISTORY_SEGMENTS must be >= 0, got %s", v))
		}
		visionHistorySegments = parsed
	}
	visionHistoryEvents := 5
	if v := os.Getenv("VISION_HISTORY_EVENTS"); v != "" {
		parsed, err := strconv.Atoi(strings.TrimSpace(v))
		if err != nil {
			panic(fmt.Sprintf("invalid VISION_HISTORY_EVENTS %q: %v", v, err))
		}
		if parsed < 0 {
			panic(fmt.Sprintf("VISION_HISTORY_EVENTS must be >= 0, got %s", v))
		}
		visionHistoryEvents = parsed
	}

	geminiAPIKey := strings.TrimSpace(os.Getenv("GEMINI_API_KEY"))

	// Validation: provider-specific required fields
	if visionProvider == "gemini" && geminiAPIKey == "" {
		panic("GEMINI_API_KEY is required when VISION_PROVIDER=gemini")
	}
	if visionProvider == "local" {
		if visionModel == "" {
			panic("VISION_MODEL is required when VISION_PROVIDER=local")
		}
		if visionModelPath == "" {
			panic("VISION_MODEL_PATH is required when VISION_PROVIDER=local")
		}
		if visionVersion == "" {
			panic("VISION_MODEL_VERSION is required when VISION_PROVIDER=local")
		}
	}

	embeddingPythonPath := os.Getenv("EMBEDDING_PYTHON_PATH")
	if embeddingPythonPath == "" {
		embeddingPythonPath = "python3"
	}
	embeddingModel := os.Getenv("EMBEDDING_MODEL")
	if embeddingModel == "" {
		embeddingModel = "BAAI/bge-small-en-v1.5"
	}
	embeddingModelVersion := os.Getenv("EMBEDDING_MODEL_VERSION")
	if embeddingModelVersion == "" {
		embeddingModelVersion = "v1.5"
	}
	embeddingTimeout := 5 * time.Minute
	if v := os.Getenv("EMBEDDING_TIMEOUT"); v != "" {
		if d, err := time.ParseDuration(v); err == nil && d > 0 {
			embeddingTimeout = d
		} else {
			panic(fmt.Sprintf("invalid EMBEDDING_TIMEOUT %q", v))
		}
	}

	searchCandidateLimit := 20
	if v := os.Getenv("SEARCH_CANDIDATE_LIMIT"); v != "" {
		parsed, err := strconv.Atoi(v)
		if err != nil || parsed < 1 {
			panic(fmt.Sprintf("invalid SEARCH_CANDIDATE_LIMIT %q: must be >= 1", v))
		}
		searchCandidateLimit = parsed
	}
	searchDefaultLimit := 10
	if v := os.Getenv("SEARCH_DEFAULT_LIMIT"); v != "" {
		parsed, err := strconv.Atoi(v)
		if err != nil || parsed < 1 {
			panic(fmt.Sprintf("invalid SEARCH_DEFAULT_LIMIT %q: must be >= 1", v))
		}
		searchDefaultLimit = parsed
	}
	searchMaxLimit := 50
	if v := os.Getenv("SEARCH_MAX_LIMIT"); v != "" {
		parsed, err := strconv.Atoi(v)
		if err != nil || parsed < 1 {
			panic(fmt.Sprintf("invalid SEARCH_MAX_LIMIT %q: must be >= 1", v))
		}
		searchMaxLimit = parsed
	}
	if searchDefaultLimit > searchMaxLimit {
		panic(fmt.Sprintf("SEARCH_DEFAULT_LIMIT (%d) must be <= SEARCH_MAX_LIMIT (%d)", searchDefaultLimit, searchMaxLimit))
	}
	if searchCandidateLimit < searchMaxLimit {
		// allow candidate < max but warn? keep as is, no panic - spec defaults 20/50 is ok
	}
	searchMinSimilarity := 0.35
	if v := os.Getenv("SEARCH_MIN_SIMILARITY"); v != "" {
		parsed, err := strconv.ParseFloat(v, 64)
		if err != nil {
			panic(fmt.Sprintf("invalid SEARCH_MIN_SIMILARITY %q: %v", v, err))
		}
		if parsed < 0 || parsed > 1 {
			panic(fmt.Sprintf("SEARCH_MIN_SIMILARITY must be 0..1, got %s", v))
		}
		searchMinSimilarity = parsed
	}

	enableVideoDescription := true
	// Primary toggle: ENABLE_VIDEO_DESCRIPTION, aliases: ENABLE_VLM, VISION_ENABLED, ENABLE_DESCRIPTION
	for _, key := range []string{"ENABLE_VIDEO_DESCRIPTION", "ENABLE_VLM", "VISION_ENABLED", "ENABLE_DESCRIPTION"} {
		if v := os.Getenv(key); v != "" {
			switch v2 := strings.ToLower(strings.TrimSpace(v)); v2 {
			case "1", "true", "yes", "y", "on", "enable", "enabled":
				enableVideoDescription = true
			case "0", "false", "no", "n", "off", "disable", "disabled":
				enableVideoDescription = false
			default:
				panic(fmt.Sprintf("invalid %s %q: must be boolean (true/false, 1/0, yes/no, on/off)", key, v))
			}
			break
		}
	}

	trackerType := os.Getenv("TRACKER_TYPE")
	if trackerType == "" {
		trackerType = "iou"
	}
	if trackerType != "iou" && trackerType != "bytetrack" {
		panic(fmt.Sprintf("invalid TRACKER_TYPE %q: must be one of [iou, bytetrack]", trackerType))
	}

	trackerHighThreshold := 0.50
	if v := os.Getenv("TRACKER_HIGH_THRESHOLD"); v != "" {
		parsed, err := strconv.ParseFloat(v, 64)
		if err != nil {
			panic(fmt.Sprintf("invalid TRACKER_HIGH_THRESHOLD %q: %v", v, err))
		}
		trackerHighThreshold = parsed
	}
	trackerLowThreshold := 0.10
	if v := os.Getenv("TRACKER_LOW_THRESHOLD"); v != "" {
		parsed, err := strconv.ParseFloat(v, 64)
		if err != nil {
			panic(fmt.Sprintf("invalid TRACKER_LOW_THRESHOLD %q: %v", v, err))
		}
		trackerLowThreshold = parsed
	}
	trackerMatchThreshold := 0.30
	if v := os.Getenv("TRACKER_MATCH_THRESHOLD"); v != "" {
		parsed, err := strconv.ParseFloat(v, 64)
		if err != nil {
			panic(fmt.Sprintf("invalid TRACKER_MATCH_THRESHOLD %q: %v", v, err))
		}
		trackerMatchThreshold = parsed
	}
	trackerTrackBuffer := 5
	if v := os.Getenv("TRACKER_TRACK_BUFFER"); v != "" {
		parsed, err := strconv.Atoi(v)
		if err != nil {
			panic(fmt.Sprintf("invalid TRACKER_TRACK_BUFFER %q: %v", v, err))
		}
		trackerTrackBuffer = parsed
	}

	trackerFuseScore := true
	if v := os.Getenv("TRACKER_FUSE_SCORE"); v != "" {
		switch strings.ToLower(strings.TrimSpace(v)) {
		case "1", "true", "yes", "y", "on":
			trackerFuseScore = true
		case "0", "false", "no", "n", "off":
			trackerFuseScore = false
		default:
			panic(fmt.Sprintf("invalid TRACKER_FUSE_SCORE %q: must be boolean", v))
		}
	}

	trackerMinHits := 2
	if v := os.Getenv("TRACKER_MIN_HITS"); v != "" {
		parsed, err := strconv.Atoi(v)
		if err != nil {
			panic(fmt.Sprintf("invalid TRACKER_MIN_HITS %q: %v", v, err))
		}
		trackerMinHits = parsed
	}

	if !(0 <= trackerLowThreshold && trackerLowThreshold < trackerHighThreshold && trackerHighThreshold <= 1) {
		panic(fmt.Sprintf("invalid tracker thresholds: must satisfy 0 <= TRACKER_LOW_THRESHOLD (%.4f) < TRACKER_HIGH_THRESHOLD (%.4f) <= 1", trackerLowThreshold, trackerHighThreshold))
	}
	if !(0 < trackerMatchThreshold && trackerMatchThreshold <= 1) {
		panic(fmt.Sprintf("invalid TRACKER_MATCH_THRESHOLD %.4f: must satisfy 0 < TRACKER_MATCH_THRESHOLD <= 1", trackerMatchThreshold))
	}
	if trackerTrackBuffer < 1 {
		panic(fmt.Sprintf("invalid TRACKER_TRACK_BUFFER %d: must be >= 1", trackerTrackBuffer))
	}
	if trackerMinHits < 1 {
		panic(fmt.Sprintf("invalid TRACKER_MIN_HITS %d: must be >= 1", trackerMinHits))
	}

	return Config{
		Env:                    env,
		Port:                   port,
		Addr:                   ":" + port,
		DatabaseURL:            dbURL,
		StorageRoot:            storageRoot,
		MaxUploadSize:          maxUploadSize,
		StabilitySeconds:       stabilitySeconds,
		StabilityDuration:      time.Duration(stabilitySeconds) * time.Second,
		PollInterval:           pollInterval,
		FFprobePath:            ffprobePath,
		FFprobeTimeout:         ffprobeTimeout,
		SegmentDuration:        segmentDuration,
		FrameSampleInterval:    frameSampleInterval,
		SamplerBeta:            samplerBeta,
		SamplerAdaptive:        samplerAdaptive,
		ProbeFPS:               probeFPS,
		ProbeSize:              probeSize,
		ProbeGrid:              probeGrid,
		ProbeNoiseK:            probeNoiseK,
		CoarseInterval:         coarseInterval,
		CoarseMaxGap:           coarseMaxGap,
		CoarseFKeep:            coarseFKeep,
		SamplerPlanTimeout:     samplerPlanTimeout,
		FFmpegPath:             ffmpegPath,
		FFmpegTimeout:          ffmpegTimeout,
		FrameJPEGQuality:       frameJPEGQuality,
		DetectionThreshold:     detectionThreshold,
		DetectorName:           detectorName,
		DetectorVersion:        detectorVersion,
		ModelPath:              modelPath,
		PythonPath:             pythonPath,
		YOLOBatchSize:          yoloBatchSize,
		EventMovementThreshold: movementThreshold,
		VisionProvider:         visionProvider,
		VisionPythonPath:       visionPythonPath,
		VisionModel:            visionModel,
		VisionModelPath:        visionModelPath,
		VisionModelVersion:     visionVersion,
		VisionMaxFrames:        visionMaxFrames,
		VisionMaxOutputTokens:  visionMaxOutputTokens,
		VisionTimeout:          visionTimeout,
		VisionHistorySegments:  visionHistorySegments,
		VisionHistoryEvents:    visionHistoryEvents,
		GeminiAPIKey:           geminiAPIKey,
		EmbeddingPythonPath:    embeddingPythonPath,
		EmbeddingModel:         embeddingModel,
		EmbeddingModelVersion:  embeddingModelVersion,
		EmbeddingTimeout:       embeddingTimeout,
		SearchCandidateLimit:   searchCandidateLimit,
		SearchDefaultLimit:     searchDefaultLimit,
		SearchMaxLimit:         searchMaxLimit,
		SearchMinSimilarity:    searchMinSimilarity,
		EnableVideoDescription: enableVideoDescription,
		TrackerType:            trackerType,
		TrackerHighThreshold:   trackerHighThreshold,
		TrackerLowThreshold:    trackerLowThreshold,
		TrackerMatchThreshold:  trackerMatchThreshold,
		TrackerTrackBuffer:     trackerTrackBuffer,
		TrackerFuseScore:       trackerFuseScore,
		TrackerMinHits:         trackerMinHits,
	}
}
