// Command codec-experiment is a TEMPORARY standalone entry point for the
// CodecSight signal inspection experiment. It runs the exact same analysis as
// the in-pipeline hook in processing.FFprobeProcessor.Process, but against any
// video file on disk, so signals can be inspected without ingesting a video.
//
// Delete this command together with services/api/internal/codec_experiment and
// workers/codec_experiment once the investigation is finished.
package main

import (
	"context"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"

	"github.com/joho/godotenv"

	"github.com/berzz26/recall/services/api/internal/codec_experiment"
)

func main() {
	_ = godotenv.Load()

	input := flag.String("input", "", "path to an encoded video file (required)")
	videoID := flag.String("video-id", "", "optional label recorded in the artifacts")
	outDir := flag.String("out", "./codec_experiment", "output directory")
	ffprobePath := flag.String("ffprobe", envOr("FFPROBE_PATH", "ffprobe"), "ffprobe binary")
	pythonPath := flag.String("python", envOr("CODEC_EXPERIMENT_PYTHON", envOr("PYTHON_PATH", "python3")), "python interpreter used for the libavcodec extractor")
	scriptPath := flag.String("script", envOr("CODEC_EXPERIMENT_SCRIPT", "workers/codec_experiment/mvdump.py"), "path to mvdump.py")
	maxFrames := flag.Int("max-frames", envIntOr("CODEC_EXPERIMENT_MAX_FRAMES", 0), "0 = no limit")
	timeout := flag.Duration("timeout", 10*time.Minute, "overall extraction timeout")
	dynamic := flag.Bool("dynamic", envBoolOr("CODECSIGHT_DYNAMIC_SAMPLING", false), "also build the experimental dynamic sampling plan")
	baselineInterval := flag.Duration("baseline-interval", envDurationOr("FRAME_SAMPLE_INTERVAL", 2*time.Second), "baseline sampling interval for the dynamic plan")
	denseInterval := flag.Duration("dense-interval", envDurationOr("CODECSIGHT_DYNAMIC_DENSE_INTERVAL", 200*time.Millisecond), "experimental dense sampling interval")
	windowBefore := flag.Duration("window-before", envDurationOr("CODECSIGHT_DYNAMIC_WINDOW_BEFORE", 1*time.Second), "experimental context before activity")
	windowAfter := flag.Duration("window-after", envDurationOr("CODECSIGHT_DYNAMIC_WINDOW_AFTER", 1*time.Second), "experimental context after activity")
	mergeGap := flag.Duration("merge-gap", envDurationOr("CODECSIGHT_DYNAMIC_MERGE_GAP", 1*time.Second), "merge nearby activity windows separated by at most this gap")
	weightMean := flag.Float64("weight-mean", envFloatOr("CODECSIGHT_DYNAMIC_WEIGHT_MEAN", 0.40), "experimental motion-mean weight")
	weightMax := flag.Float64("weight-max", envFloatOr("CODECSIGHT_DYNAMIC_WEIGHT_MAX", 0.20), "experimental motion-max weight")
	weightStatic := flag.Float64("weight-static", envFloatOr("CODECSIGHT_DYNAMIC_WEIGHT_STATIC", 0.25), "experimental static-motion weight")
	weightPacket := flag.Float64("weight-packet", envFloatOr("CODECSIGHT_DYNAMIC_WEIGHT_PACKET", 0.15), "experimental packet-size weight")
	enterThreshold := flag.Float64("enter-threshold", envFloatOr("CODECSIGHT_DYNAMIC_ENTER_THRESHOLD", 0.90), "experimental dense-mode enter threshold")
	exitThreshold := flag.Float64("exit-threshold", envFloatOr("CODECSIGHT_DYNAMIC_EXIT_THRESHOLD", 0.70), "experimental dense-mode exit threshold")
	lowPercentile := flag.Float64("low-percentile", envFloatOr("CODECSIGHT_DYNAMIC_LOW_PERCENTILE", 5), "experimental normalization low percentile")
	highPercentile := flag.Float64("high-percentile", envFloatOr("CODECSIGHT_DYNAMIC_HIGH_PERCENTILE", 95), "experimental normalization high percentile")
	flag.Parse()

	if *input == "" {
		fmt.Fprintln(os.Stderr, "codec-experiment: -input is required")
		flag.Usage()
		os.Exit(2)
	}
	if _, err := os.Stat(*input); err != nil {
		fmt.Fprintf(os.Stderr, "codec-experiment: %v\n", err)
		os.Exit(2)
	}

	svc := codec_experiment.NewService(*ffprobePath, *pythonPath, *scriptPath, *outDir, *maxFrames, *timeout)
	res, err := svc.Run(context.Background(), *input, *videoID)
	if err != nil {
		fmt.Fprintf(os.Stderr, "codec-experiment: %v\n", err)
		os.Exit(1)
	}

	fmt.Print(res.Report)

	if *dynamic {
		dynCfg := codec_experiment.DynamicConfig{
			Enabled:          true,
			BaselineInterval: *baselineInterval,
			DenseInterval:    *denseInterval,
			WindowBefore:     *windowBefore,
			WindowAfter:      *windowAfter,
			WeightMean:       *weightMean,
			WeightMax:        *weightMax,
			WeightStatic:     *weightStatic,
			WeightPacket:     *weightPacket,
			EnterThreshold:   *enterThreshold,
			ExitThreshold:    *exitThreshold,
			MergeGap:         *mergeGap,
			LowPercentile:    *lowPercentile,
			HighPercentile:   *highPercentile,
		}
		if err := codec_experiment.ValidateDynamicConfig(dynCfg, *baselineInterval); err != nil {
			fmt.Fprintf(os.Stderr, "codec-experiment: %v\n", err)
			os.Exit(2)
		}
		plan, err := codec_experiment.BuildDynamicPlan(res.Analysis, res.Analysis.Duration, *baselineInterval, dynCfg)
		if err != nil {
			fmt.Fprintf(os.Stderr, "codec-experiment: %v\n", err)
			os.Exit(1)
		}
		dynJSON, dynPNG, err := codec_experiment.WriteDynamicArtifacts(*outDir, res.Analysis, plan)
		if err != nil {
			fmt.Fprintf(os.Stderr, "codec-experiment: %v\n", err)
			os.Exit(1)
		}
		fmt.Print(codec_experiment.FormatDynamicReport(res.Analysis, plan, dynJSON, dynPNG, 0))
	}
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func envIntOr(key string, fallback int) int {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	var parsed int
	if _, err := fmt.Sscanf(v, "%d", &parsed); err != nil || parsed < 0 {
		return fallback
	}
	return parsed
}

func envBoolOr(key string, fallback bool) bool {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	switch v2 := strings.ToLower(strings.TrimSpace(v)); v2 {
	case "1", "true", "yes", "y", "on", "enable", "enabled":
		return true
	case "0", "false", "no", "n", "off", "disable", "disabled":
		return false
	default:
		return fallback
	}
}

func envFloatOr(key string, fallback float64) float64 {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	var parsed float64
	if _, err := fmt.Sscanf(v, "%f", &parsed); err != nil {
		return fallback
	}
	return parsed
}

func envDurationOr(key string, fallback time.Duration) time.Duration {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	if d, err := time.ParseDuration(v); err == nil {
		return d
	}
	return fallback
}
