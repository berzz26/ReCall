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
