package codec_experiment

// TEMPORARY EXPERIMENT -- see FINDINGS.md. Whole package is disposable.

import (
	"context"
	"encoding/csv"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

type Service struct {
	ffprobePath string
	pythonPath  string
	scriptPath  string
	outDir      string
	maxFrames   int
	timeout     time.Duration
}

func NewService(ffprobePath, pythonPath, scriptPath, outDir string, maxFrames int, timeout time.Duration) *Service {
	if ffprobePath == "" {
		ffprobePath = "ffprobe"
	}
	if pythonPath == "" {
		pythonPath = "python3"
	}
	if outDir == "" {
		outDir = "./codec_experiment"
	}
	if timeout <= 0 {
		timeout = 10 * time.Minute
	}
	return &Service{
		ffprobePath: ffprobePath,
		pythonPath:  pythonPath,
		scriptPath:  resolveScriptPath(scriptPath),
		outDir:      outDir,
		maxFrames:   maxFrames,
		timeout:     timeout,
	}
}

type Result struct {
	Analysis  *Analysis
	JSONPath  string
	CSVPath   string
	PNGPath   string
	Extractor string
	Report    string
}

// Run inspects an already-encoded video file and writes JSON/CSV/PNG artifacts
// plus a human readable report. It never mutates its input and has no effect on
// ReCall's processing: every stage is read-only and errors are returned, not
// raised to the caller as pipeline failures.
func (s *Service) Run(ctx context.Context, videoPath, videoID string) (*Result, error) {
	if videoPath == "" {
		return nil, fmt.Errorf("codec experiment: empty video path")
	}
	if _, err := os.Stat(videoPath); err != nil {
		return nil, fmt.Errorf("codec experiment: source not readable: %w", err)
	}

	runCtx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()

	start := time.Now()

	raw, extractErr := runPyAVExtractor(runCtx, s.pythonPath, s.scriptPath, videoPath, s.maxFrames)
	if extractErr != nil {
		slog.Warn("codec experiment: libavcodec extractor unavailable, falling back to ffprobe CLI",
			"video_id", videoID, "python", s.pythonPath, "script", s.scriptPath, "error", extractErr)
		fallback, err := runFFprobeFallback(runCtx, s.ffprobePath, videoPath, s.maxFrames)
		if err != nil {
			return nil, fmt.Errorf("codec experiment: %w (primary: %v)", err, extractErr)
		}
		raw = fallback
	}

	analysis, err := summarize(raw)
	if err != nil {
		return nil, fmt.Errorf("codec experiment: %w", err)
	}
	analysis.VideoID = videoID
	analysis.VideoName = filepath.Base(videoPath)
	analysis.Path = videoPath

	if err := os.MkdirAll(s.outDir, 0o755); err != nil {
		return nil, fmt.Errorf("codec experiment: %w", err)
	}

	base := artifactBase(analysis.VideoName)
	jsonPath := filepath.Join(s.outDir, base+".json")
	csvPath := filepath.Join(s.outDir, base+".csv")
	pngPath := filepath.Join(s.outDir, base+".png")

	if err := writeJSON(jsonPath, analysis); err != nil {
		return nil, err
	}
	if err := writeCSV(csvPath, analysis); err != nil {
		return nil, err
	}
	pngErr := writePNG(pngPath, analysis)

	res := &Result{
		Analysis:  analysis,
		JSONPath:  jsonPath,
		CSVPath:   csvPath,
		Extractor: analysis.Extractor,
	}
	if pngErr == nil {
		res.PNGPath = pngPath
	}
	res.Report = formatReport(analysis, res, time.Since(start))

	slog.Info("codec experiment: complete",
		"video_id", videoID,
		"video", analysis.VideoName,
		"codec", analysis.Codec,
		"frames", analysis.FrameCount,
		"extractor", analysis.Extractor,
		"motion_vectors", analysis.Motion.Available,
		"json", jsonPath, "csv", csvPath, "png", res.PNGPath,
		"duration_ms", time.Since(start).Milliseconds())
	if pngErr != nil {
		slog.Warn("codec experiment: timeline png not written", "video_id", videoID, "error", pngErr)
	}
	return res, nil
}

func artifactBase(videoName string) string {
	base := strings.TrimSuffix(filepath.Base(videoName), filepath.Ext(videoName))
	if base == "" {
		base = "video"
	}
	var b strings.Builder
	for _, r := range base {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r >= '0' && r <= '9', r == '-', r == '_':
			b.WriteRune(r)
		default:
			b.WriteByte('_')
		}
	}
	return b.String()
}

func writeJSON(path string, a *Analysis) error {
	data, err := json.MarshalIndent(a, "", "  ")
	if err != nil {
		return fmt.Errorf("codec experiment: %w", err)
	}
	data = append(data, '\n')
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return fmt.Errorf("codec experiment: %w", err)
	}
	return nil
}

func writeCSV(path string, a *Analysis) error {
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("codec experiment: %w", err)
	}
	defer f.Close()

	header := []string{
		"index", "pts", "timestamp_s", "type", "key_frame", "packet_bytes",
		"mv_count", "mv_mean_magnitude_px", "mv_median_magnitude_px", "mv_max_magnitude_px",
		"mv_mean_dx_px", "mv_mean_dy_px", "mv_mean_abs_dx_px", "mv_mean_abs_dy_px",
		"mv_zero_ratio", "mv_forward", "mv_backward", "mv_side_data",
	}
	for c := 0; c < SpatialGridSize; c++ {
		for r := 0; r < SpatialGridSize; r++ {
			header = append(header, fmt.Sprintf("mv_grid_r%dc%d", r, c))
		}
	}

	w := csv.NewWriter(f)
	if err := w.Write(header); err != nil {
		return fmt.Errorf("codec experiment: %w", err)
	}

	for i := range a.Frames {
		fr := a.Frames[i]
		mv := fr.Motion
		row := []string{
			strconv.Itoa(fr.Index),
			strconv.FormatInt(fr.PTS, 10),
			strconv.FormatFloat(fr.Timestamp, 'f', 6, 64),
			fr.Type,
			strconv.FormatBool(fr.KeyFrame),
			strconv.Itoa(fr.PacketBytes),
			strconv.Itoa(mv.Count),
			formatFloat(mv.MeanMagnitude),
			formatFloat(mv.MedianMagnitude),
			formatFloat(mv.MaxMagnitude),
			formatFloat(mv.MeanDx),
			formatFloat(mv.MeanDy),
			formatFloat(mv.MeanAbsDx),
			formatFloat(mv.MeanAbsDy),
			formatFloat(mv.ZeroRatio),
			strconv.Itoa(mv.ForwardCount),
			strconv.Itoa(mv.BackwardCount),
			strconv.FormatBool(mv.HasSideData),
		}
		for c := 0; c < SpatialGridSize; c++ {
			for r := 0; r < SpatialGridSize; r++ {
				idx := r*SpatialGridSize + c
				if idx < len(mv.SpatialGrid) {
					row = append(row, formatFloat(mv.SpatialGrid[idx]))
				} else {
					row = append(row, "")
				}
			}
		}
		if err := w.Write(row); err != nil {
			return fmt.Errorf("codec experiment: %w", err)
		}
	}
	w.Flush()
	if err := w.Error(); err != nil {
		return fmt.Errorf("codec experiment: %w", err)
	}
	return nil
}

func formatFloat(v float64) string {
	return strconv.FormatFloat(v, 'f', 4, 64)
}
