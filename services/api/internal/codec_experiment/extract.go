package codec_experiment

// TEMPORARY EXPERIMENT -- see FINDINGS.md. Whole package is disposable.

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
)

// rawFrame mirrors the per-picture record emitted by workers/codec_experiment/mvdump.py.
type rawFrame struct {
	PTS         int64       `json:"pts"`
	Timestamp   float64     `json:"timestamp"`
	Type        string      `json:"type"`
	KeyFrame    bool        `json:"key_frame"`
	PacketBytes int         `json:"packet_bytes"`
	Motion      MotionStats `json:"motion"`
}

// rawExtraction is the wire format shared by the PyAV extractor and the
// ffprobe fallback, so the rest of the package is agnostic about which ran.
type rawExtraction struct {
	Extractor       string     `json:"extractor"`
	LibavVersion    string     `json:"libav_version"`
	ContainerFormat string     `json:"container_format"`
	Codec           string     `json:"codec"`
	Profile         string     `json:"profile"`
	Width           int        `json:"width"`
	Height          int        `json:"height"`
	FPS             float64    `json:"fps"`
	Duration        float64    `json:"duration"`
	NBFrames        int        `json:"nb_frames_container"`
	HasBFrames      bool       `json:"has_b_frames"`
	ExportMVs       bool       `json:"export_mvs"`
	MotionMechanism string     `json:"motion_mechanism"`
	Truncated       bool       `json:"truncated"`
	Grid            int        `json:"grid"`
	Frames          []rawFrame `json:"frames"`

	ExtractorError string `json:"extractor_error"`
}

func resolveScriptPath(scriptPath string) string {
	if scriptPath == "" {
		scriptPath = filepath.Join("workers", "codec_experiment", "mvdump.py")
	}
	if _, err := os.Stat(scriptPath); err == nil {
		return scriptPath
	}
	if abs, err := filepath.Abs(scriptPath); err == nil {
		return abs
	}
	return scriptPath
}

// runPyAVExtractor decodes the stream through libavcodec with
// AV_CODEC_FLAG2_EXPORT_MVS and returns the codec-level motion vectors.
func runPyAVExtractor(ctx context.Context, pythonPath, scriptPath, videoPath string, maxFrames int) (*rawExtraction, error) {
	if pythonPath == "" {
		pythonPath = "python3"
	}
	args := []string{scriptPath, "--input", videoPath, "--grid", strconv.Itoa(SpatialGridSize)}
	if maxFrames > 0 {
		args = append(args, "--max-frames", strconv.Itoa(maxFrames))
	}

	cmd := exec.CommandContext(ctx, pythonPath, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if err := cmd.Run(); err != nil {
		detail := strings.TrimSpace(stderr.String())
		if detail == "" {
			detail = strings.TrimSpace(stdout.String())
		}
		if len(detail) > 600 {
			detail = detail[:600]
		}
		return nil, fmt.Errorf("pyav extractor failed: %w: %s", err, detail)
	}

	var raw rawExtraction
	if err := json.Unmarshal(stdout.Bytes(), &raw); err != nil {
		return nil, fmt.Errorf("pyav extractor output unparsable: %w", err)
	}
	if raw.ExtractorError != "" {
		return nil, fmt.Errorf("pyav extractor error: %s", raw.ExtractorError)
	}
	if raw.Grid == 0 {
		raw.Grid = SpatialGridSize
	}
	return &raw, nil
}

type ffprobeKV map[string]string

// parseCompact parses one line of `ffprobe -of compact=p=0` output, which is a
// bare `key=value|key=value` list with no section prefix. -show_entries already
// restricts each invocation to the fields we want, so no section filter is needed.
func parseCompact(line string) ffprobeKV {
	line = strings.TrimSpace(line)
	if line == "" || !strings.Contains(line, "=") {
		return nil
	}
	kv := ffprobeKV{}
	for _, part := range strings.Split(line, "|") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		idx := strings.Index(part, "=")
		if idx < 0 {
			continue
		}
		kv[strings.TrimSpace(part[:idx])] = strings.TrimSpace(part[idx+1:])
	}
	return kv
}

// parseBool handles ffprobe integer booleans, which are not always 0/1:
// has_b_frames is the reordering delay and is commonly 2.
func parseBool(s string) bool {
	switch strings.TrimSpace(s) {
	case "1", "2", "3", "true", "TRUE":
		return true
	default:
		return false
	}
}

func parseRational(s string) float64 {
	if s == "" || s == "0/0" {
		return 0
	}
	if parts := strings.SplitN(s, "/", 2); len(parts) == 2 {
		num, err1 := strconv.ParseFloat(parts[0], 64)
		den, err2 := strconv.ParseFloat(parts[1], 64)
		if err1 == nil && err2 == nil && den != 0 {
			return num / den
		}
		return 0
	}
	v, err := strconv.ParseFloat(s, 64)
	if err != nil {
		return 0
	}
	return v
}

// runFFprobeFallback extracts picture types, timestamps and per-picture
// bitstream sizes using only the ffmpeg CLI. Motion vectors are NOT available
// on this path; see residualAndVectorFindings.
func runFFprobeFallback(ctx context.Context, ffprobePath, videoPath string, maxFrames int) (*rawExtraction, error) {
	if ffprobePath == "" {
		ffprobePath = "ffprobe"
	}

	run := func(args ...string) ([]byte, error) {
		full := append([]string{"-v", "error"}, args...)
		full = append(full, videoPath)
		cmd := exec.CommandContext(ctx, ffprobePath, full...)
		var stdout, stderr bytes.Buffer
		cmd.Stdout = &stdout
		cmd.Stderr = &stderr
		if err := cmd.Run(); err != nil {
			return nil, fmt.Errorf("%s: %w: %s", ffprobePath, err, strings.TrimSpace(stderr.String()))
		}
		return stdout.Bytes(), nil
	}

	readIntervals := func() []string {
		if maxFrames > 0 {
			return []string{"-read_intervals", fmt.Sprintf("%%+#%d", maxFrames)}
		}
		return nil
	}

	raw := &rawExtraction{
		Extractor: "ffprobe-cli",
		Grid:      SpatialGridSize,
		MotionMechanism: "motion vectors require AV_CODEC_FLAG2_EXPORT_MVS at decode time, which the " +
			"ffprobe CLI cannot request; it is also why no ffprobe field exposes them",
	}

	streamOut, err := run("-select_streams", "v:0", "-show_entries",
		"stream=codec_name,profile,width,height,has_b_frames,r_frame_rate,avg_frame_rate",
		"-of", "compact=p=0")
	if err != nil {
		return nil, err
	}
	for _, line := range strings.Split(string(streamOut), "\n") {
		kv := parseCompact(line)
		if kv == nil {
			continue
		}
		raw.Codec = kv["codec_name"]
		raw.Profile = kv["profile"]
		raw.Width, _ = strconv.Atoi(kv["width"])
		raw.Height, _ = strconv.Atoi(kv["height"])
		raw.HasBFrames = parseBool(kv["has_b_frames"])
		raw.FPS = parseRational(kv["avg_frame_rate"])
		if raw.FPS == 0 {
			raw.FPS = parseRational(kv["r_frame_rate"])
		}
	}

	formatOut, err := run("-show_entries", "format=format_name,duration", "-of", "compact=p=0")
	if err != nil {
		return nil, err
	}
	for _, line := range strings.Split(string(formatOut), "\n") {
		kv := parseCompact(line)
		if kv == nil {
			continue
		}
		raw.ContainerFormat = kv["format_name"]
		raw.Duration, _ = strconv.ParseFloat(kv["duration"], 64)
	}

	// Packet pass: one entry per coded picture, giving the bitstream size.
	packetArgs := []string{"-select_streams", "v:0", "-show_entries", "packet=pts_time,size"}
	packetArgs = append(packetArgs, readIntervals()...)
	packetArgs = append(packetArgs, "-of", "compact=p=0")
	packetOut, err := run(packetArgs...)
	if err != nil {
		return nil, err
	}
	sizesByTS := map[float64]int{}
	for _, line := range strings.Split(string(packetOut), "\n") {
		kv := parseCompact(line)
		if kv == nil {
			continue
		}
		ts, err := strconv.ParseFloat(kv["pts_time"], 64)
		if err != nil {
			continue
		}
		size, err := strconv.Atoi(kv["size"])
		if err != nil {
			continue
		}
		sizesByTS[roundTS(ts)] = size
	}

	// Frame pass: requires a full decode, but yields the picture types.
	frameArgs := []string{"-select_streams", "v:0", "-show_entries",
		"frame=best_effort_timestamp_time,pict_type,key_frame"}
	frameArgs = append(frameArgs, readIntervals()...)
	frameArgs = append(frameArgs, "-of", "compact=p=0")
	frameOut, err := run(frameArgs...)
	if err != nil {
		return nil, err
	}
	for _, line := range strings.Split(string(frameOut), "\n") {
		kv := parseCompact(line)
		if kv == nil {
			continue
		}
		ts, err := strconv.ParseFloat(kv["best_effort_timestamp_time"], 64)
		if err != nil {
			continue
		}
		key := roundTS(ts)
		raw.Frames = append(raw.Frames, rawFrame{
			Timestamp:   key,
			Type:        strings.ToUpper(kv["pict_type"]),
			KeyFrame:    parseBool(kv["key_frame"]),
			PacketBytes: sizesByTS[key],
		})
	}

	raw.Truncated = maxFrames > 0 && len(raw.Frames) >= maxFrames
	if raw.Duration <= 0 && len(raw.Frames) > 0 {
		raw.Duration = raw.Frames[len(raw.Frames)-1].Timestamp
		if raw.FPS > 0 {
			raw.Duration += 1 / raw.FPS
		}
	}
	return raw, nil
}

func roundTS(ts float64) float64 {
	return float64(int64(ts*1e6+0.5)) / 1e6
}

func summarize(raw *rawExtraction) (*Analysis, error) {
	if len(raw.Frames) == 0 {
		return nil, fmt.Errorf("no coded pictures decoded")
	}

	a := &Analysis{
		Experiment:   ExperimentName,
		Container:    raw.ContainerFormat,
		Codec:        raw.Codec,
		Profile:      raw.Profile,
		Width:        raw.Width,
		Height:       raw.Height,
		FPS:          raw.FPS,
		Duration:     raw.Duration,
		FrameCount:   len(raw.Frames),
		HasBFrames:   raw.HasBFrames,
		Truncated:    raw.Truncated,
		Extractor:    raw.Extractor,
		LibavVersion: raw.LibavVersion,
		GridSize:     SpatialGridSize,
		TypeCounts:   map[string]int{},
	}

	for i := range raw.Frames {
		f := raw.Frames[i]
		a.Frames = append(a.Frames, Frame{
			Index:       i,
			PTS:         f.PTS,
			Timestamp:   f.Timestamp,
			Type:        f.Type,
			KeyFrame:    f.KeyFrame,
			PacketBytes: f.PacketBytes,
			Motion:      f.Motion,
		})
		a.TypeCounts[f.Type]++
		a.Packet.TotalBytes += f.PacketBytes
		if i == 0 || f.PacketBytes < a.Packet.MinBytes {
			a.Packet.MinBytes = f.PacketBytes
		}
		if f.PacketBytes > a.Packet.MaxBytes {
			a.Packet.MaxBytes = f.PacketBytes
		}
	}

	a.Packet.MeanBytes = float64(a.Packet.TotalBytes) / float64(len(a.Frames))
	byType := map[string]float64{}
	typeCount := map[string]int{}
	for _, f := range a.Frames {
		byType[f.Type] += float64(f.PacketBytes)
		typeCount[f.Type]++
	}
	a.Packet.MeanBytesByType = map[string]float64{}
	for t, total := range byType {
		a.Packet.MeanBytesByType[t] = total / float64(typeCount[t])
	}

	a.GOP = gopStats(a.Frames)
	a.Motion = motionSummary(a.Frames, raw.ExportMVs)
	a.Signals = signalFindings(raw)
	return a, nil
}

func gopStats(frames []Frame) GOPStats {
	var positions []int
	for i := range frames {
		if frames[i].KeyFrame || frames[i].Type == "I" {
			positions = append(positions, i)
		}
	}
	g := GOPStats{KeyframeCount: len(positions)}
	if len(positions) < 2 {
		return g
	}
	deltas := make([]int, 0, len(positions)-1)
	for i := 1; i < len(positions); i++ {
		deltas = append(deltas, positions[i]-positions[i-1])
	}
	sort.Ints(deltas)
	g.MinGOPLength = deltas[0]
	g.MaxGOPLength = deltas[len(deltas)-1]
	sum := 0
	for _, d := range deltas {
		sum += d
	}
	g.MeanGOPLength = float64(sum) / float64(len(deltas))
	mid := len(deltas) / 2
	if len(deltas)%2 == 1 {
		g.MedianGOPLength = float64(deltas[mid])
	} else {
		g.MedianGOPLength = float64(deltas[mid-1]+deltas[mid]) / 2
	}
	return g
}

func motionSummary(frames []Frame, available bool) MotionSummary {
	m := MotionSummary{Available: available}
	means := make([]float64, 0, len(frames))
	var zeroSum, countSum float64
	for i := range frames {
		mv := frames[i].Motion
		if mv.HasSideData {
			m.FramesWithSideData++
		} else {
			m.FramesWithoutSideData++
		}
		if !available || !mv.HasSideData {
			continue
		}
		means = append(means, mv.MeanMagnitude)
		zeroSum += mv.ZeroRatio
		countSum += float64(mv.Count)
		if mv.MaxMagnitude > m.MaxMagnitudeOverall {
			m.MaxMagnitudeOverall = mv.MaxMagnitude
		}
	}
	if len(means) == 0 {
		return m
	}
	sorted := append([]float64(nil), means...)
	sort.Float64s(sorted)
	sum := 0.0
	for _, v := range sorted {
		sum += v
	}
	m.MeanOfMeanMagnitude = sum / float64(len(sorted))
	mid := len(sorted) / 2
	if len(sorted)%2 == 1 {
		m.MedianMeanMagnitude = sorted[mid]
	} else {
		m.MedianMeanMagnitude = (sorted[mid-1] + sorted[mid]) / 2
	}
	n := float64(len(sorted))
	m.MeanZeroRatio = zeroSum / n
	m.MeanVectorCount = countSum / n
	return m
}

func signalFindings(raw *rawExtraction) map[string]Signal {
	frameTypes := Signal{
		Available: true,
		Source:    "decoder picture type (AVFrame.pict_type) / ffprobe frame.pict_type",
		Reason:    "Intra/inter frame coding is declared in the slice header of every coded picture, so it is readable from the source stream without any re-encoding by ReCall.",
	}

	packet := Signal{
		Available: true,
		Source:    "container demux (AVPacket.size / ffprobe packet.size)",
		Reason:    "Bitstream size per coded picture. This is encoded size, which correlates with residual energy, but it is NOT a residual measurement and must not be read as one.",
	}

	motion := Signal{Available: false}
	if raw.ExportMVs {
		motion = Signal{
			Available: true,
			Source:    raw.MotionMechanism,
			Reason:    "Real codec motion vectors. The decoder must be opened with AV_CODEC_FLAG2_EXPORT_MVS; the ffmpeg CLI alone cannot do this because the codecview filter renders motion vectors as pixels and no filter serialises them as numbers.",
		}
	} else {
		motion = Signal{
			Available: false,
			Source:    "not obtained",
			Reason:    "The current extraction path cannot reach motion vectors. " + raw.MotionMechanism,
			Requires:  "AV_CODEC_FLAG2_EXPORT_MVS at decode time and a libavcodec binding that reads AV_FRAME_DATA_MOTION_VECTORS side data (PyAV, ffmpeg-next, or a cgo decoder loop). The ffmpeg CLI is insufficient: codecview=mv=pf+bf+bb only paints vectors, it emits no numbers.",
		}
	}

	residual := Signal{
		Available: false,
		Source:    "not exposed",
		Reason:    "Residual / transform-coefficient data is never surfaced by libavcodec or the ffmpeg CLI. There is no AV_FRAME_DATA_* entry for decoded residuals, and no ffprobe field for them. This is a decoder-API limitation, not a configuration gap, so it was not substituted with pixel differences or optical flow.",
		Requires:  "A bitstream-level parser or an instrumented decoder that intercepts the inverse transform output (for example libde265/libavcodec internals, an h264/h265 NAL parser, or a custom decode loop that sums residual coefficients per block). ReCall's pipeline does not need to change to do this, but the decode must happen against the original source stream: the MJPEG re-encode in video_frame.GenerateForVideo discards all inter-frame structure, so anything captured after that point is unrecoverable.",
	}

	return map[string]Signal{
		"frame_types":    frameTypes,
		"packet_sizes":   packet,
		"motion_vectors": motion,
		"residuals":      residual,
	}
}
