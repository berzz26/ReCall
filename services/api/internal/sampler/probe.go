// Visual probe for the adaptive coarse sampler (Phase 2).
//
// The probe answers only "how much of the image has changed relative to the
// last selected frame?" It performs no semantic recognition and never sees
// YOLO, ByteTrack, events, or the VLM.
//
// Pipeline: FFmpeg decodes the video cheaply at 5 FPS, 64x64, grayscale
// (filter chain: fps=5,scale=64:64:flags=area,format=gray). Each probe frame
// is reduced to an 8x8 grid of block means (64 blocks) and compared against
// the last frame kept by the coarse planner via changed_fraction.
//
// The probe is internal to the sampler package: raw frames are reduced to
// compact block means during streaming decode and never exposed elsewhere.
package sampler

import (
	"bytes"
	"context"
	"fmt"
	"io"
	"math"
	"os/exec"
)

// Probe defaults. The probe is deliberately orders of magnitude cheaper
// than YOLO on full-resolution frames: 64x64 grayscale at 5 FPS.
const (
	// DefaultProbeFPS is the probe decode rate. YOLO frames stay much
	// sparser (coarse interval G); the probe oversamples so short visual
	// changes are observed even if they vanish before a selection.
	DefaultProbeFPS = 5.0
	// DefaultProbeSize is the square probe frame edge in pixels.
	DefaultProbeSize = 64
	// DefaultProbeGrid is the number of blocks per side; the frame is
	// divided into Grid x Grid blocks (8x8 = 64 blocks total).
	DefaultProbeGrid = 8
	// DefaultNoiseK scales the MAD term of the per-video noise floor:
	// threshold = median(block_difference) + K * MAD(block_difference).
	// K=3 is the conventional robust-outlier starting point, not a tuned
	// value.
	DefaultNoiseK = 3.0
)

// ProbeFrame is one decoded probe sample reduced to scoring information.
// The full 64x64 grayscale pixels are released after the block means are
// computed; only the 64 bytes of means are retained.
type ProbeFrame struct {
	TimestampSeconds float64
	// Means holds Grid*Grid block means (row-major) as rounded pixel
	// levels in [0,255].
	Means []uint8
}

// BlockMeans divides a size x size grayscale frame into grid x grid blocks
// and returns the mean pixel level of each block (row-major, rounded).
func BlockMeans(gray []byte, size, grid int) ([]uint8, error) {
	if size <= 0 || grid <= 0 {
		return nil, fmt.Errorf("probe size and grid must be > 0, got %d/%d", size, grid)
	}
	if size%grid != 0 {
		return nil, fmt.Errorf("probe size %d must be divisible by grid %d", size, grid)
	}
	if len(gray) != size*size {
		return nil, fmt.Errorf("malformed probe frame: got %d bytes, want %d", len(gray), size*size)
	}
	block := size / grid
	per := block * block
	means := make([]uint8, 0, grid*grid)
	for by := 0; by < grid; by++ {
		for bx := 0; bx < grid; bx++ {
			sum := 0
			for y := 0; y < block; y++ {
				row := (by*block+y)*size + bx*block
				for x := 0; x < block; x++ {
					sum += int(gray[row+x])
				}
			}
			means = append(means, uint8((sum+per/2)/per))
		}
	}
	return means, nil
}

// ChangedFraction compares two probe frames block by block against the noise
// threshold and returns changed_blocks / total_blocks in [0,1].
//
// changed = block_difference > noiseThreshold, where block_difference is the
// mean absolute pixel-level difference of the block. Terminology is
// deliberately non-semantic: this is visual change, not an event, activity,
// or motion label.
func ChangedFraction(ref, cur []uint8, noiseThreshold float64) float64 {
	if len(ref) == 0 || len(ref) != len(cur) {
		return 0
	}
	changed := 0
	for i := range ref {
		d := math.Abs(float64(cur[i]) - float64(ref[i]))
		if d > noiseThreshold {
			changed++
		}
	}
	return float64(changed) / float64(len(ref))
}

// NoiseEstimator accumulates observed block differences and derives a
// per-video noise floor: median + K * MAD (median absolute deviation).
//
// Differences are quantized to uint8 levels in a 256-bin histogram, so
// memory stays O(1) regardless of video length. Median/MAD are upper
// quantiles from the histogram; exact enough for a starting adaptive
// threshold.
type NoiseEstimator struct {
	hist  [256]int64
	total int64
}

// Add records one observed block difference (pixel levels, >= 0).
func (e *NoiseEstimator) Add(diff float64) {
	if math.IsNaN(diff) || math.IsInf(diff, 0) || diff < 0 {
		return
	}
	q := int(math.Round(diff))
	if q > 255 {
		q = 255
	}
	e.hist[q]++
	e.total++
}

// quantile returns the smallest level whose cumulative count exceeds frac of
// the total (upper quantile).
func (e *NoiseEstimator) quantile(frac float64) float64 {
	if e.total == 0 {
		return 0
	}
	target := frac * float64(e.total)
	var cum int64
	for v := 0; v < 256; v++ {
		cum += e.hist[v]
		if float64(cum) > target {
			return float64(v)
		}
	}
	return 255
}

// Threshold returns median(block_difference) + K * MAD(block_difference).
// With no observations (e.g. a single probe frame) it returns 0, in which
// case only literally identical blocks count as unchanged.
//
// Quantization floor: block differences are integer pixel levels, so a
// sub-level spread is meaningless. MAD is clamped to a minimum of one level,
// i.e. the floor always clears the median bulk by at least K levels. Without
// this, a noise distribution concentrated on a single level (MAD = 0) would
// leave the threshold at the median and mark half the noise as changed.
func (e *NoiseEstimator) Threshold(k float64) float64 {
	if e.total == 0 {
		return 0
	}
	median := e.quantile(0.5)
	// MAD via the same histogram: smallest d with
	// count(|level - median| <= d) exceeding half the total.
	target := 0.5 * float64(e.total)
	mad := 255.0
	for d := 0; d < 256; d++ {
		lo := int(median) - d
		if lo < 0 {
			lo = 0
		}
		hi := int(median) + d
		if hi > 255 {
			hi = 255
		}
		var c int64
		for v := lo; v <= hi; v++ {
			c += e.hist[v]
		}
		if float64(c) > target {
			mad = float64(d)
			break
		}
	}
	if mad < 1 {
		mad = 1
	}
	return median + k*mad
}

// DecodeProbeFrames runs the FFmpeg visual probe:
//
//	fps=<fps>,scale=<size>:<size>:flags=area,format=gray -> rawvideo gray pipe
//
// Frames stream from the pipe and are reduced to block means on the fly, so
// memory stays proportional to frame count * 64 bytes (not full video).
// Only frames with timestamp < durationSeconds are kept, matching the Phase
// 1 normalization boundary. Any decode anomaly (process failure, short
// output, malformed frames, no frames) returns an error so the caller can
// fall back to the baseline plan; the probe must never fail a video.
func DecodeProbeFrames(ctx context.Context, ffmpegPath, videoPath string, fps float64, size, grid int, durationSeconds float64) ([]ProbeFrame, error) {
	if ffmpegPath == "" {
		ffmpegPath = "ffmpeg"
	}
	if fps <= 0 {
		fps = DefaultProbeFPS
	}
	if size <= 0 {
		size = DefaultProbeSize
	}
	if grid <= 0 {
		grid = DefaultProbeGrid
	}
	if math.IsNaN(durationSeconds) || math.IsInf(durationSeconds, 0) || durationSeconds <= 0 {
		return nil, fmt.Errorf("video duration unavailable; cannot run probe")
	}
	frameBytes := size * size
	filter := fmt.Sprintf("fps=%.6f,scale=%d:%d:flags=area,format=gray", fps, size, size)
	args := []string{
		"-loglevel", "error",
		"-i", videoPath,
		"-vf", filter,
		"-f", "rawvideo",
		"-pix_fmt", "gray",
		"pipe:1",
	}
	cmd := exec.CommandContext(ctx, ffmpegPath, args...)
	var stderrBuf bytes.Buffer
	stderrBuf.Grow(8192)
	cmd.Stderr = &limitedWriter{buf: &stderrBuf, limit: 8192}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("probe stdout pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("probe failed to start: %s", probeErrMsg(stderrBuf.String(), err))
	}

	var frames []ProbeFrame
	pending := make([]byte, 0, 1<<20)
	chunk := make([]byte, 1<<20)
	index := 0
	flush := func() error {
		for len(pending) >= frameBytes {
			means, err := BlockMeans(pending[:frameBytes], size, grid)
			if err != nil {
				return err
			}
			ts := float64(index) / fps
			index++
			if ts < durationSeconds-1e-9 {
				meansCopy := make([]uint8, len(means))
				copy(meansCopy, means)
				frames = append(frames, ProbeFrame{TimestampSeconds: ts, Means: meansCopy})
			}
			pending = pending[frameBytes:]
		}
		return nil
	}
	for {
		n, readErr := stdout.Read(chunk)
		if n > 0 {
			pending = append(pending, chunk[:n]...)
			if err := flush(); err != nil {
				_ = cmd.Wait()
				return nil, err
			}
		}
		if readErr != nil {
			if readErr == io.EOF {
				break
			}
			_ = cmd.Wait()
			return nil, fmt.Errorf("probe stream read: %w", readErr)
		}
	}
	if len(pending) != 0 {
		_ = cmd.Wait()
		return nil, fmt.Errorf("malformed probe output: %d trailing bytes", len(pending))
	}
	if err := cmd.Wait(); err != nil {
		if ctx.Err() != nil {
			return nil, fmt.Errorf("probe canceled: %w", ctx.Err())
		}
		return nil, fmt.Errorf("probe ffmpeg failed: %s", probeErrMsg(stderrBuf.String(), err))
	}
	if len(frames) == 0 {
		return nil, fmt.Errorf("probe produced no frames")
	}
	return frames, nil
}

// limitedWriter caps stderr capture to avoid unbounded memory.
type limitedWriter struct {
	buf   *bytes.Buffer
	limit int
}

func (w *limitedWriter) Write(p []byte) (int, error) {
	if w.buf.Len() >= w.limit {
		return len(p), nil
	}
	remaining := w.limit - w.buf.Len()
	if len(p) > remaining {
		p = p[:remaining]
	}
	return w.buf.Write(p)
}

func probeErrMsg(stderr string, err error) string {
	msg := stderr
	if len(msg) > 500 {
		msg = msg[:500]
	}
	// Trim trailing whitespace/newlines for log readability.
	for len(msg) > 0 && (msg[len(msg)-1] == '\n' || msg[len(msg)-1] == ' ' || msg[len(msg)-1] == '\r' || msg[len(msg)-1] == '\t') {
		msg = msg[:len(msg)-1]
	}
	if msg == "" {
		msg = err.Error()
	}
	return msg
}
