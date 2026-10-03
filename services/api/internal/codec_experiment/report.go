package codec_experiment

// TEMPORARY EXPERIMENT -- see FINDINGS.md. Whole package is disposable.

import (
	"fmt"
	"sort"
	"strings"
	"time"
)

func formatReport(a *Analysis, res *Result, elapsed time.Duration) string {
	var b strings.Builder

	line := func(s string) { b.WriteString(s); b.WriteString("\n") }

	line("=== ReCall CodecSight Experiment ===")
	line("")
	line("Video: " + a.VideoName)
	if a.VideoID != "" {
		line("VideoID: " + a.VideoID)
	}
	line("Source: " + a.Path)
	line("Container: " + orDash(a.Container))
	line("Codec: " + orDash(a.Codec))
	if a.Profile != "" {
		line("Profile: " + a.Profile)
	}
	line(fmt.Sprintf("Resolution: %dx%d", a.Width, a.Height))
	line(fmt.Sprintf("Duration: %.2fs", a.Duration))
	line(fmt.Sprintf("FPS: %s", trimFloat(a.FPS)))
	line("Extractor: " + a.Extractor)
	if a.LibavVersion != "" {
		line("PyAV/libav: " + a.LibavVersion)
	}
	if a.Truncated {
		line("WARNING: frame list truncated by CODEC_EXPERIMENT_MAX_FRAMES; statistics below cover the prefix only")
	}

	line("")
	line("--- Successfully extracted ---")

	line("")
	line("Frame types (coded picture structure):")
	for _, t := range orderedTypes(a.TypeCounts) {
		line(fmt.Sprintf("  %-5s %d", t, a.TypeCounts[t]))
	}
	if a.GOP.KeyframeCount > 0 {
		line(fmt.Sprintf("  keyframes (I):      %d", a.GOP.KeyframeCount))
	}
	switch {
	case a.GOP.KeyframeCount >= 2:
		line(fmt.Sprintf("  GOP length frames:  mean %.1f  median %.1f  min %d  max %d",
			a.GOP.MeanGOPLength, a.GOP.MedianGOPLength, a.GOP.MinGOPLength, a.GOP.MaxGOPLength))
	case a.GOP.KeyframeCount == 1:
		line("  GOP length frames:  n/a (single keyframe, no interval to measure)")
	default:
		line("  GOP length frames:  n/a (no keyframe found)")
	}
	line(fmt.Sprintf("  has_b_frames:       %s", yesNo(a.HasBFrames)))

	line("")
	line("Per-picture bitstream size (NOT residual energy):")
	line(fmt.Sprintf("  total bytes:   %d", a.Packet.TotalBytes))
	line(fmt.Sprintf("  mean bytes:    %.0f", a.Packet.MeanBytes))
	line(fmt.Sprintf("  min / max:     %d / %d", a.Packet.MinBytes, a.Packet.MaxBytes))
	if len(a.Packet.MeanBytesByType) > 0 {
		keys := make([]string, 0, len(a.Packet.MeanBytesByType))
		for k := range a.Packet.MeanBytesByType {
			keys = append(keys, k)
		}
		sort.Strings(keys)
		parts := make([]string, 0, len(keys))
		for _, k := range keys {
			parts = append(parts, fmt.Sprintf("%s=%.0f", k, a.Packet.MeanBytesByType[k]))
		}
		line("  mean by type:  " + strings.Join(parts, "  "))
	}

	if sig, ok := a.Signals["motion_vectors"]; ok && sig.Available {
		m := a.Motion
		line("")
		line("Motion vectors:")
		line("  Available:            yes")
		line("  Mechanism:            " + sig.Source)
		line(fmt.Sprintf("  Frames with side data: %d", m.FramesWithSideData))
		line(fmt.Sprintf("  Frames without:       %d", m.FramesWithoutSideData))
		line(fmt.Sprintf("  Mean of mean mag (px): %.4f", m.MeanOfMeanMagnitude))
		line(fmt.Sprintf("  Median mean mag (px):  %.4f", m.MedianMeanMagnitude))
		line(fmt.Sprintf("  Max magnitude (px):    %.4f", m.MaxMagnitudeOverall))
		line(fmt.Sprintf("  Mean zero-vector ratio: %.4f", m.MeanZeroRatio))
		line(fmt.Sprintf("  Mean vectors / frame:  %.1f", m.MeanVectorCount))
		line("  Per-frame detail:     mean/median/max magnitude, mean signed dx/dy,")
		line("                        mean |dx|/|dy|, zero ratio, forward/backward split,")
		line("                        and a 4x4 spatial mean-magnitude grid (see JSON/CSV)")
	}

	line("")
	line("--- Not exposed by the current stack ---")
	for _, name := range []string{"motion_vectors", "residuals"} {
		sig, ok := a.Signals[name]
		if !ok || sig.Available {
			continue
		}
		line("")
		line(signalHeading(name))
		line("  Available: no")
		if sig.Reason != "" {
			line("  Why: " + wrap(sig.Reason, 4))
		}
		if sig.Requires != "" {
			line("  Would require: " + wrap(sig.Requires, 4))
		}
	}

	line("")
	line("--- Output ---")
	line("  " + res.JSONPath)
	line("  " + res.CSVPath)
	if res.PNGPath != "" {
		line("  " + res.PNGPath)
	}
	line("")
	line(fmt.Sprintf("Extraction wall time: %s", elapsed.Round(time.Millisecond)))
	line("")
	line("Note: this experiment observes the source stream as ReCall received it.")
	line("ReCall does not transcode. Its frame-extraction stage re-encodes sampled")
	line("frames to intra-only MJPEG (video_frame.GenerateForVideo), which discards")
	line("every signal above at that boundary.")
	return b.String()
}

func orderedTypes(counts map[string]int) []string {
	keys := make([]string, 0, len(counts))
	for k := range counts {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool {
		if counts[keys[i]] != counts[keys[j]] {
			return counts[keys[i]] > counts[keys[j]]
		}
		return keys[i] < keys[j]
	})
	return keys
}

func orDash(s string) string {
	if strings.TrimSpace(s) == "" {
		return "-"
	}
	return s
}

func yesNo(b bool) string {
	if b {
		return "yes"
	}
	return "no"
}

func trimFloat(v float64) string {
	return fmt.Sprintf("%.3f", v)
}

func signalHeading(name string) string {
	switch name {
	case "frame_types":
		return "Frame types:"
	case "packet_sizes":
		return "Encoded size:"
	case "motion_vectors":
		return "Motion vectors:"
	case "residuals":
		return "Residual information:"
	default:
		return name + ":"
	}
}

func wrap(s string, indent int) string {
	prefix := strings.Repeat(" ", indent)
	words := strings.Fields(s)
	if len(words) == 0 {
		return ""
	}
	var b strings.Builder
	lineLen := 0
	for i, w := range words {
		if i > 0 {
			if lineLen+1+len(w) > 76 {
				b.WriteString("\n" + prefix)
				lineLen = 0
			} else {
				b.WriteString(" ")
				lineLen++
			}
		}
		b.WriteString(w)
		lineLen += len(w)
	}
	return b.String()
}
