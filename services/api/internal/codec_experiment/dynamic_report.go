package codec_experiment

// TEMPORARY EXPERIMENT -- see FINDINGS.md. Whole package is disposable.

import (
	"fmt"
	"strings"
	"time"
)

// FormatDynamicReport renders the baseline-vs-dynamic experiment report. It
// describes codec activity candidates only; it never claims semantic value.
func FormatDynamicReport(a *Analysis, plan *DynamicPlan, dynJSONPath, dynPNGPath string, elapsed time.Duration) string {
	var b strings.Builder
	line := func(s string) { b.WriteString(s); b.WriteString("\n") }

	line("=== ReCall Dynamic CodecSight Experiment ===")
	line("")
	line("Video: " + a.VideoName)
	if a.VideoID != "" {
		line("VideoID: " + a.VideoID)
	}
	line(fmt.Sprintf("Duration: %.2fs", plan.Summary.DurationSeconds))
	line(fmt.Sprintf("Native FPS: %s", trimFloat(plan.Summary.NativeFPS)))
	line("Codec: " + orDash(a.Codec))
	line("Extractor: " + a.Extractor)
	line(fmt.Sprintf("Motion vectors: %s", yesNo(plan.Summary.MotionAvailable)))
	line(fmt.Sprintf("Sampling mode: %s", plan.Summary.Mode))
	if plan.Summary.Reason != "" {
		line("Reason: " + plan.Summary.Reason)
	}

	line("")
	line("Baseline sampling:")
	line(fmt.Sprintf("    interval: %s", plan.Config.BaselineInterval))
	line(fmt.Sprintf("    estimated frames: %d", plan.Summary.BaselineCount))

	line("")
	line("Dynamic sampling:")
	line(fmt.Sprintf("    normal interval: %s", plan.Config.BaselineInterval))
	line(fmt.Sprintf("    dense interval: %s", plan.Config.DenseInterval))
	line(fmt.Sprintf("    window before: %s", plan.Config.WindowBefore))
	line(fmt.Sprintf("    window after: %s", plan.Config.WindowAfter))
	line(fmt.Sprintf("    merge gap: %s", plan.Config.MergeGap))
	line(fmt.Sprintf("    weights: mean=%.2f max=%.2f static=%.2f packet=%.2f (experimental)",
		plan.Config.WeightMean, plan.Config.WeightMax, plan.Config.WeightStatic, plan.Config.WeightPacket))
	line(fmt.Sprintf("    normalization percentiles: p%.0f-p%.0f", plan.Config.LowPercentile, plan.Config.HighPercentile))

	line("")
	line("Codec activity:")
	line(fmt.Sprintf("    enter threshold: %.3f", plan.Config.EnterThreshold))
	line(fmt.Sprintf("    exit threshold: %.3f", plan.Config.ExitThreshold))
	line(fmt.Sprintf("    dynamic regions: %d", plan.Summary.RegionCount))

	line("")
	line("Frames:")
	line(fmt.Sprintf("    baseline frames: %d", plan.Summary.BaselineCount))
	line(fmt.Sprintf("    dynamic frames: %d", plan.Summary.DynamicCount))
	line(fmt.Sprintf("    additional frames: %d", plan.Summary.AddedCount))
	line(fmt.Sprintf("    dynamic frames / baseline frames: %.3f", plan.Summary.DynamicOverBaseline))
	line(fmt.Sprintf("    full native frames: %d", plan.Summary.NativeFrames))
	line(fmt.Sprintf("    baseline sampling ratio: %.4f", plan.Summary.BaselineRatio))
	line(fmt.Sprintf("    dynamic sampling ratio: %.4f", plan.Summary.DynamicRatio))
	line(fmt.Sprintf("    frame reduction vs full FPS: %.4f", plan.Summary.ReductionVsFull))

	line("")
	line("Dynamic regions:")
	if len(plan.Regions) == 0 {
		line("    none")
	}
	for i, region := range plan.Regions {
		line(fmt.Sprintf("    %d. %.1fs - %.1fs (peak %.1fs, score %.3f, %d pictures)",
			i+1, region.Start, region.End, region.Peak, region.PeakScore, region.ActiveFrames))
	}

	line("")
	line("Timestamp comparison:")
	line(fmt.Sprintf("    baseline count: %d", len(plan.BaselineTimestamps)))
	line(fmt.Sprintf("    dynamic count: %d", len(plan.SelectedTimestamps)))
	line("    exact dynamic timestamps are recorded in the dynamic JSON artifact below;")
	line("    the baseline grid is the existing 2-second sampler output.")

	line("")
	line("Output:")
	line("  " + dynJSONPath)
	if dynPNGPath != "" {
		line("  " + dynPNGPath)
	}
	line("")
	line(fmt.Sprintf("Dynamic planning wall time: %s", elapsed.Round(time.Millisecond)))
	line("")
	line("Note: selected frames are high-activity candidates, not high-value frames.")
	line("The codec only marks regions where encoded motion changed; YOLO and VLM")
	line("remain responsible for semantic understanding.")
	return b.String()
}
