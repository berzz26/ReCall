package codec_experiment

// TEMPORARY EXPERIMENT -- see FINDINGS.md. Whole package is disposable.

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"math"
	"os"
	"path/filepath"
	"sort"
	"time"
)

const DynamicExperimentName = "recall-codecsight-dynamic-sampler/1"

// DynamicConfig controls the experimental codec-activity sampler. All values
// are experimental and are exposed so thresholds and weights can be changed
// without rewriting the sampler.
type DynamicConfig struct {
	Enabled          bool          `json:"enabled"`
	BaselineInterval time.Duration `json:"baseline_interval"`
	DenseInterval    time.Duration `json:"dense_interval"`
	WindowBefore     time.Duration `json:"window_before"`
	WindowAfter      time.Duration `json:"window_after"`
	WeightMean       float64       `json:"weight_motion_mean"`
	WeightMax        float64       `json:"weight_motion_max"`
	WeightStatic     float64       `json:"weight_static"`
	WeightPacket     float64       `json:"weight_packet_size"`
	EnterThreshold   float64       `json:"enter_threshold"`
	ExitThreshold    float64       `json:"exit_threshold"`
	MergeGap         time.Duration `json:"merge_gap"`
	LowPercentile    float64       `json:"low_percentile"`
	HighPercentile   float64       `json:"high_percentile"`
}

// DefaultDynamicConfig returns the first-experiment defaults. The weights sum
// to one, but that is an experimental convenience, not an optimal calibration.
func DefaultDynamicConfig(baselineInterval time.Duration) DynamicConfig {
	if baselineInterval <= 0 {
		baselineInterval = 2 * time.Second
	}
	return DynamicConfig{
		Enabled:          false,
		BaselineInterval: baselineInterval,
		DenseInterval:    200 * time.Millisecond,
		WindowBefore:     1 * time.Second,
		WindowAfter:      1 * time.Second,
		WeightMean:       0.40,
		WeightMax:        0.20,
		WeightStatic:     0.25,
		WeightPacket:     0.15,
		EnterThreshold:   0.90,
		ExitThreshold:    0.70,
		MergeGap:         1 * time.Second,
		LowPercentile:    5,
		HighPercentile:   95,
	}
}

// ValidateDynamicConfig rejects configurations that cannot produce a denser
// experimental sampling plan.
func ValidateDynamicConfig(cfg DynamicConfig, baselineInterval time.Duration) error {
	if baselineInterval <= 0 {
		return fmt.Errorf("codec experiment: baseline interval must be > 0")
	}
	if cfg.DenseInterval <= 0 {
		return fmt.Errorf("codec experiment: dense interval must be > 0")
	}
	if cfg.DenseInterval >= baselineInterval {
		return fmt.Errorf("codec experiment: dense interval (%s) must be shorter than baseline interval (%s)", cfg.DenseInterval, baselineInterval)
	}
	if cfg.WindowBefore < 0 || cfg.WindowAfter < 0 {
		return fmt.Errorf("codec experiment: dynamic windows must be >= 0")
	}
	if cfg.MergeGap < 0 {
		return fmt.Errorf("codec experiment: merge gap must be >= 0")
	}
	if cfg.WeightMean < 0 || cfg.WeightMax < 0 || cfg.WeightStatic < 0 || cfg.WeightPacket < 0 {
		return fmt.Errorf("codec experiment: dynamic weights must be >= 0")
	}
	if cfg.WeightMean+cfg.WeightMax+cfg.WeightStatic+cfg.WeightPacket <= 0 {
		return fmt.Errorf("codec experiment: at least one dynamic weight must be > 0")
	}
	if !(cfg.EnterThreshold > cfg.ExitThreshold) {
		return fmt.Errorf("codec experiment: enter threshold must be greater than exit threshold")
	}
	if cfg.EnterThreshold <= 0 || cfg.ExitThreshold < 0 {
		return fmt.Errorf("codec experiment: thresholds must satisfy enter > 0 and exit >= 0")
	}
	if !(cfg.LowPercentile >= 0 && cfg.LowPercentile < cfg.HighPercentile && cfg.HighPercentile <= 100) {
		return fmt.Errorf("codec experiment: percentiles must satisfy 0 <= low < high <= 100")
	}
	return nil
}

// DynamicRegion is one merged high-activity temporal window, expanded to include
// context before and after the codec peak. These are codec activity candidates,
// not semantically important frames.
type DynamicRegion struct {
	Start        float64 `json:"start_s"`
	End          float64 `json:"end_s"`
	Peak         float64 `json:"peak_s"`
	PeakScore    float64 `json:"peak_score"`
	ActiveFrames int     `json:"active_pictures"`
}

// DynamicSummary compares baseline and dynamic sampling for one video.
type DynamicSummary struct {
	Mode                string  `json:"mode"`
	Reason              string  `json:"reason,omitempty"`
	DurationSeconds     float64 `json:"duration_s"`
	NativeFPS           float64 `json:"native_fps"`
	NativeFrames        int     `json:"native_frames"`
	BaselineCount       int     `json:"baseline_frames"`
	DynamicCount        int     `json:"dynamic_frames"`
	AddedCount          int     `json:"additional_frames"`
	BaselineRatio       float64 `json:"baseline_sampling_ratio"`
	DynamicRatio        float64 `json:"dynamic_sampling_ratio"`
	ReductionVsFull     float64 `json:"frame_reduction_vs_full_fps"`
	DynamicOverBaseline float64 `json:"dynamic_frames_per_baseline_frame"`
	RegionCount         int     `json:"region_count"`
	MotionAvailable     bool    `json:"motion_available"`
	Extractor           string  `json:"extractor"`
}

// DynamicPlan is the complete experimental sampling decision for one video.
type DynamicPlan struct {
	Config             DynamicConfig   `json:"config"`
	BaselineTimestamps []float64       `json:"baseline_timestamps_s"`
	SelectedTimestamps []float64       `json:"selected_timestamps_s"`
	AddedTimestamps    []float64       `json:"added_timestamps_s"`
	Regions            []DynamicRegion `json:"regions"`
	Scores             []float64       `json:"activity_scores"`
	Summary            DynamicSummary  `json:"summary"`
}

// KeepKeys returns the selected timestamps as microsecond keys. The dense
// ffmpeg stream is deterministic, so extraction can keep exactly this set.
func (p *DynamicPlan) KeepKeys() map[int64]bool {
	keys := make(map[int64]bool, len(p.SelectedTimestamps))
	for _, t := range p.SelectedTimestamps {
		keys[roundMicro(t)] = true
	}
	return keys
}

// DynamicSampler reuses the codec analyzer, then converts its per-picture
// signals into a timestamp plan. It never computes pixel differences.
type DynamicSampler struct {
	analyzer *Service
	config   DynamicConfig
}

// NewDynamicSampler binds an analyzer to experimental sampling parameters.
func NewDynamicSampler(analyzer *Service, cfg DynamicConfig) *DynamicSampler {
	return &DynamicSampler{analyzer: analyzer, config: cfg}
}

// Enabled reports whether the experimental dynamic path may run.
func (d *DynamicSampler) Enabled() bool {
	return d != nil && d.config.Enabled && d.analyzer != nil
}

// Config returns a copy of the experimental sampling parameters.
func (d *DynamicSampler) Config() DynamicConfig {
	if d == nil {
		return DynamicConfig{}
	}
	return d.config
}

// DynamicResult bundles codec analysis, the sampling plan, artifacts, and report.
type DynamicResult struct {
	Analysis *Analysis
	Plan     *DynamicPlan
	Codec    *Result
	JSONPath string
	PNGPath  string
	Report   string
}

// RunDynamic analyzes the encoded stream, builds a timestamp plan, writes the
// dynamic artifacts, and returns the report. It does not run YOLO or VLM.
func (d *DynamicSampler) RunDynamic(ctx context.Context, videoPath, videoID string, durationSeconds float64, baselineInterval time.Duration) (*DynamicResult, error) {
	if d == nil || !d.config.Enabled {
		return nil, fmt.Errorf("codec experiment: dynamic sampling is disabled")
	}
	if d.analyzer == nil {
		return nil, fmt.Errorf("codec experiment: dynamic sampler has no analyzer")
	}
	if baselineInterval <= 0 {
		return nil, fmt.Errorf("codec experiment: baseline interval must be > 0")
	}
	if err := ValidateDynamicConfig(d.config, baselineInterval); err != nil {
		return nil, err
	}

	start := time.Now()
	codecRes, err := d.analyzer.Run(ctx, videoPath, videoID)
	if err != nil {
		return nil, err
	}
	if durationSeconds <= 0 {
		durationSeconds = codecRes.Analysis.Duration
	}
	plan, err := BuildDynamicPlan(codecRes.Analysis, durationSeconds, baselineInterval, d.config)
	if err != nil {
		return nil, err
	}
	jsonPath, pngPath, err := WriteDynamicArtifacts(d.analyzer.outDir, codecRes.Analysis, plan)
	if err != nil {
		return nil, err
	}
	report := FormatDynamicReport(codecRes.Analysis, plan, jsonPath, pngPath, time.Since(start))
	slog.Info("codec experiment: dynamic plan complete",
		"video_id", videoID,
		"mode", plan.Summary.Mode,
		"baseline_frames", plan.Summary.BaselineCount,
		"dynamic_frames", plan.Summary.DynamicCount,
		"regions", plan.Summary.RegionCount,
		"dynamic_json", jsonPath,
		"dynamic_png", pngPath,
		"duration_ms", time.Since(start).Milliseconds())
	return &DynamicResult{
		Analysis: codecRes.Analysis,
		Plan:     plan,
		Codec:    codecRes,
		JSONPath: jsonPath,
		PNGPath:  pngPath,
		Report:   report,
	}, nil
}

// ScoreActivity combines normalized codec signals into one experimental score
// per coded picture. Raw motion-vector count is deliberately excluded because
// static content can still emit thousands of zero-displacement vectors.
func ScoreActivity(a *Analysis, cfg DynamicConfig) ([]float64, bool, error) {
	if a == nil {
		return nil, false, fmt.Errorf("codec experiment: nil analysis")
	}
	n := len(a.Frames)
	if n == 0 {
		return nil, false, fmt.Errorf("codec experiment: no coded pictures scored")
	}
	if !a.Motion.Available {
		return make([]float64, n), false, nil
	}

	meanVals := make([]float64, n)
	maxVals := make([]float64, n)
	zeroVals := make([]float64, n)
	packetVals := make([]float64, n)
	for i := range a.Frames {
		meanVals[i] = a.Frames[i].Motion.MeanMagnitude
		maxVals[i] = a.Frames[i].Motion.MaxMagnitude
		zeroVals[i] = a.Frames[i].Motion.ZeroRatio
		packetVals[i] = float64(a.Frames[i].PacketBytes)
	}

	meanNorm := normalizeRelative(meanVals, cfg.LowPercentile, cfg.HighPercentile)
	maxNorm := normalizeRelative(maxVals, cfg.LowPercentile, cfg.HighPercentile)
	zeroNorm := normalizeZeroRatio(zeroVals, cfg.LowPercentile, cfg.HighPercentile)
	packetNorm := normalizeRelative(packetVals, cfg.LowPercentile, cfg.HighPercentile)

	scores := make([]float64, n)
	for i := 0; i < n; i++ {
		scores[i] = cfg.WeightMean*meanNorm[i] +
			cfg.WeightMax*maxNorm[i] +
			cfg.WeightStatic*(1-zeroNorm[i]) +
			cfg.WeightPacket*packetNorm[i]
	}
	return scores, true, nil
}

// DetectActiveRegions applies hysteresis to per-picture scores, expands active
// runs into temporal context windows, and merges nearby windows.
func DetectActiveRegions(frames []Frame, scores []float64, durationSeconds float64, cfg DynamicConfig) []DynamicRegion {
	n := len(frames)
	if n == 0 || len(scores) != n || durationSeconds <= 0 {
		return nil
	}
	before := cfg.WindowBefore.Seconds()
	after := cfg.WindowAfter.Seconds()
	mergeGap := cfg.MergeGap.Seconds()

	var regions []DynamicRegion
	active := false
	start := 0
	peakScore := 0.0
	peakTime := 0.0
	finish := func(end int) {
		if end < start {
			return
		}
		startTime := frames[start].Timestamp - before
		endTime := frames[end].Timestamp + after
		if startTime < 0 {
			startTime = 0
		}
		if endTime > durationSeconds {
			endTime = durationSeconds
		}
		regions = append(regions, DynamicRegion{
			Start:        roundSeconds(startTime),
			End:          roundSeconds(endTime),
			Peak:         roundSeconds(peakTime),
			PeakScore:    peakScore,
			ActiveFrames: end - start + 1,
		})
	}

	for i := 0; i < n; i++ {
		score := scores[i]
		if !active {
			if score >= cfg.EnterThreshold {
				active = true
				start = i
				peakScore = score
				peakTime = frames[i].Timestamp
			}
			continue
		}
		if score > peakScore {
			peakScore = score
			peakTime = frames[i].Timestamp
		}
		if score <= cfg.ExitThreshold {
			finish(i - 1)
			active = false
		}
	}
	if active {
		finish(n - 1)
	}

	merged := regions[:0]
	for _, r := range regions {
		if len(merged) == 0 {
			merged = append(merged, r)
			continue
		}
		last := &merged[len(merged)-1]
		if r.Start-last.End <= mergeGap+1e-9 {
			if r.End > last.End {
				last.End = r.End
			}
			if r.PeakScore > last.PeakScore {
				last.PeakScore = r.PeakScore
				last.Peak = r.Peak
			}
			last.ActiveFrames += r.ActiveFrames
			continue
		}
		merged = append(merged, r)
	}
	return merged
}

// BuildDynamicPlan merges the baseline grid with dense timestamps inside codec
// activity regions. The output is sorted, deduplicated, and chronological.
func BuildDynamicPlan(a *Analysis, durationSeconds float64, baselineInterval time.Duration, cfg DynamicConfig) (*DynamicPlan, error) {
	if a == nil {
		return nil, fmt.Errorf("codec experiment: nil analysis")
	}
	if durationSeconds <= 0 {
		durationSeconds = a.Duration
	}
	if durationSeconds <= 0 {
		return nil, fmt.Errorf("codec experiment: duration unavailable; cannot build sampling plan")
	}
	if baselineInterval <= 0 {
		return nil, fmt.Errorf("codec experiment: baseline interval must be > 0")
	}
	cfg.BaselineInterval = baselineInterval
	if err := ValidateDynamicConfig(cfg, baselineInterval); err != nil {
		return nil, err
	}

	baseline, err := sampleGrid(durationSeconds, baselineInterval)
	if err != nil {
		return nil, err
	}
	scores, motionAvailable, err := ScoreActivity(a, cfg)
	if err != nil {
		return nil, err
	}

	plan := &DynamicPlan{Config: cfg, Scores: append([]float64(nil), scores...)}
	selected := make(map[int64]bool, len(baseline))
	for _, t := range baseline {
		selected[roundMicro(t)] = true
		plan.BaselineTimestamps = append(plan.BaselineTimestamps, canonicalSeconds(t))
	}

	mode := "dynamic_no_regions"
	reason := "no codec activity crossed the enter threshold"
	if !motionAvailable {
		mode = "baseline_fallback"
		reason = "motion vectors unavailable; dynamic sampling requires exported codec motion vectors"
	} else {
		plan.Regions = DetectActiveRegions(a.Frames, scores, durationSeconds, cfg)
		if plan.Regions == nil {
			plan.Regions = []DynamicRegion{}
		}
		if len(plan.Regions) > 0 {
			mode = "dynamic"
			reason = ""
			dense := cfg.DenseInterval.Seconds()
			for k := 0; ; k++ {
				t := float64(k) * dense
				if t >= durationSeconds-1e-9 {
					break
				}
				for _, region := range plan.Regions {
					if t+1e-9 >= region.Start && t-1e-9 <= region.End {
						selected[roundMicro(t)] = true
						break
					}
				}
			}
		}
	}

	baselineSet := make(map[int64]bool, len(plan.BaselineTimestamps))
	for _, t := range plan.BaselineTimestamps {
		baselineSet[roundMicro(t)] = true
	}
	for key := range selected {
		t := float64(key) / 1e6
		plan.SelectedTimestamps = append(plan.SelectedTimestamps, t)
		if !baselineSet[key] {
			plan.AddedTimestamps = append(plan.AddedTimestamps, t)
		}
	}
	sort.Float64s(plan.SelectedTimestamps)
	sort.Float64s(plan.AddedTimestamps)
	if plan.BaselineTimestamps == nil {
		plan.BaselineTimestamps = []float64{}
	}
	if plan.SelectedTimestamps == nil {
		plan.SelectedTimestamps = []float64{}
	}
	if plan.AddedTimestamps == nil {
		plan.AddedTimestamps = []float64{}
	}

	nativeFPS := a.FPS
	nativeFrames := len(a.Frames)
	if nativeFPS > 0 {
		nativeFrames = int(math.Round(durationSeconds * nativeFPS))
	}
	baselineCount := len(plan.BaselineTimestamps)
	dynamicCount := len(plan.SelectedTimestamps)
	addedCount := len(plan.AddedTimestamps)
	baselineRatio := 0.0
	dynamicRatio := 0.0
	if nativeFrames > 0 {
		baselineRatio = float64(baselineCount) / float64(nativeFrames)
		dynamicRatio = float64(dynamicCount) / float64(nativeFrames)
	}
	overBaseline := 0.0
	if baselineCount > 0 {
		overBaseline = float64(dynamicCount) / float64(baselineCount)
	}
	plan.Summary = DynamicSummary{
		Mode:                mode,
		Reason:              reason,
		DurationSeconds:     durationSeconds,
		NativeFPS:           nativeFPS,
		NativeFrames:        nativeFrames,
		BaselineCount:       baselineCount,
		DynamicCount:        dynamicCount,
		AddedCount:          addedCount,
		BaselineRatio:       baselineRatio,
		DynamicRatio:        dynamicRatio,
		ReductionVsFull:     1 - dynamicRatio,
		DynamicOverBaseline: overBaseline,
		RegionCount:         len(plan.Regions),
		MotionAvailable:     motionAvailable,
		Extractor:           a.Extractor,
	}
	return plan, nil
}

// WriteDynamicArtifacts writes the sampling plan JSON and sampling timeline PNG.
// The codec analyzer's own JSON/CSV/PNG are left untouched.
func WriteDynamicArtifacts(outDir string, a *Analysis, plan *DynamicPlan) (string, string, error) {
	if a == nil || plan == nil {
		return "", "", fmt.Errorf("codec experiment: nil analysis or plan")
	}
	if outDir == "" {
		outDir = "./codec_experiment"
	}
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		return "", "", fmt.Errorf("codec experiment: %w", err)
	}
	base := artifactBase(a.VideoName)
	jsonPath := filepath.Join(outDir, base+"_dynamic.json")
	pngPath := filepath.Join(outDir, base+"_dynamic.png")

	artifact := struct {
		Experiment           string          `json:"experiment"`
		Video                string          `json:"video"`
		VideoID              string          `json:"video_id,omitempty"`
		Source               string          `json:"source"`
		Codec                string          `json:"codec"`
		Extractor            string          `json:"extractor"`
		CodecAnalysis        string          `json:"codec_analysis_json"`
		Config               DynamicConfig   `json:"config"`
		Summary              DynamicSummary  `json:"summary"`
		Regions              []DynamicRegion `json:"regions"`
		BaselineTimestamps   []float64       `json:"baseline_timestamps_s"`
		SelectedTimestamps   []float64       `json:"selected_timestamps_s"`
		AddedTimestamps      []float64       `json:"added_timestamps_s"`
		ActivityScores       []float64       `json:"activity_scores"`
		ScoreFrameTimestamps []float64       `json:"score_frame_timestamps_s"`
	}{
		Experiment:           DynamicExperimentName,
		Video:                a.VideoName,
		VideoID:              a.VideoID,
		Source:               a.Path,
		Codec:                a.Codec,
		Extractor:            a.Extractor,
		CodecAnalysis:        base + ".json",
		Config:               plan.Config,
		Summary:              plan.Summary,
		Regions:              plan.Regions,
		BaselineTimestamps:   plan.BaselineTimestamps,
		SelectedTimestamps:   plan.SelectedTimestamps,
		AddedTimestamps:      plan.AddedTimestamps,
		ActivityScores:       plan.Scores,
		ScoreFrameTimestamps: frameTimestamps(a),
	}
	data, err := json.MarshalIndent(artifact, "", "  ")
	if err != nil {
		return "", "", fmt.Errorf("codec experiment: %w", err)
	}
	data = append(data, '\n')
	if err := os.WriteFile(jsonPath, data, 0o644); err != nil {
		return "", "", fmt.Errorf("codec experiment: %w", err)
	}
	if err := writeDynamicPNG(pngPath, a, plan); err != nil {
		slog.Warn("codec experiment: dynamic png not written", "video", a.VideoName, "error", err)
		return jsonPath, "", nil
	}
	return jsonPath, pngPath, nil
}

func frameTimestamps(a *Analysis) []float64 {
	out := make([]float64, 0, len(a.Frames))
	for i := range a.Frames {
		out = append(out, a.Frames[i].Timestamp)
	}
	return out
}

func sampleGrid(durationSeconds float64, step time.Duration) ([]float64, error) {
	if step <= 0 {
		return nil, fmt.Errorf("codec experiment: grid step must be > 0")
	}
	if durationSeconds <= 0 {
		return nil, fmt.Errorf("codec experiment: duration unavailable; cannot build timestamp grid")
	}
	iv := step.Seconds()
	if iv <= 0 {
		return nil, fmt.Errorf("codec experiment: grid step must be > 0")
	}
	var out []float64
	for t := 0.0; t < durationSeconds-1e-9; t += iv {
		out = append(out, canonicalSeconds(t))
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("codec experiment: duration unavailable; cannot build timestamp grid")
	}
	return out, nil
}

func roundMicro(t float64) int64 {
	return int64(math.Round(t * 1e6))
}

func canonicalSeconds(t float64) float64 {
	return float64(roundMicro(t)) / 1e6
}

func roundSeconds(t float64) float64 {
	return math.Round(t*1e6) / 1e6
}

func percentile(sorted []float64, p float64) float64 {
	if len(sorted) == 0 {
		return 0
	}
	if p <= 0 {
		return sorted[0]
	}
	if p >= 100 {
		return sorted[len(sorted)-1]
	}
	rank := p / 100 * float64(len(sorted)-1)
	low := int(math.Floor(rank))
	high := int(math.Ceil(rank))
	if low == high {
		return sorted[low]
	}
	frac := rank - float64(low)
	return sorted[low]*(1-frac) + sorted[high]*frac
}

func normalizeRelative(values []float64, lowP, highP float64) []float64 {
	out := make([]float64, len(values))
	if len(values) == 0 {
		return out
	}
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	low := percentile(sorted, lowP)
	high := percentile(sorted, highP)
	if !(high > low) {
		return out
	}
	span := high - low
	for i, v := range values {
		norm := (v - low) / span
		if norm < 0 {
			norm = 0
		}
		if norm > 1 {
			norm = 1
		}
		out[i] = norm
	}
	return out
}

func normalizeZeroRatio(values []float64, lowP, highP float64) []float64 {
	out := make([]float64, len(values))
	if len(values) == 0 {
		return out
	}
	sorted := append([]float64(nil), values...)
	sort.Float64s(sorted)
	low := percentile(sorted, lowP)
	high := percentile(sorted, highP)
	if !(high > low) {
		for i, v := range values {
			if v < 0 {
				v = 0
			}
			if v > 1 {
				v = 1
			}
			out[i] = v
		}
		return out
	}
	span := high - low
	for i, v := range values {
		norm := (v - low) / span
		if norm < 0 {
			norm = 0
		}
		if norm > 1 {
			norm = 1
		}
		out[i] = norm
	}
	return out
}
