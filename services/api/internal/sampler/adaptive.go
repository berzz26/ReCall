// Adaptive coarse planner (Phase 2).
//
// The fixed 2-second timestamp plan is replaced by a visual-probe-driven
// coarse plan: fewer YOLO frames on static footage, continued sampling
// around visually active footage, a heartbeat bounding the maximum temporal
// gap, all strictly within the Phase 1 baseline frame budget.
//
// Selection rule (probe runs at 5 FPS; decisions reference the last kept
// frame, not the previous probe frame):
//
//	last_kept = first valid probe frame; keep it (reason coarse); running_max = 0
//	for each later probe frame:
//	  changed_fraction vs last_kept; running_max = max(running_max, changed_fraction)
//	  since = current - last_kept
//	  if since >= MAX_GAP                          -> keep (heartbeat)
//	  else if since >= G and running_max >= F_KEEP -> keep (coarse)
//	  else                                         -> skip
//	  on keep: last_kept = current; running_max = 0
//
// Any probe/planning failure falls back to the Phase 1 baseline planner
// (mode baseline_fallback) so adaptive sampling can never fail a video.
package sampler

import (
	"context"
	"fmt"
	"log/slog"
	"math"
	"time"

	"github.com/google/uuid"
)

// Sampler modes for Phase 2. ModeBaseline is unchanged from Phase 1.
const (
	// ModeAdaptive marks a visual-probe-driven coarse plan.
	ModeAdaptive = "adaptive"
	// ModeBaselineFallback marks a fixed 2-second plan produced because the
	// visual probe or planning failed. uniform_fallback is intentionally
	// deferred to the robustness phase.
	ModeBaselineFallback = "baseline_fallback"
)

// Adaptive planner defaults.
const (
	// DefaultCoarseInterval (G) is the minimum spacing between
	// activity-driven selections. The probe oversamples at 5 FPS while YOLO
	// frames stay much sparser.
	DefaultCoarseInterval = 4 * time.Second
	// DefaultMaxGap (MAX_GAP) is the hard temporal coverage guard: even
	// perfectly static footage yields a frame at least this often.
	DefaultMaxGap = 10 * time.Second
	// DefaultFKeep is the running-maximum changed_fraction required to keep
	// a coarse frame: at least a quarter of the 64 probe blocks (16 blocks)
	// must differ from the last kept frame. Deliberately conservative and
	// untuned: heartbeat coverage applies regardless, and the noise floor
	// already suppresses sensor/compression residuals, so anything crossing
	// 0.25 is unambiguous frame-level visual change.
	DefaultFKeep = 0.25
	// DefaultPlanTimeout bounds the whole probe+planning operation. The
	// sampler is an optimization; on timeout the baseline plan is used.
	DefaultPlanTimeout = 5 * time.Minute
)

// AdaptiveConfig configures the visual probe and the coarse selection rule.
// Zero values fall back to the defaults above in NewAdaptiveCoarsePlanner.
type AdaptiveConfig struct {
	// BaselineInterval is the Phase 1 reference grid (2s) used only for the
	// budget: B = ceil(duration / BaselineInterval) * beta.
	BaselineInterval time.Duration
	Beta             float64
	ProbeFPS         float64
	ProbeSize        int
	ProbeGrid        int
	NoiseK           float64
	// CoarseInterval (G) is the minimum spacing of coarse selections.
	CoarseInterval time.Duration
	// MaxGap (MAX_GAP) forces a heartbeat selection when exceeded.
	MaxGap     time.Duration
	FKeep      float64
	FFmpegPath string
	// PlanTimeout bounds probe decoding plus planning.
	PlanTimeout time.Duration
}

// DefaultAdaptiveConfig returns the Phase 2 starting configuration.
func DefaultAdaptiveConfig() AdaptiveConfig {
	return AdaptiveConfig{
		BaselineInterval: DefaultBaselineInterval,
		Beta:             DefaultBeta,
		ProbeFPS:         DefaultProbeFPS,
		ProbeSize:        DefaultProbeSize,
		ProbeGrid:        DefaultProbeGrid,
		NoiseK:           DefaultNoiseK,
		CoarseInterval:   DefaultCoarseInterval,
		MaxGap:           DefaultMaxGap,
		FKeep:            DefaultFKeep,
		FFmpegPath:       "ffmpeg",
		PlanTimeout:      DefaultPlanTimeout,
	}
}

// ProbePoint retains one probe decision for debugging and as the Phase 4
// refinement prior. It is internal to the sampler and never attached to
// VideoFrame or persisted.
type ProbePoint struct {
	TimestampSeconds float64
	// ChangedFraction is visual change vs the last kept frame at this time.
	ChangedFraction float64
	// RunningMax is the running maximum since the last selection, after
	// incorporating this frame.
	RunningMax float64
	Selected   bool
	// Reason is coarse/heartbeat when Selected, empty otherwise.
	Reason Reason
}

// ProbeReport is the retained probe score series for one video.
type ProbeReport struct {
	Points        []ProbePoint
	NoiseFloor    float64
	FramesDecoded int
}

// AdaptivePlanResult is the outcome of coarse planning for one video.
type AdaptivePlanResult struct {
	Plan SamplingPlan
	// Probe is the retained series (nil on baseline_fallback).
	Probe *ProbeReport
	// NoiseFloor is the per-video adaptive threshold (0 on fallback).
	NoiseFloor float64
	// FallbackReason is empty for adaptive plans.
	FallbackReason string
}

// VideoPlanner plans timestamps for a concrete video file. The extractor
// consumes only the resulting SamplingPlan and stays unaware of how
// timestamps were selected.
type VideoPlanner interface {
	PlanVideo(ctx context.Context, videoID uuid.UUID, videoPath string, durationSeconds float64) (AdaptivePlanResult, error)
}

// AdaptiveCoarsePlanner implements VideoPlanner with the visual probe plus
// the coarse selection rule.
type AdaptiveCoarsePlanner struct {
	cfg      AdaptiveConfig
	baseline *BaselinePlanner
}

// NewAdaptiveCoarsePlanner sanitizes cfg (non-positive/NaN values revert to
// defaults) and returns a planner sharing the Phase 1 budget semantics.
func NewAdaptiveCoarsePlanner(cfg AdaptiveConfig) *AdaptiveCoarsePlanner {
	if cfg.BaselineInterval <= 0 {
		cfg.BaselineInterval = DefaultBaselineInterval
	}
	if math.IsNaN(cfg.Beta) || math.IsInf(cfg.Beta, 0) || cfg.Beta <= 0 {
		cfg.Beta = DefaultBeta
	}
	if cfg.ProbeFPS <= 0 || math.IsNaN(cfg.ProbeFPS) || math.IsInf(cfg.ProbeFPS, 0) {
		cfg.ProbeFPS = DefaultProbeFPS
	}
	if cfg.ProbeSize <= 0 {
		cfg.ProbeSize = DefaultProbeSize
	}
	if cfg.ProbeGrid <= 0 {
		cfg.ProbeGrid = DefaultProbeGrid
	}
	if math.IsNaN(cfg.NoiseK) || cfg.NoiseK < 0 {
		cfg.NoiseK = DefaultNoiseK
	}
	if cfg.CoarseInterval <= 0 {
		cfg.CoarseInterval = DefaultCoarseInterval
	}
	if cfg.MaxGap <= 0 {
		cfg.MaxGap = DefaultMaxGap
	}
	if math.IsNaN(cfg.FKeep) || cfg.FKeep < 0 {
		cfg.FKeep = DefaultFKeep
	}
	if cfg.FKeep > 1 {
		cfg.FKeep = 1
	}
	if cfg.FFmpegPath == "" {
		cfg.FFmpegPath = "ffmpeg"
	}
	if cfg.PlanTimeout <= 0 {
		cfg.PlanTimeout = DefaultPlanTimeout
	}
	return &AdaptiveCoarsePlanner{
		cfg:      cfg,
		baseline: NewBaselinePlanner(cfg.BaselineInterval, cfg.Beta),
	}
}

// Config returns the sanitized configuration (for logging/tests).
func (p *AdaptiveCoarsePlanner) Config() AdaptiveConfig { return p.cfg }

// fallback builds the Phase 1 baseline plan with mode baseline_fallback.
// It errors only when the baseline itself cannot plan (e.g. bad duration),
// which is a genuine pipeline error rather than a probe failure.
func (p *AdaptiveCoarsePlanner) fallback(videoID uuid.UUID, durationSeconds float64, reason string) (AdaptivePlanResult, error) {
	plan, err := p.baseline.Plan(videoID, durationSeconds)
	if err != nil {
		return AdaptivePlanResult{}, err
	}
	plan.Mode = ModeBaselineFallback
	return AdaptivePlanResult{Plan: plan, FallbackReason: reason}, nil
}

// PlanVideo decodes the visual probe, estimates the per-video noise floor,
// applies the coarse selection rule, and returns a budget-capped plan.
// Any probe/planning failure (or timeout) yields the baseline_fallback plan
// instead of an error, so adaptive sampling never fails a video.
func (p *AdaptiveCoarsePlanner) PlanVideo(ctx context.Context, videoID uuid.UUID, videoPath string, durationSeconds float64) (AdaptivePlanResult, error) {
	budget, err := BaselineBudget(durationSeconds, p.cfg.BaselineInterval, p.cfg.Beta)
	if err != nil {
		return AdaptivePlanResult{}, err
	}

	planCtx, cancel := context.WithTimeout(ctx, p.cfg.PlanTimeout)
	defer cancel()

	frames, err := DecodeProbeFrames(planCtx, p.cfg.FFmpegPath, videoPath, p.cfg.ProbeFPS, p.cfg.ProbeSize, p.cfg.ProbeGrid, durationSeconds)
	if err != nil {
		reason := fmt.Sprintf("visual probe failed: %v", err)
		slog.Warn("sampler: probe failure, baseline fallback", "video_id", videoID.String(), "reason", reason)
		return p.fallback(videoID, durationSeconds, reason)
	}

	// Per-video noise floor from consecutive-frame block differences:
	// sensor/compression noise shows up frame-to-frame, while real visual
	// change is an outlier the median+MAD statistic resists.
	est := &NoiseEstimator{}
	for i := 1; i < len(frames); i++ {
		a, b := frames[i-1].Means, frames[i].Means
		if len(a) != len(b) {
			continue
		}
		for k := range a {
			d := math.Abs(float64(b[k]) - float64(a[k]))
			est.Add(d)
		}
	}
	noiseFloor := est.Threshold(p.cfg.NoiseK)

	plan, report := p.planFromFrames(videoID, durationSeconds, frames, noiseFloor, budget)

	coarse, heartbeat := 0, 0
	for _, e := range plan.Entries {
		switch e.Reason {
		case ReasonCoarse:
			coarse++
		case ReasonHeartbeat:
			heartbeat++
		}
	}
	saved := budget.BaselineCount - len(plan.Entries)
	if saved < 0 {
		saved = 0
	}
	slog.Info("sampler: coarse plan",
		"video_id", videoID.String(),
		"duration_seconds", durationSeconds,
		"mode", plan.Mode,
		"baseline_budget", budget.BaselineCount,
		"budget_cap", budget.Cap,
		"selected", len(plan.Entries),
		"frames_saved_vs_baseline", saved,
		"coarse", coarse, "heartbeat", heartbeat,
		"probe_fps", p.cfg.ProbeFPS,
		"probe_resolution", fmt.Sprintf("%dx%d", p.cfg.ProbeSize, p.cfg.ProbeSize),
		"probe_grid", fmt.Sprintf("%dx%d", p.cfg.ProbeGrid, p.cfg.ProbeGrid),
		"noise_floor", noiseFloor,
		"noise_k", p.cfg.NoiseK,
		"f_keep", p.cfg.FKeep,
		"g_seconds", p.cfg.CoarseInterval.Seconds(),
		"max_gap_seconds", p.cfg.MaxGap.Seconds(),
		"probe_frames", len(frames),
	)
	slog.Debug("sampler: probe series retained",
		"video_id", videoID.String(),
		"points", len(report.Points),
	)
	return AdaptivePlanResult{Plan: plan, Probe: report, NoiseFloor: noiseFloor}, nil
}

// planFromFrames is the pure selection core: probe frames plus a noise
// threshold yield a budget-capped SamplingPlan and the retained series.
// Exported indirectly for tests via PlanVideoWithFrames.
func (p *AdaptiveCoarsePlanner) planFromFrames(videoID uuid.UUID, durationSeconds float64, frames []ProbeFrame, noiseFloor float64, budget Budget) (SamplingPlan, *ProbeReport) {
	gSec := p.cfg.CoarseInterval.Seconds()
	maxGapSec := p.cfg.MaxGap.Seconds()

	// First valid probe frame is always kept (reason coarse) and becomes
	// the comparison reference.
	lastKeptTs := frames[0].TimestampSeconds
	ref := frames[0].Means
	runningMax := 0.0

	selected := []float64{lastKeptTs}
	reasons := []Reason{ReasonCoarse}
	scores := []float64{0}
	report := &ProbeReport{
		Points:        make([]ProbePoint, 0, len(frames)),
		NoiseFloor:    noiseFloor,
		FramesDecoded: len(frames),
	}
	report.Points = append(report.Points, ProbePoint{
		TimestampSeconds: lastKeptTs,
		ChangedFraction:  0,
		RunningMax:       0,
		Selected:         true,
		Reason:           ReasonCoarse,
	})

	for i := 1; i < len(frames); i++ {
		f := frames[i]
		frac := ChangedFraction(ref, f.Means, noiseFloor)
		if frac > runningMax {
			runningMax = frac
		}
		since := f.TimestampSeconds - lastKeptTs
		var keep bool
		var reason Reason
		switch {
		case since >= maxGapSec:
			// Heartbeat ignores changed_fraction: hard coverage guard.
			keep, reason = true, ReasonHeartbeat
		case since >= gSec && runningMax >= p.cfg.FKeep:
			keep, reason = true, ReasonCoarse
		}
		report.Points = append(report.Points, ProbePoint{
			TimestampSeconds: f.TimestampSeconds,
			ChangedFraction:  frac,
			RunningMax:       runningMax,
			Selected:         keep,
			Reason:           reason,
		})
		if keep {
			selected = append(selected, f.TimestampSeconds)
			reasons = append(reasons, reason)
			scores = append(scores, runningMax)
			lastKeptTs = f.TimestampSeconds
			ref = f.Means
			runningMax = 0
		}
	}

	// Existing Phase 1 boundary behavior, then the hard budget cap (keeps
	// the earliest timestamps).
	ts := CapTimestamps(NormalizeTimestamps(selected, durationSeconds), budget.Cap)
	entries := make([]PlannedTimestamp, 0, len(ts))
	for i, t := range ts {
		score := scores[i]
		entries = append(entries, PlannedTimestamp{
			TimestampSeconds: t,
			Reason:           reasons[i],
			Score:            &score,
		})
	}
	plan := SamplingPlan{
		VideoID:         videoID,
		DurationSeconds: durationSeconds,
		Mode:            ModeAdaptive,
		Budget:          budget,
		Entries:         entries,
	}
	return plan, report
}

// PlanVideoWithFrames runs the pure selection core on caller-supplied probe
// frames (tests, debugging). It shares the exact code path as PlanVideo
// minus FFmpeg, and applies the same budget cap.
func (p *AdaptiveCoarsePlanner) PlanVideoWithFrames(videoID uuid.UUID, durationSeconds float64, frames []ProbeFrame, noiseFloor float64) (SamplingPlan, *ProbeReport, error) {
	if len(frames) == 0 {
		return SamplingPlan{}, nil, fmt.Errorf("no probe frames to plan from")
	}
	budget, err := BaselineBudget(durationSeconds, p.cfg.BaselineInterval, p.cfg.Beta)
	if err != nil {
		return SamplingPlan{}, nil, err
	}
	plan, report := p.planFromFrames(videoID, durationSeconds, frames, noiseFloor, budget)
	return plan, report, nil
}
