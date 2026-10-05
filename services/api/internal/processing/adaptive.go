package processing

// Adaptive-sampling production orchestrator (Phase 5).
//
// This is the single production boundary for optional adaptive planning.
// It sits between metadata/segmentation and frame extraction in the
// pipeline and decides, per video, which frame plan downstream stages
// consume:
//
//	SAMPLER_ADAPTIVE=false → baseline path untouched (this file not used)
//	SAMPLER_ADAPTIVE=true  → probe → coarse plan → busy? → refine → validate
//
// Safety properties:
//   - One panic boundary (recover) around the whole adaptive block: any
//     unexpected panic becomes a baseline fallback, never a pipeline crash.
//     The recover lives ONLY here, not scattered through the sampler.
//   - One planning timeout (SAMPLER_PLAN_TIMEOUT) covering probe, coarse
//     planning, planning YOLO, disagreement, refinement, and midpoint YOLO.
//     Expiry falls back to baseline. Parent-context cancellation propagates
//     as cancellation (not misreported as sampler failure).
//   - Planning detections are EPHEMERAL (in-memory only). Production
//     detections are produced exactly once, later, by the pipeline's visual
//     stage over the final frame set. Planning YOLO and production YOLO are
//     therefore distinct passes that can never double-persist or collide.
//   - Any adaptive failure discards adaptive state and re-extracts the
//     baseline plan via GenerateForVideoWithPlan (which replaces whatever
//     partial frames exist). Downstream only ever sees one complete,
//     validated frame set.
//   - Hard budget: every DetectFunc timestamp counts against the cap before
//     spending; the final plan is validated (sorted, unique, in range,
//     within cap, known reasons). A violation aborts to baseline.
//
// Checkpoint resume: when frames already exist for the video (and the
// pipeline runs with checkpoints), the orchestrator skips planning entirely
// and preserves the previous frame set — same semantics as the legacy
// skip-if-exists path it replaces. Refinement is best-effort within a
// single frame-stage execution; any extracted frame set is a valid input to
// the visual stage, which always YOLOs whatever frames exist.

import (
	"context"
	"fmt"
	"log/slog"
	"time"

	"github.com/google/uuid"

	"github.com/berzz26/recall/services/api/internal/sampler"
	"github.com/berzz26/recall/services/api/internal/tracker"
	"github.com/berzz26/recall/services/api/internal/video"
	"github.com/berzz26/recall/services/api/internal/video_frame"
	"github.com/berzz26/recall/services/api/internal/video_segment"
)

// VideoFrameExtractor is the frame-extraction surface the orchestrator
// needs. *video_frame.Service satisfies it; tests inject fakes.
type VideoFrameExtractor interface {
	GenerateForVideoWithPlan(ctx context.Context, v *video.Video, segments []video_segment.VideoSegment, plan sampler.SamplingPlan, width, height int) ([]video_frame.VideoFrame, error)
	ExtractAdditional(ctx context.Context, v *video.Video, segments []video_segment.VideoSegment, timestamps []float64, durationSeconds float64, width, height int) ([]video_frame.VideoFrame, error)
	GetByVideoID(ctx context.Context, videoID uuid.UUID) ([]video_frame.VideoFrame, error)
}

// PlanningDetector is the ephemeral planning-detection surface.
// *visual.Service satisfies it; planning detections are never persisted.
type PlanningDetector interface {
	DetectFrames(ctx context.Context, videoID uuid.UUID, frames []video_frame.VideoFrame) ([]tracker.DetectionInput, error)
}

// AdaptiveSamplerConfig wires the orchestrator. All sampler tuning lives in
// sampler.*Config values; this struct only carries them plus orchestration
// behavior (timeout, shadow, baseline reference for fallbacks).
type AdaptiveSamplerConfig struct {
	// Planner performs probe + coarse planning (sampler.VideoPlanner).
	Planner sampler.VideoPlanner
	// Refiner performs Phase 4 best-first refinement.
	Refiner *sampler.Refiner
	// Busy configures global-busy detection.
	Busy sampler.BusyConfig
	// Timeout bounds the complete adaptive operation (probe through
	// midpoint YOLO). Expiry falls back to baseline.
	Timeout time.Duration
	// Shadow computes (but never applies) the adaptive plan: baseline
	// frames remain authoritative while adaptive statistics are logged.
	Shadow bool
	// BaselineInterval and Beta rebuild the baseline plan for fallbacks
	// and uniform mode.
	BaselineInterval time.Duration
	Beta             float64
}

// AdaptiveStats records one adaptive sampling run for logging and tests.
// Field names mirror §17 shadow metrics plus planning-time breakdown.
type AdaptiveStats struct {
	VideoID          string
	DurationSeconds  float64
	Mode             string
	Version          string
	FallbackReason   string
	BaselineFrames   int
	BudgetCap        int
	CoarseFrames     int
	RefinementFrames int
	FinalFrames      int
	EstimatedSavings int
	// Reason counts over the final plan (why each frame was selected).
	ReasonCoarse     int
	ReasonHeartbeat  int
	ReasonRefinement int
	Rounds           int
	MaxPriority      float64
	MeanPriority     float64
	BusyFraction     float64
	BusyPoints       int
	ProbePoints      int
	NoiseFloor       float64
	// ShadowAdaptiveFrames records what the adaptive plan would have
	// selected in shadow mode (informational only; baseline authoritative).
	ShadowAdaptiveFrames int
	ProbeMs              int64
	PlanningYoloMs       int64
	RefineMs             int64
	TotalMs              int64
	Resumed              bool
	Shadow               bool
}

// AdaptiveSampler runs the production adaptive-sampling boundary.
type AdaptiveSampler struct {
	frames  VideoFrameExtractor
	visual  PlanningDetector
	planner sampler.VideoPlanner
	refiner *sampler.Refiner
	cfg     AdaptiveSamplerConfig
}

// NewAdaptiveSampler builds the orchestrator. Nil planner/refiner is
// rejected: a half-wired adaptive path must fail fast at startup, not
// mid-video.
func NewAdaptiveSampler(frames VideoFrameExtractor, visual PlanningDetector, planner sampler.VideoPlanner, refiner *sampler.Refiner, cfg AdaptiveSamplerConfig) (*AdaptiveSampler, error) {
	if frames == nil || visual == nil {
		return nil, fmt.Errorf("adaptive sampler requires frame extractor and detector")
	}
	if planner == nil || refiner == nil {
		return nil, fmt.Errorf("adaptive sampler requires planner and refiner")
	}
	cfg.Busy = sampler.NewBusyConfig(cfg.Busy)
	if cfg.Timeout <= 0 {
		cfg.Timeout = sampler.DefaultPlanTimeout
	}
	if cfg.BaselineInterval <= 0 {
		cfg.BaselineInterval = sampler.DefaultBaselineInterval
	}
	if cfg.Beta <= 0 {
		cfg.Beta = sampler.DefaultBeta
	}
	return &AdaptiveSampler{frames: frames, visual: visual, planner: planner, refiner: refiner, cfg: cfg}, nil
}

// baselinePlan rebuilds the source-of-truth baseline plan. It fails only on
// invalid duration, which the pipeline already rejects upstream.
func (a *AdaptiveSampler) baselinePlan(videoID uuid.UUID, duration float64) (sampler.SamplingPlan, error) {
	return sampler.NewBaselinePlanner(a.cfg.BaselineInterval, a.cfg.Beta).Plan(videoID, duration)
}

// PlanAndExtract runs the full adaptive boundary and returns the final
// validated plan (nil on resume-skip) plus run statistics. Downstream
// stages read frames/detections from storage as usual; the plan object
// itself is observability, not a downstream input.
// allowResume mirrors the legacy checkpoint skip-if-exists semantics.
func (a *AdaptiveSampler) PlanAndExtract(ctx context.Context, v *video.Video, segments []video_segment.VideoSegment, duration float64, w, h int, videoPath string, allowResume bool) (plan *sampler.SamplingPlan, stats *AdaptiveStats, err error) {
	stats = &AdaptiveStats{VideoID: v.ID.String(), DurationSeconds: duration, Version: sampler.SamplerVersion, Shadow: a.cfg.Shadow}
	start := time.Now()
	defer func() { stats.TotalMs = time.Since(start).Milliseconds() }()

	// Single panic boundary for the optional adaptive path. There is
	// deliberately exactly one recover: adaptive planning must never crash
	// the pipeline, and scattering recovers would hide bugs elsewhere.
	defer func() {
		if r := recover(); r != nil {
			slog.Error("sampler: panic contained, baseline fallback",
				"video_id", v.ID.String(), "panic", fmt.Sprintf("%v", r))
			// A cancelled parent must still propagate as cancellation.
			if ctx.Err() != nil {
				stats.Mode = "cancelled"
				plan, err = nil, ctx.Err()
				return
			}
			fb, _, fbErr := a.extractBaseline(ctx, v, segments, duration, w, h, stats, fmt.Sprintf("planner panic: %v", r))
			if fbErr != nil {
				plan, err = nil, fbErr
				return
			}
			stats.FinalFrames = len(fb.Timestamps())
			plan, err = fb, nil
		}
	}()

	if allowResume {
		if err := ctx.Err(); err != nil {
			return nil, stats, err
		}
		existing, err := a.frames.GetByVideoID(ctx, v.ID)
		if err == nil && len(existing) > 0 {
			stats.Mode = "resume"
			stats.Resumed = true
			stats.FinalFrames = len(existing)
			if budget, berr := sampler.BaselineBudget(duration, a.cfg.BaselineInterval, a.cfg.Beta); berr == nil {
				stats.BudgetCap = budget.Cap
				stats.BaselineFrames = budget.BaselineCount
			}
			slog.Info("sampler: resume, preserving existing frames",
				"video_id", v.ID.String(), "existing_frames", len(existing))
			return nil, stats, nil
		}
	}

	planCtx, cancel := context.WithTimeout(ctx, a.cfg.Timeout)
	defer cancel()

	p, st, err := a.adaptiveInner(planCtx, ctx, v, segments, duration, w, h, videoPath)
	*stats = *st
	stats.VideoID = v.ID.String()
	stats.DurationSeconds = duration
	stats.Version = sampler.SamplerVersion
	stats.Shadow = a.cfg.Shadow
	return p, stats, err
}

// adaptiveInner is the timeout-bounded adaptive flow (panic boundary above).
// parentCtx distinguishes our deadline expiry (baseline fallback) from real
// caller cancellation (propagate).
func (a *AdaptiveSampler) adaptiveInner(planCtx, parentCtx context.Context, v *video.Video, segments []video_segment.VideoSegment, duration float64, w, h int, videoPath string) (*sampler.SamplingPlan, *AdaptiveStats, error) {
	stats := &AdaptiveStats{}
	cancelled := func() bool { return parentCtx.Err() != nil }

	// Budget first: every later branch (including fallbacks and shadow)
	// reports against the same cap.
	budget, err := sampler.BaselineBudget(duration, a.cfg.BaselineInterval, a.cfg.Beta)
	if err != nil {
		return nil, stats, fmt.Errorf("budget failed: %w", err)
	}
	stats.BudgetCap = budget.Cap
	stats.BaselineFrames = budget.BaselineCount

	// 1. Probe + coarse plan.
	probeStart := time.Now()
	pres, err := a.planner.PlanVideo(planCtx, v.ID, videoPath, duration)
	stats.ProbeMs = time.Since(probeStart).Milliseconds()
	if err != nil {
		if cancelled() {
			return nil, stats, parentCtx.Err()
		}
		if planCtx.Err() == context.DeadlineExceeded {
			// Our planning timeout expired during probe/coarse planning:
			// fall back rather than failing the video.
			return a.fallbackToBaseline(planCtx, parentCtx, v, segments, duration, w, h, stats, "planning timeout during probe/coarse planning")
		}
		// Baseline planner itself failed (e.g. invalid duration): genuine
		// error, nothing safer to fall back to.
		return nil, stats, fmt.Errorf("coarse planning failed: %w", err)
	}
	stats.NoiseFloor = pres.NoiseFloor
	if pres.Probe != nil {
		stats.ProbePoints = len(pres.Probe.Points)
	}

	// 2. Planner-level fallback (probe/ffmpeg/planning failure inside PlanVideo).
	if pres.Plan.Mode == sampler.ModeBaselineFallback {
		return a.finishWithPlan(planCtx, parentCtx, v, segments, duration, w, h, pres.Plan, pres.FallbackReason, stats)
	}

	// 3. Global-busy short-circuit: uniform allocation, no refinement spend.
	busy := sampler.IsGloballyBusy(pres.Probe, a.cfg.Busy)
	stats.BusyFraction = busy.BusyFraction
	stats.BusyPoints = busy.BusyPoints
	if busy.Busy {
		uplan, err := a.baselinePlan(v.ID, duration)
		if err != nil {
			return nil, stats, err
		}
		uplan.Mode = sampler.ModeUniformFallback
		if verr := sampler.ValidatePlan(uplan); verr != nil {
			return nil, stats, fmt.Errorf("uniform plan invalid: %w", verr)
		}
		if a.cfg.Shadow {
			return a.finishShadow(planCtx, parentCtx, v, segments, duration, w, h, uplan, busy, stats, sampler.ModeUniformFallback)
		}
		// Parent context: fallback-adjacent extraction must not inherit an
		// expired planning deadline.
		extracted, err := a.frames.GenerateForVideoWithPlan(parentCtx, v, segments, uplan, w, h)
		if err != nil {
			if cancelled() {
				return nil, stats, parentCtx.Err()
			}
			return a.fallbackToBaseline(planCtx, parentCtx, v, segments, duration, w, h, stats, fmt.Sprintf("uniform extraction failed: %v", err))
		}
		a.fillFinalStats(stats, uplan, len(extracted), 0, 0)
		stats.Mode = sampler.ModeUniformFallback
		a.logDecision(v, stats)
		return &uplan, stats, nil
	}

	// 4. Shadow mode: record what adaptive planning would do, keep baseline
	// authoritative. No refinement, no planning YOLO beyond the probe.
	if a.cfg.Shadow {
		return a.finishShadow(planCtx, parentCtx, v, segments, duration, w, h, pres.Plan, busy, stats, sampler.ModeAdaptive)
	}

	// 5. Adaptive + refinement.
	coarse := pres.Plan
	if err := sampler.ValidatePlan(coarse); err != nil {
		return a.fallbackToBaseline(planCtx, parentCtx, v, segments, duration, w, h, stats, fmt.Sprintf("coarse plan invalid: %v", err))
	}
	if cancelled() {
		return nil, stats, parentCtx.Err()
	}
	coarseFrames, err := a.frames.GenerateForVideoWithPlan(planCtx, v, segments, coarse, w, h)
	if err != nil {
		if cancelled() {
			return nil, stats, parentCtx.Err()
		}
		return a.fallbackToBaseline(planCtx, parentCtx, v, segments, duration, w, h, stats, fmt.Sprintf("coarse extraction failed: %v", err))
	}
	stats.CoarseFrames = len(coarseFrames)
	if processed := len(coarseFrames); processed > budget.Cap {
		return a.fallbackToBaseline(planCtx, parentCtx, v, segments, duration, w, h, stats, fmt.Sprintf("coarse frames %d exceed cap %d", processed, budget.Cap))
	}

	// 6. Ephemeral planning YOLO on coarse frames only.
	yoloStart := time.Now()
	coarseDets, err := a.visual.DetectFrames(planCtx, v.ID, coarseFrames)
	stats.PlanningYoloMs = time.Since(yoloStart).Milliseconds()
	if err != nil {
		if cancelled() {
			return nil, stats, parentCtx.Err()
		}
		return a.fallbackToBaseline(planCtx, parentCtx, v, segments, duration, w, h, stats, fmt.Sprintf("planning detection failed: %v", err))
	}
	in, err := refinementInput(coarse, coarseFrames, coarseDets)
	if err != nil {
		return a.fallbackToBaseline(planCtx, parentCtx, v, segments, duration, w, h, stats, fmt.Sprintf("refinement input invalid: %v", err))
	}

	// 7. Best-first refinement; DetectFunc spends budget on midpoints only.
	spent := len(coarseFrames)
	refineStart := time.Now()
	res, err := a.refiner.Refine(planCtx, in, func(rctx context.Context, midpoints []float64) ([]sampler.MidpointDetections, error) {
		if err := rctx.Err(); err != nil {
			return nil, err
		}
		if spent+len(midpoints) > budget.Cap {
			return nil, fmt.Errorf("midpoint batch %d would exceed cap %d (spent %d)", len(midpoints), budget.Cap, spent)
		}
		newFrames, err := a.frames.ExtractAdditional(rctx, v, segments, midpoints, duration, w, h)
		if err != nil {
			return nil, err
		}
		md, err := a.visual.DetectFrames(rctx, v.ID, newFrames)
		if err != nil {
			return nil, err
		}
		spent += len(newFrames)
		if spent > budget.Cap {
			return nil, fmt.Errorf("YOLO budget exceeded: %d > %d", spent, budget.Cap)
		}
		byID := make(map[uuid.UUID][]tracker.DetectionInput, len(newFrames))
		for _, d := range md {
			byID[d.FrameID] = append(byID[d.FrameID], d)
		}
		out := make([]sampler.MidpointDetections, 0, len(newFrames))
		for _, f := range newFrames {
			out = append(out, sampler.MidpointDetections{Timestamp: f.TimestampSeconds, Detections: byID[f.ID]})
		}
		return out, nil
	})
	stats.RefineMs = time.Since(refineStart).Milliseconds()
	if err != nil {
		if cancelled() {
			return nil, stats, parentCtx.Err()
		}
		if planCtx.Err() == context.DeadlineExceeded {
			return a.fallbackToBaseline(planCtx, parentCtx, v, segments, duration, w, h, stats, "planning timeout during refinement")
		}
		return a.fallbackToBaseline(planCtx, parentCtx, v, segments, duration, w, h, stats, fmt.Sprintf("refinement failed: %v", err))
	}
	final := res.Plan
	if err := sampler.ValidatePlan(final); err != nil {
		return a.fallbackToBaseline(planCtx, parentCtx, v, segments, duration, w, h, stats, fmt.Sprintf("final plan invalid: %v", err))
	}
	if len(final.Timestamps()) > budget.Cap {
		return a.fallbackToBaseline(planCtx, parentCtx, v, segments, duration, w, h, stats, fmt.Sprintf("final frames %d exceed cap %d", len(final.Timestamps()), budget.Cap))
	}

	a.fillFinalStats(stats, final, len(coarseFrames), res.FramesAdded, res.Rounds)
	stats.Mode = sampler.ModeAdaptive
	stats.MaxPriority = res.MaxPriority
	stats.MeanPriority = meanPriority(res.History)
	a.logDecision(v, stats)
	return &final, stats, nil
}

// finishWithPlan extracts a planner-supplied fallback plan (already
// validated) and records the outcome.
func (a *AdaptiveSampler) finishWithPlan(planCtx, parentCtx context.Context, v *video.Video, segments []video_segment.VideoSegment, duration float64, w, h int, plan sampler.SamplingPlan, reason string, stats *AdaptiveStats) (*sampler.SamplingPlan, *AdaptiveStats, error) {
	if err := sampler.ValidatePlan(plan); err != nil {
		return nil, stats, fmt.Errorf("fallback plan invalid: %w", err)
	}
	// Parent context: the rescue extraction must survive an expired
	// planning deadline.
	extracted, err := a.frames.GenerateForVideoWithPlan(parentCtx, v, segments, plan, w, h)
	if err != nil {
		if parentCtx.Err() != nil {
			return nil, stats, parentCtx.Err()
		}
		return nil, stats, fmt.Errorf("fallback extraction failed: %w", err)
	}
	a.fillFinalStats(stats, plan, len(extracted), 0, 0)
	stats.Mode = plan.Mode
	stats.FallbackReason = reason
	a.logDecision(v, stats)
	out := plan
	return &out, stats, nil
}

// fallbackToBaseline discards all adaptive state and re-extracts the
// baseline plan (replacing any partial frames). Cancellation still
// propagates instead of falling back.
func (a *AdaptiveSampler) fallbackToBaseline(planCtx, parentCtx context.Context, v *video.Video, segments []video_segment.VideoSegment, duration float64, w, h int, stats *AdaptiveStats, reason string) (*sampler.SamplingPlan, *AdaptiveStats, error) {
	if parentCtx.Err() != nil {
		return nil, stats, parentCtx.Err()
	}
	// Rescue extraction runs on the parent context so an expired planning
	// deadline cannot fail the fallback itself.
	return a.extractBaseline(parentCtx, v, segments, duration, w, h, stats, reason)
}

func (a *AdaptiveSampler) extractBaseline(ctx context.Context, v *video.Video, segments []video_segment.VideoSegment, duration float64, w, h int, stats *AdaptiveStats, reason string) (*sampler.SamplingPlan, *AdaptiveStats, error) {
	fb, err := a.baselinePlan(v.ID, duration)
	if err != nil {
		return nil, stats, err
	}
	if verr := sampler.ValidatePlan(fb); verr != nil {
		return nil, stats, fmt.Errorf("baseline plan invalid: %w", verr)
	}
	extracted, err := a.frames.GenerateForVideoWithPlan(ctx, v, segments, fb, w, h)
	if err != nil {
		return nil, stats, fmt.Errorf("baseline extraction failed: %w", err)
	}
	a.fillFinalStats(stats, fb, len(extracted), 0, 0)
	stats.Mode = sampler.ModeBaselineFallback
	stats.FallbackReason = reason
	a.logDecision(v, stats)
	out := fb
	return &out, stats, nil
}

// finishShadow logs what adaptive planning would have done (mode, counts,
// busy evaluation) and extracts the authoritative baseline plan instead. No
// refinement runs, and no adaptive frame is ever sent through YOLO for
// planning purposes.
func (a *AdaptiveSampler) finishShadow(planCtx, parentCtx context.Context, v *video.Video, segments []video_segment.VideoSegment, duration float64, w, h int, wouldBe sampler.SamplingPlan, busy sampler.BusyStats, stats *AdaptiveStats, wouldBeMode string) (*sampler.SamplingPlan, *AdaptiveStats, error) {
	if parentCtx.Err() != nil {
		return nil, stats, parentCtx.Err()
	}
	fb, err := a.baselinePlan(v.ID, duration)
	if err != nil {
		return nil, stats, err
	}
	if verr := sampler.ValidatePlan(fb); verr != nil {
		return nil, stats, fmt.Errorf("shadow baseline plan invalid: %w", verr)
	}
	extracted, err := a.frames.GenerateForVideoWithPlan(parentCtx, v, segments, fb, w, h)
	if err != nil {
		if parentCtx.Err() != nil {
			return nil, stats, parentCtx.Err()
		}
		return nil, stats, fmt.Errorf("shadow baseline extraction failed: %w", err)
	}
	stats.Mode = sampler.ModeBaseline
	stats.BusyFraction = busy.BusyFraction
	stats.BusyPoints = busy.BusyPoints
	stats.ShadowAdaptiveFrames = len(wouldBe.Timestamps())
	a.fillFinalStats(stats, fb, len(extracted), 0, 0)
	slog.Info("sampler: shadow",
		"video_id", v.ID.String(),
		"sampler_version", sampler.SamplerVersion,
		"duration_seconds", duration,
		"baseline_frame_count", stats.BaselineFrames,
		"adaptive_frame_count", stats.ShadowAdaptiveFrames,
		"estimated_savings", stats.BaselineFrames-stats.ShadowAdaptiveFrames,
		"sampler_mode", stats.Mode,
		"shadow_adaptive_mode", wouldBeMode,
		"planning_duration_ms", stats.ProbeMs,
		"coarse_frame_count", stats.ShadowAdaptiveFrames,
		"refinement_frame_count", 0,
		"refinement_rounds", 0,
		"busy_fraction", stats.BusyFraction,
		"probe_points", stats.ProbePoints,
		"fallback_reason", "",
	)
	out := fb
	return &out, stats, nil
}

func (a *AdaptiveSampler) fillFinalStats(stats *AdaptiveStats, plan sampler.SamplingPlan, coarse, added, rounds int) {
	stats.CoarseFrames = coarse
	stats.RefinementFrames = added
	stats.FinalFrames = len(plan.Timestamps())
	stats.EstimatedSavings = stats.BaselineFrames - stats.FinalFrames
	if stats.EstimatedSavings < 0 {
		stats.EstimatedSavings = 0
	}
	stats.Rounds = rounds
	for _, e := range plan.Entries {
		switch e.Reason {
		case sampler.ReasonCoarse:
			stats.ReasonCoarse++
		case sampler.ReasonHeartbeat:
			stats.ReasonHeartbeat++
		case sampler.ReasonRefinement:
			stats.ReasonRefinement++
		}
	}
}

func (a *AdaptiveSampler) logDecision(v *video.Video, stats *AdaptiveStats) {
	slog.Info("sampler: adaptive decision",
		"video_id", v.ID.String(),
		"sampler_mode", stats.Mode,
		"sampler_version", sampler.SamplerVersion,
		"duration_seconds", stats.DurationSeconds,
		"baseline_frame_count", stats.BaselineFrames,
		"budget_cap", stats.BudgetCap,
		"coarse_frame_count", stats.CoarseFrames,
		"refinement_frame_count", stats.RefinementFrames,
		"final_frame_count", stats.FinalFrames,
		"estimated_savings", stats.EstimatedSavings,
		"refinement_rounds", stats.Rounds,
		"reason_coarse", stats.ReasonCoarse,
		"reason_heartbeat", stats.ReasonHeartbeat,
		"reason_refinement", stats.ReasonRefinement,
		"max_priority", stats.MaxPriority,
		"mean_priority", stats.MeanPriority,
		"busy_fraction", stats.BusyFraction,
		"busy_points", stats.BusyPoints,
		"probe_points", stats.ProbePoints,
		"noise_floor", stats.NoiseFloor,
		"fallback_reason", stats.FallbackReason,
		"probe_ms", stats.ProbeMs,
		"planning_yolo_ms", stats.PlanningYoloMs,
		"refine_ms", stats.RefineMs,
		"total_sampler_ms", stats.TotalMs,
		"shadow", stats.Shadow,
	)
}

func meanPriority(history []sampler.RoundInfo) float64 {
	var sum float64
	var n int
	for _, h := range history {
		for _, c := range h.Selected {
			sum += c.Priority
			n++
		}
	}
	if n == 0 {
		return 0
	}
	return sum / float64(n)
}

// refinementInput aligns extracted coarse frames (authoritative) with plan
// timestamps and groups ephemeral planning detections by frame.
func refinementInput(plan sampler.SamplingPlan, frames []video_frame.VideoFrame, dets []tracker.DetectionInput) (sampler.RefinementInput, error) {
	ts := plan.Timestamps()
	if len(ts) == 0 {
		return sampler.RefinementInput{}, fmt.Errorf("coarse plan is empty")
	}
	byTs := make(map[float64]video_frame.VideoFrame, len(frames))
	for _, f := range frames {
		matched := false
		for _, t := range ts {
			if absDiff(f.TimestampSeconds, t) < 1e-6 {
				byTs[t] = f
				matched = true
				break
			}
		}
		if !matched {
			return sampler.RefinementInput{}, fmt.Errorf("extracted frame %v not in plan", f.TimestampSeconds)
		}
	}
	if len(byTs) != len(ts) {
		return sampler.RefinementInput{}, fmt.Errorf("extracted %d frames for %d planned timestamps", len(byTs), len(ts))
	}
	byID := make(map[uuid.UUID][]tracker.DetectionInput)
	tframes := make([]tracker.FrameInput, 0, len(ts))
	for _, t := range ts {
		f := byTs[t]
		tframes = append(tframes, tracker.FrameInput{FrameID: f.ID, Timestamp: t})
	}
	for _, d := range dets {
		byID[d.FrameID] = append(byID[d.FrameID], d)
	}
	return sampler.RefinementInput{Plan: plan, Frames: tframes, Detections: byID}, nil
}

func absDiff(a, b float64) float64 {
	if a > b {
		return a - b
	}
	return b - a
}
