package sampler

// Unit tests for priority-based adaptive refinement (Phase 4, Tests 1-15).

import (
	"context"
	"math"
	"reflect"
	"testing"
	"time"

	"github.com/google/uuid"

	"github.com/berzz26/recall/services/api/internal/tracker"
)

// refineFixture builds RefinementInput from per-frame detection lists with
// an explicit budget cap and probe report.
func refineFixture(timestamps []float64, perFrame [][]tracker.DetectionInput, probe *ProbeReport, duration float64, capN int) RefinementInput {
	frames := make([]tracker.FrameInput, 0, len(timestamps))
	byFrame := make(map[uuid.UUID][]tracker.DetectionInput)
	var entries []PlannedTimestamp
	for i, ts := range timestamps {
		fid := uuid.New()
		frames = append(frames, tracker.FrameInput{FrameID: fid, Timestamp: ts})
		var dets []tracker.DetectionInput
		if i < len(perFrame) {
			for _, d := range perFrame[i] {
				d.FrameID = fid
				d.Timestamp = ts
				dets = append(dets, d)
			}
		}
		if len(dets) > 0 {
			byFrame[fid] = dets
		}
		reason := ReasonCoarse
		if i > 0 && i == len(timestamps)-1 {
			reason = ReasonHeartbeat
		}
		entries = append(entries, PlannedTimestamp{TimestampSeconds: ts, Reason: reason})
	}
	budget := Budget{
		BaselineInterval: 2 * time.Second,
		Beta:             1.0,
		BaselineCount:    capN,
		Cap:              capN,
	}
	plan := SamplingPlan{
		VideoID:         uuid.New(),
		DurationSeconds: duration,
		Mode:            ModeAdaptive,
		Budget:          budget,
		Entries:         entries,
	}
	return RefinementInput{Plan: plan, Probe: probe, Frames: frames, Detections: byFrame}
}

// detectStub scripts midpoint detections and records every YOLO invocation.
type detectStub struct {
	calls [][]float64
	fn    func(ts float64) []tracker.DetectionInput
}

func (s *detectStub) Detect(_ context.Context, timestamps []float64) ([]MidpointDetections, error) {
	cp := append([]float64(nil), timestamps...)
	s.calls = append(s.calls, cp)
	var out []MidpointDetections
	for _, ts := range timestamps {
		out = append(out, MidpointDetections{Timestamp: ts, Detections: s.fn(ts)})
	}
	return out, nil
}

func (s *detectStub) total() int {
	n := 0
	for _, c := range s.calls {
		n += len(c)
	}
	return n
}

func emptyDets(_ float64) []tracker.DetectionInput { return nil }

func boxAt(x float64) func(float64) []tracker.DetectionInput {
	return func(_ float64) []tracker.DetectionInput {
		return []tracker.DetectionInput{box("person", 0.9, x)}
	}
}

func finalTimestamps(plan SamplingPlan) []float64 { return plan.Timestamps() }

func reasonAt(plan SamplingPlan, ts float64) Reason {
	for _, e := range plan.Entries {
		if math.Abs(e.TimestampSeconds-ts) < dedupeEpsilon {
			return e.Reason
		}
	}
	return ""
}

// Test 1 — one high-priority interval selects midpoint 2 from 0 -> 4.
func TestRefineHighPriorityMidpoint(t *testing.T) {
	in := refineFixture(
		[]float64{0, 4},
		[][]tracker.DetectionInput{{}, {box("person", 0.9, 0.3)}},
		nil, 8, 4,
	)
	stub := &detectStub{fn: emptyDets}
	res, err := NewRefiner(RefinementConfig{MaxRounds: 1}).Refine(context.Background(), in, stub.Detect)
	if err != nil {
		t.Fatalf("Refine: %v", err)
	}
	got := finalTimestamps(res.Plan)
	want := []float64{0, 2, 4}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("want %v, got %v", want, got)
	}
	if reasonAt(res.Plan, 2) != ReasonRefinement {
		t.Fatalf("midpoint reason must be refinement")
	}
	// Original reasons preserved.
	if reasonAt(res.Plan, 0) != ReasonCoarse || reasonAt(res.Plan, 4) != ReasonHeartbeat {
		t.Fatalf("coarse/heartbeat reasons destroyed")
	}
	if len(stub.calls) != 1 || !reflect.DeepEqual(stub.calls[0], []float64{2}) {
		t.Fatalf("YOLO must run exactly once on [2], got %v", stub.calls)
	}
}

// Test 2 — priority <= epsilon selects nothing.
func TestRefineLowPriorityNoSelection(t *testing.T) {
	in := refineFixture(
		[]float64{0, 4, 8},
		[][]tracker.DetectionInput{{}, {}, {}},
		&ProbeReport{Points: []ProbePoint{
			{TimestampSeconds: 1, ChangedFraction: 0},
			{TimestampSeconds: 5, ChangedFraction: 0},
		}},
		12, 6,
	)
	res, err := NewRefiner(DefaultRefinementConfig()).Refine(context.Background(), in, nil)
	if err != nil {
		t.Fatalf("Refine: %v", err)
	}
	if res.Rounds != 0 || res.FramesAdded != 0 {
		t.Fatalf("nothing eligible: rounds=%d added=%d", res.Rounds, res.FramesAdded)
	}
	if got := finalTimestamps(res.Plan); !reflect.DeepEqual(got, []float64{0, 4, 8}) {
		t.Fatalf("plan must be unchanged, got %v", got)
	}
}

// Test 3 — gap <= MIN_GAP is never refined, even with high disagreement.
func TestRefineMinGapBlocks(t *testing.T) {
	in := refineFixture(
		[]float64{8, 8.4},
		[][]tracker.DetectionInput{{}, {box("person", 0.9, 0.3)}},
		nil, 12, 6,
	)
	res, err := NewRefiner(DefaultRefinementConfig()).Refine(context.Background(), in, nil)
	if err != nil {
		t.Fatalf("Refine: %v", err)
	}
	if res.FramesAdded != 0 {
		t.Fatalf("sub-gap interval refined: %+v", res.Plan.Timestamps())
	}
}

// Test 4 — best-first order B(5) > C(3) > A(1), verified at candidate level.
func TestCandidateBestFirstOrder(t *testing.T) {
	cfg := DefaultRefinementConfig()
	cfg.MinGapSeconds = 0.1
	cfg.Epsilon = 0
	scores := []IntervalScore{
		{StartTimestamp: 0, EndTimestamp: 4, Disagreement: 0.25},  // pri 1.0
		{StartTimestamp: 4, EndTimestamp: 6.5, Disagreement: 2.0}, // pri 5.0
		{StartTimestamp: 8, EndTimestamp: 11, Disagreement: 1.0},  // pri 3.0
	}
	cands := BuildCandidates(scores, []float64{0, 0, 0}, cfg)
	if len(cands) != 3 {
		t.Fatalf("want 3 candidates, got %d", len(cands))
	}
	if cands[0].StartTimestamp != 4 || cands[1].StartTimestamp != 8 || cands[2].StartTimestamp != 0 {
		t.Fatalf("wrong order: %+v", cands)
	}
	if cands[0].Midpoint != 5.25 || cands[1].Midpoint != 9.5 || cands[2].Midpoint != 2.0 {
		t.Fatalf("wrong midpoints: %+v", cands)
	}
}

// Test 5 — identical priorities break ties by earlier start timestamp.
func TestCandidateDeterministicTies(t *testing.T) {
	cfg := DefaultRefinementConfig()
	cfg.Epsilon = 0
	scores := []IntervalScore{
		{StartTimestamp: 4, EndTimestamp: 8, Disagreement: 1.0},
		{StartTimestamp: 0, EndTimestamp: 4, Disagreement: 1.0},
	}
	cands := BuildCandidates(scores, []float64{0, 0}, cfg)
	if len(cands) != 2 || cands[0].StartTimestamp != 0 || cands[1].StartTimestamp != 4 {
		t.Fatalf("ties must break by start: %+v", cands)
	}
	// End timestamp as final tie-break: same start is impossible for
	// adjacent intervals, but identical (start,end) pairs stay stable.
	dup := append([]RefinementCandidate(nil), cands...)
	SortCandidates(dup)
	if !reflect.DeepEqual(cands, dup) {
		t.Fatalf("sort not stable on ties")
	}
}

// Test 6 — remaining budget 2 with 5 eligible candidates selects the top 2.
func TestRefineBudgetLimitedSelection(t *testing.T) {
	ts := []float64{0, 2, 4, 6, 8, 10}
	var perFrame [][]tracker.DetectionInput
	for i := range ts {
		// Every frame a far-apart box: each interval births+deaths equally.
		perFrame = append(perFrame, []tracker.DetectionInput{box("person", 0.9, 0.05+0.3*float64(i%3))})
	}
	in := refineFixture(ts, perFrame, nil, 16, 8) // 6 initial, cap 8 -> remaining 2
	stub := &detectStub{fn: emptyDets}
	res, err := NewRefiner(RefinementConfig{MaxRounds: 1}).Refine(context.Background(), in, stub.Detect)
	if err != nil {
		t.Fatalf("Refine: %v", err)
	}
	if res.FramesAdded != 2 {
		t.Fatalf("want exactly 2 added, got %d", res.FramesAdded)
	}
	if len(stub.calls) != 1 || !reflect.DeepEqual(stub.calls[0], []float64{1, 3}) {
		t.Fatalf("want top-2 midpoints [1 3], got %v", stub.calls)
	}
	if len(finalTimestamps(res.Plan)) != 8 {
		t.Fatalf("final plan must have 8 timestamps")
	}
}

// Test 7 — fewer eligible candidates than budget adds only those.
func TestRefineFewerCandidatesThanBudget(t *testing.T) {
	ts := []float64{0, 4, 8, 12}
	var perFrame [][]tracker.DetectionInput
	for i := range ts {
		perFrame = append(perFrame, []tracker.DetectionInput{box("person", 0.9, 0.05+0.3*float64(i%3))})
	}
	in := refineFixture(ts, perFrame, nil, 28, 14) // 4 initial, cap 14 -> remaining 10
	stub := &detectStub{fn: emptyDets}
	res, err := NewRefiner(RefinementConfig{MaxRounds: 1}).Refine(context.Background(), in, stub.Detect)
	if err != nil {
		t.Fatalf("Refine: %v", err)
	}
	if res.FramesAdded != 3 {
		t.Fatalf("want exactly 3 added, got %d", res.FramesAdded)
	}
	if res.RemainingBudget != 14-7 {
		t.Fatalf("remaining must be 7, got %d", res.RemainingBudget)
	}
}

// Test 8 — 8 -> 12 becomes 8 -> 10 -> 12 after insertion.
func TestRefineMidpointSplit(t *testing.T) {
	in := refineFixture(
		[]float64{8, 12},
		[][]tracker.DetectionInput{{}, {box("person", 0.9, 0.3)}},
		nil, 16, 8,
	)
	stub := &detectStub{fn: emptyDets}
	res, err := NewRefiner(RefinementConfig{MaxRounds: 1}).Refine(context.Background(), in, stub.Detect)
	if err != nil {
		t.Fatalf("Refine: %v", err)
	}
	if got := finalTimestamps(res.Plan); !reflect.DeepEqual(got, []float64{8, 10, 12}) {
		t.Fatalf("want [8 10 12], got %v", got)
	}
	if MidpointOf(8, 12) != 10 {
		t.Fatalf("midpoint formula broken")
	}
}

// Test 9 — midpoints coinciding with existing timestamps are never
// re-processed: the refiner never requests an already-processed timestamp
// across rounds, so no duplicate YOLO invocation can occur.
func TestRefineNoDuplicateProcessing(t *testing.T) {
	in := refineFixture(
		[]float64{0, 4},
		[][]tracker.DetectionInput{{box("person", 0.9, 0.1)}, {box("person", 0.9, 0.7)}},
		nil, 20, 10,
	)
	stub := &detectStub{fn: boxAt(0.4)}
	res, err := NewRefiner(RefinementConfig{MaxRounds: 3}).Refine(context.Background(), in, stub.Detect)
	if err != nil {
		t.Fatalf("Refine: %v", err)
	}
	seen := map[float64]bool{0: true, 4: true}
	for _, call := range stub.calls {
		for _, ts := range call {
			if seen[ts] {
				t.Fatalf("duplicate YOLO on %v", ts)
			}
			seen[ts] = true
		}
	}
	ts := finalTimestamps(res.Plan)
	for i := 1; i < len(ts); i++ {
		if ts[i]-ts[i-1] < dedupeEpsilon {
			t.Fatalf("duplicate timestamp in final plan: %v", ts)
		}
	}
}

// Test 10 — after inserting 10, the next round scores 8->10 and 10->12.
func TestRefineRescoresSplitIntervals(t *testing.T) {
	in := refineFixture(
		[]float64{8, 12},
		[][]tracker.DetectionInput{{}, {box("person", 0.9, 0.3)}},
		nil, 20, 10,
	)
	stub := &detectStub{fn: emptyDets}
	res, err := NewRefiner(RefinementConfig{MaxRounds: 2}).Refine(context.Background(), in, stub.Detect)
	if err != nil {
		t.Fatalf("Refine: %v", err)
	}
	if len(res.History) != 2 {
		t.Fatalf("want 2 rounds, got %d", len(res.History))
	}
	var bounds [][2]float64
	for _, s := range res.History[1].Scores {
		bounds = append(bounds, [2]float64{s.StartTimestamp, s.EndTimestamp})
	}
	want := [][2]float64{{8, 10}, {10, 12}}
	if !reflect.DeepEqual(bounds, want) {
		t.Fatalf("round 2 must score split intervals, got %v", bounds)
	}
}

// Test 11 — a static sequence consumes no refinement budget.
func TestRefineStaticSequence(t *testing.T) {
	in := refineFixture(
		[]float64{0, 4, 8, 12},
		[][]tracker.DetectionInput{{}, {}, {}, {}},
		&ProbeReport{Points: []ProbePoint{
			{TimestampSeconds: 0, ChangedFraction: 0},
			{TimestampSeconds: 2, ChangedFraction: 0},
			{TimestampSeconds: 6, ChangedFraction: 0},
			{TimestampSeconds: 10, ChangedFraction: 0},
		}},
		16, 8,
	)
	res, err := NewRefiner(DefaultRefinementConfig()).Refine(context.Background(), in, nil)
	if err != nil {
		t.Fatalf("Refine: %v", err)
	}
	if res.Rounds != 0 || res.FramesAdded != 0 {
		t.Fatalf("static must not refine: rounds=%d added=%d", res.Rounds, res.FramesAdded)
	}
	if res.RemainingBudget != 8-4 {
		t.Fatalf("remaining must be 4, got %d", res.RemainingBudget)
	}
}

// Test 12 — budget invariant holds at every stage across rounds.
func TestRefineBudgetInvariant(t *testing.T) {
	ts := []float64{0, 4, 8}
	var perFrame [][]tracker.DetectionInput
	for i := range ts {
		perFrame = append(perFrame, []tracker.DetectionInput{box("person", 0.9, 0.05+0.3*float64(i))})
	}
	in := refineFixture(ts, perFrame, nil, 12, 6) // 3 initial, cap 6
	processed := len(ts)
	stub := &detectStub{fn: func(ts float64) []tracker.DetectionInput {
		_ = ts
		return nil
	}}
	res, err := NewRefiner(DefaultRefinementConfig()).Refine(context.Background(), in,
		func(ctx context.Context, want []float64) ([]MidpointDetections, error) {
			processed += len(want)
			if processed > 6 {
				t.Fatalf("budget violated mid-refinement: %d > 6", processed)
			}
			return stub.Detect(ctx, want)
		})
	if err != nil {
		t.Fatalf("Refine: %v", err)
	}
	if len(finalTimestamps(res.Plan)) != 6 {
		t.Fatalf("want final count at cap 6, got %v", finalTimestamps(res.Plan))
	}
	if res.FramesProcessed > 6 || res.RemainingBudget != 0 {
		t.Fatalf("final invariant broken: %+v", res)
	}
}

// Test 13 — refinement stops after MaxRounds with eligible work remaining.
func TestRefineRoundLimit(t *testing.T) {
	in := refineFixture(
		[]float64{0, 4},
		[][]tracker.DetectionInput{{}, {box("person", 0.9, 0.3)}},
		nil, 20, 10, // remaining 8: budget is not the limiter
	)
	stub := &detectStub{fn: emptyDets}
	res, err := NewRefiner(RefinementConfig{MaxRounds: 2}).Refine(context.Background(), in, stub.Detect)
	if err != nil {
		t.Fatalf("Refine: %v", err)
	}
	if res.Rounds != 2 {
		t.Fatalf("want 2 rounds, got %d", res.Rounds)
	}
	got := finalTimestamps(res.Plan)
	if !reflect.DeepEqual(got, []float64{0, 2, 3, 4}) {
		t.Fatalf("want [0 2 3 4], got %v", got)
	}
}

// Test 14 — remainingBudget == 0 stops all further work immediately.
func TestRefineBudgetExhaustion(t *testing.T) {
	ts := []float64{0, 4, 8, 12}
	var perFrame [][]tracker.DetectionInput
	for i := range ts {
		perFrame = append(perFrame, []tracker.DetectionInput{box("person", 0.9, 0.05+0.3*float64(i))})
	}
	in := refineFixture(ts, perFrame, nil, 16, 8) // 4 initial, remaining 4
	stub := &detectStub{fn: emptyDets}
	res, err := NewRefiner(DefaultRefinementConfig()).Refine(context.Background(), in, stub.Detect)
	if err != nil {
		t.Fatalf("Refine: %v", err)
	}
	if stub.total() != 4 {
		t.Fatalf("want exactly 4 YOLO frames, got %d in %v", stub.total(), stub.calls)
	}
	if len(stub.calls) != 2 || len(stub.calls[0]) != 3 || len(stub.calls[1]) != 1 {
		t.Fatalf("want batches [3 1], got %v", stub.calls)
	}
	if res.RemainingBudget != 0 || len(finalTimestamps(res.Plan)) != 8 {
		t.Fatalf("must end exactly at cap: %+v", res)
	}
}

// Test 15 — identical inputs built twice (fresh UUIDs) refine identically.
func TestRefineDeterministic(t *testing.T) {
	vid := uuid.New() // fixed: VideoID is input identity, not randomness
	build := func() RefinementInput {
		ts := []float64{0, 4, 8, 12}
		confs := []float64{0.91, 0.85, 0.93, 0.87}
		var perFrame [][]tracker.DetectionInput
		for i := range ts {
			perFrame = append(perFrame, []tracker.DetectionInput{box("person", confs[i], 0.05+0.3*float64(i))})
		}
		probe := &ProbeReport{Points: []ProbePoint{
			{TimestampSeconds: 2, ChangedFraction: 0.4},
			{TimestampSeconds: 6, ChangedFraction: 0.1},
			{TimestampSeconds: 10, ChangedFraction: 0.7},
		}}
		in := refineFixture(ts, perFrame, probe, 16, 8)
		in.Plan.VideoID = vid
		return in
	}
	run := func() RefinementResult {
		stub := &detectStub{fn: func(ts float64) []tracker.DetectionInput {
			return []tracker.DetectionInput{box("person", 0.9-0.001*ts, 0.5)}
		}}
		res, err := NewRefiner(DefaultRefinementConfig()).Refine(context.Background(), build(), stub.Detect)
		if err != nil {
			t.Fatalf("Refine: %v", err)
		}
		return res
	}
	a, b := run(), run()
	if !reflect.DeepEqual(a.Plan, b.Plan) || a.Rounds != b.Rounds {
		t.Fatalf("nondeterministic plans:\n%+v\n%+v", a.Plan, b.Plan)
	}
}

// Probe rank unit checks: max-in-interval aggregation + rank normalization.
func TestProbeRankAggregation(t *testing.T) {
	probe := &ProbeReport{Points: []ProbePoint{
		{TimestampSeconds: 1, ChangedFraction: 0.1},
		{TimestampSeconds: 2, ChangedFraction: 0.9},
		{TimestampSeconds: 3, ChangedFraction: 0.2},
		{TimestampSeconds: 5, ChangedFraction: 0.5},
		{TimestampSeconds: 7, ChangedFraction: 0.0},
	}}
	ranks := ProbeRankForIntervals(probe, []float64{0, 4}, []float64{4, 8})
	if !reflect.DeepEqual(ranks, []float64{1, 0}) {
		t.Fatalf("want [1 0], got %v", ranks)
	}
	// All equal -> all zero; nil probe -> zeros; single interval -> zero.
	flat := &ProbeReport{Points: []ProbePoint{
		{TimestampSeconds: 1, ChangedFraction: 0},
		{TimestampSeconds: 5, ChangedFraction: 0},
	}}
	if got := ProbeRankForIntervals(flat, []float64{0, 4}, []float64{4, 8}); !reflect.DeepEqual(got, []float64{0, 0}) {
		t.Fatalf("flat probe must rank zero, got %v", got)
	}
	if got := ProbeRankForIntervals(nil, []float64{0, 4}, []float64{4, 8}); !reflect.DeepEqual(got, []float64{0, 0}) {
		t.Fatalf("nil probe must rank zero, got %v", got)
	}
	if got := ProbeRankForIntervals(probe, []float64{0}, []float64{8}); !reflect.DeepEqual(got, []float64{0}) {
		t.Fatalf("single interval must rank zero, got %v", got)
	}
}

// Priority formula: max(disagreement, gamma*rank) * gap, exactly.
func TestPriorityFormula(t *testing.T) {
	if got := PriorityOf(0.8, 0.2, 4, 0.5); got != 0.8*4 {
		t.Fatalf("disagreement must win: %v", got)
	}
	if got := PriorityOf(0.1, 1.0, 4, 0.5); got != 0.5*4 {
		t.Fatalf("probe prior must win: %v", got)
	}
	if got := PriorityOf(0, 0, 4, 0.5); got != 0 {
		t.Fatalf("zero signal must be zero: %v", got)
	}
}
