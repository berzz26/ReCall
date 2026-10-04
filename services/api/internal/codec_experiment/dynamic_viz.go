package codec_experiment

// TEMPORARY EXPERIMENT -- see FINDINGS.md. Whole package is disposable.

import (
	"fmt"
	"image/color"
	"image/png"
	"math"
	"os"
)

// writeDynamicPNG extends the codec timeline with the sampling decision: codec
// activity, the baseline 2-second grid, the dynamically selected timestamps,
// and the merged dense regions.
func writeDynamicPNG(path string, a *Analysis, plan *DynamicPlan) error {
	if len(a.Frames) == 0 {
		return fmt.Errorf("codec experiment: no frames to plot")
	}
	if plan == nil {
		return fmt.Errorf("codec experiment: nil dynamic plan")
	}
	duration := plan.Summary.DurationSeconds
	if duration <= 0 {
		duration = a.Duration
	}
	if duration <= 0 {
		duration = a.Frames[len(a.Frames)-1].Timestamp
	}
	if duration <= 0 {
		return fmt.Errorf("codec experiment: duration unavailable; cannot plot sampling plan")
	}

	lanes := []lane{
		{label: "ACTIVITY SCORE", sub: "0-1 ENTER/EXIT"},
		{label: "BASELINE SAMPLES", sub: fmt.Sprintf("%d SAMPLES", len(plan.BaselineTimestamps))},
		{label: "DYNAMIC SAMPLES", sub: fmt.Sprintf("%d SAMPLES", len(plan.SelectedTimestamps))},
		{label: "DYNAMIC REGIONS", sub: fmt.Sprintf("%d REGIONS", len(plan.Regions))},
	}
	height := vizTop + len(lanes)*(vizLaneH+vizLaneGap) + vizAxisH
	c := newCanvas(vizWidth, height)
	plotW := vizWidth - vizLeft - vizRight
	if plotW < 32 {
		plotW = 32
	}
	for i := range lanes {
		lanes[i].y = vizTop + i*(vizLaneH+vizLaneGap)
		lanes[i].h = vizLaneH
		lanes[i].plotW = plotW
	}

	c.text(14, 12, "RECALL DYNAMIC CODECSIGHT EXPERIMENT", vizTextScal, colText)
	c.text(14, 32, fmt.Sprintf("%s  MODE %s  BASELINE %d  DYNAMIC %d  ADDED %d",
		a.VideoName, plan.Summary.Mode, plan.Summary.BaselineCount, plan.Summary.DynamicCount, plan.Summary.AddedCount), 1, colMuted)

	activity := &lanes[0]
	c.rect(vizLeft, activity.y, plotW, activity.h, colPanel)
	drawActivityLane(c, a, plan, vizLeft, activity.y, plotW, activity.h, duration)
	c.text(14, activity.y+2, activity.label, 1, colText)
	c.text(14, activity.y+11, activity.sub, 1, colMuted)

	baseline := &lanes[1]
	c.rect(vizLeft, baseline.y, plotW, baseline.h, colPanel)
	drawTimestampTicks(c, plan.BaselineTimestamps, duration, vizLeft, baseline.y, plotW, baseline.h, colText, 2)
	c.text(14, baseline.y+2, baseline.label, 1, colText)
	c.text(14, baseline.y+11, baseline.sub, 1, colMuted)

	dynamic := &lanes[2]
	c.rect(vizLeft, dynamic.y, plotW, dynamic.h, colPanel)
	for _, region := range plan.Regions {
		x0 := vizLeft + xForTime(region.Start, duration, plotW)
		x1 := vizLeft + xForTime(region.End, duration, plotW)
		w := x1 - x0 + 1
		if w < 1 {
			w = 1
		}
		c.rect(x0, dynamic.y, w, dynamic.h, colRegion)
	}
	drawTimestampTicks(c, plan.SelectedTimestamps, duration, vizLeft, dynamic.y, plotW, dynamic.h, colDynamic, 1)
	c.text(14, dynamic.y+2, dynamic.label, 1, colText)
	c.text(14, dynamic.y+11, dynamic.sub, 1, colMuted)

	regions := &lanes[3]
	c.rect(vizLeft, regions.y, plotW, regions.h, colPanel)
	for i, region := range plan.Regions {
		x0 := vizLeft + xForTime(region.Start, duration, plotW)
		x1 := vizLeft + xForTime(region.End, duration, plotW)
		w := x1 - x0 + 1
		if w < 1 {
			w = 1
		}
		c.rect(x0, regions.y, w, regions.h, colRegion)
		peakX := vizLeft + xForTime(region.Peak, duration, plotW)
		c.rect(peakX-1, regions.y, 3, regions.h, colEnter)
		if w > 40 {
			c.text(x0+3, regions.y+regions.h/2-4, fmt.Sprintf("%d", i+1), 1, colText)
		}
	}
	c.text(14, regions.y+2, regions.label, 1, colText)
	c.text(14, regions.y+11, regions.sub, 1, colMuted)

	drawTimeAxis(c, a, vizLeft, vizTop+len(lanes)*(vizLaneH+vizLaneGap), plotW, vizAxisH)

	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("codec experiment: %w", err)
	}
	defer f.Close()
	if err := png.Encode(f, c.img); err != nil {
		return fmt.Errorf("codec experiment: %w", err)
	}
	return nil
}

func xForTime(ts, duration float64, plotW int) int {
	if duration <= 0 || plotW <= 1 {
		return 0
	}
	x := int(ts/duration*float64(plotW-1) + 0.5)
	if x < 0 {
		return 0
	}
	if x >= plotW {
		return plotW - 1
	}
	return x
}

func drawActivityLane(c *canvas, a *Analysis, plan *DynamicPlan, x, y, w, h int, duration float64) {
	if w <= 0 || h <= 0 || len(a.Frames) == 0 {
		return
	}
	n := len(a.Frames)
	scores := plan.Scores
	if len(scores) != n {
		return
	}
	maxScore := 0.0
	for _, score := range scores {
		if score > maxScore {
			maxScore = score
		}
	}
	displayMax := math.Max(1.0, math.Max(maxScore, plan.Config.EnterThreshold))
	if displayMax <= 0 {
		displayMax = 1.0
	}
	cols := make([]float64, w)
	for i := 0; i < n; i++ {
		colIdx := xForTime(a.Frames[i].Timestamp, duration, w)
		v := scores[i] / displayMax
		if v > cols[colIdx] {
			cols[colIdx] = v
		}
	}
	for i, v := range cols {
		if v <= 0 {
			continue
		}
		if v > 1 {
			v = 1
		}
		bh := int(v*float64(h) + 0.5)
		if bh < 1 {
			bh = 1
		}
		c.rect(x+i, y+h-bh, 1, bh, colMeanMag)
	}
	drawThresholdLine(c, plan.Config.EnterThreshold/displayMax, x, y, w, h, colEnter)
	drawThresholdLine(c, plan.Config.ExitThreshold/displayMax, x, y, w, h, colExit)
}

func drawThresholdLine(c *canvas, fraction float64, x, y, w, h int, col color.RGBA) {
	if fraction < 0 {
		fraction = 0
	}
	if fraction > 1 {
		fraction = 1
	}
	yy := y + h - int(fraction*float64(h)+0.5)
	if yy < y {
		yy = y
	}
	if yy >= y+h {
		yy = y + h - 1
	}
	c.rect(x, yy, w, 1, col)
}

func drawTimestampTicks(c *canvas, timestamps []float64, duration float64, x, y, w, h int, col color.RGBA, width int) {
	if w <= 0 || h <= 0 || width <= 0 {
		return
	}
	for _, ts := range timestamps {
		xi := x + xForTime(ts, duration, w)
		c.rect(xi-width/2, y, width, h, col)
	}
}
