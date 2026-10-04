package codec_experiment

// TEMPORARY EXPERIMENT -- see FINDINGS.md. Whole package is disposable.
//
// Renders a codec timeline so frame types, motion signal and encoded size can be
// correlated against time. Stdlib image/png only; no plotting dependency added.

import (
	"fmt"
	"image"
	"image/color"
	"image/png"
	"math"
	"os"
	"strings"
)

const (
	vizWidth    = 1400
	vizLeft     = 168
	vizRight    = 16
	vizTop      = 52
	vizLaneH    = 30
	vizLaneGap  = 9
	vizAxisH    = 34
	vizTextScal = 2
)

var (
	colBg       = color.RGBA{0x12, 0x15, 0x1a, 0xff}
	colPanel    = color.RGBA{0x1b, 0x1f, 0x27, 0xff}
	colText     = color.RGBA{0xd6, 0xdd, 0xe6, 0xff}
	colMuted    = color.RGBA{0x77, 0x82, 0x90, 0xff}
	colFaint    = color.RGBA{0x33, 0x3a, 0x44, 0xff}
	colTypeI    = color.RGBA{0xff, 0x5f, 0x56, 0xff}
	colTypeP    = color.RGBA{0x5a, 0xc8, 0x5a, 0xff}
	colTypeB    = color.RGBA{0x4a, 0x9e, 0xff, 0xff}
	colTypeO    = color.RGBA{0x7a, 0x7f, 0x87, 0xff}
	colMeanMag  = color.RGBA{0xff, 0xb4, 0x54, 0xff}
	colMaxMag   = color.RGBA{0xff, 0x7b, 0x54, 0xff}
	colVecCount = color.RGBA{0xa7, 0x8b, 0xfa, 0xff}
	colZeroRat  = color.RGBA{0x4d, 0xd0, 0xe1, 0xff}
	colBytes    = color.RGBA{0xff, 0xd1, 0x66, 0xff}
	colMissing  = color.RGBA{0xff, 0x6b, 0x6b, 0xff}
	colDynamic  = color.RGBA{0x7c, 0xe8, 0x6b, 0xff}
	colRegion   = color.RGBA{0x2c, 0x3a, 0x24, 0xff}
	colEnter    = color.RGBA{0xff, 0x6b, 0x6b, 0xff}
	colExit     = color.RGBA{0x9a, 0xa7, 0xb5, 0xff}
)

type canvas struct {
	img *image.RGBA
	w   int
	h   int
}

func newCanvas(w, h int) *canvas {
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	c := &canvas{img: img, w: w, h: h}
	c.fill(colBg)
	return c
}

func (c *canvas) fill(col color.RGBA) {
	for y := 0; y < c.h; y++ {
		for x := 0; x < c.w; x++ {
			c.img.SetRGBA(x, y, col)
		}
	}
}

func (c *canvas) rect(x, y, w, h int, col color.RGBA) {
	if w <= 0 || h <= 0 {
		return
	}
	if x < 0 {
		w += x
		x = 0
	}
	if y < 0 {
		h += y
		y = 0
	}
	if x+w > c.w {
		w = c.w - x
	}
	if y+h > c.h {
		h = c.h - y
	}
	if w <= 0 || h <= 0 {
		return
	}
	for yy := y; yy < y+h; yy++ {
		for xx := x; xx < x+w; xx++ {
			c.img.SetRGBA(xx, yy, col)
		}
	}
}

func (c *canvas) vline(x, y0, y1 int, col color.RGBA) {
	if x < 0 || x >= c.w {
		return
	}
	if y0 < 0 {
		y0 = 0
	}
	if y1 > c.h {
		y1 = c.h
	}
	for y := y0; y < y1; y++ {
		c.img.SetRGBA(x, y, col)
	}
}

func (c *canvas) text(x, y int, s string, scale int, col color.RGBA) {
	cx := x
	for _, r := range strings.ToUpper(s) {
		cols := glyphCols[r]
		for gx := 0; gx < glyphW; gx++ {
			bits := cols[gx]
			for gy := 0; gy < glyphH; gy++ {
				if bits&(1<<uint(gy)) == 0 {
					continue
				}
				c.rect(cx+gx*scale, y+gy*scale, scale, scale, col)
			}
		}
		cx += glyphAdv * scale
	}
}

func typeColor(t string) color.RGBA {
	switch t {
	case "I":
		return colTypeI
	case "P":
		return colTypeP
	case "B":
		return colTypeB
	default:
		return colTypeO
	}
}

// writePNG draws the timeline. Any failure is non fatal: the JSON and CSV carry
// the same data.
func writePNG(path string, a *Analysis) error {
	n := len(a.Frames)
	if n == 0 {
		return fmt.Errorf("codec experiment: no frames to plot")
	}

	lanes := []lane{
		{label: "FRAME TYPE", sub: "I/P/B"},
		{label: "MOTION MEAN MAG", sub: "PX, PEAK/COL"},
		{label: "MOTION MAX MAG", sub: "PX, PEAK/COL"},
		{label: "MOTION VECTORS", sub: "COUNT, PEAK"},
		{label: "MOTION ZERO RATIO", sub: "0-1, MEAN"},
		{label: "ENCODED SIZE", sub: "BYTES, LOG"},
		{label: "RESIDUAL", sub: "NOT EXPOSED"},
	}

	height := vizTop + len(lanes)*(vizLaneH+vizLaneGap) + vizAxisH
	c := newCanvas(vizWidth, height)

	plotW := vizWidth - vizLeft - vizRight
	if plotW < 32 {
		plotW = 32
	}

	c.text(14, 12, "RECALL CODECSIGHT EXPERIMENT", vizTextScal, colText)
	c.text(14, 32, fmt.Sprintf("%s  CODEC %s  %s  %d FRAMES  %.2FS",
		a.VideoName, orDash(a.Codec), trimFloat(a.FPS)+" FPS", a.FrameCount, a.Duration), 1, colMuted)

	for i := range lanes {
		y := vizTop + i*(vizLaneH+vizLaneGap)
		lanes[i].y = y
		lanes[i].h = vizLaneH
		lanes[i].plotW = plotW
	}

	// Frame type lane: colour per picture, letters when cells are wide enough.
	lt := &lanes[0]
	c.rect(vizLeft, lt.y, plotW, lt.h, colPanel)
	drawTypeLane(c, a, vizLeft, lt.y, plotW, lt.h)
	c.text(14, lt.y+2, lt.label, 1, colText)
	c.text(14, lt.y+11, lt.sub, 1, colMuted)
	if plotW/n >= 8 {
		for i := range a.Frames {
			cellW := plotW / n
			if cellW < 8 {
				break
			}
			col := i * cellW
			if col+cellW > plotW {
				cellW = plotW - col
			}
			c.text(col+(cellW-textWidth(a.Frames[i].Type, vizTextScal))/2, lt.y+(lt.h-glyphH*vizTextScal)/2,
				a.Frames[i].Type, vizTextScal, color.RGBA{0x0a, 0x0c, 0x10, 0xff})
		}
	}

	barLanes := []struct {
		lane  int
		color color.RGBA
		value func(f *Frame) float64
		log   bool
	}{
		{lane: 1, color: colMeanMag, value: func(f *Frame) float64 { return f.Motion.MeanMagnitude }},
		{lane: 2, color: colMaxMag, value: func(f *Frame) float64 { return f.Motion.MaxMagnitude }},
		{lane: 3, color: colVecCount, value: func(f *Frame) float64 { return float64(f.Motion.Count) }},
		{lane: 4, color: colZeroRat, value: func(f *Frame) float64 { return f.Motion.ZeroRatio }},
		{lane: 5, color: colBytes, value: func(f *Frame) float64 { return float64(f.PacketBytes) }, log: true},
	}

	for _, bl := range barLanes {
		ln := &lanes[bl.lane]
		c.rect(vizLeft, ln.y, plotW, ln.h, colPanel)
		maxV := 0.0
		for i := range a.Frames {
			if v := bl.value(&a.Frames[i]); v > maxV {
				maxV = v
			}
		}
		if bl.lane == 4 {
			maxV = 1.0
		}
		plotBars(c, a, vizLeft, ln.y, plotW, ln.h, bl.color, bl.value, maxV, bl.log)
		c.text(14, ln.y+2, ln.label, 1, colText)
		c.text(14, ln.y+11, ln.sub, 1, colMuted)
		c.text(14, ln.y+20, "MAX "+trimFloat(maxV), 1, colMuted)
	}

	lr := &lanes[6]
	c.rect(vizLeft, lr.y, plotW, lr.h, colPanel)
	c.text(vizLeft+10, lr.y+(lr.h-glyphH*vizTextScal)/2,
		"RESIDUAL INFORMATION NOT EXPOSED BY LIBAVCODEC OR THE FFMPEG CLI", vizTextScal, colMissing)
	c.text(14, lr.y+2, lr.label, 1, colText)
	c.text(14, lr.y+11, lr.sub, 1, colMissing)

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

type lane struct {
	label string
	sub   string
	y     int
	h     int
	plotW int
}

// drawTypeLane collapses each pixel column to the most significant picture type
// present in it, so keyframes stay visible when many pictures share a column.
// Columns beyond the last picture are left as background rather than being
// filled from a fallback range.
func drawTypeLane(c *canvas, a *Analysis, x, y, w, h int) {
	if w <= 0 || len(a.Frames) == 0 {
		return
	}
	n := len(a.Frames)
	per := float64(n) / float64(w)
	rank := map[string]int{"B": 1, "P": 2, "I": 3}
	for col := 0; col < w; col++ {
		lo := int(float64(col) * per)
		if lo >= n {
			break
		}
		hi := int(float64(col+1) * per)
		if hi > n {
			hi = n
		}
		if hi <= lo {
			hi = lo + 1
		}
		best := ""
		bestRank := 0
		for i := lo; i < hi; i++ {
			tp := a.Frames[i].Type
			if r := rank[tp]; r > bestRank {
				best, bestRank = tp, r
			}
		}
		if bestRank == 0 {
			best = a.Frames[lo].Type
		}
		c.rect(x+col, y, 1, h, typeColor(best))
	}
}

// plotBars downsamples by taking the peak per pixel column for magnitude and
// count lanes, so short spikes stay visible. zero_ratio uses the mean because
// its meaning is an aggregate.
func plotBars(c *canvas, a *Analysis, x, y, w, h int, col color.RGBA, value func(*Frame) float64, maxV float64, useLog bool) {
	if w <= 0 || len(a.Frames) == 0 || maxV <= 0 {
		return
	}
	cols := make([]float64, w)
	if useLog {
		den := math.Log1p(maxV)
		if den <= 0 {
			den = 1
		}
		for i := range a.Frames {
			colIdx := i * w / len(a.Frames)
			if colIdx >= w {
				colIdx = w - 1
			}
			v := math.Log1p(value(&a.Frames[i])) / den
			if v > cols[colIdx] {
				cols[colIdx] = v
			}
		}
	} else {
		for i := range a.Frames {
			colIdx := i * w / len(a.Frames)
			if colIdx >= w {
				colIdx = w - 1
			}
			v := value(&a.Frames[i]) / maxV
			if v > cols[colIdx] {
				cols[colIdx] = v
			}
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
		c.rect(x+i, y+h-bh, 1, bh, col)
	}
}

func drawTimeAxis(c *canvas, a *Analysis, x, y, w, h int) {
	c.vline(x, y, y+h-12, colFaint)
	c.vline(x+w-1, y, y+h-12, colFaint)

	n := len(a.Frames)
	if n == 0 || w <= 0 {
		return
	}
	ticks := 10
	if w < 400 {
		ticks = 5
	}
	for i := 0; i <= ticks; i++ {
		px := x + i*w/ticks
		if px >= x+w {
			px = x + w - 1
		}
		c.vline(px, y, y+5, colFaint)
		frameIdx := i * n / ticks
		if frameIdx >= n {
			frameIdx = n - 1
		}
		lbl := trimFloat(a.Frames[frameIdx].Timestamp) + "S"
		lx := px - textWidth(lbl, 1)/2
		if lx < x {
			lx = x
		}
		if lx+textWidth(lbl, 1) > x+w {
			lx = x + w - textWidth(lbl, 1)
		}
		c.text(lx, y+10, lbl, 1, colMuted)
	}
	c.text(vizLeft, y+h-9, fmt.Sprintf("TIME 0-%s  FRAME 0-%d", trimFloat(a.Duration), n-1), 1, colFaint)
}
