package tracker

import (
	"context"
	"math"
	"sort"
	"strings"

	"github.com/google/uuid"
)

// Class-confusion groups: YOLO flips categories between frames for the same
// physical object (car↔truck, dog↔cat). Identity must not be broken by a
// label flip, so association matches on the canonical group instead of the
// raw label. The track's stored label is the one it was born with.
func confusionGroup(label string) string {
	l := strings.ToLower(strings.TrimSpace(label))
	switch l {
	case "car", "truck", "bus", "van", "vehicle", "suv", "minivan", "license_plate":
		return "vehicle"
	case "bicycle", "motorcycle", "bike", "scooter", "electric_scooter":
		return "two_wheeler"
	case "chair", "couch", "bench", "sofa", "potted_plant":
		return "furniture"
	case "dog", "cat", "bear", "bird", "horse", "sheep", "cow", "animal":
		return "animal"
	case "suitcase", "backpack", "handbag":
		return "luggage"
	default:
		return l
	}
}

func sameClass(a, b string) bool {
	return confusionGroup(a) == confusionGroup(b)
}

// Sparse-video association. Sampled YOLO frames can be seconds apart (see
// FRAME_SAMPLE_INTERVAL and the adaptive sampler), so hard Mahalanobis/IoU
// gates tuned for ~30 FPS video forbid otherwise obvious re-associations and
// fragment tracks. When the time between frames exceeds Sparse.DtThreshold the
// association switches to a dt-aware spatial gate instead:
//
//	center-distance( predicted-state | last-observed, detection ) <= allowedDrift(dt)
//
// allowedDrift lifts gently with dt (a floor keeps slow/stationary objects
// associated, a speed limit covers walking/crossing motion). IoU overlap is
// still honored and ranks closer matches cheaper.
const (
	sparseDtThreshold = 2.5  // seconds between frames before sparse gating kicks in
	sparseAllowFloor  = 0.04 // normalized center-distance floor (slow/stationary objects)
	sparseMaxSpeed    = 0.28 // normalized frame units per second (walk/vehicle speed headroom)
	// Cross-gap re-association: a track that was removed while unobserved gets
	// re-attached to a later track of the same class if the gap since its last
	// observation is within this limit and the first box fits its trajectory.
	reattachMaxGap = 30.0
)

// allowedDrift is the maximum normalized center distance that still counts as
// "same object" after dt seconds without observation.
func (t *ByteTrack) allowedDrift(dt float64) float64 {
	return t.Sparse.AllowFloor + t.Sparse.MaxSpeed*dt
}

// ByteTrack implements the ByteTrack two-stage association in pure Go.
// High/low split, Kalman prediction, Hungarian assignment, label-aware matching,
// Tracked/Lost/Removed lifecycle with count-based buffer and MinHits confirmation.
type ByteTrack struct {
	HighThreshold  float64
	LowThreshold   float64
	MatchThreshold float64
	TrackBuffer    int
	FuseScore      bool
	MinHits        int
	Sparse         SparseParams
}

func NewByteTrack(high, low, match float64, buffer int, fuseScore bool, minHits int, sparse SparseParams) *ByteTrack {
	if minHits < 1 {
		minHits = 2
	}
	return &ByteTrack{
		HighThreshold:  high,
		LowThreshold:   low,
		MatchThreshold: match,
		TrackBuffer:    buffer,
		FuseScore:      fuseScore,
		MinHits:        minHits,
		Sparse:         sparse.withDefaults(),
	}
}

func (t *ByteTrack) Name() string    { return "bytetrack" }
func (t *ByteTrack) Version() string { return "3" }

// TrackState represents ByteTrack track lifecycle.
type TrackState int

const (
	StateNew TrackState = iota
	StateTracked
	StateLost
	StateRemoved
)

type btTrack struct {
	Index           int
	Label           string
	State           TrackState
	Mean            [8]float64
	Cov             [8][8]float64
	LastBBox        DetectionInput // last observed bbox (for fallback if Kalman not yet)
	PredictedBBox   DetectionInput // predicted bbox for association (derived from Mean)
	Age             int
	Hits            int
	Misses          int
	StartTimestamp  float64
	LastTimestamp   float64
	LastMatchedTime float64
	HitCount        int // number of successful observations
}

// Kalman helpers

const (
	stdWeightPosition = 1.0 / 20.0
	stdWeightVelocity = 1.0 / 160.0
)

// tlwhToXyah converts normalized xywh to [cx, cy, a, h]
func tlwhToXyah(b DetectionInput) [4]float64 {
	cx := b.BBoxX + b.BBoxWidth/2
	cy := b.BBoxY + b.BBoxHeight/2
	h := b.BBoxHeight
	a := 1.0
	if h > 1e-6 {
		a = b.BBoxWidth / h
	}
	return [4]float64{cx, cy, a, h}
}

// xyahToTLWH converts Kalman state [cx,cy,a,h] to normalized xywh DetectionInput (for IoU)
func xyahToTLWH(mean [8]float64, label string) DetectionInput {
	cx, cy, a, h := mean[0], mean[1], mean[2], mean[3]
	if h < 1e-6 {
		h = 1e-6
	}
	w := a * h
	x := cx - w/2
	y := cy - h/2
	// Clamp to [0,1] for IoU stability
	if x < 0 {
		x = 0
	}
	if y < 0 {
		y = 0
	}
	if w < 0 {
		w = 0
	}
	if h < 0 {
		h = 0
	}
	if x+w > 1 {
		w = 1 - x
	}
	if y+h > 1 {
		h = 1 - y
	}
	return DetectionInput{BBoxX: x, BBoxY: y, BBoxWidth: w, BBoxHeight: h, Label: label}
}

func kalmanInitiate(z [4]float64) ([8]float64, [8][8]float64) {
	var mean [8]float64
	mean[0] = z[0]
	mean[1] = z[1]
	mean[2] = z[2]
	mean[3] = z[3]
	// velocities 0
	h := z[3]
	if h < 1e-6 {
		h = 1e-6
	}
	std := [8]float64{
		2 * stdWeightPosition * h,
		2 * stdWeightPosition * h,
		1e-2,
		2 * stdWeightPosition * h,
		10 * stdWeightVelocity * h,
		10 * stdWeightVelocity * h,
		1e-5,
		10 * stdWeightVelocity * h,
	}
	var cov [8][8]float64
	for i := 0; i < 8; i++ {
		cov[i][i] = std[i] * std[i]
	}
	return mean, cov
}

func kalmanPredict(mean [8]float64, cov [8][8]float64, dt float64) ([8]float64, [8][8]float64) {
	// F
	var F [8][8]float64
	for i := 0; i < 8; i++ {
		F[i][i] = 1
	}
	F[0][4] = dt
	F[1][5] = dt
	F[2][6] = dt
	F[3][7] = dt

	// newMean = F*mean
	var newMean [8]float64
	for i := 0; i < 8; i++ {
		sum := 0.0
		for k := 0; k < 8; k++ {
			sum += F[i][k] * mean[k]
		}
		newMean[i] = sum
	}

	// Q - process noise
	h := mean[3]
	if h < 1e-6 {
		h = 1e-6
	}
	// Use dt to scale velocity noise? simple scaling
	stdPos := stdWeightPosition * h
	stdVel := stdWeightVelocity * h
	// Q diagonal with dt factor for velocities
	var qDiag [8]float64
	qDiag[0] = stdPos * stdPos
	qDiag[1] = stdPos * stdPos
	qDiag[2] = 1e-4
	qDiag[3] = stdPos * stdPos
	qDiag[4] = stdVel*stdVel*dt*dt + 1e-4
	qDiag[5] = stdVel*stdVel*dt*dt + 1e-4
	qDiag[6] = 1e-6
	qDiag[7] = stdVel*stdVel*dt*dt + 1e-4
	var Q [8][8]float64
	for i := 0; i < 8; i++ {
		Q[i][i] = qDiag[i]
	}

	// newCov = F*cov*F^T + Q
	// temp = F*cov
	var temp [8][8]float64
	for i := 0; i < 8; i++ {
		for j := 0; j < 8; j++ {
			sum := 0.0
			for k := 0; k < 8; k++ {
				sum += F[i][k] * cov[k][j]
			}
			temp[i][j] = sum
		}
	}
	// newCov = temp * F^T
	var newCov [8][8]float64
	for i := 0; i < 8; i++ {
		for j := 0; j < 8; j++ {
			sum := 0.0
			for k := 0; k < 8; k++ {
				sum += temp[i][k] * F[j][k] // F^T[k][j] = F[j][k]
			}
			newCov[i][j] = sum + Q[i][j]
		}
	}
	return newMean, newCov
}

func kalmanUpdate(mean [8]float64, cov [8][8]float64, z [4]float64) ([8]float64, [8][8]float64) {
	// H is 4x8: first 4 cols identity
	// y = z - H*mean
	var y [4]float64
	for i := 0; i < 4; i++ {
		y[i] = z[i] - mean[i]
	}
	// S = H*P*H^T + R
	// H*P is 4x8: first 4 cols of P
	var HP [4][8]float64
	for i := 0; i < 4; i++ {
		for j := 0; j < 8; j++ {
			HP[i][j] = cov[i][j]
		}
	}
	// S = HP * H^T + R ; H^T is 8x4 with identity top
	var S [4][4]float64
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			S[i][j] = HP[i][j] // because H^T[j][j] etc = HP*H^T = HP[:,0:4]
		}
	}
	// R
	h := z[3]
	if h < 1e-6 {
		h = 1e-6
	}
	var rDiag [4]float64
	rDiag[0] = stdWeightPosition * h
	rDiag[0] = rDiag[0] * rDiag[0]
	rDiag[1] = stdWeightPosition * h
	rDiag[1] = rDiag[1] * rDiag[1]
	rDiag[2] = 1e-2
	rDiag[2] = rDiag[2] * rDiag[2]
	rDiag[3] = stdWeightPosition * h
	rDiag[3] = rDiag[3] * rDiag[3]
	for i := 0; i < 4; i++ {
		S[i][i] += rDiag[i]
	}

	// K = P*H^T * S^{-1}
	// P*H^T is 8x4: first 4 cols of P transposed? P*H^T = P[:,0:4] (8x4)
	var PHt [8][4]float64
	for i := 0; i < 8; i++ {
		for j := 0; j < 4; j++ {
			PHt[i][j] = cov[i][j]
		}
	}
	// S^{-1}
	Sinv := matInv4(S)
	// K = PHt * Sinv (8x4)
	var K [8][4]float64
	for i := 0; i < 8; i++ {
		for j := 0; j < 4; j++ {
			sum := 0.0
			for k := 0; k < 4; k++ {
				sum += PHt[i][k] * Sinv[k][j]
			}
			K[i][j] = sum
		}
	}
	// newMean = mean + K*y
	var newMean [8]float64
	for i := 0; i < 8; i++ {
		sum := mean[i]
		for j := 0; j < 4; j++ {
			sum += K[i][j] * y[j]
		}
		newMean[i] = sum
	}
	// newCov = (I - K*H) * P
	// K*H is 8x8: K (8x4) * H (4x8) = K*H where H = [I 0]
	var KH [8][8]float64
	for i := 0; i < 8; i++ {
		for j := 0; j < 8; j++ {
			sum := 0.0
			for k := 0; k < 4; k++ {
				// H[k][j] = 1 if k==j and j<4 else 0
				var hkj float64
				if k == j && j < 4 {
					hkj = 1
				}
				sum += K[i][k] * hkj
			}
			KH[i][j] = sum
		}
	}
	var IKH [8][8]float64
	for i := 0; i < 8; i++ {
		for j := 0; j < 8; j++ {
			if i == j {
				IKH[i][j] = 1 - KH[i][j]
			} else {
				IKH[i][j] = -KH[i][j]
			}
		}
	}
	var newCov [8][8]float64
	for i := 0; i < 8; i++ {
		for j := 0; j < 8; j++ {
			sum := 0.0
			for k := 0; k < 8; k++ {
				sum += IKH[i][k] * cov[k][j]
			}
			newCov[i][j] = sum
		}
	}
	return newMean, newCov
}

func matInv4(m [4][4]float64) [4][4]float64 {
	// Gauss-Jordan for 4x4
	var aug [4][8]float64
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			aug[i][j] = m[i][j]
		}
		for j := 4; j < 8; j++ {
			if j-4 == i {
				aug[i][j] = 1
			}
		}
	}
	for col := 0; col < 4; col++ {
		// pivot
		pivot := col
		maxVal := math.Abs(aug[pivot][col])
		for row := col + 1; row < 4; row++ {
			if v := math.Abs(aug[row][col]); v > maxVal {
				maxVal = v
				pivot = row
			}
		}
		if maxVal < 1e-12 {
			// singular, return identity
			var id [4][4]float64
			for i := 0; i < 4; i++ {
				id[i][i] = 1
			}
			return id
		}
		if pivot != col {
			aug[col], aug[pivot] = aug[pivot], aug[col]
		}
		piv := aug[col][col]
		for j := 0; j < 8; j++ {
			aug[col][j] /= piv
		}
		for row := 0; row < 4; row++ {
			if row == col {
				continue
			}
			factor := aug[row][col]
			if factor != 0 {
				for j := 0; j < 8; j++ {
					aug[row][j] -= factor * aug[col][j]
				}
			}
		}
	}
	var inv [4][4]float64
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			inv[i][j] = aug[i][j+4]
		}
	}
	return inv
}

// iou computes IoU for normalized xywh
func iouByteTrack(a, b DetectionInput) float64 {
	ax1, ay1 := a.BBoxX, a.BBoxY
	ax2, ay2 := a.BBoxX+a.BBoxWidth, a.BBoxY+a.BBoxHeight
	bx1, by1 := b.BBoxX, b.BBoxY
	bx2, by2 := b.BBoxX+b.BBoxWidth, b.BBoxY+b.BBoxHeight
	ix1 := math.Max(ax1, bx1)
	iy1 := math.Max(ay1, by1)
	ix2 := math.Min(ax2, bx2)
	iy2 := math.Min(ay2, by2)
	if ix2 <= ix1 || iy2 <= iy1 {
		return 0
	}
	inter := (ix2 - ix1) * (iy2 - iy1)
	areaA := a.BBoxWidth * a.BBoxHeight
	areaB := b.BBoxWidth * b.BBoxHeight
	union := areaA + areaB - inter
	if union <= 0 {
		return 0
	}
	return inter / union
}

func (t *ByteTrack) Track(ctx context.Context, frames []FrameInput, detectionsByFrame map[uuid.UUID][]DetectionInput) (map[uuid.UUID]int, error) {
	// Deterministic ordering
	sort.Slice(frames, func(i, j int) bool {
		if frames[i].Timestamp == frames[j].Timestamp {
			return frames[i].FrameID.String() < frames[j].FrameID.String()
		}
		return frames[i].Timestamp < frames[j].Timestamp
	})

	assignments := make(map[uuid.UUID]int)
	// active tracks: Tracked, Lost, New (not Removed)
	active := make(map[int]*btTrack)
	nextIndex := 0
	var prevTimestamp *float64

	// For MinHits filtering, we need to keep all assignments but later filter
	// Keep per-track hit count
	for _, fr := range frames {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// dt for Kalman prediction
		var dt float64
		if prevTimestamp != nil {
			dt = fr.Timestamp - *prevTimestamp
			if dt < 0 {
				dt = 0
			}
			// Clamp dt to avoid huge jumps (e.g., 2 sec is expected, but allow up to 10 sec)
			if dt > 10 {
				dt = 10
			}
		} else {
			dt = 0
		}
		// Predict all Tracked and Lost (do not modify LastTimestamp/LastMatchedTime; they represent last actual match)
		for _, tr := range active {
			if tr.State == StateTracked || tr.State == StateLost || tr.State == StateNew {
				if dt > 1e-6 {
					tr.Mean, tr.Cov = kalmanPredict(tr.Mean, tr.Cov, dt)
				}
				// Update predicted bbox
				pred := xyahToTLWH(tr.Mean, tr.Label)
				tr.PredictedBBox = pred
			}
		}

		dets := detectionsByFrame[fr.FrameID]

		var highDets []DetectionInput
		var lowDets []DetectionInput
		for _, d := range dets {
			if d.Confidence >= t.HighThreshold {
				highDets = append(highDets, d)
			} else if d.Confidence >= t.LowThreshold {
				lowDets = append(lowDets, d)
			}
		}
		// Deterministic: confidence desc, UUID asc tie-breaker
		sort.Slice(highDets, func(i, j int) bool {
			if highDets[i].Confidence == highDets[j].Confidence {
				return highDets[i].ID.String() < highDets[j].ID.String()
			}
			return highDets[i].Confidence > highDets[j].Confidence
		})
		sort.Slice(lowDets, func(i, j int) bool {
			if lowDets[i].Confidence == lowDets[j].Confidence {
				return lowDets[i].ID.String() < lowDets[j].ID.String()
			}
			return lowDets[i].Confidence > lowDets[j].Confidence
		})

		// Build sorted active list for deterministic ordering
		var trackedLost []*btTrack
		for _, tr := range active {
			if tr.State == StateTracked || tr.State == StateLost || tr.State == StateNew {
				trackedLost = append(trackedLost, tr)
			}
		}
		sort.Slice(trackedLost, func(i, j int) bool { return trackedLost[i].Index < trackedLost[j].Index })

		// For second stage, we need list of Tracked only (not Lost, not New) that were unmatched in first stage
		// So we will compute after first stage.

		// First stage: Tracked+Lost+New vs high
		matchedTrack := make(map[int]bool)
		matchedHigh := make(map[int]bool)

		if len(trackedLost) > 0 && len(highDets) > 0 {
			cost := buildCostMatrixByteTrack(trackedLost, highDets, t.MatchThreshold, t.FuseScore, dt, t.Sparse)
			assign := hungarianRect(cost, len(trackedLost), len(highDets))
			for i, tr := range trackedLost {
				j := assign[i]
				if j >= 0 && j < len(highDets) {
					// Check if not INF
					if cost[i][j] >= INF/2 {
						continue
					}
					d := highDets[j]
					// Update Kalman
					z := tlwhToXyah(d)
					tr.Mean, tr.Cov = kalmanUpdate(tr.Mean, tr.Cov, z)
					tr.LastBBox = d
					tr.PredictedBBox = d // after update, predicted equals observed for next association? but keep updated mean
					tr.State = StateTracked
					tr.Hits++
					tr.HitCount++
					tr.Misses = 0
					tr.LastMatchedTime = fr.Timestamp
					tr.LastTimestamp = fr.Timestamp
					assignments[d.ID] = tr.Index
					matchedTrack[tr.Index] = true
					matchedHigh[j] = true
				}
			}
		}

		// Collect unmatched Tracked (only Tracked, not Lost/New) for second stage
		var remainingTracked []*btTrack
		for _, tr := range trackedLost {
			if matchedTrack[tr.Index] {
				continue
			}
			if tr.State == StateTracked {
				remainingTracked = append(remainingTracked, tr)
			}
		}
		sort.Slice(remainingTracked, func(i, j int) bool { return remainingTracked[i].Index < remainingTracked[j].Index })

		// Second stage: remaining Tracked vs low
		matchedLow := make(map[int]bool)
		if len(remainingTracked) > 0 && len(lowDets) > 0 {
			cost := buildCostMatrixByteTrack(remainingTracked, lowDets, t.MatchThreshold, t.FuseScore, dt, t.Sparse)
			assign := hungarianRect(cost, len(remainingTracked), len(lowDets))
			for i, tr := range remainingTracked {
				j := assign[i]
				if j >= 0 && j < len(lowDets) {
					if cost[i][j] >= INF/2 {
						continue
					}
					d := lowDets[j]
					z := tlwhToXyah(d)
					tr.Mean, tr.Cov = kalmanUpdate(tr.Mean, tr.Cov, z)
					tr.LastBBox = d
					tr.PredictedBBox = d
					tr.State = StateTracked
					tr.Hits++
					tr.HitCount++
					tr.Misses = 0
					tr.LastMatchedTime = fr.Timestamp
					tr.LastTimestamp = fr.Timestamp
					assignments[d.ID] = tr.Index
					matchedTrack[tr.Index] = true
					matchedLow[j] = true
				}
			}
		}

		// Mark unmatched tracks: increment consecutive misses, Tracked -> Lost
		for _, tr := range trackedLost {
			if matchedTrack[tr.Index] {
				continue
			}
			if tr.State == StateTracked {
				tr.State = StateLost
				tr.Misses = 1
			} else if tr.State == StateLost {
				tr.Misses++
			} else if tr.State == StateNew {
				tr.Misses++
			}
		}

		// New tracks from unmatched high
		for idx, d := range highDets {
			if matchedHigh[idx] {
				continue
			}
			// Do not create from low (already filtered)
			z := tlwhToXyah(d)
			mean, cov := kalmanInitiate(z)
			newIdx := nextIndex
			nextIndex++
			tr := &btTrack{
				Index:           newIdx,
				Label:           d.Label,
				State:           StateNew,
				Mean:            mean,
				Cov:             cov,
				LastBBox:        d,
				PredictedBBox:   d,
				Age:             0,
				Hits:            1,
				HitCount:        1,
				StartTimestamp:  fr.Timestamp,
				LastTimestamp:   fr.Timestamp,
				LastMatchedTime: fr.Timestamp,
			}
			// If MinHits ==1, new track immediately Tracked
			if tr.HitCount >= t.MinHits {
				tr.State = StateTracked
			}
			active[newIdx] = tr
			// Only assign if track is already confirmed? But per MinHits logic, we should not return unconfirmed.
			// We will assign now, but later filter at return time.
			// For now, assign but will filter.
			assignments[d.ID] = newIdx
			// If not yet confirmed, this assignment will be filtered later, but we keep it for internal matching.
			// To avoid polluting output with tentative, we will keep assignments but filter at end.
		}

		// Promote New tracks that have reached MinHits via second hit (already handled above via HitCount increment)
		// For tracks that were New and just matched, they may now have HitCount >= MinHits, promote to Tracked
		for _, tr := range active {
			if tr.State == StateNew && tr.HitCount >= t.MinHits {
				tr.State = StateTracked
			}
		}

		// Remove Lost/New that exceed TrackBuffer sampled observations (count-based, not wall-clock)
		for _, tr := range active {
			if tr.State == StateLost || tr.State == StateNew {
				if tr.Misses > t.TrackBuffer {
					tr.State = StateRemoved
				}
			}
		}

		timestampCopy := fr.Timestamp
		prevTimestamp = &timestampCopy
	}

	// Reverse lookup: detectionID -> (timestamp, detection box), and the first
	// observation per track index, used by the cross-gap re-association pass.
	type detInfo struct {
		t   float64
		det DetectionInput
	}
	detInfoByID := make(map[uuid.UUID]detInfo)
	frameTsByFrameID := make(map[uuid.UUID]float64)
	for _, f := range frames {
		frameTsByFrameID[f.FrameID] = f.Timestamp
	}
	for frID, ds := range detectionsByFrame {
		ts := frameTsByFrameID[frID]
		for _, d := range ds {
			detInfoByID[d.ID] = detInfo{t: ts, det: d}
		}
	}
	firstObs := make(map[int]detInfo) // trackIndex -> earliest assigned observation
	for detID, idx := range assignments {
		di, ok := detInfoByID[detID]
		if !ok {
			continue
		}
		if fo, ok := firstObs[idx]; !ok || di.t < fo.t {
			firstObs[idx] = di
		}
	}

	// -- Cross-gap re-association ---------------------------------------------
	// With sparse sampling an object is routinely unobserved long enough for
	// its track to die (state Removed), so the next observation spawns a new
	// track index — tracks fragment every gap. Here each removed track's
	// trajectory endpoint is matched against later tracks' first observation:
	// if the gap is within reattachMaxGap, classes match, and the first box is
	// within the dt-aware drift limit, the later track's detections are
	// re-pointed to the older index, restoring one identity across the gap.
	type deadSeed struct {
		idx      int
		label    string
		lastT    float64
		lastBBox DetectionInput
	}
	var deads []deadSeed
	for _, tr := range active {
		// Only truly dead tracks participate; a Lost-but-alive track is still
		// matching in the main loop and Removed tracks keep their state here.
		if tr.State == StateRemoved && tr.HitCount >= t.MinHits {
			deads = append(deads, deadSeed{idx: tr.Index, label: tr.Label, lastT: tr.LastTimestamp, lastBBox: tr.LastBBox})
		}
	}
	if len(deads) > 0 {
		type cand struct {
			deadIdx, lateIdx int
			ratio            float64
		}
		var cands []cand
		for _, dc := range deads {
			for lateIdx, fo := range firstObs {
				if lateIdx == dc.idx {
					continue
				}
				gap := fo.t - dc.lastT
				if gap <= 0.05 || gap > t.Sparse.ReattachGap {
					continue
				}
				if !sameClass(dc.label, fo.det.Label) {
					continue
				}
				limit := t.allowedDrift(gap)
				ratio := centerDistance(dc.lastBBox, fo.det) / limit
				if ratio <= 1 {
					cands = append(cands, cand{deadIdx: dc.idx, lateIdx: lateIdx, ratio: ratio})
				}
			}
		}
		if len(cands) > 0 {
			sort.Slice(cands, func(i, j int) bool {
				if cands[i].ratio == cands[j].ratio {
					return cands[i].deadIdx < cands[j].deadIdx
				}
				return cands[i].ratio < cands[j].ratio
			})
			detIDsByIndex := make(map[int][]uuid.UUID)
			for detID, idx := range assignments {
				detIDsByIndex[idx] = append(detIDsByIndex[idx], detID)
			}
			usedDead := make(map[int]bool)
			usedLate := make(map[int]bool)
			for _, c := range cands {
				if usedDead[c.deadIdx] || usedLate[c.lateIdx] {
					continue
				}
				usedDead[c.deadIdx] = true
				usedLate[c.lateIdx] = true
				for _, detID := range detIDsByIndex[c.lateIdx] {
					assignments[detID] = c.deadIdx
				}
				delete(firstObs, c.lateIdx)
			}
		}
	}

	// Filter assignments to only confirmed tracks (HitCount >= MinHits)
	// Keep all tracks' HitCount, including Removed (still in active map)
	filtered := make(map[uuid.UUID]int)
	for detID, trackIdx := range assignments {
		if tr, ok := active[trackIdx]; ok {
			if tr.HitCount >= t.MinHits {
				filtered[detID] = trackIdx
			}
		}
	}
	return filtered, nil
}

// gatingDistance computes Mahalanobis gating distance for association (ByteTrack/SORT).
// y = z - H*mean, S = H*P*H^T + R, distance = y^T * S^{-1} * y
func gatingDistance(mean [8]float64, cov [8][8]float64, z [4]float64) float64 {
	var y [4]float64
	for i := 0; i < 4; i++ {
		y[i] = z[i] - mean[i]
	}
	// S as in kalmanUpdate
	var HP [4][8]float64
	for i := 0; i < 4; i++ {
		for j := 0; j < 8; j++ {
			HP[i][j] = cov[i][j]
		}
	}
	var S [4][4]float64
	for i := 0; i < 4; i++ {
		for j := 0; j < 4; j++ {
			S[i][j] = HP[i][j]
		}
	}
	h := z[3]
	if h < 1e-6 {
		h = 1e-6
	}
	var rDiag [4]float64
	rDiag[0] = stdWeightPosition * h
	rDiag[0] = rDiag[0] * rDiag[0]
	rDiag[1] = stdWeightPosition * h
	rDiag[1] = rDiag[1] * rDiag[1]
	rDiag[2] = 1e-4
	rDiag[3] = stdWeightPosition * h
	rDiag[3] = rDiag[3] * rDiag[3]
	for i := 0; i < 4; i++ {
		S[i][i] += rDiag[i]
	}
	Sinv := matInv4(S)
	var tmp [4]float64
	for i := 0; i < 4; i++ {
		sum := 0.0
		for j := 0; j < 4; j++ {
			sum += Sinv[i][j] * y[j]
		}
		tmp[i] = sum
	}
	dist := 0.0
	for i := 0; i < 4; i++ {
		dist += y[i] * tmp[i]
	}
	return dist
}

const INF = 1e9

const chiSquareThreshold = 9.4877

func buildCostMatrixByteTrack(tracks []*btTrack, dets []DetectionInput, matchThreshold float64, fuseScore bool, dt float64, sparse SparseParams) [][]float64 {
	nRows := len(tracks)
	nCols := len(dets)
	cost := make([][]float64, nRows)
	for i := range cost {
		cost[i] = make([]float64, nCols)
		for j := range cost[i] {
			cost[i][j] = INF
		}
	}
	allowedDrift := func(dt float64) float64 {
		return sparse.AllowFloor + sparse.MaxSpeed*dt
	}
	sparseGating := dt > sparse.DtThreshold
	for i, tr := range tracks {
		// Use predicted bbox for tracks
		predBBox := tr.PredictedBBox
		// Fallback if PredictedBBox empty (not yet predicted)
		if predBBox.BBoxWidth == 0 && predBBox.BBoxHeight == 0 {
			// Use LastBBox or derive from mean
			predBBox = xyahToTLWH(tr.Mean, tr.Label)
		}
		for j, d := range dets {
			// Soft label gate: match on confusion group so a YOLO category
			// flip between frames does not break the object's identity.
			if !sameClass(tr.Label, d.Label) {
				continue
			}
			z := tlwhToXyah(d)
			if sparseGating {
				// Sparse gating: the Mahalanobis/IoU prediction gates are
				// meaningless across multi-second gaps, so gate on dt-aware
				// spatial plausibility instead. Ranking still prefers
				// overlapping and closer boxes.
				dPred := centerDistance(predBBox, d)
				dLast := centerDistance(tr.LastBBox, d)
				limit := allowedDrift(dt)
				overlap := iouByteTrack(predBBox, d) > 0 || iouByteTrack(tr.LastBBox, d) > 0
				ratio := math.Min(dPred, dLast) / limit
				if ratio > 1 && !overlap {
					continue
				}
				if ratio > 1 {
					ratio = 1
				}
				iou := iouByteTrack(predBBox, d)
				if iou > 0 {
					cost[i][j] = (1-iou)*0.6 + 0.4*ratio
				} else {
					cost[i][j] = 0.5 + 0.5*ratio
				}
				continue
			}
			// Mahalanobis gating on predicted Kalman state (4-dim chi-square 95%)
			if gatingDistance(tr.Mean, tr.Cov, z) > chiSquareThreshold {
				continue
			}
			score := iouByteTrack(predBBox, d)
			if score < matchThreshold {
				continue
			}
			c := 1 - score
			if fuseScore {
				// Fuse with detection confidence: higher confidence -> lower cost
				// Ultralytics: fused = IoU * detection_score
				fused := score * d.Confidence
				c = 1 - fused
			}
			cost[i][j] = c
		}
	}
	return cost
}

// hungarianRect solves rectangular assignment with INF forbidden.
// Returns assign array size nRows, where assign[i]=j or -1 for unmatched.
func hungarianRect(cost [][]float64, nRows, nCols int) []int {
	if nRows == 0 || nCols == 0 {
		return make([]int, nRows)
	}
	n := nRows
	if nCols > n {
		n = nCols
	}
	// Build square cost with dummy cost 0 (so unmatched goes to dummy)
	square := make([][]float64, n)
	for i := 0; i < n; i++ {
		square[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			if i < nRows && j < nCols {
				square[i][j] = cost[i][j]
			} else {
				square[i][j] = 0 // dummy
			}
		}
	}
	assignSquare := hungarian(square, n)
	// Map back to rectangular
	result := make([]int, nRows)
	for i := 0; i < nRows; i++ {
		j := assignSquare[i]
		if j < 0 || j >= nCols {
			result[i] = -1
			continue
		}
		if cost[i][j] >= INF/2 {
			result[i] = -1
			continue
		}
		result[i] = j
	}
	return result
}

// hungarian solves the assignment problem for a square cost matrix N x N (minimization).
// Returns assignment array where assign[row] = col.
func hungarian(cost [][]float64, n int) []int {
	if n == 0 {
		return []int{}
	}
	u := make([]float64, n+1)
	v := make([]float64, n+1)
	p := make([]int, n+1)
	way := make([]int, n+1)

	for i := 1; i <= n; i++ {
		p[0] = i
		j0 := 0
		minv := make([]float64, n+1)
		for j := 0; j <= n; j++ {
			minv[j] = 1e18
		}
		used := make([]bool, n+1)
		for {
			used[j0] = true
			i0 := p[j0]
			delta := 1e18
			j1 := 0
			for j := 1; j <= n; j++ {
				if !used[j] {
					cur := cost[i0-1][j-1] - u[i0] - v[j]
					if cur < minv[j] {
						minv[j] = cur
						way[j] = j0
					}
					if minv[j] < delta {
						delta = minv[j]
						j1 = j
					}
				}
			}
			for j := 0; j <= n; j++ {
				if used[j] {
					u[p[j]] += delta
					v[j] -= delta
				} else {
					minv[j] -= delta
				}
			}
			j0 = j1
			if p[j0] == 0 {
				break
			}
		}
		for {
			j1 := way[j0]
			p[j0] = p[j1]
			j0 = j1
			if j0 == 0 {
				break
			}
		}
	}

	ans := make([]int, n)
	for j := 1; j <= n; j++ {
		if p[j] != 0 {
			ans[p[j]-1] = j - 1
		}
	}
	return ans
}
