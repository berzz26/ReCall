package sampler

// Unit tests for the visual-probe scoring primitives: block means,
// changed_fraction, and the per-video noise floor.

import (
	"math"
	"math/rand"
	"testing"
)

// uniformGray builds a size x size frame filled with one level.
func uniformGray(size int, level uint8) []byte {
	f := make([]byte, size*size)
	for i := range f {
		f[i] = level
	}
	return f
}

// paintBlock sets one grid block (bi = block index, row-major) to level.
func paintBlock(f []byte, size, grid, bi int, level uint8) {
	block := size / grid
	by, bx := bi/grid, bi%grid
	for y := 0; y < block; y++ {
		row := (by*block+y)*size + bx*block
		for x := 0; x < block; x++ {
			f[row+x] = level
		}
	}
}

func meansOf(t *testing.T, f []byte) []uint8 {
	t.Helper()
	m, err := BlockMeans(f, 64, 8)
	if err != nil {
		t.Fatalf("BlockMeans: %v", err)
	}
	if len(m) != 64 {
		t.Fatalf("want 64 blocks, got %d", len(m))
	}
	return m
}

// Test 1 — identical frames produce changed_fraction = 0.
func TestIdenticalFramesZeroChange(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	f := make([]byte, 64*64)
	for i := range f {
		f[i] = uint8(rng.Intn(256))
	}
	ref, cur := meansOf(t, f), meansOf(t, append([]byte(nil), f...))
	if got := ChangedFraction(ref, cur, 0); got != 0 {
		t.Fatalf("identical frames: want 0, got %v", got)
	}
}

// Test 2 — exactly one changed block gives 1/64.
func TestOneChangedBlock(t *testing.T) {
	ref := meansOf(t, uniformGray(64, 100))
	cur := meansOf(t, uniformGray(64, 100))
	changed := append([]byte(nil), uniformGray(64, 100)...)
	paintBlock(changed, 64, 8, 17, 200)
	cur = meansOf(t, changed)
	if got := ChangedFraction(ref, cur, 5); math.Abs(got-1.0/64) > 1e-9 {
		t.Fatalf("one block: want %v, got %v", 1.0/64, got)
	}
}

// Test 3 — half the blocks changed gives 0.5.
func TestHalfImageChange(t *testing.T) {
	base := uniformGray(64, 100)
	half := append([]byte(nil), base...)
	for b := 0; b < 32; b++ {
		paintBlock(half, 64, 8, b, 200)
	}
	ref, cur := meansOf(t, base), meansOf(t, half)
	if got := ChangedFraction(ref, cur, 5); math.Abs(got-0.5) > 1e-9 {
		t.Fatalf("half image: want 0.5, got %v", got)
	}
	// All blocks changed gives 1.0.
	all := append([]byte(nil), base...)
	for b := 0; b < 64; b++ {
		paintBlock(all, 64, 8, b, 200)
	}
	if got := ChangedFraction(ref, meansOf(t, all), 5); got != 1.0 {
		t.Fatalf("full image: want 1.0, got %v", got)
	}
}

// Test 9 — synthetic low-level noise stays below the noise floor: with
// block differences ~ uniform[0,2], median+3*MAD must leave essentially no
// block marked changed.
func TestNoiseFloorSuppressesLowLevelNoise(t *testing.T) {
	rng := rand.New(rand.NewSource(7))
	est := &NoiseEstimator{}
	// Simulate 300 probe frames of pure sensor noise: consecutive-frame
	// block differences uniform in [0,2].
	for i := 0; i < 300*64; i++ {
		est.Add(rng.Float64() * 2)
	}
	threshold := est.Threshold(DefaultNoiseK)
	if threshold < 2 {
		t.Fatalf("noise floor %v should exceed the max noise level 2", threshold)
	}
	// Fresh noisy frame vs reference: ~no changed blocks.
	ref := make([]uint8, 64)
	cur := make([]uint8, 64)
	for i := range cur {
		cur[i] = uint8(rng.Intn(3)) // diff in [0,2]
	}
	if got := ChangedFraction(ref, cur, threshold); got != 0 {
		t.Fatalf("noise-only frame: want 0 changed, got %v", got)
	}
}

// Noise floor sanity: strong motion outliers must not drag the robust
// threshold up with them.
func TestNoiseFloorRobustToOutliers(t *testing.T) {
	est := &NoiseEstimator{}
	for i := 0; i < 1000; i++ {
		est.Add(1)
	}
	for i := 0; i < 50; i++ {
		est.Add(200) // a few frames of large motion
	}
	if got := est.Threshold(DefaultNoiseK); got > 10 {
		t.Fatalf("robust threshold should stay near noise, got %v", got)
	}
}

// Empty estimator (single probe frame) yields threshold 0.
func TestNoiseFloorEmpty(t *testing.T) {
	est := &NoiseEstimator{}
	if got := est.Threshold(DefaultNoiseK); got != 0 {
		t.Fatalf("empty estimator: want 0, got %v", got)
	}
}

func TestBlockMeansValidation(t *testing.T) {
	if _, err := BlockMeans(make([]byte, 10), 64, 8); err == nil {
		t.Fatalf("want error for short frame")
	}
	if _, err := BlockMeans(make([]byte, 64*64), 64, 7); err == nil {
		t.Fatalf("want error for indivisible grid")
	}
}
