package config

// Configuration tests for Phase 5 sampler keys (no database needed).
// Load() reads the process environment; t.Setenv isolates each case.

import (
	"testing"
	"time"
)

func TestSamplerRobustnessDefaults(t *testing.T) {
	cfg := Load()
	if cfg.SamplerShadow {
		t.Fatalf("shadow must default off (probe still costs a decode)")
	}
	if cfg.SamplerBusyThreshold != 0.25 {
		t.Fatalf("busy threshold default must be 0.25, got %v", cfg.SamplerBusyThreshold)
	}
	if cfg.SamplerBusyFraction != 0.6 {
		t.Fatalf("busy fraction default must be 0.6, got %v", cfg.SamplerBusyFraction)
	}
}

func TestSamplerRobustnessEnv(t *testing.T) {
	t.Setenv("SAMPLER_SHADOW", "true")
	t.Setenv("SAMPLER_BUSY_THRESHOLD", "0.4")
	t.Setenv("SAMPLER_BUSY_FRACTION", "0.75")
	cfg := Load()
	if !cfg.SamplerShadow {
		t.Fatalf("shadow must parse true")
	}
	if cfg.SamplerBusyThreshold != 0.4 {
		t.Fatalf("busy threshold must parse, got %v", cfg.SamplerBusyThreshold)
	}
	if cfg.SamplerBusyFraction != 0.75 {
		t.Fatalf("busy fraction must parse, got %v", cfg.SamplerBusyFraction)
	}
}

func TestSamplerAdaptiveFlagMatrix(t *testing.T) {
	// Unset in this process environment defaults to adaptive enabled
	// (development default); explicit false restores baseline.
	t.Setenv("SAMPLER_ADAPTIVE", "false")
	if cfg := Load(); cfg.SamplerAdaptive {
		t.Fatalf("SAMPLER_ADAPTIVE=false must disable adaptive")
	}
	t.Setenv("SAMPLER_ADAPTIVE", "true")
	if cfg := Load(); !cfg.SamplerAdaptive {
		t.Fatalf("SAMPLER_ADAPTIVE=true must enable adaptive")
	}
}

func mustPanic(t *testing.T, key, value string) {
	t.Helper()
	t.Setenv(key, value)
	defer func() {
		if recover() == nil {
			t.Fatalf("%s=%q must panic", key, value)
		}
	}()
	Load()
}

func TestSamplerRobustnessInvalid(t *testing.T) {
	mustPanic(t, "SAMPLER_SHADOW", "maybe")
	mustPanic(t, "SAMPLER_BUSY_THRESHOLD", "-0.1")
	mustPanic(t, "SAMPLER_BUSY_FRACTION", "1.5")
	mustPanic(t, "SAMPLER_BUSY_FRACTION", "nan-x")
	mustPanic(t, "SAMPLER_GAMMA", "-1")
	mustPanic(t, "SAMPLER_MIN_GAP", "0s")
	mustPanic(t, "SAMPLER_EPSILON", "-0.5")
	mustPanic(t, "SAMPLER_MAX_ROUNDS", "0")
}

func TestSamplerTimeoutSane(t *testing.T) {
	if cfg := Load(); cfg.SamplerPlanTimeout != 5*time.Minute {
		t.Fatalf("plan timeout default must be 5m, got %v", cfg.SamplerPlanTimeout)
	}
	t.Setenv("SAMPLER_PLAN_TIMEOUT", "2m")
	if cfg := Load(); cfg.SamplerPlanTimeout != 2*time.Minute {
		t.Fatalf("plan timeout must parse, got %v", cfg.SamplerPlanTimeout)
	}
}
