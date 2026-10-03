package codec_experiment

// TEMPORARY EXPERIMENT -- see FINDINGS.md. Whole package is disposable.

const ExperimentName = "recall-codecsight-experiment/1"

// SpatialGridSize is the number of cells per axis in the per-frame motion
// spatial distribution. The extractor emits grid*grid mean magnitudes per frame.
const SpatialGridSize = 4

// MotionStats aggregates the codec motion vectors exported for a single coded
// picture. All displacements are in pixels (motion_x/motion_scale).
//
// A frame that carries no MOTION_VECTORS side data (typically an intra picture,
// or an inter picture the encoder coded entirely as skip blocks) reports
// HasSideData=false rather than being reported as genuinely static.
type MotionStats struct {
	HasSideData     bool      `json:"has_side_data"`
	Count           int       `json:"count"`
	MeanMagnitude   float64   `json:"mean_magnitude"`
	MedianMagnitude float64   `json:"median_magnitude"`
	MaxMagnitude    float64   `json:"max_magnitude"`
	MeanDx          float64   `json:"mean_dx"`
	MeanDy          float64   `json:"mean_dy"`
	MeanAbsDx       float64   `json:"mean_abs_dx"`
	MeanAbsDy       float64   `json:"mean_abs_dy"`
	ZeroRatio       float64   `json:"zero_ratio"`
	ForwardCount    int       `json:"forward_count"`
	BackwardCount   int       `json:"backward_count"`
	SpatialGrid     []float64 `json:"spatial_grid,omitempty"`
}

// Frame is one coded picture in presentation order.
type Frame struct {
	Index       int         `json:"index"`
	PTS         int64       `json:"pts"`
	Timestamp   float64     `json:"timestamp"`
	Type        string      `json:"type"`
	KeyFrame    bool        `json:"key_frame"`
	PacketBytes int         `json:"packet_bytes"`
	Motion      MotionStats `json:"motion"`
}

// Signal is one investigated codec signal plus an explicit availability verdict.
//
// The distinction between Available and the accompanying Reason/Requires is the
// point of the experiment: it records what the current ReCall + FFmpeg stack
// actually exposes versus what it does not.
type Signal struct {
	Available bool   `json:"available"`
	Source    string `json:"source,omitempty"`
	Reason    string `json:"reason,omitempty"`
	Requires  string `json:"requires,omitempty"`
}

// GOPStats describes keyframe spacing, derived from picture types.
type GOPStats struct {
	KeyframeCount   int     `json:"keyframe_count"`
	MeanGOPLength   float64 `json:"mean_gop_length"`
	MedianGOPLength float64 `json:"median_gop_length"`
	MinGOPLength    int     `json:"min_gop_length"`
	MaxGOPLength    int     `json:"max_gop_length"`
}

// MotionSummary aggregates motion across all frames.
type MotionSummary struct {
	Available             bool    `json:"available"`
	FramesWithSideData    int     `json:"frames_with_side_data"`
	FramesWithoutSideData int     `json:"frames_without_side_data"`
	MeanOfMeanMagnitude   float64 `json:"mean_of_mean_magnitude"`
	MedianMeanMagnitude   float64 `json:"median_mean_magnitude"`
	MaxMagnitudeOverall   float64 `json:"max_magnitude_overall"`
	MeanZeroRatio         float64 `json:"mean_zero_ratio"`
	MeanVectorCount       float64 `json:"mean_vector_count"`
}

// PacketSummary aggregates bitstream cost per coded picture. This is encoded
// size, not residual energy: it is reported separately and never presented as
// a residual measurement.
type PacketSummary struct {
	TotalBytes      int                `json:"total_bytes"`
	MeanBytes       float64            `json:"mean_bytes"`
	MinBytes        int                `json:"min_bytes"`
	MaxBytes        int                `json:"max_bytes"`
	MeanBytesByType map[string]float64 `json:"mean_bytes_by_type,omitempty"`
}

// Analysis is the complete artifact written to <video>.json.
type Analysis struct {
	Experiment   string            `json:"experiment"`
	VideoID      string            `json:"video_id,omitempty"`
	VideoName    string            `json:"video"`
	Path         string            `json:"path"`
	Container    string            `json:"container,omitempty"`
	Codec        string            `json:"codec"`
	Profile      string            `json:"profile,omitempty"`
	Width        int               `json:"width"`
	Height       int               `json:"height"`
	FPS          float64           `json:"fps"`
	Duration     float64           `json:"duration"`
	FrameCount   int               `json:"frame_count"`
	HasBFrames   bool              `json:"has_b_frames"`
	Truncated    bool              `json:"truncated"`
	Extractor    string            `json:"extractor"`
	LibavVersion string            `json:"libav_version,omitempty"`
	GridSize     int               `json:"spatial_grid_size"`
	TypeCounts   map[string]int    `json:"type_counts"`
	GOP          GOPStats          `json:"gop_stats"`
	Motion       MotionSummary     `json:"motion_summary"`
	Packet       PacketSummary     `json:"packet_summary"`
	Signals      map[string]Signal `json:"signals"`
	Frames       []Frame           `json:"frames"`
}
