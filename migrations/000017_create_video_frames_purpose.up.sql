-- Frame-set designation: distinguishes which sampling policy a frame
-- belongs to.
--
--   perception: produced by the regular fixed-rate perception sampler
--               (PERCEPTION_SAMPLE_FPS) and consumed by production YOLO
--               detection, ByteTrack tracking, events, and the frame UI.
--   semantic:   produced by the adaptive sampler as the frame set for VLM
--               description generation.
--   both:       member of both sets. Timestamps shared between the two
--               policies reuse a single physical row (and storage asset);
--               sharing never overwrites the other set's membership.
--
-- Backfill: frames persisted before this column existed were the single
-- adaptive/baseline selection consumed by both detection and the VLM, so
-- they become 'both'. This keeps already-processed videos consistent with
-- the new consumers.
ALTER TABLE video_frames
    ADD COLUMN purpose TEXT NOT NULL DEFAULT 'both'
    CHECK (purpose IN ('perception', 'semantic', 'both'));
