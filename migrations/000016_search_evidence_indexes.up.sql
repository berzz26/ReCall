-- composite indexes for bounded search enrichment + evidence endpoint.
CREATE INDEX IF NOT EXISTS idx_video_events_video_start
ON video_events(video_id, start_timestamp);
CREATE INDEX IF NOT EXISTS idx_video_events_video_start_end
ON video_events(video_id, start_timestamp, end_timestamp);
CREATE INDEX IF NOT EXISTS idx_video_tracks_video_start_end
ON video_tracks(video_id, start_timestamp, end_timestamp);
CREATE INDEX IF NOT EXISTS idx_video_frames_video_ts
ON video_frames(video_id, timestamp_seconds);
CREATE INDEX IF NOT EXISTS idx_video_frames_segment_ts
ON video_frames(segment_id, timestamp_seconds);
CREATE INDEX IF NOT EXISTS idx_detections_video_frame
ON video_frame_detections(video_id, frame_id);
