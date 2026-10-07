// Package playable generates and serves browser-playable proxies.
//
// Background: browsers only decode a handful of codecs in <video> (H.264,
// VP9, AV1). Surveillance corpora such as VIRAT ship MPEG-4 Part 2 in .mp4,
// which the video pipeline (ffmpeg/YOLO) handles fine but no browser plays —
// the player fails with "no video with supported format and MIME type".
//
// The fix keeps the original authoritative for processing and generates a
// sidecar proxy (H.264 + faststart, videos/<id>/play.mp4) that /stream
// prefers when present. Proxies are content, not state: idempotent,
// regenerable, deleted with the video.
package playable

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"os"
	"os/exec"
	"path/filepath"
	"time"

	"github.com/berzz26/recall/services/api/internal/storage"
	"github.com/berzz26/recall/services/api/internal/video"
	"github.com/google/uuid"
)

// ProxyFileName is the sidecar name under the video's storage directory.
const ProxyFileName = "play.mp4"

// browserPlayable video codecs (in an MP4 container). HEVC is deliberately
// excluded: Safari plays it, Firefox/Chromium generally do not.
var browserPlayable = map[string]bool{
	"h264": true,
	"vp9":  true,
	"av1":  true,
	"vp8":  true,
}

// NeedsProxy reports whether a video codec requires a transcoded proxy.
func NeedsProxy(videoCodec string) bool {
	if videoCodec == "" {
		return false
	}
	return !browserPlayable[videoCodec]
}

type probeStream struct {
	CodecName string `json:"codec_name"`
	CodecType string `json:"codec_type"`
}

// ProbeCodecs returns the primary video and audio codec names for a file.
func ProbeCodecs(ctx context.Context, ffprobePath, filePath string) (vcodec, acodec string, err error) {
	if ffprobePath == "" {
		ffprobePath = "ffprobe"
	}
	ctx, cancel := context.WithTimeout(ctx, 60*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, ffprobePath,
		"-v", "error",
		"-select_streams", "v:0",
		"-show_entries", "stream=codec_name,codec_type",
		"-of", "json",
		filePath,
	)
	var out bytes.Buffer
	cmd.Stdout = &out
	if err := cmd.Run(); err != nil {
		return "", "", fmt.Errorf("ffprobe video stream: %w", err)
	}
	var parsed struct {
		Streams []probeStream `json:"streams"`
	}
	if err := json.Unmarshal(out.Bytes(), &parsed); err != nil {
		return "", "", fmt.Errorf("parse ffprobe output: %w", err)
	}
	if len(parsed.Streams) > 0 {
		vcodec = parsed.Streams[0].CodecName
	}
	// Audio is optional (VIRAT clips have none); probe separately so a
	// missing audio stream is not an error.
	cmd = exec.CommandContext(ctx, ffprobePath,
		"-v", "error",
		"-select_streams", "a:0",
		"-show_entries", "stream=codec_name",
		"-of", "json",
		filePath,
	)
	out.Reset()
	cmd.Stdout = &out
	if err := cmd.Run(); err == nil {
		var a struct {
			Streams []probeStream `json:"streams"`
		}
		if json.Unmarshal(out.Bytes(), &a) == nil && len(a.Streams) > 0 {
			acodec = a.Streams[0].CodecName
		}
	}
	return vcodec, acodec, nil
}

// Transcode builds the H.264 + faststart proxy. The destination is written
// atomically (temp file + rename) so /stream never serves a partial file.
func Transcode(ctx context.Context, ffmpegPath, src, dst string, hasAudio bool) error {
	if ffmpegPath == "" {
		ffmpegPath = "ffmpeg"
	}
	if err := os.MkdirAll(filepath.Dir(dst), 0o755); err != nil {
		return fmt.Errorf("create proxy dir: %w", err)
	}
	tmp, err := os.CreateTemp(filepath.Dir(dst), "play-*.tmp.mp4")
	if err != nil {
		return fmt.Errorf("create proxy temp file: %w", err)
	}
	tmpPath := tmp.Name()
	_ = tmp.Close()
	defer os.Remove(tmpPath) // no-op after successful rename

	args := []string{"-y", "-i", src,
		"-map", "0:v:0",
		"-c:v", "libx264", "-preset", "veryfast", "-crf", "21",
		"-pix_fmt", "yuv420p",
	}
	if hasAudio {
		args = append(args, "-map", "0:a:0?", "-c:a", "aac", "-b:a", "128k")
	}
	args = append(args, "-movflags", "+faststart", tmpPath)
	cmd := exec.CommandContext(ctx, ffmpegPath, args...)
	var stderr bytes.Buffer
	cmd.Stderr = &stderr
	if err := cmd.Run(); err != nil {
		return fmt.Errorf("ffmpeg transcode: %w (ffmpeg: %s)", err, firstLines(stderr.String(), 5))
	}
	if err := os.Rename(tmpPath, dst); err != nil {
		return fmt.Errorf("publish proxy: %w", err)
	}
	return nil
}

func firstLines(s string, n int) string {
	out := ""
	lines := 0
	for _, r := range s {
		if r == '\n' {
			lines++
			if lines >= n {
				break
			}
		}
		out += string(r)
		if len(out) > 2000 {
			break
		}
	}
	return out
}

// Service generates proxies and records them on the video row.
type Service struct {
	videoRepo *video.Repository
	store     storage.Storage
	ffprobe   string
	ffmpeg    string
	timeout   time.Duration
	log       *slog.Logger
}

// NewService wires proxy generation. timeout bounds a single transcode;
// 0 defaults to 30 minutes (a 500 MB 1080p file takes minutes).
func NewService(videoRepo *video.Repository, store storage.Storage, ffprobe, ffmpeg string, timeout time.Duration, log *slog.Logger) *Service {
	if timeout <= 0 {
		timeout = 30 * time.Minute
	}
	if log == nil {
		log = slog.Default()
	}
	return &Service{videoRepo: videoRepo, store: store, ffprobe: ffprobe, ffmpeg: ffmpeg, timeout: timeout, log: log}
}

// ProxyKeyFor returns the canonical storage key for a video's proxy.
func ProxyKeyFor(id uuid.UUID) string {
	return "videos/" + id.String() + "/" + ProxyFileName
}

// resolveLocal maps the video to its on-disk original. Proxies are only
// supported on local storage (the only backend); anything else errors.
func (s *Service) resolveLocal(v *video.Video) (string, error) {
	ls, ok := s.store.(*storage.LocalStorage)
	if !ok {
		return "", fmt.Errorf("playable proxy requires local storage")
	}
	if v.StorageKey != nil && *v.StorageKey != "" {
		return filepath.Join(ls.Root(), *v.StorageKey), nil
	}
	if v.SourcePath != nil && *v.SourcePath != "" {
		return *v.SourcePath, nil
	}
	return "", fmt.Errorf("video has no file backing")
}

// ProxyPath resolves a recorded playable key to its on-disk path.
func (s *Service) ProxyPath(key string) (string, error) {
	return LocalProxyPath(s.store, key)
}

// LocalProxyPath resolves a proxy storage key against local storage.
func LocalProxyPath(store storage.Storage, key string) (string, error) {
	ls, ok := store.(*storage.LocalStorage)
	if !ok || key == "" {
		return "", fmt.Errorf("no local proxy")
	}
	return filepath.Join(ls.Root(), key), nil
}

// Status reports the proxy state without doing work.
func (s *Service) Status(ctx context.Context, id uuid.UUID) (state string, key string, err error) {
	v, err := s.videoRepo.GetByID(ctx, id)
	if err != nil {
		return "", "", err
	}
	if v.PlayableKey != nil && *v.PlayableKey != "" {
		if p, perr := s.ProxyPath(*v.PlayableKey); perr == nil {
			if _, serr := os.Stat(p); serr == nil {
				return "ready", *v.PlayableKey, nil
			}
		}
		// Key recorded but file gone — regenerate below.
	}
	src, err := s.resolveLocal(v)
	if err != nil {
		return "", "", err
	}
	vcodec, _, err := ProbeCodecs(ctx, s.ffprobe, src)
	if err != nil {
		return "", "", err
	}
	if !NeedsProxy(vcodec) {
		return "native", "", nil
	}
	return "missing", "", nil
}

// Ensure generates the proxy if the original needs one and none exists.
// It is idempotent: a present proxy (or a natively playable original) is a
// no-op returning ("ready"|"native", key, nil). A failed transcode returns
// an error but leaves no partial state (atomic rename + key written last).
func (s *Service) Ensure(ctx context.Context, id uuid.UUID) (state string, key string, err error) {
	state, key, err = s.Status(ctx, id)
	if err != nil || state != "missing" {
		return state, key, err
	}
	v, err := s.videoRepo.GetByID(ctx, id)
	if err != nil {
		return "", "", err
	}
	src, err := s.resolveLocal(v)
	if err != nil {
		return "", "", err
	}
	vcodec, acodec, err := ProbeCodecs(ctx, s.ffprobe, src)
	if err != nil {
		return "", "", err
	}
	key = ProxyKeyFor(id)
	dst, err := s.ProxyPath(key)
	if err != nil {
		return "", "", err
	}
	// Another worker may have won the race while we probed.
	if _, serr := os.Stat(dst); serr == nil {
		if _, uerr := s.videoRepo.SetPlayableKey(ctx, id, key); uerr != nil {
			return "", "", uerr
		}
		return "ready", key, nil
	}
	s.log.Info("playable proxy transcode started", "video_id", id.String(), "codec", vcodec)
	tctx, cancel := context.WithTimeout(ctx, s.timeout)
	defer cancel()
	if terr := Transcode(tctx, s.ffmpeg, src, dst, acodec != ""); terr != nil {
		s.log.Error("playable proxy transcode failed", "video_id", id.String(), "error", terr)
		return "", "", terr
	}
	if _, uerr := s.videoRepo.SetPlayableKey(ctx, id, key); uerr != nil {
		return "", "", uerr
	}
	s.log.Info("playable proxy ready", "video_id", id.String(), "playable_key", key)
	return "ready", key, nil
}
