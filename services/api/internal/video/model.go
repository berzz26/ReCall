package video

import (
	"time"

	"github.com/google/uuid"
)

type Status string

const (
	StatusUploading  Status = "UPLOADING"
	StatusUploaded   Status = "UPLOADED"
	StatusProcessing Status = "PROCESSING"
	StatusReady      Status = "READY"
	StatusFailed     Status = "FAILED"
)

type SourceType string

const (
	SourceTypeLocal  SourceType = "LOCAL"
	SourceTypeUpload SourceType = "UPLOAD"
)

type Video struct {
	ID          uuid.UUID  `json:"id"`
	Filename    string     `json:"filename"`
	ContentHash string     `json:"content_hash"`
	MimeType    string     `json:"mime_type"`
	SizeBytes   int64      `json:"size_bytes"`
	SourceType  SourceType `json:"source_type"`
	SourcePath  *string    `json:"source_path"`
	StorageKey  *string    `json:"storage_key"`
	// PlayableKey is the storage key of the browser-playable H.264 proxy
	// (videos/<id>/play.mp4). Nil when the original is directly playable
	// or no proxy has been generated yet. Served by /stream when present.
	PlayableKey     *string    `json:"playable_key,omitempty"`
	SourceMtime     *time.Time `json:"source_mtime"`
	ProcessingError *string    `json:"processing_error"`
	Status          Status     `json:"status"`
	CreatedAt       time.Time  `json:"created_at"`
	UpdatedAt       time.Time  `json:"updated_at"`
}
