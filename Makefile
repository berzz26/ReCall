# Reranker environment plumbing for prepare-rerank. Precedence:
# command line > shell environment > .env > prepare_onnx.py defaults.
# `?=` makes command-line/shell values win; `export` makes .env-derived
# values visible to the recipe and the Python script (which defaults its
# flags from RERANKER_* when not passed explicitly).
rerank_env = $(shell grep -E "^$(1)=" .env 2>/dev/null | tail -1 | cut -d= -f2- | sed "s/[[:space:]]*#.*$$//")

RERANKER_ENABLED ?= $(call rerank_env,RERANKER_ENABLED)
RERANKER_BACKEND ?= $(call rerank_env,RERANKER_BACKEND)
RERANKER_QUANTIZATION ?= $(call rerank_env,RERANKER_QUANTIZATION)
RERANKER_MODEL ?= $(call rerank_env,RERANKER_MODEL)
RERANKER_ONNX_DIR ?= $(call rerank_env,RERANKER_ONNX_DIR)
RERANKER_PYTHON_PATH ?= $(call rerank_env,RERANKER_PYTHON_PATH)

EMBEDDING_BACKEND ?= $(call rerank_env,EMBEDDING_BACKEND)
EMBEDDING_PYTHON_PATH ?= $(call rerank_env,EMBEDDING_PYTHON_PATH)
EMBEDDING_MODEL ?= $(call rerank_env,EMBEDDING_MODEL)
EMBEDDING_MODEL_REVISION ?= $(call rerank_env,EMBEDDING_MODEL_REVISION)
EMBEDDING_ONNX_DIR ?= $(call rerank_env,EMBEDDING_ONNX_DIR)

export RERANKER_ENABLED RERANKER_BACKEND RERANKER_QUANTIZATION RERANKER_MODEL RERANKER_ONNX_DIR RERANKER_PYTHON_PATH
export EMBEDDING_BACKEND EMBEDDING_PYTHON_PATH EMBEDDING_MODEL EMBEDDING_MODEL_REVISION EMBEDDING_ONNX_DIR

.PHONY: run prepare-rerank prepare-embedding build tidy fmt test docker-build docker-up docker-down db-up migrate-up migrate-down migrate-create migrate-install fetch-onnxruntime

# Prepares the ONNX rerank artifact backend=onnx targets require. Runs before
# every `make run`; prepare_onnx.py --ensure skips in milliseconds when a valid
# artifact already exists for the configured model/opset/quantization and
# re-prepares when parameters changed. Never invoked when the reranker is
# disabled or the torch backend is selected.
prepare-rerank:
	@if [ "$(RERANKER_ENABLED)" != "true" ]; then \
	  echo "prepare-rerank: RERANKER_ENABLED is not true, skipping"; \
	elif [ "$(RERANKER_BACKEND)" != "onnx" ] && [ "$(RERANKER_BACKEND)" != "native" ]; then \
	  echo "prepare-rerank: RERANKER_BACKEND is neither onnx nor native, skipping"; \
	else \
	  py="$(RERANKER_PYTHON_PATH)"; \
	  [ -n "$$py" ] || py="python3"; \
	  echo "prepare-rerank: ensuring rerank ONNX artifact (model=$(RERANKER_MODEL) quantization=$(RERANKER_QUANTIZATION))"; \
	  exec "$$py" workers/rerank/prepare_onnx.py --ensure; \
	fi

prepare-embedding:
	@backend="$(EMBEDDING_BACKEND)"; backend=$${backend:-onnx}; \
	if [ "$$backend" != "onnx" ] && [ "$$backend" != "native" ]; then \
	  echo "prepare-embedding: EMBEDDING_BACKEND=$$backend, skipping ONNX artifact"; \
	else \
	  py="$(EMBEDDING_PYTHON_PATH)"; [ -n "$$py" ] || py="python3"; \
	  echo "prepare-embedding: ensuring embedding ONNX artifact (model=$(EMBEDDING_MODEL), dir=$(EMBEDDING_ONNX_DIR))"; \
	  exec "$$py" workers/embedding/prepare_onnx.py --ensure; \
	fi

# `run` builds a real binary instead of using `go run`: `go run` recompiles
# and re-links the whole API on every invocation (tens of seconds on a cold
# toolchain), while `go build` reuses the build cache incrementally and the
# binary starts instantly afterwards.
run: prepare-rerank prepare-embedding fetch-onnxruntime
	go build -o bin/api ./services/api/cmd/server
	./bin/api

run-web:
	cd services/web && npm run dev

build: fetch-onnxruntime
	go build -o bin/api ./services/api/cmd/server

# Downloads the pinned ONNX Runtime distribution the native (in-process)
# embedding/rerank backends link against. Idempotent: exits immediately when
# the library is already present, so `make run` stays a no-op fast path
# locally. third_party/ is gitignored; fresh clones fetch it from here.
fetch-onnxruntime:
	./scripts/fetch-onnxruntime.sh

tidy:
	go mod tidy

fmt:
	go fmt ./...

test:
	go test ./...

docker-build:
	docker build -f services/api/Dockerfile -t recall-api .

docker-up:
	docker compose up --build

docker-down:
	docker compose down

db-up:
	docker compose up -d postgres

migrate-up:
	migrate -path migrations -database "${DATABASE_URL}" up

migrate-down:
	migrate -path migrations -database "${DATABASE_URL}" down 1

migrate-create:
	@read -p "Enter migration name: " name; \
	migrate create -ext sql -dir migrations -seq $$name

migrate-install:
	go install -tags 'postgres' github.com/golang-migrate/migrate/v4/cmd/migrate@latest
