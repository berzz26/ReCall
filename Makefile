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

export RERANKER_ENABLED RERANKER_BACKEND RERANKER_QUANTIZATION RERANKER_MODEL RERANKER_ONNX_DIR RERANKER_PYTHON_PATH

.PHONY: run prepare-rerank build tidy fmt test docker-build docker-up docker-down db-up migrate-up migrate-down migrate-create migrate-install

# Prepares the ONNX rerank artifact backend=onnx targets require. Runs before
# every `make run`; prepare_onnx.py --ensure skips in milliseconds when a valid
# artifact already exists for the configured model/opset/quantization and
# re-prepares when parameters changed. Never invoked when the reranker is
# disabled or the torch backend is selected.
prepare-rerank:
	@if [ "$(RERANKER_ENABLED)" != "true" ]; then \
	  echo "prepare-rerank: RERANKER_ENABLED is not true, skipping"; \
	elif [ "$(RERANKER_BACKEND)" != "onnx" ]; then \
	  echo "prepare-rerank: RERANKER_BACKEND is not onnx, skipping"; \
	else \
	  py="$(RERANKER_PYTHON_PATH)"; \
	  [ -n "$$py" ] || py="python3"; \
	  echo "prepare-rerank: ensuring rerank ONNX artifact (model=$(RERANKER_MODEL) quantization=$(RERANKER_QUANTIZATION))"; \
	  exec "$$py" workers/rerank/prepare_onnx.py --ensure; \
	fi

run: prepare-rerank
	go run ./services/api/cmd/server

run-web:
	cd services/web && npm run dev

build:
	go build -o bin/api ./services/api/cmd/server

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
