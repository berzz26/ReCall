.PHONY: run build tidy fmt test docker-build docker-up docker-down db-up migrate-up migrate-down migrate-create migrate-install codec-experiment

run:
	go run ./services/api/cmd/server

# TEMPORARY: standalone CodecSight signal inspection. Usage:
#   make codec-experiment INPUT=/path/to/video.mp4
codec-experiment:
	go run ./services/api/cmd/codec-experiment -input "$(INPUT)"
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
