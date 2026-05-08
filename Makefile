.PHONY: help dev dev-api dev-frontend build clean test lint

# Default target
help:
	@echo "QuantNodes Development Commands:"
	@echo ""
	@echo "  make dev          - Start both frontend and API servers"
	@echo "  make dev-api      - Start API server only"
	@echo "  make dev-frontend - Start frontend server only"
	@echo "  make build        - Build frontend for production"
	@echo "  make docker-up    - Start all services with Docker"
	@echo "  make docker-down  - Stop all services"
	@echo "  make clean        - Clean build artifacts"
	@echo "  make test         - Run tests"
	@echo "  make lint         - Run linter"

# Development servers
dev:
	@echo "Starting development servers..."
	@cd frontend && npm run dev &
	@cd api && uvicorn main:app --reload --port 8000

dev-api:
	@echo "Starting API server..."
	@cd api && uvicorn main:app --reload --port 8000

dev-frontend:
	@echo "Starting frontend server..."
	@cd frontend && npm run dev

# Build
build:
	@echo "Building frontend..."
	@cd frontend && npm run build

# Docker
docker-up:
	@echo "Starting Docker services..."
	@docker-compose up -d

docker-down:
	@echo "Stopping Docker services..."
	@docker-compose down

docker-build:
	@echo "Building Docker images..."
	@docker-compose build

# Clean
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf frontend/dist
	@rm -rf frontend/node_modules
	@cd api && find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Test
test:
	@echo "Running tests..."
	@cd api && python -m pytest

# Lint
lint:
	@echo "Running linter..."
	@cd frontend && npm run lint
