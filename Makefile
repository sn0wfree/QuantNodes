.PHONY: help init dev dev-api dev-frontend build clean test lint

# Default target
help:
	@echo "QuantNodes Development Commands:"
	@echo ""
	@echo "  make init          - Initialize current directory (interactive)"
	@echo "  make dev           - Start both frontend and API servers"
	@echo "  make dev-api       - Start API server only"
	@echo "  make dev-frontend  - Start frontend server only"
	@echo "  make build         - Build frontend for production"
	@echo "  make clean         - Clean build artifacts"
	@echo "  make test          - Run tests"
	@echo "  make lint          - Run linter"
	@echo ""
	@echo "Or use CLI directly:"
	@echo "  quantnodes init"
	@echo "  quantnodes run --daemon"
	@echo "  quantnodes run --port 8080"

# Initialize
init:
	@echo "Running QuantNodes initialization..."
	@python -m quantnodes init

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

# Clean
clean:
	@echo "Cleaning build artifacts..."
	@rm -rf frontend/dist
	@rm -rf frontend/node_modules
	@rm -rf logs/*.log
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true

# Test
test:
	@echo "Running tests..."
	@python -m pytest tests/ -v

# Lint
lint:
	@echo "Running linter..."
	@python -m ruff check QuantNodes/
	@cd frontend && npm run lint

# Reinstall (useful after pulling updates)
reinstall:
	@echo "Reinstalling QuantNodes..."
	@pip install -e .
	@echo "QuantNodes reinstalled. Run 'make init' if needed."
