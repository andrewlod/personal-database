.PHONY: test test-backend test-frontend test-coverage lint typecheck

test: test-backend test-frontend

test-backend: test-backend-unit test-backend-integration

test-backend-unit:
	@echo "Running backend unit tests..."
	python -m pytest tests/unit/backend/ -v

test-backend-integration:
	@echo "Running backend integration tests..."
	python -m pytest tests/integration/ -v

test-backend-coverage:
	@echo "Running backend tests with coverage..."
	python -m pytest tests/unit/backend/ tests/integration/ --cov=src --cov-report=html

test-frontend: test-frontend-unit test-frontend-e2e

test-frontend-unit:
	@echo "Running frontend unit tests..."
	cd src/web && npm test -- --ci --watchAll=false

test-frontend-e2e:
	@echo "Running frontend E2E tests..."
	cd src/web && npx playwright test

test-frontend-e2e-headed:
	@echo "Running frontend E2E tests (headed)..."
	cd src/web && npx playwright test --headed

lint:
	@echo "Running linters..."
	# Python linting
	ruff check src/ tests/
	# JavaScript/TypeScript linting
	cd src/web && npm run lint

typecheck:
	@echo "Running type checkers..."
	# Python type checking
	mypy src/
	# JavaScript/TypeScript type checking
	cd src/web && npm run typecheck

test-coverage: test-backend-coverage
	@echo "Frontend coverage available via npm test -- --coverage"