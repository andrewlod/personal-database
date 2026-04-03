"""
Unit tests for the health check endpoints.
"""

from unittest.mock import Mock, AsyncMock
from fastapi.testclient import TestClient
from src.api.main import app
from src.api.dependencies import (
    get_vector_db_service,
    get_embedding_service,
    get_rag_service,
)


class TestHealthEndpoints:
    """Test cases for health check endpoints."""

    def test_health_check_success(self):
        """Test successful health check."""
        mock_vector_db_service = Mock()
        mock_vector_db_service.health_check = AsyncMock(
            return_value={
                "status": "healthy",
                "class_name": "PersonalKnowledge",
                "object_count": 50,
            }
        )

        mock_embedding_service = Mock()
        mock_embedding_service.health_check = AsyncMock(
            return_value={
                "status": "healthy",
                "model": "all-MiniLM-L6-v2",
                "device": "cpu",
            }
        )

        mock_rag_service = Mock()

        app.dependency_overrides[get_vector_db_service] = lambda: mock_vector_db_service
        app.dependency_overrides[get_embedding_service] = lambda: mock_embedding_service
        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service

        try:
            client = TestClient(app)
            response = client.get("/api/health/")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert "services" in data
            assert data["services"]["vector_database"]["status"] == "healthy"
            assert data["services"]["embedding_service"]["status"] == "healthy"
        finally:
            app.dependency_overrides.clear()

    def test_health_check_unhealthy(self):
        """Test health check when services are unhealthy."""
        mock_vector_db_service = Mock()
        mock_vector_db_service.health_check = AsyncMock(
            return_value={"status": "unhealthy", "error": "Connection failed"}
        )

        mock_embedding_service = Mock()
        mock_embedding_service.health_check = AsyncMock(
            return_value={
                "status": "healthy",
                "model": "all-MiniLM-L6-v2",
                "device": "cpu",
            }
        )

        mock_rag_service = Mock()

        app.dependency_overrides[get_vector_db_service] = lambda: mock_vector_db_service
        app.dependency_overrides[get_embedding_service] = lambda: mock_embedding_service
        app.dependency_overrides[get_rag_service] = lambda: mock_rag_service

        try:
            client = TestClient(app)
            response = client.get("/api/health/")

            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "unhealthy"
            assert data["services"]["vector_database"]["status"] == "unhealthy"
        finally:
            app.dependency_overrides.clear()

    def test_ping_endpoint(self):
        """Test the ping endpoint."""
        client = TestClient(app)
        response = client.get("/api/health/ping")

        assert response.status_code == 200
        assert response.json() == {"message": "pong"}
