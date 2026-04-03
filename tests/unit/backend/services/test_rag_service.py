"""
Unit tests for the RAG service.
"""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from src.api.services.rag_service import RAGService


class TestRAGService:
    """Test cases for the RAGService class."""

    def test_init(self):
        """Test RAGService initialization."""
        mock_vector_db_service = Mock()
        mock_embedding_service = Mock()

        service = RAGService(mock_vector_db_service, mock_embedding_service)

        assert service.vector_db_service == mock_vector_db_service
        assert service.embedding_service == mock_embedding_service
        assert service.default_temperature == 0.7
        assert service.default_max_tokens == 1000
        assert service.default_top_p == 0.9
        assert service.default_model == "anthropic/claude-3-haiku"

    @pytest.mark.asyncio
    async def test_query_success(self):
        """Test successful query processing."""
        # Setup mocks
        mock_vector_db_service = Mock()
        mock_embedding_service = Mock()

        # Mock embedding generation
        mock_embedding_service.embed_query = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        # Mock vector search
        mock_vector_db_service.search_vectors = AsyncMock(
            return_value=[
                {
                    "content": "Test content about machine learning.",
                    "title": "ML Document",
                    "score": 0.85,
                    "document_id": "doc1",
                    "chunk_id": "chunk1",
                    "chunk_index": 0,
                    "source_url": "http://example.com/ml",
                    "timestamp": 1640995200.0,
                    "word_count": 10,
                    "metadata": {},
                }
            ]
        )

        mock_async_context_manager = AsyncMock()
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Machine learning is a subset of artificial intelligence."
                    }
                }
            ],
            "usage": {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75},
        }
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_async_context_manager.__aenter__ = AsyncMock(return_value=mock_client)
        mock_async_context_manager.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_async_context_manager):
            # Create service and test
            service = RAGService(mock_vector_db_service, mock_embedding_service)
            result = await service.query(
                question="What is machine learning?",
                top_k=5,
                score_threshold=0.7,
                temperature=0.7,
                max_tokens=1000,
            )

            # Assertions
            assert "answer" in result
            assert "machine learning" in result["answer"].lower()
            assert result["question"] == "What is machine learning?"
            assert result["sources"] is not None
            assert len(result["sources"]) == 1
            assert result["sources"][0]["title"] == "ML Document"
            assert result["model_used"] == "anthropic/claude-3-haiku"
            assert "processing_time_seconds" in result

            # Verify mocks were called
            mock_embedding_service.embed_query.assert_called_once_with(
                "What is machine learning?"
            )
            mock_vector_db_service.search_vectors.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_no_results(self):
        """Test query when no relevant chunks are found."""
        # Setup mocks
        mock_vector_db_service = Mock()
        mock_embedding_service = Mock()

        # Mock embedding generation
        mock_embedding_service.embed_query = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        # Mock vector search returning empty results
        mock_vector_db_service.search_vectors = AsyncMock(return_value=[])

        # Create service and test
        service = RAGService(mock_vector_db_service, mock_embedding_service)
        result = await service.query(
            question="What is machine learning?", top_k=5, score_threshold=0.7
        )

        # Assertions
        assert "answer" in result
        assert "couldn't find any relevant information" in result["answer"].lower()
        assert result["sources"] == []
        assert result["question"] == "What is machine learning?"

    @pytest.mark.asyncio
    async def test_query_embedding_failure(self):
        """Test query when embedding generation fails."""
        mock_vector_db_service = Mock()
        mock_embedding_service = Mock()

        mock_embedding_service.embed_query = AsyncMock(return_value=None)

        service = RAGService(mock_vector_db_service, mock_embedding_service)

        with pytest.raises(Exception, match="Failed to generate embedding"):
            await service.query(
                question="What is machine learning?", top_k=5, score_threshold=0.7
            )

    @pytest.mark.asyncio
    async def test_query_llm_fallback(self):
        """Test query falls back to extractive answer when LLM fails."""
        # Setup mocks
        mock_vector_db_service = Mock()
        mock_embedding_service = Mock()

        # Mock embedding generation
        mock_embedding_service.embed_query = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

        # Mock vector search
        mock_vector_db_service.search_vectors = AsyncMock(
            return_value=[
                {
                    "content": "Machine learning is a method of data analysis that automates analytical model building.",
                    "title": "ML Document",
                    "score": 0.9,
                    "document_id": "doc1",
                    "chunk_id": "chunk1",
                    "chunk_index": 0,
                    "source_url": "http://example.com/ml",
                    "timestamp": 1640995200.0,
                    "word_count": 15,
                    "metadata": {},
                }
            ]
        )

        # Mock LLM failure (non-200 response)
        mock_async_context_manager = AsyncMock()
        mock_client = AsyncMock()
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_async_context_manager.__aenter__ = AsyncMock(return_value=mock_client)
        mock_async_context_manager.__aexit__ = AsyncMock(return_value=None)

        with patch("httpx.AsyncClient", return_value=mock_async_context_manager):
            service = RAGService(mock_vector_db_service, mock_embedding_service)
            result = await service.query(
                question="What is machine learning?", top_k=5, score_threshold=0.7
            )

            assert "answer" in result
            assert "Machine learning is a method" in result["answer"]
            assert "extractive answer" in result["answer"]
            assert result["sources"] is not None
            assert len(result["sources"]) == 1

    @pytest.mark.asyncio
    async def test_add_documents_success(self):
        """Test successful document addition."""
        # Setup mocks
        mock_vector_db_service = Mock()
        mock_embedding_service = Mock()

        # Mock embedding generation
        mock_embedding_service.embed_texts = AsyncMock(
            return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        )

        # Mock vector database addition
        mock_vector_db_service.add_vectors = AsyncMock(return_value=True)

        # Create service and test
        service = RAGService(mock_vector_db_service, mock_embedding_service)
        documents = [
            {
                "text": "First test document.",
                "document_id": "doc1",
                "chunk_id": "chunk1",
                "chunk_index": 0,
                "title": "Test Doc 1",
                "source_url": "http://example.com/1",
                "timestamp": 1640995200.0,
                "word_count": 3,
                "metadata": {},
            },
            {
                "text": "Second test document.",
                "document_id": "doc2",
                "chunk_id": "chunk2",
                "chunk_index": 0,
                "title": "Test Doc 2",
                "source_url": "http://example.com/2",
                "timestamp": 1640995260.0,
                "word_count": 3,
                "metadata": {},
            },
        ]

        result = await service.add_documents(documents)

        # Assertions
        assert result["message"] == "Successfully added 2 documents to knowledge base"
        assert result["documents_processed"] == 2
        assert result["chunks_created"] == 2
        assert result["vectors_stored"] == 2
        assert "processing_time_seconds" in result

        # Verify mocks were called
        mock_embedding_service.embed_texts.assert_called_once()
        mock_vector_db_service.add_vectors.assert_called_once()

    @pytest.mark.asyncio
    async def test_add_documents_empty(self):
        """Test adding empty document list."""
        # Setup mocks
        mock_vector_db_service = Mock()
        mock_embedding_service = Mock()

        # Create service and test
        service = RAGService(mock_vector_db_service, mock_embedding_service)
        result = await service.add_documents([])

        # Assertions
        assert result["message"] == "No valid documents to process"
        assert result["documents_processed"] == 0
        assert result["chunks_created"] == 0
        assert result["vectors_stored"] == 0

        # Verify mocks were not called
        mock_embedding_service.embed_texts.assert_not_called()
        mock_vector_db_service.add_vectors.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_documents_embedding_failure(self):
        """Test document addition when embedding generation fails."""
        mock_vector_db_service = Mock()
        mock_embedding_service = Mock()

        mock_embedding_service.embed_texts = AsyncMock(return_value=[])

        service = RAGService(mock_vector_db_service, mock_embedding_service)
        documents = [{"text": "Test document.", "document_id": "doc1"}]

        with pytest.raises(Exception, match="Failed to generate embeddings"):
            await service.add_documents(documents)

        mock_embedding_service.embed_texts.assert_called_once()
        mock_vector_db_service.add_vectors.assert_not_called()

    @pytest.mark.asyncio
    async def test_add_documents_db_failure(self):
        """Test document addition when database fails."""
        mock_vector_db_service = Mock()
        mock_embedding_service = Mock()

        mock_embedding_service.embed_texts = AsyncMock(return_value=[[0.1, 0.2, 0.3]])
        mock_vector_db_service.add_vectors = AsyncMock(return_value=False)

        service = RAGService(mock_vector_db_service, mock_embedding_service)
        documents = [{"text": "Test document.", "document_id": "doc1"}]

        with pytest.raises(Exception, match="Failed to add vectors"):
            await service.add_documents(documents)

        mock_embedding_service.embed_texts.assert_called_once()
        mock_vector_db_service.add_vectors.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_stats(self):
        """Test getting service statistics."""
        # Setup mocks
        mock_vector_db_service = Mock()
        mock_embedding_service = Mock()

        # Mock service stats
        mock_vector_db_service.get_stats = AsyncMock(
            return_value={"class_name": "PersonalKnowledge", "object_count": 100}
        )
        mock_embedding_service.get_stats = Mock(
            return_value={"model": "all-MiniLM-L6-v2", "device": "cpu"}
        )

        # Create service and test
        service = RAGService(mock_vector_db_service, mock_embedding_service)
        result = await service.get_stats()

        # Assertions
        assert "rag_service" in result
        assert result["rag_service"]["status"] == "healthy"
        assert "vector_database" in result
        assert result["vector_database"]["class_name"] == "PersonalKnowledge"
        assert "embedding_service" in result
        assert result["embedding_service"]["model"] == "all-MiniLM-L6-v2"

        # Verify mocks were called
        mock_vector_db_service.get_stats.assert_called_once()
        mock_embedding_service.get_stats.assert_called_once()
