"""
Dependency providers for the Personal Database API.
"""

from .services.vector_db_service import VectorDBService
from .services.embedding_service import EmbeddingService
from .services.rag_service import RAGService

# Global service instances
_vector_db_service: VectorDBService = None
_embedding_service: EmbeddingService = None
_rag_service: RAGService = None


def get_vector_db_service() -> VectorDBService:
    """Dependency to get vector database service instance."""
    global _vector_db_service
    if _vector_db_service is None:
        _vector_db_service = VectorDBService()
    return _vector_db_service


def get_embedding_service() -> EmbeddingService:
    """Dependency to get embedding service instance."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service


def get_rag_service() -> RAGService:
    """Dependency to get RAG service instance."""
    global _rag_service, _vector_db_service, _embedding_service
    if _rag_service is None:
        # Initialize services if not already done
        if _vector_db_service is None:
            _vector_db_service = VectorDBService()
        if _embedding_service is None:
            _embedding_service = EmbeddingService()
        _rag_service = RAGService(_vector_db_service, _embedding_service)
    return _rag_service