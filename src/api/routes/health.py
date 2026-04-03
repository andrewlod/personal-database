"""
Health check endpoints for the Personal Database RAG API.
"""

from fastapi import APIRouter, Depends
from src.api.services.vector_db_service import VectorDBService
from src.api.services.embedding_service import EmbeddingService
from src.api.services.rag_service import RAGService
from src.api.dependencies import get_vector_db_service, get_embedding_service, get_rag_service

router = APIRouter()


@router.get("/")
async def health_check(
    vector_db_service: VectorDBService = Depends(get_vector_db_service),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Health check endpoint to verify all services are operational.
    
    Returns:
        Dictionary with health status of all services
    """
    try:
        # Check vector database connection
        vector_db_status = await vector_db_service.health_check()
        
        # Check embedding service
        embedding_status = await embedding_service.health_check()
        
        # Overall status
        overall_status = "healthy" if (
            vector_db_status.get("status") == "healthy" and 
            embedding_status.get("status") == "healthy"
        ) else "unhealthy"
        
        return {
            "status": overall_status,
            "services": {
                "vector_database": vector_db_status,
                "embedding_service": embedding_status,
                "rag_service": {"status": "healthy" if overall_status == "healthy" else "unhealthy"}
            }
        }
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "services": {
                "vector_database": {"status": "unknown", "error": str(e)},
                "embedding_service": {"status": "unknown", "error": str(e)},
                "rag_service": {"status": "unknown", "error": str(e)}
            }
        }


@router.get("/ping")
async def ping():
    """Simple ping endpoint for basic connectivity check."""
    return {"message": "pong"}