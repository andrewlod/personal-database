"""
Query endpoints for the Personal Database RAG API.
Handles user queries and returns RAG-generated responses.
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import time
import logging

from src.api.dependencies import get_rag_service
from src.api.services.rag_service import RAGService

router = APIRouter()
logger = logging.getLogger(__name__)


class QueryRequest(BaseModel):
    """Request model for query operations."""
    question: str = Field(..., min_length=1, description="The question to ask")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of chunks to retrieve")
    score_threshold: Optional[float] = Field(None, ge=0.0, le=1.0, description="Minimum similarity score")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="LLM temperature")
    max_tokens: Optional[int] = Field(None, ge=1, le=4000, description="Maximum tokens for LLM response")
    include_sources: bool = Field(True, description="Whether to include source chunks in response")


class QueryResponse(BaseModel):
    """Response model for query operations."""
    answer: str
    question: str
    processing_time_seconds: float
    sources: Optional[List[Dict[str, Any]]] = None
    usage: Optional[Dict[str, Any]] = None
    model_used: str


@router.post("/")  # Removed Depends from function signature
async def query_knowledge_base(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Query the knowledge base using RAG (Retrieval-Augmented Generation).
    
    This endpoint retrieves relevant chunks from the vector database and uses
    an LLM to generate a response based on the retrieved context.
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing query: '{request.question[:100]}...'")
        
        # Validate question
        if not request.question.strip():
            raise HTTPException(status_code=400, detail="Question cannot be empty")
        
        # Process the query through the RAG service
        result = await rag_service.query(
            question=request.question,
            top_k=request.top_k,
            score_threshold=request.score_threshold,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            include_sources=request.include_sources
        )
        
        processing_time = time.time() - start_time
        
        # Prepare response
        response = QueryResponse(
            answer=result.get('answer', ''),
            question=request.question,
            processing_time_seconds=round(processing_time, 2),
            sources=result.get('sources'),
            usage=result.get('usage'),
            model_used=result.get('model_used', 'unknown')
        )
        
        logger.info(
            f"Query processed successfully in {processing_time:.2f}s. "
            f"Answer length: {len(response.answer)} chars"
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to process query: {str(e)}"
        )


@router.get("/suggest")  # Removed Depends from function signature
async def get_query_suggestions(
    partial_query: str,
    limit: int = 5,
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Get query suggestions based on partial input.
    
    This is a placeholder implementation - in a full system, this might
    use query logs or semantic search to suggest similar questions.
    """
    # For now, return some generic suggestions
    suggestions = [
        "What is the main topic of the documents?",
        "Can you summarize the key points?",
        "What are the most important findings?",
        "How does this relate to [topic]?",
        "What are the implications of this information?"
    ]
    
    # Filter suggestions based on partial query (simple matching)
    if partial_query.strip():
        filtered = [s for s in suggestions if partial_query.lower() in s.lower()]
        return {"suggestions": filtered[:limit]}
    else:
        return {"suggestions": suggestions[:limit]}


@router.get("/stats")  # Removed Depends from function signature
async def get_query_stats(
    rag_service: RAGService = Depends(get_rag_service)
):
    """Get statistics about the query service and underlying systems."""
    try:
        stats = await rag_service.get_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting query stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get query statistics: {str(e)}"
        )