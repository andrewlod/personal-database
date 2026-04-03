"""
Main FastAPI application for the Personal Database RAG system.
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import service classes
from .services.vector_db_service import VectorDBService
from .services.embedding_service import EmbeddingService
from .services.rag_service import RAGService
from . import dependencies as deps

# Import API routers
from .routes import ingest, query, health

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Personal Database RAG API",
    description="A Retrieval-Augmented Generation system for personal knowledge management",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # React frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router, prefix="/api/health", tags=["health"])
app.include_router(ingest.router, prefix="/api/ingest", tags=["ingestion"])
app.include_router(query.router, prefix="/api/query", tags=["query"])

# Startup event to initialize services
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    try:
        deps._vector_db_service = VectorDBService()
        await deps._vector_db_service.initialize()
        
        deps._embedding_service = EmbeddingService()
        await deps._embedding_service.initialize()
        
        deps._rag_service = RAGService(deps._vector_db_service, deps._embedding_service)
        
        logger.info("All services initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize services: {e}")

# Root endpoint
@app.get("/", tags=["root"])
async def root():
    """Root endpoint with basic information."""
    return {
        "message": "Personal Database RAG API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }

if __name__ == "__main__":
    # Run the application
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )