"""
Embedding service for the Personal Database API.
Handles text embedding generation using sentence transformers.
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the project root to Python path so we can import from src.scripts
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.embedder import Embedder

logger = logging.getLogger(__name__)


class EmbeddingService:
    """Service for generating text embeddings."""
    
    def __init__(self):
        """Initialize the embedding service."""
        self.embedder = None
        self.initialized = False
        logger.info("EmbeddingService initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize the embedding model.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Initialize embedder with default settings
            # These should match what's used in the embedder script
            self.embedder = Embedder(
                model_name="all-MiniLM-L6-v2",
                device="cpu",
                vector_db_provider="weaviate",  # We'll initialize this separately if needed
                vector_db_config={
                    "host": "localhost",
                    "port": 8080,
                    "scheme": "http",
                    "api_key": None
                }
            )
            
            # Initialize the vector database connection (needed for some operations)
            db_success = self.embedder.initialize_vector_db()
            
            self.initialized = True  # We consider the service initialized if the embedder is ready
            
            if self.initialized:
                logger.info("Embedding service initialized successfully")
                if not db_success:
                    logger.warning("Embedding service initialized but vector database connection failed")
            else:
                logger.error("Failed to initialize embedding service")
                
            return self.initialized
            
        except Exception as e:
            logger.error(f"Error initializing embedding service: {e}")
            self.initialized = False
            return False
    
    async def embed_query(self, query: str) -> Optional[List[float]]:
        """
        Generate embedding for a single query.
        
        Args:
            query: Text query to embed
            
        Returns:
            Embedding vector or None if failed
        """
        if not self.initialized or not self.embedder:
            logger.error("Embedding service not initialized")
            return None
        
        try:
            embeddings = self.embedder.embed_texts([query])
            if embeddings and len(embeddings) > 0:
                return embeddings[0]
            return None
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            return None
    
    async def embed_texts(self, texts: List[str]) -> Optional[List[List[float]]]:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors or None if failed
        """
        if not self.initialized or not self.embedder:
            logger.error("Embedding service not initialized")
            return None
        
        try:
            return self.embedder.embed_texts(texts)
        except Exception as e:
            logger.error(f"Error embedding texts: {e}")
            return None
    
    async def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Add document chunks to the vector database.
        
        Args:
            chunks: List of chunk dictionaries with 'text' and metadata
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized or not self.embedder:
            logger.error("Embedding service not initialized")
            return False
        
        try:
            return self.embedder.add_documents(chunks)
        except Exception as e:
            logger.error(f"Error adding documents: {e}")
            return False
    
    async def search(self, query: str, limit: int = 5, 
                     score_threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Search for similar documents using a text query.
        
        Args:
            query: Text query to search for
            limit: Maximum number of results
            score_threshold: Minimum similarity score (0-1)
            
        Returns:
            List of search results
        """
        if not self.initialized or not self.embedder:
            logger.error("Embedding service not initialized")
            return []
        
        try:
            return self.embedder.search(query, limit, score_threshold)
        except Exception as e:
            logger.error(f"Error searching: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the embedding service.
        
        Returns:
            Dictionary with service statistics
        """
        if not self.initialized or not self.embedder:
            return {"status": "uninitialized", "error": "Embedding service not initialized"}
        
        try:
            return self.embedder.get_stats()
        except Exception as e:
            logger.error(f"Error getting embedding stats: {e}")
            return {"status": "error", "error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the embedding service.
        
        Returns:
            Dictionary with health status
        """
        if not self.initialized or not self.embedder:
            return {"status": "unhealthy", "error": "Embedding service not initialized"}
        
        try:
            stats = self.get_stats()
            if "error" in stats:
                return {"status": "unhealthy", "error": stats["error"]}
            else:
                return {
                    "status": "healthy",
                    "model": stats.get("model", "unknown"),
                    "device": stats.get("device", "unknown"),
                    "provider": stats.get("provider", "unknown")
                }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# Singleton instance
embedding_service = EmbeddingService()