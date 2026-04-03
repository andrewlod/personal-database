"""
Vector database service for the Personal Database API.
Handles connections and operations with the vector database.
"""

import logging
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add the project root to Python path so we can import from scripts
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from scripts.embedder import get_vector_database

logger = logging.getLogger(__name__)


class VectorDBService:
    """Service for vector database operations."""
    
    def __init__(self):
        """Initialize the vector database service."""
        self.vector_db = None
        self.initialized = False
        logger.info("VectorDBService initialized")
    
    async def initialize(self) -> bool:
        """
        Initialize the vector database connection.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Get vector database configuration from environment or defaults
            # For now, we'll use defaults that match our docker-compose setup
            self.vector_db = get_vector_database(
                provider="weaviate",
                host="localhost",
                port=8080,
                scheme="http",
                api_key=None
            )
            
            # Initialize the connection
            success = self.vector_db.initialize()
            self.initialized = success
            
            if success:
                logger.info("Vector database initialized successfully")
            else:
                logger.error("Failed to initialize vector database")
                
            return success
            
        except Exception as e:
            logger.error(f"Error initializing vector database: {e}")
            self.initialized = False
            return False
    
    async def add_vectors(self, vectors: List[List[float]], 
                         payloads: List[Dict[str, Any]],
                         ids: Optional[List[str]] = None) -> bool:
        """
        Add vectors to the database.
        
        Args:
            vectors: List of embedding vectors
            payloads: List of metadata payloads
            ids: Optional list of custom IDs
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized or not self.vector_db:
            logger.error("Vector database not initialized")
            return False
        
        try:
            return self.vector_db.add_vectors(vectors, payloads, ids)
        except Exception as e:
            logger.error(f"Error adding vectors to database: {e}")
            return False
    
    async def search_vectors(self, query_vector: List[float], 
                            limit: int = 10,
                            score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Search for similar vectors.
        
        Args:
            query_vector: Query embedding vector
            limit: Maximum number of results
            score_threshold: Minimum similarity score
            
        Returns:
            List of search results with payloads and scores
        """
        if not self.initialized or not self.vector_db:
            logger.error("Vector database not initialized")
            return []
        
        try:
            return self.vector_db.search_vectors(query_vector, limit, score_threshold)
        except Exception as e:
            logger.error(f"Error searching vectors: {e}")
            return []
    
    async def delete_vectors(self, ids: List[str]) -> bool:
        """
        Delete vectors by ID.
        
        Args:
            ids: List of vector IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        if not self.initialized or not self.vector_db:
            logger.error("Vector database not initialized")
            return False
        
        try:
            return self.vector_db.delete_vectors(ids)
        except Exception as e:
            logger.error(f"Error deleting vectors: {e}")
            return False
    
    async def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection/index.
        
        Returns:
            Dictionary with collection information
        """
        if not self.initialized or not self.vector_db:
            logger.error("Vector database not initialized")
            return {"error": "Vector database not initialized"}
        
        try:
            return self.vector_db.get_collection_info()
        except Exception as e:
            logger.error(f"Error getting collection info: {e}")
            return {"error": str(e)}
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check the health of the vector database connection.
        
        Returns:
            Dictionary with health status
        """
        if not self.initialized or not self.vector_db:
            return {"status": "unhealthy", "error": "Vector database not initialized"}
        
        try:
            info = self.vector_db.get_collection_info()
            if "error" in info:
                return {"status": "unhealthy", "error": info["error"]}
            else:
                return {
                    "status": "healthy",
                    "class_name": info.get("class_name", "unknown"),
                    "object_count": info.get("object_count", 0),
                    "status_detail": info.get("status", "unknown")
                }
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# Singleton instance
vector_db_service = VectorDBService()