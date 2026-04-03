"""
Embedding script for generating embeddings and storing them in a vector database.
Supports multiple vector database backends via abstraction layer.
"""

import logging
import json
import os
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger(__name__)


class VectorDatabase(ABC):
    """Abstract base class for vector database operations."""
    
    @abstractmethod
    def initialize(self, **kwargs) -> bool:
        """
        Initialize the vector database connection.
        
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def add_vectors(self, vectors: List[List[float]], 
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
        pass
    
    @abstractmethod
    def search_vectors(self, query_vector: List[float], 
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
        pass
    
    @abstractmethod
    def delete_vectors(self, ids: List[str]) -> bool:
        """
        Delete vectors by ID.
        
        Args:
            ids: List of vector IDs to delete
            
        Returns:
            True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Get information about the collection/index.
        
        Returns:
            Dictionary with collection information
        """
        pass


class WeaviateVectorDB(VectorDatabase):
    """Weaviate vector database implementation (v4 API)."""
    
    def __init__(self, host: str = "localhost", port: int = 8080,
                 scheme: str = "http", api_key: Optional[str] = None):
        """
        Initialize Weaviate client.
        
        Args:
            host: Weaviate host
            port: Weaviate port
            scheme: Connection scheme (http/https)
            api_key: API key for authentication (if required)
        """
        self.host = host
        self.port = port
        self.scheme = scheme
        self.api_key = api_key
        self.client = None
        self.collection = None
        self.class_name = "PersonalKnowledge"
    
    def initialize(self, **kwargs) -> bool:
        """Initialize Weaviate connection and schema."""
        try:
            import weaviate
            
            if self.api_key:
                self.client = weaviate.connect_to_local(
                    host=self.host,
                    port=self.port,
                    grpc_port=50051,
                    auth_credentials=weaviate.auth.AuthApiKey(api_key=self.api_key),
                )
            else:
                self.client = weaviate.connect_to_local(
                    host=self.host,
                    port=self.port,
                    grpc_port=50051,
                )
            
            if not self.client.is_ready():
                logger.error("Weaviate is not ready")
                return False
            
            self._create_schema_if_not_exists(kwargs.get('class_name', self.class_name))
            
            self.collection = self.client.collections.get(self.class_name)
            
            url = f"{self.scheme}://{self.host}:{self.port}"
            logger.info(f"Weaviate initialized successfully at {url}")
            return True
            
        except ImportError:
            logger.error("weaviate-client package not installed")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate: {str(e)}")
            return False
    
    def _create_schema_if_not_exists(self, class_name: str):
        """Create Weaviate collection for the knowledge base."""
        from weaviate.classes.config import Property, DataType
        
        self.class_name = class_name
        
        if self.client.collections.exists(self.class_name):
            logger.info(f"Weaviate collection '{self.class_name}' already exists")
            return
        
        try:
            self.client.collections.create(
                name=self.class_name,
                description="Personal knowledge base documents",
                vectorizer_config=None,
                properties=[
                    Property(name="content", data_type=DataType.TEXT,
                             description="The text content of the chunk"),
                    Property(name="document_id", data_type=DataType.TEXT,
                             description="ID of the source document"),
                    Property(name="chunk_id", data_type=DataType.TEXT,
                             description="ID of this specific chunk"),
                    Property(name="chunk_index", data_type=DataType.INT,
                             description="Index of this chunk within the document"),
                    Property(name="title", data_type=DataType.TEXT,
                             description="Title of the source document"),
                    Property(name="source_url", data_type=DataType.TEXT,
                             description="URL of the source document (if applicable)"),
                    Property(name="timestamp", data_type=DataType.NUMBER,
                             description="Timestamp when the document was processed"),
                    Property(name="word_count", data_type=DataType.INT,
                             description="Number of words in the chunk"),
                    Property(name="metadata_json", data_type=DataType.TEXT,
                             description="Additional metadata as JSON string"),
                ]
            )
            logger.info(f"Weaviate collection '{self.class_name}' created successfully")
        except Exception as e:
            logger.error(f"Failed to create Weaviate collection: {str(e)}")
            raise
    
    def add_vectors(self, vectors: List[List[float]], 
                   payloads: List[Dict[str, Any]],
                   ids: Optional[List[str]] = None) -> bool:
        """Add vectors to Weaviate using v4 batch API."""
        import json
        import uuid as uuid_module
        try:
            if len(vectors) != len(payloads):
                logger.error("Vectors and payloads must have the same length")
                return False
            
            if ids is None:
                ids = [f"id_{i}" for i in range(len(vectors))]
            elif len(ids) != len(vectors):
                logger.error("IDs must have the same length as vectors")
                return False
            
            collection = self.collection or self.client.collections.get(self.class_name)
            
            with self.client.batch.fixed_size(batch_size=100, concurrent_requests=2) as batch:
                for i, (vector, payload, doc_id) in enumerate(zip(vectors, payloads, ids)):
                    properties = {}
                    
                    # Flatten metadata from nested dict into top-level properties
                    nested_meta = payload.get("metadata", {})
                    if isinstance(nested_meta, dict):
                        for k, v in nested_meta.items():
                            if k not in ("metadata",):
                                payload.setdefault(k, v)
                    
                    if "content" in payload:
                        properties["content"] = payload["content"]
                    elif "text" in payload:
                        properties["content"] = payload["text"]
                    else:
                        properties["content"] = ""
                    
                    properties["document_id"] = payload.get("document_id", "")
                    
                    if "chunk_id" in payload:
                        properties["chunk_id"] = payload["chunk_id"]
                    elif "id" in payload:
                        properties["chunk_id"] = payload["id"]
                    else:
                        properties["chunk_id"] = doc_id
                    
                    properties["chunk_index"] = payload.get("chunk_index", 0)
                    properties["title"] = payload.get("title", "")
                    properties["source_url"] = payload.get("source_url", "") or payload.get("url", "")
                    properties["timestamp"] = payload.get("timestamp", 0)
                    properties["word_count"] = payload.get("word_count", 0)
                    
                    extra_metadata = {}
                    for key in ("start_char", "end_char", "token_count", "category", "language"):
                        if key in payload:
                            extra_metadata[key] = payload[key]
                    
                    if "metadata" in payload:
                        if isinstance(payload["metadata"], dict):
                            extra_metadata.update(payload["metadata"])
                        else:
                            extra_metadata["raw"] = str(payload["metadata"])
                    
                    properties["metadata_json"] = json.dumps(extra_metadata) if extra_metadata else "{}"
                    
                    try:
                        obj_uuid = uuid_module.UUID(doc_id)
                    except (ValueError, AttributeError):
                        obj_uuid = uuid_module.uuid5(uuid_module.NAMESPACE_DNS, doc_id)
                    
                    batch.add_object(
                        collection=self.class_name,
                        properties=properties,
                        vector=vector,
                        uuid=obj_uuid,
                    )
                    
                    if (i + 1) % 50 == 0:
                        logger.debug(f"Processed {i + 1}/{len(vectors)} vectors")
            
            if self.client.batch.failed_objects:
                for failed in self.client.batch.failed_objects:
                    logger.warning(f"Failed object: {failed.message}")
                logger.warning(f"Batch had {len(self.client.batch.failed_objects)} failed objects")
            
            logger.info(f"Successfully added {len(vectors)} vectors to Weaviate")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add vectors to Weaviate: {str(e)}")
            return False
    
    def search_vectors(self, query_vector: List[float], 
                      limit: int = 10,
                      score_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """Search for similar vectors in Weaviate using v4 query API."""
        import json
        from weaviate.classes.query import MetadataQuery
        try:
            collection = self.collection or self.client.collections.get(self.class_name)
            
            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=limit,
                include_vector=False,
                return_metadata=MetadataQuery(distance=True, certainty=True),
            )
            
            search_results = []
            for obj in response.objects:
                certainty = obj.metadata.certainty if obj.metadata.certainty is not None else 0.0
                distance = obj.metadata.distance if obj.metadata.distance is not None else 1.0
                score = certainty
                
                if score >= score_threshold:
                    props = obj.properties
                    metadata_raw = props.get("metadata_json", "{}")
                    try:
                        metadata = json.loads(metadata_raw) if metadata_raw else {}
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}
                    
                    result_item = {
                        'id': props.get('chunk_id', ''),
                        'content': props.get('content', ''),
                        'document_id': props.get('document_id', ''),
                        'chunk_id': props.get('chunk_id', ''),
                        'chunk_index': props.get('chunk_index', 0),
                        'title': props.get('title', ''),
                        'source_url': props.get('source_url', ''),
                        'timestamp': props.get('timestamp', 0),
                        'word_count': props.get('word_count', 0),
                        'metadata': metadata,
                        'score': score,
                        'distance': distance
                    }
                    search_results.append(result_item)
            
            logger.debug(f"Weaviate search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Failed to search vectors in Weaviate: {str(e)}")
            return []
    
    def delete_vectors(self, ids: List[str]) -> bool:
        """Delete vectors by ID from Weaviate using v4 API."""
        import uuid as uuid_module
        try:
            collection = self.collection or self.client.collections.get(self.class_name)
            
            for doc_id in ids:
                try:
                    obj_uuid = uuid_module.UUID(doc_id)
                except (ValueError, AttributeError):
                    obj_uuid = uuid_module.uuid5(uuid_module.NAMESPACE_DNS, doc_id)
                collection.data.delete_by_id(obj_uuid)
            
            logger.info(f"Deleted {len(ids)} vectors from Weaviate")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete vectors from Weaviate: {str(e)}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the Weaviate collection."""
        try:
            if not self.client.collections.exists(self.class_name):
                return {'error': f'Collection {self.class_name} not found'}
            
            collection = self.collection or self.client.collections.get(self.class_name)
            
            count = len(collection)
            
            config = collection.config.get()
            
            return {
                'class_name': self.class_name,
                'description': config.description or '',
                'vectorizer': config.vectorizer_config.config.class_ if config.vectorizer_config else 'none',
                'properties_count': len(config.properties),
                'object_count': count,
                'status': 'healthy' if self.client.is_ready() else 'unhealthy'
            }
                
        except Exception as e:
            logger.error(f"Failed to get collection info: {str(e)}")
            return {'error': str(e)}


# Factory function for getting vector database instances
def get_vector_database(provider: str = "weaviate", **kwargs) -> VectorDatabase:
    """
    Get a vector database instance.
    
    Args:
        provider: Database provider ('weaviate', 'pinecone', etc.)
        **kwargs: Provider-specific configuration
        
    Returns:
        VectorDatabase instance
    """
    provider = provider.lower()
    
    if provider == 'weaviate':
        return WeaviateVectorDB(
            host=kwargs.get('host', 'localhost'),
            port=kwargs.get('port', 8080),
            scheme=kwargs.get('scheme', 'http'),
            api_key=kwargs.get('api_key')
        )
    else:
        raise ValueError(f"Unsupported vector database provider: {provider}")


class Embedder:
    """Main embedder class for generating embeddings and storing in vector database."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", 
                 device: str = "cpu",
                 vector_db_provider: str = "weaviate",
                 vector_db_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the embedder.
        
        Args:
            model_name: Sentence transformer model name
            device: Device to run model on ('cpu' or 'cuda')
            vector_db_provider: Vector database provider
            vector_db_config: Configuration for vector database
        """
        self.model_name = model_name
        self.device = device
        self.vector_db_provider = vector_db_provider
        self.vector_db_config = vector_db_config or {}
        
        # Initialize sentence transformer model
        self._init_model()
        
        # Initialize vector database
        self.vector_db = get_vector_database(vector_db_provider, **self.vector_db_config)
        
        logger.info(f"Embedder initialized with model {model_name} on {device}")
    
    def _init_model(self):
        """Initialize the sentence transformer model."""
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)
            self.model.to(self.device)
            logger.info(f"Loaded sentence transformer model: {self.model_name}")
        except ImportError:
            logger.error("sentence-transformers package not installed")
            raise
        except Exception as e:
            logger.error(f"Failed to load sentence transformer model: {str(e)}")
            raise
    
    def initialize_vector_db(self, **kwargs) -> bool:
        """
        Initialize the vector database connection.
        
        Args:
            **kwargs: Additional arguments for vector database initialization
            
        Returns:
            True if successful, False otherwise
        """
        return self.vector_db.initialize(**kwargs)
    
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        try:
            # Generate embeddings
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True  # Normalize for cosine similarity
            )
            
            # Convert to list of lists
            embeddings_list = embeddings.tolist()
            
            logger.debug(f"Generated embeddings for {len(texts)} texts")
            return embeddings_list
            
        except Exception as e:
            logger.error(f"Failed to generate embeddings: {str(e)}")
            return []
    
    def add_documents(self, chunks: List[Dict[str, Any]]) -> bool:
        """
        Add document chunks to the vector database.
        
        Args:
            chunks: List of chunk dictionaries with 'text' and metadata
            
        Returns:
            True if successful, False otherwise
        """
        if not chunks:
            logger.warning("No chunks provided for embedding")
            return False
        
        try:
            # Extract texts for embedding
            texts = [chunk.get('text', '') for chunk in chunks]
            
            # Filter out empty texts
            valid_chunks = []
            valid_texts = []
            for chunk, text in zip(chunks, texts):
                if text.strip():
                    valid_chunks.append(chunk)
                    valid_texts.append(text)
            
            if not valid_texts:
                logger.warning("No valid texts to embed")
                return False
            
            # Generate embeddings
            embeddings = self.embed_texts(valid_texts)
            
            if not embeddings:
                logger.error("Failed to generate embeddings")
                return False
            
            # Prepare payloads for vector database
            payloads = []
            ids = []
            
            for i, chunk in enumerate(valid_chunks):
                # Create payload with all metadata
                payload = {
                    'content': chunk.get('text', ''),
                    'document_id': chunk.get('document_id', ''),
                    'chunk_id': chunk.get('chunk_id', ''),
                    'chunk_index': chunk.get('chunk_index', 0),
                    'title': chunk.get('title', ''),
                    'source_url': chunk.get('source_url', ''),
                    'timestamp': chunk.get('timestamp', 0),
                    'word_count': chunk.get('word_count', 0),
                    'metadata': chunk.get('metadata', {})
                }
                
                payloads.append(payload)
                ids.append(chunk.get('chunk_id', f"chunk_{i}"))
            
            # Add to vector database
            success = self.vector_db.add_vectors(embeddings, payloads, ids)
            
            if success:
                logger.info(f"Successfully added {len(valid_chunks)} chunks to vector database")
            else:
                logger.error("Failed to add chunks to vector database")
            
            return success
            
        except Exception as e:
            logger.error(f"Error adding documents to vector database: {str(e)}")
            return False
    
    def search(self, query: str, limit: int = 5, 
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
        if not query.strip():
            return []
        
        try:
            # Embed the query
            query_embeddings = self.embed_texts([query])
            
            if not query_embeddings:
                logger.error("Failed to embed query")
                return []
            
            # Search vector database
            results = self.vector_db.search_vectors(
                query_vector=query_embeddings[0],
                limit=limit,
                score_threshold=score_threshold
            )
            
            logger.debug(f"Search for '{query}' returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Error searching vector database: {str(e)}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            info = self.vector_db.get_collection_info()
            return {
                'provider': self.vector_db_provider,
                'model': self.model_name,
                'device': self.device,
                'database_info': info
            }
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return {'error': str(e)}


def embed_directory(input_dir: str, 
                   model_name: str = "all-MiniLM-L6-v2",
                   device: str = "cpu",
                   vector_db_provider: str = "weaviate",
                   vector_db_config: Optional[Dict[str, Any]] = None) -> bool:
    """
    Convenience function to embed all JSON chunk files in a directory.
    
    Args:
        input_dir: Directory containing chunk JSON files
        model_name: Sentence transformer model name
        device: Device to run model on
        vector_db_provider: Vector database provider
        vector_db_config: Vector database configuration
        
    Returns:
        True if successful, False otherwise
    """
    from pathlib import Path
    import json
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    try:
        # Initialize embedder
        embedder = Embedder(
            model_name=model_name,
            device=device,
            vector_db_provider=vector_db_provider,
            vector_db_config=vector_db_config or {}
        )
        
        # Initialize vector database
        if not embedder.initialize_vector_db():
            logger.error("Failed to initialize vector database")
            return False
        
        # Find all JSON files
        input_path = Path(input_dir)
        if not input_path.exists():
            logger.error(f"Input directory not found: {input_dir}")
            return False
        
        json_files = list(input_path.glob("*.json"))
        logger.info(f"Found {len(json_files)} JSON files to process")
        
        if not json_files:
            logger.warning("No JSON files found to process")
            return False
        
        # Process files in batches
        batch_size = 50
        all_chunks = []
        
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                
                # Validate required fields
                if 'text' not in chunk_data:
                    logger.warning(f"Skipping file {json_file}: missing 'text' field")
                    continue
                
                # Ensure we have required IDs
                if 'chunk_id' not in chunk_data:
                    chunk_data['chunk_id'] = json_file.stem
                
                all_chunks.append(chunk_data)
                
            except Exception as e:
                logger.error(f"Error reading file {json_file}: {str(e)}")
        
        if not all_chunks:
            logger.error("No valid chunks found to process")
            return False
        
        # Add chunks to vector database
        success = embedder.add_documents(all_chunks)
        
        if success:
            stats = embedder.get_stats()
            logger.info(f"Embedding completed successfully: {stats}")
        else:
            logger.error("Embedding failed")
        
        return success
        
    except Exception as e:
        logger.error(f"Error in embed_directory: {str(e)}")
        return False


def main():
    """Command line interface for the embedder."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate embeddings and store in vector database')
    parser.add_argument('input', help='Directory containing chunk JSON files')
    parser.add_argument('-m', '--model', default='all-MiniLM-L6-v2',
                       help='Sentence transformer model name')
    parser.add_argument('-d', '--device', choices=['cpu', 'cuda'], default='cpu',
                       help='Device to run model on')
    parser.add_argument('-p', '--provider', choices=['weaviate'], default='weaviate',
                       help='Vector database provider')
    parser.add_argument('--host', default='localhost',
                       help='Weaviate host')
    parser.add_argument('--port', type=int, default=8080,
                       help='Weaviate port')
    parser.add_argument('--scheme', choices=['http', 'https'], default='http',
                       help='Weaviate scheme')
    parser.add_argument('--api-key', default='',
                       help='Weaviate API key (if required)')
    parser.add_argument('--class-name', default='PersonalKnowledge',
                       help='Weaviate class name')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Prepare vector database config
    vector_db_config = {
        'host': args.host,
        'port': args.port,
        'scheme': args.scheme,
        'api_key': args.api_key if args.api_key else None
    }
    
    # Run embedding
    success = embed_directory(
        input_dir=args.input,
        model_name=args.model,
        device=args.device,
        vector_db_provider=args.provider,
        vector_db_config=vector_db_config
    )
    
    if success:
        print("Embedding completed successfully")
        return 0
    else:
        print("Embedding failed")
        return 1


if __name__ == "__main__":
    exit(main())