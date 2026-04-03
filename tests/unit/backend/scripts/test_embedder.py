"""
Unit tests for the embedder script.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np
from scripts.embedder import Embedder, VectorDatabase, WeaviateVectorDB, get_vector_database


class TestVectorDatabase:
    """Test cases for the VectorDatabase abstract base class."""
    
    def test_is_abstract(self):
        """Test that VectorDatabase cannot be instantiated directly."""
        with pytest.raises(TypeError):
            VectorDatabase()


class TestWeaviateVectorDB:
    """Test cases for the WeaviateVectorDB class."""
    
    def test_init(self):
        """Test WeaviateVectorDB initialization."""
        db = WeaviateVectorDB(host="localhost", port=8080, scheme="http")
        assert db.host == "localhost"
        assert db.port == 8080
        assert db.scheme == "http"
        assert db.api_key is None
        assert db.client is None
        assert db.collection is None
        assert db.class_name == "PersonalKnowledge"
    
    def test_init_with_api_key(self):
        """Test WeaviateVectorDB initialization with API key."""
        db = WeaviateVectorDB(host="localhost", port=8080, scheme="http", api_key="test-key")
        assert db.api_key == "test-key"
    
    @patch('weaviate.connect_to_local')
    @patch('weaviate.WeaviateClient')
    def test_initialize_success(self, mock_client_class, mock_connect):
        """Test successful Weaviate initialization."""
        mock_client = Mock()
        mock_client.is_ready.return_value = True
        mock_client.collections.exists.return_value = True
        mock_collection = Mock()
        mock_client.collections.get.return_value = mock_collection
        mock_connect.return_value = mock_client
        
        db = WeaviateVectorDB(host="localhost", port=8080, scheme="http")
        result = db.initialize()
        
        assert result is True
        assert db.client == mock_client
        mock_connect.assert_called_once()
        mock_client.is_ready.assert_called_once()
    
    @patch('weaviate.connect_to_local')
    def test_initialize_failure_not_ready(self, mock_connect):
        """Test Weaviate initialization when not ready."""
        mock_client = Mock()
        mock_client.is_ready.return_value = False
        mock_connect.return_value = mock_client
        
        db = WeaviateVectorDB(host="localhost", port=8080, scheme="http")
        result = db.initialize()
        
        assert result is False
        assert db.client == mock_client
    
    @patch('builtins.__import__', side_effect=ImportError("No module named 'weaviate'"))
    def test_initialize_import_error(self, mock_import):
        """Test Weaviate initialization when package not installed."""
        db = WeaviateVectorDB(host="localhost", port=8080, scheme="http")
        result = db.initialize()
        
        assert result is False
    
    @patch('weaviate.connect_to_local')
    def test_initialize_general_exception(self, mock_connect):
        """Test Weaviate initialization with general exception."""
        mock_connect.side_effect = Exception("Connection failed")
        
        db = WeaviateVectorDB(host="localhost", port=8080, scheme="http")
        result = db.initialize()
        
        assert result is False


class TestGetVectorDatabase:
    """Test cases for the get_vector_database factory function."""
    
    def test_get_weaviate_database(self):
        """Test getting Weaviate database instance."""
        db = get_vector_database("weaviate", host="localhost", port=8080)
        assert isinstance(db, WeaviateVectorDB)
        assert db.host == "localhost"
        assert db.port == 8080
    
    def test_get_unsupported_database(self):
        """Test getting unsupported database raises error."""
        with pytest.raises(ValueError, match="Unsupported vector database provider"):
            get_vector_database("unsupported", host="localhost", port=8080)


class TestEmbedder:
    """Test cases for the Embedder class."""
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_init(self, mock_sentence_transformer):
        """Test Embedder initialization."""
        # Mock the sentence transformer model
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        embedder = Embedder(
            model_name="test-model",
            device="cpu",
            vector_db_provider="weaviate",
            vector_db_config={"host": "localhost", "port": 8080}
        )
        
        assert embedder.model_name == "test-model"
        assert embedder.device == "cpu"
        assert embedder.vector_db_provider == "weaviate"
        assert embedder.vector_db_config == {"host": "localhost", "port": 8080}
        assert embedder.model == mock_model
        mock_sentence_transformer.assert_called_once_with("test-model")
        mock_model.to.assert_called_once_with("cpu")
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_init_sentence_transformer_error(self, mock_sentence_transformer):
        """Test Embedder initialization when sentence transformer fails."""
        mock_sentence_transformer.side_effect = Exception("Model loading failed")
        
        with pytest.raises(Exception, match="Model loading failed"):
            Embedder(model_name="test-model", device="cpu")
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_embed_texts(self, mock_sentence_transformer):
        """Test text embedding generation."""
        # Mock the sentence transformer model and its encode method
        mock_model = Mock()
        mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_model.encode.return_value = mock_embeddings
        mock_sentence_transformer.return_value = mock_model
        
        embedder = Embedder(model_name="test-model", device="cpu")
        
        texts = ["First test text", "Second test text"]
        embeddings = embedder.embed_texts(texts)
        
        assert embeddings == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_model.encode.assert_called_once_with(
            texts,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_embed_texts_empty(self, mock_sentence_transformer):
        """Test embedding empty text list."""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        embedder = Embedder(model_name="test-model", device="cpu")
        
        embeddings = embedder.embed_texts([])
        
        assert embeddings == []
        mock_model.encode.assert_not_called()
    
    @patch('sentence_transformers.SentenceTransformer')
    def test_embed_texts_error(self, mock_sentence_transformer):
        """Test embedding generation error handling."""
        mock_model = Mock()
        mock_model.encode.side_effect = Exception("Encoding failed")
        mock_sentence_transformer.return_value = mock_model
        
        embedder = Embedder(model_name="test-model", device="cpu")
        
        embeddings = embedder.embed_texts(["test text"])
        
        assert embeddings == []
    
    @patch('sentence_transformers.SentenceTransformer')
    @patch('scripts.embedder.WeaviateVectorDB')
    def test_initialize_vector_db_success(self, mock_weaviate_db, mock_sentence_transformer):
        """Test successful vector database initialization."""
        # Mock dependencies
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        mock_db_instance = Mock()
        mock_db_instance.initialize.return_value = True
        mock_weaviate_db.return_value = mock_db_instance
        
        embedder = Embedder(
            model_name="test-model",
            device="cpu",
            vector_db_provider="weaviate",
            vector_db_config={"host": "localhost", "port": 8080}
        )
        
        result = embedder.initialize_vector_db()
        
        assert result is True
        assert embedder.vector_db == mock_db_instance
        mock_weaviate_db.assert_called_once_with(
            host="localhost",
            port=8080,
            scheme="http",
            api_key=None
        )
        mock_db_instance.initialize.assert_called_once()
    
    @patch('sentence_transformers.SentenceTransformer')
    @patch('scripts.embedder.WeaviateVectorDB')
    def test_initialize_vector_db_failure(self, mock_weaviate_db, mock_sentence_transformer):
        """Test failed vector database initialization."""
        # Mock dependencies
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        mock_db_instance = Mock()
        mock_db_instance.initialize.return_value = False
        mock_weaviate_db.return_value = mock_db_instance
        
        embedder = Embedder(
            model_name="test-model",
            device="cpu",
            vector_db_provider="weaviate",
            vector_db_config={"host": "localhost", "port": 8080}
        )
        
        result = embedder.initialize_vector_db()
        
        assert result is False
        assert embedder.vector_db == mock_db_instance
    
    @patch('sentence_transformers.SentenceTransformer')
    @patch('scripts.embedder.WeaviateVectorDB')
    def test_add_documents_success(self, mock_weaviate_db, mock_sentence_transformer):
        """Test successful document addition."""
        # Mock dependencies
        mock_model = Mock()
        mock_embeddings = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        mock_model.encode.return_value = mock_embeddings
        mock_sentence_transformer.return_value = mock_model
        
        mock_db_instance = Mock()
        mock_db_instance.add_vectors.return_value = True
        mock_weaviate_db.return_value = mock_db_instance
        
        embedder = Embedder(
            model_name="test-model",
            device="cpu",
            vector_db_provider="weaviate",
            vector_db_config={"host": "localhost", "port": 8080}
        )
        
        # Mock the vector db initialization
        embedder.vector_db = mock_db_instance
        
        chunks = [
            {
                'text': 'First test chunk',
                'document_id': 'doc1',
                'chunk_id': 'chunk1',
                'chunk_index': 0,
                'title': 'Test Doc 1',
                'source_url': 'http://example.com/1',
                'timestamp': 1640995200.0,
                'word_count': 3,
                'metadata': {'category': 'test'}
            },
            {
                'text': 'Second test chunk',
                'document_id': 'doc2',
                'chunk_id': 'chunk2',
                'chunk_index': 0,
                'title': 'Test Doc 2',
                'source_url': 'http://example.com/2',
                'timestamp': 1640995260.0,
                'word_count': 3,
                'metadata': {'category': 'test'}
            }
        ]
        
        result = embedder.add_documents(chunks)
        
        assert result is True
        mock_model.encode.assert_called_once()
        mock_db_instance.add_vectors.assert_called_once()
        
        # Check the arguments passed to add_vectors
        call_args = mock_db_instance.add_vectors.call_args
        vectors = call_args[0][0]  # First positional argument
        payloads = call_args[0][1]  # Second positional argument
        ids = call_args[0][2] if len(call_args[0]) > 2 else None  # Third positional argument
        
        assert vectors == [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        assert len(payloads) == 2
        assert payloads[0]['content'] == 'First test chunk'
        assert payloads[1]['content'] == 'Second test chunk'
        assert ids == ['chunk1', 'chunk2']
    
    @patch('sentence_transformers.SentenceTransformer')
    @patch('scripts.embedder.WeaviateVectorDB')
    def test_add_documents_empty(self, mock_weaviate_db, mock_sentence_transformer):
        """Test adding empty document list."""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        mock_db_instance = Mock()
        mock_weaviate_db.return_value = mock_db_instance
        
        embedder = Embedder(
            model_name="test-model",
            device="cpu",
            vector_db_provider="weaviate",
            vector_db_config={"host": "localhost", "port": 8080}
        )
        
        # Mock the vector db initialization
        embedder.vector_db = mock_db_instance
        
        result = embedder.add_documents([])
        
        assert result is False
        mock_model.encode.assert_not_called()
        mock_db_instance.add_vectors.assert_not_called()
    
    @patch('sentence_transformers.SentenceTransformer')
    @patch('scripts.embedder.WeaviateVectorDB')
    def test_add_documents_no_valid_texts(self, mock_weaviate_db, mock_sentence_transformer):
        """Test adding documents with no valid texts."""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        mock_db_instance = Mock()
        mock_weaviate_db.return_value = mock_db_instance
        
        embedder = Embedder(
            model_name="test-model",
            device="cpu",
            vector_db_provider="weaviate",
            vector_db_config={"host": "localhost", "port": 8080}
        )
        
        # Mock the vector db initialization
        embedder.vector_db = mock_db_instance
        
        chunks = [
            {
                'text': '',  # Empty text
                'document_id': 'doc1',
                'chunk_id': 'chunk1',
                'chunk_index': 0,
                'title': 'Test Doc 1',
                'source_url': 'http://example.com/1',
                'timestamp': 1640995200.0,
                'word_count': 0,
                'metadata': {}
            },
            {
                'text': '   ',  # Whitespace only
                'document_id': 'doc2',
                'chunk_id': 'chunk2',
                'chunk_index': 0,
                'title': 'Test Doc 2',
                'source_url': 'http://example.com/2',
                'timestamp': 1640995260.0,
                'word_count': 0,
                'metadata': {}
            }
        ]
        
        result = embedder.add_documents(chunks)
        
        assert result is False
        mock_model.encode.assert_not_called()
        mock_db_instance.add_vectors.assert_not_called()
    
    @patch('sentence_transformers.SentenceTransformer')
    @patch('scripts.embedder.WeaviateVectorDB')
    def test_search_success(self, mock_weaviate_db, mock_sentence_transformer):
        """Test successful search."""
        # Mock dependencies
        mock_model = Mock()
        mock_embedding = np.array([[0.1, 0.2, 0.3]])
        mock_model.encode.return_value = mock_embedding
        mock_sentence_transformer.return_value = mock_model
        
        mock_db_instance = Mock()
        mock_search_results = [
            {
                'id': 'chunk1',
                'content': 'Test content',
                'document_id': 'doc1',
                'chunk_id': 'chunk1',
                'chunk_index': 0,
                'title': 'Test Doc',
                'source_url': 'http://example.com',
                'timestamp': 1640995200.0,
                'word_count': 3,
                'metadata': {},
                'score': 0.85,
                'distance': 0.15
            }
        ]
        mock_db_instance.search_vectors.return_value = mock_search_results
        mock_weaviate_db.return_value = mock_db_instance
        
        embedder = Embedder(
            model_name="test-model",
            device="cpu",
            vector_db_provider="weaviate",
            vector_db_config={"host": "localhost", "port": 8080}
        )
        
        # Mock the vector db initialization
        embedder.vector_db = mock_db_instance
        
        results = embedder.search("test query", limit=5, score_threshold=0.7)
        
        assert len(results) == 1
        assert results[0]['content'] == 'Test content'
        assert results[0]['score'] == 0.85
        mock_model.encode.assert_called_once()
        mock_db_instance.search_vectors.assert_called_once_with(
            query_vector=[0.1, 0.2, 0.3],
            limit=5,
            score_threshold=0.7
        )
    
    @patch('sentence_transformers.SentenceTransformer')
    @patch('scripts.embedder.WeaviateVectorDB')
    def test_search_empty_query(self, mock_weaviate_db, mock_sentence_transformer):
        """Test search with empty query."""
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        mock_db_instance = Mock()
        mock_weaviate_db.return_value = mock_db_instance
        
        embedder = Embedder(
            model_name="test-model",
            device="cpu",
            vector_db_provider="weaviate",
            vector_db_config={"host": "localhost", "port": 8080}
        )
        
        # Mock the vector db initialization
        embedder.vector_db = mock_db_instance
        
        results = embedder.search("", limit=5, score_threshold=0.7)
        
        assert results == []
        mock_model.encode.assert_not_called()
        mock_db_instance.search_vectors.assert_not_called()
    
    @patch('sentence_transformers.SentenceTransformer')
    @patch('scripts.embedder.WeaviateVectorDB')
    def test_search_error(self, mock_weaviate_db, mock_sentence_transformer):
        """Test search error handling."""
        mock_model = Mock()
        mock_embedding = np.array([[0.1, 0.2, 0.3]])
        mock_model.encode.return_value = mock_embedding
        mock_sentence_transformer.return_value = mock_model
        
        mock_db_instance = Mock()
        mock_db_instance.search_vectors.side_effect = Exception("Search failed")
        mock_weaviate_db.return_value = mock_db_instance
        
        embedder = Embedder(
            model_name="test-model",
            device="cpu",
            vector_db_provider="weaviate",
            vector_db_config={"host": "localhost", "port": 8080}
        )
        
        # Mock the vector db initialization
        embedder.vector_db = mock_db_instance
        
        results = embedder.search("test query", limit=5, score_threshold=0.7)
        
        assert results == []
        mock_model.encode.assert_called_once()
    
    @patch('sentence_transformers.SentenceTransformer')
    @patch('scripts.embedder.WeaviateVectorDB')
    def test_get_stats(self, mock_weaviate_db, mock_sentence_transformer):
        """Test getting statistics."""
        # Mock dependencies
        mock_model = Mock()
        mock_sentence_transformer.return_value = mock_model
        
        mock_db_instance = Mock()
        mock_db_info = {
            'class_name': 'PersonalKnowledge',
            'description': 'Test description',
            'vectorizer': 'none',
            'properties_count': 5,
            'object_count': 100,
            'status': 'healthy'
        }
        mock_db_instance.get_collection_info.return_value = mock_db_info
        mock_weaviate_db.return_value = mock_db_instance
        
        embedder = Embedder(
            model_name="test-model",
            device="cpu",
            vector_db_provider="weaviate",
            vector_db_config={"host": "localhost", "port": 8080}
        )
        
        # Mock the vector db initialization
        embedder.vector_db = mock_db_instance
        
        stats = embedder.get_stats()
        
        assert stats['provider'] == 'weaviate'
        assert stats['model'] == 'test-model'
        assert stats['device'] == 'cpu'
        assert stats['database_info'] == mock_db_info


if __name__ == "__main__":
    pytest.main([__file__])