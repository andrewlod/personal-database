"""
Unit tests for the text chunker script.
"""

import pytest
from scripts.chunker import (
    CharacterChunker, TokenChunker, SemanticChunker, PatternChunker,
    get_chunker, chunk_document, TextChunk
)


class TestCharacterChunker:
    """Test cases for the CharacterChunker."""
    
    def test_init(self):
        """Test CharacterChunker initialization."""
        chunker = CharacterChunker(chunk_size=100, overlap=20)
        assert chunker.chunk_size == 100
        assert chunker.overlap == 20
    
    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        chunker = CharacterChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk_text("", "doc1")
        assert chunks == []
    
    def test_chunk_short_text(self):
        """Test chunking text shorter than chunk size."""
        chunker = CharacterChunker(chunk_size=100, overlap=20)
        text = "This is a short text."
        chunks = chunker.chunk_text(text, "doc1")
        
        assert len(chunks) == 1
        assert chunks[0].text == text
        assert chunks[0].chunk_index == 0
        assert chunks[0].document_id == "doc1"
    
    def test_chunk_long_text(self):
        """Test chunking text longer than chunk size."""
        chunker = CharacterChunker(chunk_size=50, overlap=10)
        text = "This is a longer text that should be split into multiple chunks based on character count."
        chunks = chunker.chunk_text(text, "doc1")
        
        assert len(chunks) > 1
        # Check that chunks have correct positions
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
            assert chunk.document_id == "doc1"
            # Check overlap (except for first chunk)
            if i > 0:
                prev_chunk = chunks[i-1]
                # There should be some overlap or continuity
                assert chunk.start_char <= prev_chunk.end_char
    
    def test_chunk_with_metadata(self):
        """Test chunking preserves metadata."""
        chunker = CharacterChunker(chunk_size=50, overlap=10)
        text = "This is a test text for checking metadata preservation."
        metadata = {"source": "test", "author": "tester"}
        chunks = chunker.chunk_text(text, "doc1", metadata)
        
        assert len(chunks) >= 1
        for chunk in chunks:
            assert chunk.metadata["source"] == "test"
            assert chunk.metadata["author"] == "tester"


class TestTokenChunker:
    """Test cases for the TokenChunker."""
    
    def test_init(self):
        """Test TokenChunker initialization."""
        chunker = TokenChunker(chunk_size=50, overlap=10)
        assert chunker.chunk_size == 50
        assert chunker.overlap == 10
    
    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        chunker = TokenChunker(chunk_size=50, overlap=10)
        chunks = chunker.chunk_text("", "doc1")
        assert chunks == []
    
    def test_chunk_short_text(self):
        """Test chunking text shorter than chunk size."""
        chunker = TokenChunker(chunk_size=50, overlap=10)
        text = "This is a short text."
        chunks = chunker.chunk_text(text, "doc1")
        
        assert len(chunks) == 1
        assert chunks[0].text == text


class TestPatternChunker:
    """Test cases for the PatternChunker."""
    
    def test_init(self):
        """Test PatternChunker initialization."""
        chunker = PatternChunker(chunk_size=100, overlap=20)
        assert chunker.chunk_size == 100
        assert chunker.overlap == 20
        assert len(chunker.separators) > 0
    
    def test_init_with_custom_separators(self):
        """Test PatternChunker with custom separators."""
        separators = ["\n\n", "\n"]
        chunker = PatternChunker(chunk_size=100, overlap=20, separators=separators)
        assert chunker.separators == separators
    
    def test_chunk_empty_text(self):
        """Test chunking empty text."""
        chunker = PatternChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk_text("", "doc1")
        assert chunks == []
    
    def test_chunk_paragraphs(self):
        """Test chunking text with paragraph separators."""
        chunker = PatternChunker(chunk_size=100, overlap=20)
        text = "First paragraph.\n\nSecond paragraph with more content.\n\nThird paragraph."
        chunks = chunker.chunk_text(text, "doc1")
        
        assert len(chunks) >= 1
        # Should respect paragraph boundaries when possible
        for chunk in chunks:
            assert chunk.text.strip() != ""


class TestChunkerFactory:
    """Test cases for the chunker factory functions."""
    
    def test_get_chunker_character(self):
        """Test getting character chunker."""
        chunker = get_chunker("character", 50, 10)
        assert isinstance(chunker, CharacterChunker)
    
    def test_get_chunker_token(self):
        """Test getting token chunker."""
        chunker = get_chunker("token", 50, 10)
        assert isinstance(chunker, TokenChunker)
    
    def test_get_chunker_semantic(self):
        """Test getting semantic chunker."""
        chunker = get_chunker("semantic", 5, 1)
        assert isinstance(chunker, SemanticChunker)
    
    def test_get_chunker_pattern(self):
        """Test getting pattern chunker."""
        chunker = get_chunker("pattern", 50, 10)
        assert isinstance(chunker, PatternChunker)
    
    def test_get_chunker_invalid(self):
        """Test getting invalid chunker raises error."""
        with pytest.raises(ValueError):
            get_chunker("invalid", 50, 10)
    
    def test_chunk_document_convenience_function(self):
        """Test the chunk_document convenience function."""
        text = "This is a test text for the convenience function."
        chunks = chunk_document(
            text=text,
            document_id="test_doc",
            strategy="character",
            chunk_size=20,
            overlap=5
        )
        
        assert len(chunks) >= 1
        assert all(isinstance(chunk, TextChunk) for chunk in chunks)
        assert chunks[0].document_id == "test_doc"


class TestTextChunk:
    """Test cases for the TextChunk dataclass."""
    
    def test_text_chunk_creation(self):
        """Test TextChunk creation."""
        chunk = TextChunk(
            id="test_chunk_001",
            text="This is a test chunk.",
            chunk_index=0,
            document_id="test_doc",
            start_char=0,
            end_char=25,
            token_count=5,
            metadata={"source": "test"}
        )
        
        assert chunk.id == "test_chunk_001"
        assert chunk.text == "This is a test chunk."
        assert chunk.chunk_index == 0
        assert chunk.document_id == "test_doc"
        assert chunk.start_char == 0
        assert chunk.end_char == 25
        assert chunk.token_count == 5
        assert chunk.metadata == {"source": "test"}
    
    def test_text_chunk_to_dict(self):
        """Test TextChunk to_dict conversion."""
        chunk = TextChunk(
            id="test_chunk_001",
            text="This is a test chunk.",
            chunk_index=0,
            document_id="test_doc",
            start_char=0,
            end_char=25,
            token_count=5,
            metadata={"source": "test"}
        )
        
        chunk_dict = chunk.to_dict()
        assert chunk_dict["id"] == "test_chunk_001"
        assert chunk_dict["text"] == "This is a test chunk."
        assert chunk_dict["chunk_index"] == 0
        assert chunk_dict["document_id"] == "test_doc"
        assert chunk_dict["start_char"] == 0
        assert chunk_dict["end_char"] == 25
        assert chunk_dict["token_count"] == 5
        assert chunk_dict["metadata"] == {"source": "test"}