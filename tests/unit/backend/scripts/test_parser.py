"""
Unit tests for the text parser script.
"""

import pytest
from scripts.parser import TextParser, ParsedDocument


class TestTextParser:
    """Test cases for the TextParser class."""
    
    def test_init(self):
        """Test TextParser initialization."""
        parser = TextParser()
        assert parser is not None
        assert hasattr(parser, 'boilerplate_patterns')
        assert hasattr(parser, 'compiled_patterns')
    
    def test_clean_content(self):
        """Test text content cleaning."""
        parser = TextParser()
        
        # Test boilerplate removal
        dirty_text = """
        This is the main content.
        
        © 2023 All rights reserved
        Privacy Policy | Terms of Service
        Subscribe to our newsletter
        
        More content here.
        """
        cleaned = parser._clean_content(dirty_text)
        assert "© 2023 All rights reserved" not in cleaned
        assert "Privacy Policy" not in cleaned
        assert "Subscribe to our newsletter" not in cleaned
        assert "This is the main content." in cleaned
        assert "More content here." in cleaned
        
        # Test whitespace normalization - newlines preserved, excess collapsed
        messy_text = "Too   many   spaces\n\n\n\nand newlines"
        cleaned = parser._clean_content(messy_text)
        assert cleaned == "Too   many   spaces\n\nand newlines"
        
        # Test empty content
        assert parser._clean_content("") == ""
        assert parser._clean_content("   \n\n   ") == ""
    
    def test_extract_metadata(self):
        """Test metadata extraction from file content."""
        parser = TextParser()
        
        content = """Title: Test Document
URL: http://example.com
Timestamp: 1640995200
Word Count: 10

This is the actual content.
It has multiple lines.
"""
        metadata, text_content = parser._extract_metadata(content)
        
        assert metadata['title'] == 'Test Document'
        assert metadata['url'] == 'http://example.com'
        assert metadata['timestamp'] == 1640995200.0
        assert metadata['word_count'] == 10
        assert "This is the actual content." in text_content
        assert "It has multiple lines." in text_content
    
    def test_extract_metadata_defaults(self):
        """Test metadata extraction with missing fields."""
        parser = TextParser()
        
        content = """Some random content
without proper metadata headers
"""
        metadata, text_content = parser._extract_metadata(content)
        
        assert metadata['title'] == 'Untitled'  # Default
        assert metadata['url'] is None
        assert metadata['timestamp'] is None
        assert "Some random content" in text_content
    
    def test_parse_file_not_found(self):
        """Test parsing non-existent file."""
        parser = TextParser()
        result = parser.parse_file('/non/existent/file.txt')
        assert result is None
    
    def test_parse_file_empty_content(self, temp_dir):
        """Test parsing file with insufficient content after cleaning."""
        parser = TextParser()
        
        # Create a file with mostly boilerplate
        file_path = temp_dir / "test.txt"
        file_path.write_text("""
        © 2023 All rights reserved
        Privacy Policy
        Terms of Service
        """)
        
        result = parser.parse_file(str(file_path))
        assert result is None  # Should be filtered out as too short
    
    def test_parse_file_valid_content(self, temp_dir):
        """Test parsing file with valid content."""
        parser = TextParser()
        
        # Create a test file
        file_path = temp_dir / "test.txt"
        content = """Title: Test Document
URL: http://example.com
Timestamp: 1640995200
Word Count: 5

This is a test document.
It has sufficient content for processing.
"""
        file_path.write_text(content)
        
        result = parser.parse_file(str(file_path))
        
        assert result is not None
        assert isinstance(result, ParsedDocument)
        assert result.title == "Test Document"
        assert result.source_url == "http://example.com"
        assert result.timestamp == 1640995200.0
        assert result.word_count > 0
        assert "test document" in result.content.lower()
    
    def test_save_parsed_documents_json(self, temp_dir):
        """Test saving parsed documents in JSON format."""
        parser = TextParser()
        
        # Create sample documents
        docs = [
            ParsedDocument(
                id="doc1",
                title="Test Document 1",
                content="This is the first test document.",
                source_url="http://example.com/1",
                timestamp=1640995200.0,
                word_count=7,
                char_count=35,
                language="en",
                metadata={"category": "test"}
            ),
            ParsedDocument(
                id="doc2",
                title="Test Document 2",
                content="This is the second test document with different content.",
                source_url="http://example.com/2",
                timestamp=1640995260.0,
                word_count=10,
                char_count=59,
                language="en",
                metadata={"category": "test"}
            )
        ]
        
        output_dir = temp_dir / "output"
        saved_files = parser.save_parsed_documents(docs, str(output_dir), format="json")
        
        assert len(saved_files) == 2
        assert (output_dir / "doc1.json").exists()
        assert (output_dir / "doc2.json").exists()
        
        # Check content of saved files
        import json
        with open(output_dir / "doc1.json", 'r') as f:
            saved_doc = json.load(f)
        assert saved_doc['title'] == "Test Document 1"
        assert saved_doc['content'] == "This is the first test document."
        assert saved_doc['metadata']['category'] == "test"


class TestParsedDocument:
    """Test cases for the ParsedDocument dataclass."""
    
    def test_parsed_document_creation(self):
        """Test ParsedDocument creation."""
        doc = ParsedDocument(
            id="test123",
            title="Test Title",
            content="Test content here.",
            source_url="http://example.com",
            timestamp=1640995200.0,
            word_count=4,
            char_count=20,
            language="en",
            metadata={"key": "value"}
        )
        
        assert doc.id == "test123"
        assert doc.title == "Test Title"
        assert doc.content == "Test content here."
        assert doc.source_url == "http://example.com"
        assert doc.timestamp == 1640995200.0
        assert doc.word_count == 4
        assert doc.char_count == 20
        assert doc.language == "en"
        assert doc.metadata == {"key": "value"}
    
    def test_parsed_document_default_metadata(self):
        """Test ParsedDocument with default metadata."""
        doc = ParsedDocument(
            id="test123",
            title="Test Title",
            content="Test content here.",
            source_url=None,
            timestamp=1640995200.0,
            word_count=4,
            char_count=20
        )
        
        assert doc.metadata == {}