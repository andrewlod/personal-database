"""
Pytest configuration and shared fixtures for testing.
"""

import pytest
import tempfile
import sys
from pathlib import Path

# Add project root to Python path so scripts can be imported
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


# Test fixtures
@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_text():
    """Sample text for testing."""
    return """
    This is a sample text for testing purposes.
    It contains multiple sentences and paragraphs.
    
    This is the second paragraph with some additional information.
    We can use this to test various text processing functions.
    """


@pytest.fixture
def sample_document():
    """Sample document dictionary for testing."""
    return {
        "text": "This is a sample document for testing.",
        "document_id": "test_doc_001",
        "title": "Test Document",
        "source_url": "https://example.com/test",
        "timestamp": 1640995200.0,
        "word_count": 8,
        "metadata": {"author": "Test Author", "category": "testing"},
    }


@pytest.fixture
def sample_chunks():
    """Sample chunks for testing."""
    return [
        {
            "id": "test_doc_001_chunk_0000",
            "text": "This is a sample document for testing.",
            "chunk_index": 0,
            "document_id": "test_doc_001",
            "start_char": 0,
            "end_char": 42,
            "token_count": 8,
            "metadata": {
                "title": "Test Document",
                "source_url": "https://example.com/test",
                "timestamp": 1640995200.0,
                "word_count": 8,
            },
        }
    ]
