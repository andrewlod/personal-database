"""
Unit tests for the web scraper script.
"""

from unittest.mock import Mock, patch
import requests
from scripts.scraper import WebScraper


class TestWebScraper:
    """Test cases for the WebScraper class."""

    def test_init(self):
        """Test WebScraper initialization."""
        scraper = WebScraper(delay=2.0, timeout=60)
        assert scraper.delay == 2.0
        assert scraper.timeout == 60
        assert scraper.session is not None

    @patch("requests.Session")
    def test_scrape_url_success(self, mock_session_class):
        """Test successful URL scraping."""
        mock_response = Mock()
        mock_response.content = b"<html><head><title>Test Page</title></head><body><article><p>This is test content that is long enough to pass the minimum content length check of 100 characters in the scraper. We need to add more words to ensure it passes the validation threshold.</p></article></body></html>"
        mock_response.raise_for_status.return_value = None

        mock_session_instance = Mock()
        mock_session_instance.get.return_value = mock_response
        mock_session_class.return_value = mock_session_instance

        scraper = WebScraper()
        result = scraper.scrape_url("http://example.com")

        assert result is not None
        assert result["title"] == "Test Page"
        assert "test content" in result["content"].lower()
        assert result["url"] == "http://example.com"
        assert "timestamp" in result
        assert result["word_count"] > 0

    @patch("requests.Session")
    def test_scrape_url_request_error(self, mock_session_class):
        """Test URL scraping with request error."""
        mock_session_instance = Mock()
        mock_session_instance.get.side_effect = requests.RequestException(
            "Network error"
        )
        mock_session_class.return_value = mock_session_instance

        scraper = WebScraper()
        result = scraper.scrape_url("http://example.com")

        assert result is None

    @patch("requests.Session")
    def test_scrape_url_invalid_url(self, mock_session_class):
        """Test URL scraping with invalid URL."""
        mock_session_instance = Mock()
        mock_session_class.return_value = mock_session_instance

        scraper = WebScraper()
        result = scraper.scrape_url("not-a-url")

        assert result is None

    def test_clean_text(self):
        """Test text cleaning functionality."""
        scraper = WebScraper()

        # Test excessive newlines are collapsed to double newline
        dirty_text = "This is a test\n\n\n\nwith excessive newlines."
        cleaned = scraper._clean_text(dirty_text)
        assert cleaned == "This is a test\n\nwith excessive newlines."

        # Test trailing whitespace on lines is stripped
        dirty_text = "line one   \nline two  \n"
        cleaned = scraper._clean_text(dirty_text)
        assert cleaned == "line one\nline two"

        # Test non-printable character removal
        dirty_text = "Hello\x00\x01World\x02"
        cleaned = scraper._clean_text(dirty_text)
        assert cleaned == "HelloWorld"

    @patch.object(WebScraper, "scrape_url")
    def test_scrape_urls(self, mock_scrape_url):
        """Test scraping multiple URLs."""
        mock_scrape_url.side_effect = [
            {
                "title": "Page 1",
                "content": "Content 1",
                "url": "http://example1.com",
                "timestamp": 123,
                "word_count": 10,
            },
            {
                "title": "Page 2",
                "content": "Content 2",
                "url": "http://example2.com",
                "timestamp": 456,
                "word_count": 20,
            },
            None,
        ]

        scraper = WebScraper(delay=0)
        urls = ["http://example1.com", "http://example2.com", "http://example3.com"]
        results = scraper.scrape_urls(urls)

        assert len(results) == 2
        assert results[0]["title"] == "Page 1"
        assert results[1]["title"] == "Page 2"
        assert mock_scrape_url.call_count == 3

    def test_scrape_from_file_not_found(self):
        """Test scraping from non-existent file."""
        scraper = WebScraper()
        results = scraper.scrape_from_file("/non/existent/file.txt")
        assert results == []
