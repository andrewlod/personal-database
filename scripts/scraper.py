"""
Web scraper script for extracting text content from web pages.
Supports single URLs and batch processing from file.
"""

import requests
from bs4 import BeautifulSoup
import re
import json
import time
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class WebScraper:
    """Web scraper for extracting clean text content from web pages."""
    
    def __init__(self, delay: float = 1.0, timeout: int = 30):
        """
        Initialize the web scraper.
        
        Args:
            delay: Delay between requests in seconds
            timeout: Request timeout in seconds
        """
        self.delay = delay
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def scrape_url(self, url: str) -> Optional[Dict[str, str]]:
        """
        Scrape a single URL and extract text content.
        
        Args:
            url: URL to scrape
            
        Returns:
            Dictionary with title, content, url, and timestamp, or None if failed
        """
        try:
            logger.info(f"Scraping URL: {url}")
            
            # Validate URL
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                logger.error(f"Invalid URL: {url}")
                return None
            
            # Make request
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 
                               'aside', 'advertisement', 'ads', '.ad', '#ads']):
                element.decompose()
            
            # Extract title
            title_tag = soup.find('title')
            title = title_tag.get_text().strip() if title_tag else "Untitled"
            
            # Extract main content - try common content selectors first
            content_selectors = [
                'article', '.content', '#content', '.post', '.entry',
                '.article-body', '.story-body', '.main-content', 'main'
            ]
            
            main_content = None
            for selector in content_selectors:
                elements = soup.select(selector)
                if elements:
                    # Take the largest element by text length
                    main_content = max(elements, key=lambda e: len(e.get_text()))
                    break
            
            # Fallback to body if no content found
            if not main_content:
                main_content = soup.find('body') or soup
            
            # Extract text preserving structure
            text = self._extract_structured_text(main_content)
            
            # Clean text
            text = self._clean_text(text)
            
            # Skip if content is too short
            if len(text.strip()) < 100:
                logger.warning(f"Content too short for URL: {url}")
                return None
            
            result = {
                'title': title,
                'content': text,
                'url': url,
                'timestamp': time.time(),
                'word_count': len(text.split())
            }
            
            logger.info(f"Successfully scraped {url}: {len(text)} characters")
            return result
            
        except requests.RequestException as e:
            logger.error(f"Request error scraping {url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Error scraping {url}: {str(e)}")
            return None
    
    def scrape_urls(self, urls: List[str]) -> List[Dict[str, str]]:
        """
        Scrape multiple URLs.
        
        Args:
            urls: List of URLs to scrape
            
        Returns:
            List of successfully scraped documents
        """
        results = []
        for i, url in enumerate(urls):
            result = self.scrape_url(url)
            if result:
                results.append(result)
            
            # Delay between requests (except for last one)
            if i < len(urls) - 1 and self.delay > 0:
                time.sleep(self.delay)
        
        logger.info(f"Scraped {len(results)}/{len(urls)} URLs successfully")
        return results
    
    def scrape_from_file(self, filepath: str) -> List[Dict[str, str]]:
        """
        Scrape URLs from a text file (one URL per line).
        
        Args:
            filepath: Path to file containing URLs
            
        Returns:
            List of successfully scraped documents
        """
        try:
            with open(filepath, 'r') as f:
                urls = [line.strip() for line in f if line.strip()]
            
            logger.info(f"Loaded {len(urls)} URLs from {filepath}")
            return self.scrape_urls(urls)
            
        except FileNotFoundError:
            logger.error(f"File not found: {filepath}")
            return []
        except Exception as e:
            logger.error(f"Error reading file {filepath}: {str(e)}")
            return []
    
    def _extract_structured_text(self, element) -> str:
        """
        Extract text from an HTML element while preserving document structure.
        Headings, paragraphs, list items, and block elements each get their own lines.
        
        Args:
            element: BeautifulSoup element
            
        Returns:
            Text with structural newlines preserved
        """
        block_tags = {
            'p', 'div', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
            'li', 'ul', 'ol', 'dl', 'dt', 'dd',
            'blockquote', 'pre', 'figure', 'figcaption',
            'table', 'tr', 'th', 'td',
            'section', 'article', 'aside', 'main', 'header', 'footer',
            'hr', 'br',
            'details', 'summary',
        }
        
        lines = []
        self._walk_element(element, block_tags, lines)
        return '\n'.join(lines)
    
    def _walk_element(self, element, block_tags: set, lines: List[str]):
        """Recursively walk the DOM, collecting text with structural newlines."""
        from bs4 import NavigableString, Tag
        
        if isinstance(element, NavigableString):
            text = element.strip()
            if text:
                lines.append(text)
            return
        
        if not isinstance(element, Tag):
            return
        
        tag = element.name
        
        if tag in ('script', 'style', 'noscript'):
            return
        
        if tag == 'br':
            lines.append('')
            return
        
        if tag == 'hr':
            lines.append('')
            lines.append('-' * 40)
            lines.append('')
            return
        
        if tag.startswith('h') and tag[1:].isdigit():
            level = int(tag[1])
            prefix = '#' * level + ' '
            inner = self._inline_text(element)
            if inner:
                lines.append('')
                lines.append(f'{prefix}{inner}')
                lines.append('')
            return
        
        if tag == 'li':
            inner = self._inline_text(element)
            if inner:
                lines.append(f'  - {inner}')
            return
        
        if tag == 'blockquote':
            inner_lines = []
            for child in element.children:
                self._walk_element(child, block_tags, inner_lines)
            if inner_lines:
                lines.append('')
                for line in inner_lines:
                    if line.strip():
                        lines.append(f'> {line}')
                    else:
                        lines.append('')
                lines.append('')
            return
        
        if tag in ('pre', 'code'):
            inner = element.get_text()
            if inner:
                lines.append('')
                for line in inner.splitlines():
                    lines.append(line)
                lines.append('')
            return
        
        if tag == 'a':
            href = element.get('href', '')
            inner = self._inline_text(element)
            if inner and href and not href.startswith('#'):
                lines.append(f'{inner} ({href})')
            elif inner:
                lines.append(inner)
            return
        
        is_block = tag in block_tags
        
        if is_block:
            lines.append('')
        
        for child in element.children:
            self._walk_element(child, block_tags, lines)
        
        if is_block:
            lines.append('')
    
    def _inline_text(self, element) -> str:
        """Get inline text from an element, collapsing whitespace."""
        text = element.get_text(separator=' ', strip=True)
        return re.sub(r'\s+', ' ', text).strip()
    
    def _clean_text(self, text: str) -> str:
        """
        Clean extracted text while preserving paragraph structure.
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text with paragraph breaks intact
        """
        # Remove non-printable characters except newlines
        text = ''.join(char for char in text if char.isprintable() or char == '\n')
        
        # Collapse more than two consecutive newlines into exactly two
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Strip trailing whitespace from each line
        lines = [line.rstrip() for line in text.splitlines()]
        text = '\n'.join(lines)
        
        # Strip leading/trailing whitespace of the whole document
        return text.strip()
    
    def save_results(self, results: List[Dict[str, str]], output_dir: str) -> List[str]:
        """
        Save scraped results to individual text files.
        
        Args:
            results: List of scraped documents
            output_dir: Directory to save files
            
        Returns:
            List of saved file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        for i, result in enumerate(results):
            # Create filename from URL or use index
            url = result['url']
            parsed = urlparse(url)
            filename = parsed.netloc.replace('.', '_')
            if not filename:
                filename = f"document_{i+1}"
            
            # Add timestamp to avoid conflicts
            timestamp = int(result['timestamp'])
            filename = f"{filename}_{timestamp}.txt"
            
            filepath = output_path / filename
            
            # Prepare content with metadata
            content = f"""Title: {result['title']}
URL: {result['url']}
Timestamp: {result['timestamp']}
Word Count: {result['word_count']}
{'='*50}

{result['content']}
"""
            
            try:
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
                saved_files.append(str(filepath))
                logger.info(f"Saved scraped content to {filepath}")
            except Exception as e:
                logger.error(f"Error saving file {filepath}: {str(e)}")
        
        return saved_files


def main():
    """Command line interface for the scraper."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Scrape web pages for text content')
    parser.add_argument('input', help='URL or file containing URLs (one per line)')
    parser.add_argument('-o', '--output', default='scraped_data', 
                       help='Output directory for scraped files')
    parser.add_argument('-d', '--delay', type=float, default=1.0,
                       help='Delay between requests in seconds')
    parser.add_argument('-t', '--timeout', type=int, default=30,
                       help='Request timeout in seconds')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize scraper
    scraper = WebScraper(delay=args.delay, timeout=args.timeout)
    
    # Determine if input is URL or file
    if args.input.startswith(('http://', 'https://')):
        # Single URL
        results = scraper.scrape_urls([args.input])
    else:
        # File containing URLs
        results = scraper.scrape_from_file(args.input)
    
    if results:
        saved_files = scraper.save_results(results, args.output)
        print(f"Successfully scraped {len(results)} URLs")
        print(f"Saved {len(saved_files)} files to {args.output}/")
    else:
        print("No content was successfully scraped")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())