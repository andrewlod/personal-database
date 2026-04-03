"""
Text parser script for structuring raw scraped text into knowledge base-ready files.
Cleans, normalizes, and structures text for better chunking and embedding.
"""

import re
import json
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ParsedDocument:
    """Data class representing a parsed document."""
    id: str
    title: str
    content: str
    source_url: Optional[str]
    timestamp: float
    word_count: int
    char_count: int
    language: str = "en"
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class TextParser:
    """Text parser for cleaning and structuring scraped content."""
    
    def __init__(self):
        """Initialize the text parser."""
        # Common boilerplate patterns to remove
        self.boilerplate_patterns = [
            r'©.*?All rights reserved',
            r'Copyright\s+.*?\d{4}',
            r'Privacy Policy|Terms of Service|Cookie Policy',
            r'Subscribe to our newsletter',
            r'Follow us on.*?(Facebook|Twitter|Instagram|LinkedIn)',
            r'Share this article',
            r'Related articles?',
            r'Advertisement|Sponsored Content',
            r'Sign up|Log in|Register',
            r'Comments?\s*\d*',  # Match "Comment" or "Comments" followed by optional numbers
            r'Leave a reply',
            r'Post a comment',
        ]
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = [re.compile(pattern, re.IGNORECASE) 
                                 for pattern in self.boilerplate_patterns]
    
    def parse_file(self, filepath: str) -> Optional[ParsedDocument]:
        """
        Parse a single scraped text file.
        
        Args:
            filepath: Path to the scraped text file
            
        Returns:
            ParsedDocument object or None if parsing failed
        """
        try:
            logger.debug(f"Parsing file: {filepath}")
            
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract metadata from header
            metadata, text_content = self._extract_metadata(content)
            
            # Clean and structure the content
            cleaned_content = self._clean_content(text_content)
            
            # Skip if content is too short after cleaning
            if len(cleaned_content.strip()) < 50:
                logger.warning(f"Content too short after cleaning: {filepath}")
                return None
            
            # Generate unique ID based on content hash
            content_hash = hashlib.md5(cleaned_content.encode()).hexdigest()[:12]
            doc_id = f"doc_{content_hash}"
            
            # Create parsed document
            parsed_doc = ParsedDocument(
                id=doc_id,
                title=metadata.get('title', 'Untitled'),
                content=cleaned_content,
                source_url=metadata.get('url'),
                timestamp=metadata.get('timestamp', datetime.now().timestamp()),
                word_count=len(cleaned_content.split()),
                char_count=len(cleaned_content),
                language=metadata.get('language', 'en'),
                metadata=metadata
            )
            
            logger.debug(f"Parsed document: {parsed_doc.title} ({parsed_doc.word_count} words)")
            return parsed_doc
            
        except Exception as e:
            logger.error(f"Error parsing file {filepath}: {str(e)}")
            return None
    
    def parse_directory(self, directory: str) -> List[ParsedDocument]:
        """
        Parse all text files in a directory.
        
        Args:
            directory: Path to directory containing scraped files
            
        Returns:
            List of successfully parsed documents
        """
        directory_path = Path(directory)
        if not directory_path.exists():
            logger.error(f"Directory not found: {directory}")
            return []
        
        # Find all text files
        text_files = list(directory_path.glob("*.txt"))
        logger.info(f"Found {len(text_files)} text files to parse in {directory}")
        
        parsed_docs = []
        for filepath in text_files:
            parsed_doc = self.parse_file(str(filepath))
            if parsed_doc:
                parsed_docs.append(parsed_doc)
        
        logger.info(f"Successfully parsed {len(parsed_docs)}/{len(text_files)} files")
        return parsed_docs
    
    def save_parsed_documents(self, documents: List[ParsedDocument], 
                            output_dir: str, format: str = "json") -> List[str]:
        """
        Save parsed documents to files.
        
        Args:
            documents: List of parsed documents
            output_dir: Directory to save parsed files
            format: Output format ("json" or "txt")
            
        Returns:
            List of saved file paths
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        saved_files = []
        for doc in documents:
            if format.lower() == "json":
                filename = f"{doc.id}.json"
                filepath = output_path / filename
                
                # Convert to dict for JSON serialization
                doc_dict = asdict(doc)
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(doc_dict, f, indent=2, ensure_ascii=False)
                    
            else:  # txt format
                # Create a readable text file
                filename = f"{doc.id}.txt"
                filepath = output_path / filename
                
                content = f"""Title: {doc.title}
ID: {doc.id}
Source URL: {doc.source_url or 'N/A'}
Timestamp: {doc.timestamp}
Word Count: {doc.word_count}
Character Count: {doc.char_count}
Language: {doc.language}
{'='*60}

{doc.content}
"""
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            saved_files.append(str(filepath))
            logger.debug(f"Saved parsed document to {filepath}")
        
        logger.info(f"Saved {len(saved_files)} parsed documents to {output_dir}")
        return saved_files
    
    def _extract_metadata(self, content: str) -> tuple[Dict[str, Any], str]:
        """
        Extract metadata from the header of scraped content.
        
        Args:
            content: Raw file content with metadata header
            
        Returns:
            Tuple of (metadata_dict, content_text)
        """
        metadata = {}
        content_text = content
        
        lines = content.split('\n')
        content_start = 0
        
        # Look for metadata in the header (before the separator)
        for i, line in enumerate(lines):
            if line.startswith('='*40) or line.startswith('='*50):
                content_start = i + 1
                break
            
            # Parse metadata lines
            if ': ' in line and not line.startswith(' ') and i < 10:  # Only first 10 lines
                key, value = line.split(': ', 1)
                key = key.lower().replace(' ', '_')
                metadata[key] = value.strip()
        
        # Extract the actual content
        content_text = '\n'.join(lines[content_start:]).strip()
        
        # Process specific metadata fields
        if 'timestamp' in metadata:
            try:
                metadata['timestamp'] = float(metadata['timestamp'])
            except ValueError:
                metadata['timestamp'] = None
        
        if 'word_count' in metadata:
            try:
                metadata['word_count'] = int(metadata['word_count'])
            except ValueError:
                pass  # Will be recalculated
        
        # Set defaults
        metadata.setdefault('title', 'Untitled')
        metadata.setdefault('url', None)
        metadata.setdefault('timestamp', None)
        
        return metadata, content_text
    
    def _clean_content(self, content: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            content: Raw text content
            
        Returns:
            Cleaned text content
        """
        if not content:
            return ""
        
        # Remove boilerplate content
        cleaned = content
        for pattern in self.compiled_patterns:
            cleaned = pattern.sub('', cleaned)
        
        # Remove non-printable characters except newlines
        cleaned = ''.join(char for char in cleaned if char.isprintable() or char == '\n')
        
        # Collapse 3+ consecutive newlines into exactly 2
        cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
        
        # Strip trailing whitespace from each line
        lines = [line.rstrip() for line in cleaned.splitlines()]
        cleaned = '\n'.join(lines)
        
        # Remove empty lines that are just whitespace
        cleaned = re.sub(r'\n[ \t]+\n', '\n\n', cleaned)
        
        cleaned = cleaned.strip()
        
        return cleaned


def main():
    """Command line interface for the parser."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse and structure scraped text files')
    parser.add_argument('input', help='File or directory containing scraped text files')
    parser.add_argument('-o', '--output', default='parsed_data', 
                       help='Output directory for parsed files')
    parser.add_argument('-f', '--format', choices=['json', 'txt'], default='json',
                       help='Output format')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize parser
    parser = TextParser()
    
    # Process input
    input_path = Path(args.input)
    if input_path.is_file():
        # Single file
        parsed_doc = parser.parse_file(str(input_path))
        documents = [parsed_doc] if parsed_doc else []
    elif input_path.is_dir():
        # Directory of files
        documents = parser.parse_directory(str(input_path))
    else:
        logger.error(f"Input path not found: {args.input}")
        return 1
    
    if documents:
        saved_files = parser.save_parsed_documents(documents, args.output, args.format)
        print(f"Successfully parsed {len(documents)} documents")
        print(f"Saved {len(saved_files)} files to {args.output}/")
    else:
        print("No documents were successfully parsed")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())