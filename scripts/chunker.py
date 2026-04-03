"""
Text chunker script implementing multiple chunking strategies:
- Character-based (fixed size with overlap)
- Token-based (using tokenizer from embedding model)
- Semantic-based (using sentence transformers for coherence)
- Pattern-based (headers, paragraphs, sections)
"""

import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import hashlib
from abc import ABC, abstractmethod
from pathlib import Path

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    SEMANTIC_CHUNKING_AVAILABLE = True
except ImportError:
    SEMANTIC_CHUNKING_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TextChunk:
    """Data class representing a text chunk."""
    id: str
    text: str
    chunk_index: int
    document_id: str
    start_char: int
    end_char: int
    token_count: int
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'text': self.text,
            'chunk_index': self.chunk_index,
            'document_id': self.document_id,
            'start_char': self.start_char,
            'end_char': self.end_char,
            'token_count': self.token_count,
            'metadata': self.metadata
        }


class BaseChunker(ABC):
    """Abstract base class for chunkers."""
    
    def __init__(self, chunk_size: int, overlap: int):
        """
        Initialize chunker.
        
        Args:
            chunk_size: Target size for chunks (meaning depends on strategy)
            overlap: Overlap between consecutive chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    @abstractmethod
    def chunk_text(self, text: str, document_id: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """
        Chunk text into pieces.
        
        Args:
            text: Text to chunk
            document_id: ID of source document
            metadata: Additional metadata to include with chunks
            
        Returns:
            List of TextChunk objects
        """
        pass
    
    def _create_chunk_id(self, document_id: str, chunk_index: int) -> str:
        """Create a unique chunk ID."""
        return f"{document_id}_chunk_{chunk_index:04d}"
    
    def _estimate_token_count(self, text: str) -> int:
        """
        Rough estimate of token count (4 chars per token average).
        Override in subclasses for more accurate estimates.
        """
        return max(1, len(text) // 4)


class CharacterChunker(BaseChunker):
    """Character-based chunker with configurable size and overlap."""
    
    def chunk_text(self, text: str, document_id: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """
        Chunk text by character count with overlap.
        
        Args:
            text: Text to chunk
            document_id: ID of source document
            metadata: Additional metadata
            
        Returns:
            List of TextChunk objects
        """
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # Adjust end to not cut words if possible
            if end < len(text):
                # Look for word boundary within overlap distance
                search_start = max(start, end - self.overlap)
                # Find last space in the search window
                last_space = text.rfind(' ', search_start, end)
                if last_space > start:
                    end = last_space
            
            # Extract chunk text
            chunk_text = text[start:end].strip()
            
            if chunk_text:  # Only add non-empty chunks
                chunk_id = self._create_chunk_id(document_id, chunk_index)
                
                chunk = TextChunk(
                    id=chunk_id,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    document_id=document_id,
                    start_char=start,
                    end_char=end,
                    token_count=self._estimate_token_count(chunk_text),
                    metadata=metadata or {}
                )
                
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position (with overlap)
            if end >= len(text):
                break
            start = end - self.overlap
            if start < 0:
                start = 0
        
        logger.debug(f"Created {len(chunks)} character-based chunks for document {document_id}")
        return chunks


class TokenChunker(BaseChunker):
    """Token-based chunker using approximate token counting."""
    
    def __init__(self, chunk_size: int, overlap: int):
        """
        Initialize token chunker.
        
        Args:
            chunk_size: Target token count per chunk
            overlap: Overlap in tokens between chunks
        """
        super().__init__(chunk_size, overlap)
        # Try to load a tokenizer for more accurate counting
        self.tokenizer = None
        try:
            import tiktoken
            self.tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 tokenizer
        except ImportError:
            logger.warning("tiktoken not available, using character-based token estimation")
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text))
        else:
            # Fallback: rough estimate
            return max(1, len(text) // 4)
    
    def chunk_text(self, text: str, document_id: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """
        Chunk text by token count with overlap.
        
        Args:
            text: Text to chunk
            document_id: ID of source document
            metadata: Additional metadata
            
        Returns:
            List of TextChunk objects
        """
        if not text.strip():
            return []
        
        chunks = []
        sentences = self._split_into_sentences(text)
        
        current_chunk = []
        current_tokens = 0
        start_char = 0
        chunk_index = 0
        
        for sentence in sentences:
            sentence_tokens = self._count_tokens(sentence)
            
            # If adding this sentence would exceed chunk size, finalize current chunk
            if current_tokens + sentence_tokens > self.chunk_size and current_chunk:
                # Create chunk from accumulated sentences
                chunk_text = ' '.join(current_chunk)
                end_char = start_char + len(chunk_text)
                
                chunk_id = self._create_chunk_id(document_id, chunk_index)
                
                chunk = TextChunk(
                    id=chunk_id,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    document_id=document_id,
                    start_char=start_char,
                    end_char=end_char,
                    token_count=current_tokens,
                    metadata=metadata or {}
                )
                
                chunks.append(chunk)
                chunk_index += 1
                
                # Handle overlap: keep some sentences from the end
                overlap_tokens = 0
                overlap_sentences = []
                for sent in reversed(current_chunk):
                    sent_tokens = self._count_tokens(sent)
                    if overlap_tokens + sent_tokens <= self.overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_tokens += sent_tokens
                    else:
                        break
                
                # Start new chunk with overlap
                current_chunk = overlap_sentences[:]
                current_tokens = overlap_tokens
                start_char = end_char - len(' '.join(overlap_sentences)) if overlap_sentences else end_char
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_tokens += sentence_tokens
        
        # Handle remaining sentences
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            end_char = start_char + len(chunk_text)
            
            chunk_id = self._create_chunk_id(document_id, chunk_index)
            
            chunk = TextChunk(
                id=chunk_id,
                text=chunk_text,
                chunk_index=chunk_index,
                document_id=document_id,
                start_char=start_char,
                end_char=end_char,
                token_count=current_tokens,
                metadata=metadata or {}
            )
            
            chunks.append(chunk)
        
        logger.debug(f"Created {len(chunks)} token-based chunks for document {document_id}")
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - can be improved with NLTK or spaCy
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


class SemanticChunker(BaseChunker):
    """Semantic-based chunker using sentence embeddings for coherence.
    
    Uses a sliding window approach: computes cosine similarity between
    consecutive sentences and splits when similarity drops below the
    threshold. A hard token-count cap prevents runaway chunks.
    """
    
    MAX_TOKENS_PER_CHUNK = 500
    MIN_SENTENCES_PER_CHUNK = 3
    
    def __init__(self, chunk_size: int = 5, overlap: int = 1, 
                 similarity_threshold: float = 0.5):
        """
        Initialize semantic chunker.
        
        Args:
            chunk_size: Target number of sentences per chunk (default 5)
            overlap: Number of overlapping sentences between chunks
            similarity_threshold: Cosine similarity below which a split is triggered
        """
        super().__init__(chunk_size, overlap)
        self.similarity_threshold = similarity_threshold
        self.encoder = None
        
        if SEMANTIC_CHUNKING_AVAILABLE:
            try:
                self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
                logger.info("Loaded sentence transformer model for semantic chunking")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")
                self.encoder = None
        else:
            logger.warning("Sentence transformers not available, falling back to token chunking")
    
    def chunk_text(self, text: str, document_id: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """
        Chunk text semantically based on sentence similarity.
        
        Splits are triggered when consecutive-sentence cosine similarity drops
        below the threshold AND the current chunk has at least MIN_SENTENCES_PER_CHUNK
        sentences. A hard token cap (MAX_TOKENS_PER_CHUNK) forces a split regardless.
        Small segments that don't meet the minimum are merged with adjacent chunks
        rather than dropped.
        """
        if not text.strip():
            return []
        
        if not self.encoder:
            logger.warning("Semantic chunking not available, falling back to token chunking")
            token_chunker = TokenChunker(self.chunk_size * 4, self.overlap)
            return token_chunker.chunk_text(text, document_id, metadata)
        
        try:
            sentences = self._split_into_sentences(text)
            if len(sentences) <= self.MIN_SENTENCES_PER_CHUNK:
                if self._estimate_token_count(text) > self.MAX_TOKENS_PER_CHUNK:
                    token_chunker = TokenChunker(self.MAX_TOKENS_PER_CHUNK, self.overlap)
                    return token_chunker.chunk_text(text, document_id, metadata)
                return [self._make_chunk(text, document_id, 0, metadata)]
            
            embeddings = self.encoder.encode(sentences)
            similarities = [
                float(cosine_similarity([embeddings[i]], [embeddings[i+1]])[0][0])
                for i in range(len(sentences) - 1)
            ]
            
            breakpoints = []
            for i, sim in enumerate(similarities):
                if sim < self.similarity_threshold:
                    breakpoints.append(i + 1)
            
            # Build raw segments from breakpoints
            raw_segments = []
            prev_end = 0
            for bp in breakpoints:
                raw_segments.append(sentences[prev_end:bp])
                prev_end = bp
            raw_segments.append(sentences[prev_end:])
            
            # Merge small segments with adjacent ones instead of dropping them
            merged_segments = self._merge_small_segments(raw_segments)
            
            # Convert segments to chunks
            chunks = []
            chunk_index = 0
            prev_char_end = 0
            
            for segment_sentences in merged_segments:
                chunk_text = ' '.join(segment_sentences)
                token_count = self._estimate_token_count(chunk_text)
                char_start = text.find(segment_sentences[0], prev_char_end)
                char_end = char_start + len(chunk_text)
                
                if token_count > self.MAX_TOKENS_PER_CHUNK and len(segment_sentences) > self.MIN_SENTENCES_PER_CHUNK:
                    sub_chunks = self._split_by_tokens(segment_sentences, document_id, chunk_index, metadata)
                    chunks.extend(sub_chunks)
                    chunk_index += len(sub_chunks)
                else:
                    chunks.append(self._make_chunk(chunk_text, document_id, chunk_index, metadata,
                                                   char_start, char_end))
                    chunk_index += 1
                
                prev_char_end = char_end
            
            if not chunks:
                return [self._make_chunk(text, document_id, 0, metadata)]
            
            logger.debug(f"Created {len(chunks)} semantic-based chunks for document {document_id}")
            return chunks
            
        except Exception as e:
            logger.error(f"Error in semantic chunking: {e}, falling back to token chunking")
            token_chunker = TokenChunker(self.chunk_size * 4, self.overlap)
            return token_chunker.chunk_text(text, document_id, metadata)
    
    def _merge_small_segments(self, segments: List[List[str]]) -> List[List[str]]:
        """Merge segments smaller than MIN_SENTENCES_PER_CHUNK with adjacent segments.
        
        Strategy: merge small segments with the larger neighbor. If both neighbors
        exist, merge with the one that results in fewer total tokens.
        """
        if not segments:
            return segments
        
        merged = []
        for i, seg in enumerate(segments):
            if len(seg) >= self.MIN_SENTENCES_PER_CHUNK:
                merged.append(seg)
            else:
                # Small segment - merge with best neighbor
                left_count = len(merged[-1]) if merged else float('inf')
                right_count = len(segments[i+1]) if i+1 < len(segments) else float('inf')
                
                if left_count <= right_count and merged:
                    merged[-1].extend(seg)
                elif i+1 < len(segments):
                    segments[i+1] = seg + segments[i+1]
                elif merged:
                    merged[-1].extend(seg)
                else:
                    merged.append(seg)
        
        return merged
    
    def _split_by_tokens(self, sentences: List[str], document_id: str,
                         start_index: int, metadata: Optional[Dict[str, Any]] = None) -> List:
        """Split a group of sentences into chunks that fit within MAX_TOKENS_PER_CHUNK."""
        text = ' '.join(sentences)
        token_chunker = TokenChunker(self.MAX_TOKENS_PER_CHUNK, self.overlap)
        return token_chunker.chunk_text(text, document_id, metadata)
    
    def _make_chunk(self, text: str, document_id: str, index: int,
                    metadata: Optional[Dict[str, Any]] = None,
                    start_char: int = 0, end_char: int = 0):
        """Create a TextChunk with semantic-chunking metadata."""
        chunk_id = self._create_chunk_id(document_id, index)
        if end_char == 0:
            end_char = len(text)
        return TextChunk(
            id=chunk_id,
            text=text,
            chunk_index=index,
            document_id=document_id,
            start_char=start_char,
            end_char=end_char,
            token_count=self._estimate_token_count(text),
            metadata={**(metadata or {}), 'chunking_method': 'semantic'}
        )
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Use regex for sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]


class PatternChunker(BaseChunker):
    """Pattern-based chunker using document structure (headers, paragraphs, etc.)."""
    
    def __init__(self, chunk_size: int, overlap: int, 
                 separators: Optional[List[str]] = None):
        """
        Initialize pattern chunker.
        
        Args:
            chunk_size: Target size for chunks (in characters by default)
            overlap: Overlap between chunks
            separators: List of separators to respect (e.g., ['\n\n', '\n', '. '])
        """
        super().__init__(chunk_size, overlap)
        self.separators = separators or ["\n\n", "\n", ". ", "! ", "? ", "; "]
        # Sort separators by length (descending) to try larger splits first
        self.separators.sort(key=len, reverse=True)
    
    def chunk_text(self, text: str, document_id: str, 
                   metadata: Optional[Dict[str, Any]] = None) -> List[TextChunk]:
        """
        Chunk text respecting structural patterns.
        
        Args:
            text: Text to chunk
            document_id: ID of source document
            metadata: Additional metadata
            
        Returns:
            List of TextChunk objects
        """
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(text):
            # Determine ideal end position
            ideal_end = start + self.chunk_size
            
            # If we're at the end of text, take what remains
            if ideal_end >= len(text):
                chunk_text = text[start:].strip()
                if chunk_text:
                    chunk_id = self._create_chunk_id(document_id, chunk_index)
                    chunk = TextChunk(
                        id=chunk_id,
                        text=chunk_text,
                        chunk_index=chunk_index,
                        document_id=document_id,
                        start_char=start,
                        end_char=len(text),
                        token_count=self._estimate_token_count(chunk_text),
                        metadata=metadata or {}
                    )
                    chunks.append(chunk)
                break
            
            # Look for the best separator before or near the ideal end
            best_split = ideal_end
            found_separator = False
            
            # Search backwards from ideal_end to find a good separator
            search_start = max(start, ideal_end - 200)  # Don't search too far back
            search_text = text[search_start:ideal_end]
            
            for separator in self.separators:
                # Look for separator in the search window
                pos = search_text.rfind(separator)
                if pos != -1:
                    # Found a separator, calculate actual position
                    actual_pos = search_start + pos + len(separator)
                    # Only use if it's not too far from our target
                    if actual_pos >= search_start and actual_pos <= ideal_end + 50:
                        best_split = actual_pos
                        found_separator = True
                        break
            
            # If no good separator found, look forward for one
            if not found_separator:
                search_end = min(len(text), ideal_end + 200)
                search_text = text[ideal_end:search_end]
                
                for separator in self.separators:
                    pos = search_text.find(separator)
                    if pos != -1:
                        actual_pos = ideal_end + pos + len(separator)
                        if actual_pos <= search_end:
                            best_split = actual_pos
                            found_separator = True
                            break
            
            # Extract chunk
            chunk_text = text[start:best_split].strip()
            
            if chunk_text:
                chunk_id = self._create_chunk_id(document_id, chunk_index)
                
                chunk = TextChunk(
                    id=chunk_id,
                    text=chunk_text,
                    chunk_index=chunk_index,
                    document_id=document_id,
                    start_char=start,
                    end_char=best_split,
                    token_count=self._estimate_token_count(chunk_text),
                    metadata={**(metadata or {}), 'chunking_method': 'pattern'}
                )
                
                chunks.append(chunk)
                chunk_index += 1
            
            # Move start position with overlap
            if best_split >= len(text):
                break
            start = best_split - self.overlap
            if start < 0:
                start = 0
        
        logger.debug(f"Created {len(chunks)} pattern-based chunks for document {document_id}")
        return chunks


def get_chunker(strategy: str, chunk_size: int, overlap: int, 
                **kwargs) -> BaseChunker:
    """
    Factory function to get a chunker instance.
    
    Args:
        strategy: Chunking strategy ('character', 'token', 'semantic', 'pattern')
        chunk_size: Size parameter for chunks
        overlap: Overlap between chunks
        **kwargs: Additional strategy-specific parameters
        
    Returns:
        BaseChunker instance
    """
    strategy = strategy.lower()
    
    if strategy == 'character':
        return CharacterChunker(chunk_size, overlap)
    elif strategy == 'token':
        return TokenChunker(chunk_size, overlap)
    elif strategy == 'semantic':
        similarity_threshold = kwargs.get('similarity_threshold', 0.7)
        return SemanticChunker(chunk_size, overlap, similarity_threshold)
    elif strategy == 'pattern':
        separators = kwargs.get('separators', ["\n\n", "\n", ". ", "! ", "? ", "; "])
        return PatternChunker(chunk_size, overlap, separators)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")


def chunk_document(text: str, document_id: str, 
                   strategy: str = "semantic",
                   chunk_size: int = 100,
                   overlap: int = 20,
                   metadata: Optional[Dict[str, Any]] = None,
                   **kwargs) -> List[TextChunk]:
    """
    Convenience function to chunk a document.
    
    Args:
        text: Text to chunk
        document_id: ID of source document
        strategy: Chunking strategy to use
        chunk_size: Size parameter for chunks
        overlap: Overlap between chunks
        metadata: Additional metadata
        **kwargs: Additional strategy-specific parameters
        
    Returns:
        List of TextChunk objects
    """
    chunker = get_chunker(strategy, chunk_size, overlap, **kwargs)
    return chunker.chunk_text(text, document_id, metadata)


def save_chunks(chunks: List[TextChunk], output_dir: str) -> List[str]:
    """
    Save chunks to JSON files.
    
    Args:
        chunks: List of TextChunk objects
        output_dir: Directory to save chunk files
        
    Returns:
        List of saved file paths
    """
    from pathlib import Path
    import json
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    for chunk in chunks:
        filename = f"{chunk.id}.json"
        filepath = output_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(chunk.to_dict(), f, indent=2, ensure_ascii=False)
        
        saved_files.append(str(filepath))
    
    logger.info(f"Saved {len(saved_files)} chunks to {output_dir}")
    return saved_files


def main():
    """Command line interface for the chunker."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Chunk text documents using various strategies')
    parser.add_argument('input', help='File or directory containing text files to chunk')
    parser.add_argument('-o', '--output', default='chunked_data', 
                       help='Output directory for chunk files')
    parser.add_argument('-s', '--strategy', choices=['character', 'token', 'semantic', 'pattern'],
                       default='semantic', help='Chunking strategy to use')
    parser.add_argument('-c', '--chunk-size', type=int, default=100,
                       help='Target chunk size (meaning depends on strategy)')
    parser.add_argument('--overlap', type=int, default=20,
                       help='Overlap between chunks')
    parser.add_argument('--similarity-threshold', type=float, default=0.7,
                       help='Similarity threshold for semantic chunking (0.0-1.0)')
    parser.add_argument('--separators', type=str, 
                       help='Comma-separated list of separators for pattern chunking')
    parser.add_argument('-v', '--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse separators if provided
    separators = None
    if args.separators:
        separators = [s.strip() for s in args.separators.split(',')]
    
    # Prepare kwargs for chunker
    kwargs = {}
    if args.strategy == 'semantic':
        kwargs['similarity_threshold'] = args.similarity_threshold
    elif args.strategy == 'pattern' and separators:
        kwargs['separators'] = separators
    
    # Determine input type
    input_path = Path(args.input)
    if input_path.is_file():
        files_to_process = [input_path]
    elif input_path.is_dir():
        files_to_process = list(input_path.glob("*.txt")) + list(input_path.glob("*.json"))
    else:
        logger.error(f"Input path not found: {args.input}")
        return 1
    
    if not files_to_process:
        logger.error(f"No text files found in {args.input}")
        return 1
    
    # Process each file
    all_chunks = []
    for filepath in files_to_process:
        try:
            # Load text content
            if filepath.suffix == '.json':
                import json
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    text = data.get('content', '')
                    doc_id = data.get('id', filepath.stem)
                    existing_metadata = {k: v for k, v in data.items() 
                                       if k not in ['content', 'id']}
            else:  # .txt file
                with open(filepath, 'r', encoding='utf-8') as f:
                    raw_text = f.read()
                
                # Extract metadata from header (Title:, URL:, etc. before the separator line)
                existing_metadata = {}
                text = raw_text
                lines = raw_text.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('=' * 40):
                        text = '\n'.join(lines[i + 1:]).strip()
                        break
                    if ': ' in line and not line.startswith(' ') and i < 10:
                        key, value = line.split(': ', 1)
                        existing_metadata[key.lower().replace(' ', '_')] = value.strip()
                
                doc_id = filepath.stem
                text = raw_text
                lines = raw_text.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('=' * 40):
                        text = '\n'.join(lines[i + 1:]).strip()
                        break
                    if ': ' in line and not line.startswith(' ') and i < 10:
                        key, value = line.split(': ', 1)
                        existing_metadata[key.lower().replace(' ', '_')] = value.strip()
                
                doc_id = filepath.stem
                text = raw_text
                lines = raw_text.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('=' * 40):
                        text = '\n'.join(lines[i + 1:]).strip()
                        break
                    if ': ' in line and not line.startswith(' ') and i < 10:
                        key, value = line.split(': ', 1)
                        existing_metadata[key.lower().replace(' ', '_')] = value.strip()
                
                doc_id = filepath.stem
                text = raw_text
                lines = raw_text.split('\n')
                for i, line in enumerate(lines):
                    if line.startswith('=' * 40):
                        text = '\n'.join(lines[i + 1:]).strip()
                        break
                    if ': ' in line and not line.startswith(' ') and i < 10:
                        key, value = line.split(': ', 1)
                        existing_metadata[key.lower().replace(' ', '_')] = value.strip()
                
                doc_id = filepath.stem
            
            if not text.strip():
                logger.warning(f"Skipping empty file: {filepath}")
                continue
            
            # Chunk the text
            chunks = chunk_document(
                text=text,
                document_id=doc_id,
                strategy=args.strategy,
                chunk_size=args.chunk_size,
                overlap=args.overlap,
                metadata=existing_metadata,
                **kwargs
            )
            
            all_chunks.extend(chunks)
            logger.info(f"Chunked {filepath}: {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error processing file {filepath}: {str(e)}")
    
    if all_chunks:
        saved_files = save_chunks(all_chunks, args.output)
        print(f"Successfully chunked {len(files_to_process)} files into {len(all_chunks)} chunks")
        print(f"Saved {len(saved_files)} chunk files to {args.output}/")
    else:
        print("No chunks were generated")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())