"""
Ingestion endpoints for the Personal Database RAG API.
Handles document processing pipeline: scrape -> parse -> chunk -> embed.
"""

import sys
import tempfile
import shutil
from pathlib import Path

# Add the project root to Python path so we can import from src.scripts
project_root = Path(__file__).parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import APIRouter, HTTPException, BackgroundTasks, UploadFile, File, Depends
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import os
import logging

from src.api.dependencies import get_rag_service

router = APIRouter()
logger = logging.getLogger(__name__)


class IngestURLRequest(BaseModel):
    """Request model for ingesting a single URL."""
    url: str = Field(..., description="URL to scrape and ingest")
    processing_options: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Optional processing overrides"
    )


class IngestURLsRequest(BaseModel):
    """Request model for ingesting multiple URLs."""
    urls: List[str] = Field(..., description="List of URLs to scrape and ingest")
    processing_options: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Optional processing overrides"
    )


class IngestFileRequest(BaseModel):
    """Request model for ingesting uploaded files."""
    processing_options: Optional[Dict[str, Any]] = Field(
        default=None, 
        description="Optional processing overrides"
    )


# Simplest possible response model
class IngestResponse(BaseModel):
    message: str


@router.post("/url")
async def ingest_url(
    request: IngestURLRequest,
    background_tasks: BackgroundTasks,
    rag_service = Depends(get_rag_service)
):
    """
    Ingest a single URL by scraping, processing, chunking, and embedding.
    
    This endpoint processes the request in the background to avoid timeouts.
    """
    try:
        # Add background task for processing
        background_tasks.add_task(
            process_single_url,
            request.url,
            request.processing_options or {},
            rag_service
        )
        
        return IngestResponse(
            message="Ingestion started in background"
        )
        
    except Exception as e:
        logger.error(f"Error starting URL ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/urls")
async def ingest_urls(
    request: IngestURLsRequest,
    background_tasks: BackgroundTasks,
    rag_service = Depends(get_rag_service)
):
    """
    Ingest multiple URLs by scraping, processing, chunking, and embedding.
    
    This endpoint processes the request in the background to avoid timeouts.
    """
    try:
        background_tasks.add_task(
            process_multiple_urls,
            request.urls,
            request.processing_options or {},
            rag_service
        )
        
        return IngestResponse(
            message=f"Ingestion started for {len(request.urls)} URLs in background"
        )
        
    except Exception as e:
        logger.error(f"Error starting URLs ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload")
async def ingest_uploaded_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    processing_options: Optional[str] = None,
    rag_service = Depends(get_rag_service)
):
    """
    Ingest uploaded files by processing, chunking, and embedding.
    
    Accepts .json and .txt files. Processing happens in the background.
    """
    try:
        import tempfile
        temp_dir = Path(tempfile.mkdtemp())
        file_paths = []
        
        for file in files:
            file_path = temp_dir / file.filename
            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)
            file_paths.append(str(file_path))
        
        options = json.loads(processing_options) if processing_options else {}
        
        background_tasks.add_task(
            process_uploaded_files,
            file_paths,
            options,
            rag_service,
            temp_dir
        )
        
        return IngestResponse(
            message=f"Processing started for {len(files)} file(s) in background"
        )
        
    except Exception as e:
        logger.error(f"Error starting file upload ingestion: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background processing functions
import logging
import time
from scripts.scraper import WebScraper
from scripts.parser import TextParser
from scripts.chunker import chunk_document, get_chunker
from scripts.embedder import Embedder

logger = logging.getLogger(__name__)


async def process_single_url(url: str, processing_options: Dict[str, Any], rag_service):
    """Process a single URL in the background."""
    start_time = time.time()
    try:
        logger.info(f"Starting background processing for URL: {url}")
        
        # Step 1: Scrape
        scraper = WebScraper(
            delay=processing_options.get('scrape_delay', 1.0),
            timeout=processing_options.get('scrape_timeout', 30)
        )
        scraped_results = scraper.scrape_urls([url])
        
        if not scraped_results:
            logger.error(f"Failed to scrape URL: {url}")
            return
        
        # Step 2: Parse
        parser = TextParser()
        # Save scraped results temporarily for parsing
        temp_scrape_dir = Path(tempfile.mkdtemp())
        try:
            scraper.save_results(scraped_results, str(temp_scrape_dir))
            parsed_docs = parser.parse_directory(str(temp_scrape_dir))
        finally:
            shutil.rmtree(temp_scrape_dir, ignore_errors=True)
        
        if not parsed_docs:
            logger.error(f"Failed to parse scraped content for URL: {url}")
            return
        
        # Step 3: Chunk
        chunking_strategy = processing_options.get('chunking_strategy', 'semantic')
        chunk_size = processing_options.get('chunk_size', 100)
        overlap = processing_options.get('overlap', 20)
        
        all_chunks = []
        for doc in parsed_docs:
            # Convert ParsedDocument to dict for chunker
            doc_dict = {
                'text': doc.content,
                'document_id': doc.id,
                'title': doc.title,
                'source_url': doc.source_url,
                'timestamp': doc.timestamp,
                'word_count': doc.word_count,
                'metadata': doc.metadata
            }
            
            chunks = chunk_document(
                text=doc_dict['text'],
                document_id=doc_dict['document_id'],
                strategy=chunking_strategy,
                chunk_size=chunk_size,
                overlap=overlap,
                metadata={
                    'title': doc_dict['title'],
                    'source_url': doc_dict['source_url'],
                    'timestamp': doc_dict['timestamp'],
                    'word_count': doc_dict['word_count'],
                    **doc_dict.get('metadata', {})
                }
            )
            
            # Add chunk metadata
            for chunk in chunks:
                chunk.metadata.update({
                    'title': doc_dict['title'],
                    'source_url': doc_dict['source_url'],
                    'timestamp': doc_dict['timestamp'],
                    'original_word_count': doc_dict['word_count']
                })
            
            all_chunks.extend([chunk.to_dict() for chunk in chunks])
        
        if not all_chunks:
            logger.error(f"No chunks generated for URL: {url}")
            return
        
        # Step 4: Embed and store
        result = await rag_service.add_documents(all_chunks)
        
        processing_time = time.time() - start_time
        logger.info(
            f"Completed background processing for URL {url}: "
            f"{len(scraped_results)} documents, {len(all_chunks)} chunks, "
            f"{result.get('vectors_stored', 0)} vectors stored in {processing_time:.2f}s"
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error in background processing for URL {url}: {e}")


async def process_multiple_urls(urls: List[str], processing_options: Dict[str, Any], rag_service):
    """Process multiple URLs in the background."""
    start_time = time.time()
    try:
        logger.info(f"Starting background processing for {len(urls)} URLs")
        
        # Step 1: Scrape all URLs
        scraper = WebScraper(
            delay=processing_options.get('scrape_delay', 1.0),
            timeout=processing_options.get('scrape_timeout', 30)
        )
        scraped_results = scraper.scrape_urls(urls)
        
        if not scraped_results:
            logger.error("Failed to scrape any URLs")
            return
        
        logger.info(f"Successfully scraped {len(scraped_results)}/{len(urls)} URLs")
        
        # Step 2: Parse all scraped content
        parser = TextParser()
        temp_scrape_dir = Path(tempfile.mkdtemp())
        try:
            scraper.save_results(scraped_results, str(temp_scrape_dir))
            parsed_docs = parser.parse_directory(str(temp_scrape_dir))
        finally:
            shutil.rmtree(temp_scrape_dir, ignore_errors=True)
        
        if not parsed_docs:
            logger.error("Failed to parse any scraped content")
            return
        
        logger.info(f"Successfully parsed {len(parsed_docs)} documents")
        
        # Step 3: Chunk all documents
        chunking_strategy = processing_options.get('chunking_strategy', 'semantic')
        chunk_size = processing_options.get('chunk_size', 100)
        overlap = processing_options.get('overlap', 20)
        
        all_chunks = []
        for doc in parsed_docs:
            # Convert ParsedDocument to dict for chunker
            doc_dict = {
                'text': doc.content,
                'document_id': doc.id,
                'title': doc.title,
                'source_url': doc.source_url,
                'timestamp': doc.timestamp,
                'word_count': doc.word_count,
                'metadata': doc.metadata
            }
            
            chunks = chunk_document(
                text=doc_dict['text'],
                document_id=doc_dict['document_id'],
                strategy=chunking_strategy,
                chunk_size=chunk_size,
                overlap=overlap,
                metadata={
                    'title': doc_dict['title'],
                    'source_url': doc_dict['source_url'],
                    'timestamp': doc_dict['timestamp'],
                    'word_count': doc_dict['word_count'],
                    **doc_dict.get('metadata', {})
                }
            )
            
            # Add chunk metadata
            for chunk in chunks:
                chunk.metadata.update({
                    'title': doc_dict['title'],
                    'source_url': doc_dict['source_url'],
                    'timestamp': doc_dict['timestamp'],
                    'original_word_count': doc_dict['word_count']
                })
            
            all_chunks.extend([chunk.to_dict() for chunk in chunks])
        
        if not all_chunks:
            logger.error("No chunks generated from any documents")
            return
        
        logger.info(f"Generated {len(all_chunks)} chunks from {len(parsed_docs)} documents")
        
        # Step 4: Embed and store
        result = await rag_service.add_documents(all_chunks)
        
        processing_time = time.time() - start_time
        logger.info(
            f"Completed background processing for {len(urls)} URLs: "
            f"{len(scraped_results)} documents scraped, {len(parsed_docs)} parsed, "
            f"{len(all_chunks)} chunks created, {result.get('vectors_stored', 0)} vectors stored "
            f"in {processing_time:.2f}s"
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error in background processing for multiple URLs: {e}")


async def process_uploaded_files(file_paths: List[str], processing_options: Dict[str, Any], 
                               rag_service, temp_dir: Path):
    """Process uploaded files in the background."""
    start_time = time.time()
    try:
        logger.info(f"Starting background processing for {len(file_paths)} uploaded files")
        
        all_chunks = []
        
        for file_path in file_paths:
            file_path_obj = Path(file_path)
            
            try:
                # Load file content based on extension
                if file_path_obj.suffix == '.json':
                    # JSON file - expect document data
                    with open(file_path_obj, 'r', encoding='utf-8') as f:
                        doc_data = json.load(f)
                    
                    # Validate required fields
                    if 'text' not in doc_data:
                        logger.warning(f"Skipping {file_path_obj.name}: missing 'text' field")
                        continue
                    
                    # Convert to our internal format
                    doc_dict = {
                        'text': doc_data.get('text', ''),
                        'document_id': doc_data.get('id', file_path_obj.stem),
                        'title': doc_data.get('title', 'Untitled'),
                        'source_url': doc_data.get('source_url'),
                        'timestamp': doc_data.get('timestamp', 0),
                        'word_count': doc_data.get('word_count', 0),
                        'metadata': doc_data.get('metadata', {})
                    }
                    
                elif file_path_obj.suffix == '.txt':
                    # Plain text file
                    with open(file_path_obj, 'r', encoding='utf-8') as f:
                        text_content = f.read()
                    
                    if not text_content.strip():
                        logger.warning(f"Skipping {file_path_obj.name}: empty file")
                        continue
                    
                    # Convert to our internal format
                    doc_dict = {
                        'text': text_content,
                        'document_id': file_path_obj.stem,
                        'title': file_path_obj.stem.replace('_', ' ').title(),
                        'source_url': None,
                        'timestamp': 0,
                        'word_count': len(text_content.split()),
                        'metadata': {}
                    }
                
                else:
                    logger.warning(f"Skipping unsupported file type: {file_path_obj.name}")
                    continue
                
                # Skip if no meaningful content
                if not doc_dict['text'].strip():
                    logger.warning(f"Skipping {file_path_obj.name}: no text content")
                    continue
                
                # Chunk the document
                chunking_strategy = processing_options.get('chunking_strategy', 'semantic')
                chunk_size = processing_options.get('chunk_size', 100)
                overlap = processing_options.get('overlap', 20)
                
                chunks = chunk_document(
                    text=doc_dict['text'],
                    document_id=doc_dict['document_id'],
                    strategy=chunking_strategy,
                    chunk_size=chunk_size,
                    overlap=overlap,
                    metadata={
                        'title': doc_dict['title'],
                        'source_url': doc_dict['source_url'],
                        'timestamp': doc_dict['timestamp'],
                        'word_count': doc_dict['word_count'],
                        **doc_dict.get('metadata', {})
                    }
                )
                
                # Add chunk metadata
                for chunk in chunks:
                    chunk.metadata.update({
                        'title': doc_dict['title'],
                        'source_url': doc_dict['source_url'],
                        'timestamp': doc_dict['timestamp'],
                        'original_word_count': doc_dict['word_count']
                    })
                
                all_chunks.extend([chunk.to_dict() for chunk in chunks])
                logger.debug(f"Processed {file_path_obj.name}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Error processing file {file_path_obj.name}: {e}")
                continue
        
        if not all_chunks:
            logger.error("No chunks generated from uploaded files")
            return
        
        logger.info(f"Generated {len(all_chunks)} chunks from {len(file_paths)} uploaded files")
        
        # Embed and store
        result = await rag_service.add_documents(all_chunks)
        
        processing_time = time.time() - start_time
        logger.info(
            f"Completed background processing for {len(file_paths)} uploaded files: "
            f"{len(all_chunks)} chunks created, {result.get('vectors_stored', 0)} vectors stored "
            f"in {processing_time:.2f}s"
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error in background processing for uploaded files: {e}")
    finally:
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception as e:
            logger.warning(f"Error cleaning up temp directory: {e}")