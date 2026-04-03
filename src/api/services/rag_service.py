"""
RAG (Retrieval-Augmented Generation) service for the Personal Database API.
Handles the retrieval of relevant chunks and generation of responses using LLMs.
"""

import logging
import time
from typing import List, Dict, Any, Optional

from src.api.services.vector_db_service import VectorDBService
from src.api.services.embedding_service import EmbeddingService

logger = logging.getLogger(__name__)


class RAGService:
    """Retrieval-Augmented Generation service."""

    def __init__(
        self, vector_db_service: VectorDBService, embedding_service: EmbeddingService
    ):
        """
        Initialize the RAG service.

        Args:
            vector_db_service: Service for vector database operations
            embedding_service: Service for generating embeddings
        """
        self.vector_db_service = vector_db_service
        self.embedding_service = embedding_service

        # Default LLM parameters (can be overridden per request)
        self.default_temperature = 0.7
        self.default_max_tokens = 1000
        self.default_top_p = 0.9
        self.default_model = "anthropic/claude-3-haiku"

        logger.info("RAG service initialized")

    async def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        score_threshold: Optional[float] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        include_sources: bool = True,
    ) -> Dict[str, Any]:
        """
        Process a query using the RAG pipeline.

        Args:
            question: The user's question
            top_k: Number of chunks to retrieve (default: 5)
            score_threshold: Minimum similarity score (default: 0.7)
            temperature: LLM temperature (default: 0.7)
            max_tokens: Maximum tokens for LLM response (default: 1000)
            include_sources: Whether to include source chunks in response

        Returns:
            Dictionary containing the answer and optional sources
        """
        start_time = time.time()

        try:
            # Set defaults
            top_k = top_k or 5
            score_threshold = score_threshold if score_threshold is not None else 0.5
            temperature = (
                temperature if temperature is not None else self.default_temperature
            )
            max_tokens = (
                max_tokens if max_tokens is not None else self.default_max_tokens
            )

            logger.debug(
                f"Processing RAG query: '{question[:50]}...' "
                f"(top_k={top_k}, score_threshold={score_threshold})"
            )

            # Step 1: Generate embedding for the question
            question_embedding = await self.embedding_service.embed_query(question)
            if not question_embedding:
                raise Exception("Failed to generate embedding for question")

            # Step 2: Retrieve relevant chunks from vector database
            search_results = await self.vector_db_service.search_vectors(
                query_vector=question_embedding,
                limit=top_k,
                score_threshold=score_threshold,
            )

            if not search_results:
                logger.warning("No relevant chunks found for query")
                return {
                    "answer": "I couldn't find any relevant information in the knowledge base to answer your question. Please try rephrasing or adding more documents to your personal database.",
                    "question": question,
                    "sources": [] if include_sources else None,
                    "usage": None,
                    "model_used": self.default_model,
                }

            # Step 3: Prepare context from retrieved chunks
            context = self._prepare_context(search_results)

            # Step 4: Generate response using LLM (via OpenRouter)
            answer, usage = await self._generate_response(
                question=question,
                context=context,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Step 5: Prepare sources information if requested
            sources = None
            if include_sources:
                sources = self._prepare_sources(search_results)

            processing_time = time.time() - start_time
            logger.info(
                f"RAG query completed in {processing_time:.2f}s. "
                f"Retrieved {len(search_results)} chunks, "
                f"generated answer of {len(answer)} characters"
            )

            return {
                "answer": answer,
                "question": question,
                "sources": sources,
                "usage": usage,
                "model_used": self.default_model,
                "processing_time_seconds": round(processing_time, 2),
            }

        except Exception as e:
            logger.error(f"Error in RAG query processing: {e}")
            raise

    async def add_documents(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Add documents to the knowledge base.

        Args:
            documents: List of document dictionaries with 'text' and metadata

        Returns:
            Dictionary with processing results
        """
        start_time = time.time()

        try:
            logger.info(f"Adding {len(documents)} documents to knowledge base")

            # Extract texts for embedding
            texts = [doc.get("text", "") for doc in documents]

            # Filter out empty texts
            valid_docs = []
            valid_texts = []
            for doc, text in zip(documents, texts):
                if text.strip():
                    valid_docs.append(doc)
                    valid_texts.append(text)

            if not valid_texts:
                return {
                    "message": "No valid documents to process",
                    "documents_processed": 0,
                    "chunks_created": 0,
                    "vectors_stored": 0,
                }

            # Generate embeddings
            embeddings = await self.embedding_service.embed_texts(valid_texts)
            if not embeddings:
                raise Exception("Failed to generate embeddings for documents")

            # Add to vector database
            success = await self.vector_db_service.add_vectors(
                vectors=embeddings,
                payloads=valid_docs,
                ids=[
                    doc.get("chunk_id", f"doc_{i}") for i, doc in enumerate(valid_docs)
                ],
            )

            if not success:
                raise Exception("Failed to add vectors to database")

            processing_time = time.time() - start_time
            logger.info(
                f"Successfully added {len(valid_docs)} documents to knowledge base "
                f"in {processing_time:.2f}s"
            )

            return {
                "message": f"Successfully added {len(valid_docs)} documents to knowledge base",
                "documents_processed": len(valid_docs),
                "chunks_created": len(
                    valid_docs
                ),  # Assuming 1 chunk per document for simplicity
                "vectors_stored": len(valid_docs),
                "processing_time_seconds": round(processing_time, 2),
            }

        except Exception as e:
            logger.error(f"Error adding documents to knowledge base: {e}")
            raise

    async def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the RAG service and underlying systems.

        Returns:
            Dictionary with service statistics
        """
        try:
            vector_db_stats = await self.vector_db_service.get_stats()
            embedding_stats = self.embedding_service.get_stats()

            return {
                "rag_service": {
                    "status": "healthy",
                    "default_model": self.default_model,
                    "default_temperature": self.default_temperature,
                    "default_max_tokens": self.default_max_tokens,
                },
                "vector_database": vector_db_stats,
                "embedding_service": embedding_stats,
            }
        except Exception as e:
            logger.error(f"Error getting RAG service stats: {e}")
            return {
                "rag_service": {"status": "unhealthy", "error": str(e)},
                "vector_database": {"status": "unknown"},
                "embedding_service": {"status": "unknown"},
            }

    def _prepare_context(self, search_results: List[Dict[str, Any]]) -> str:
        """
        Prepare context string from search results.

        Args:
            search_results: List of search result dictionaries

        Returns:
            Formatted context string
        """
        if not search_results:
            return ""

        context_parts = []
        for i, result in enumerate(search_results, 1):
            # Extract relevant information
            content = result.get("content", "").strip()
            title = result.get("title", "Untitled")
            score = result.get("score", 0.0)

            if content:
                context_part = (
                    f"[Source {i}: {title} (Relevance: {score:.2f})]\n{content}\n"
                )
                context_parts.append(context_part)

        return "\n---\n".join(context_parts)

    def _prepare_sources(
        self, search_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Prepare sources information for response.

        Args:
            search_results: List of search result dictionaries

        Returns:
            List of source dictionaries
        """
        import json
        import re

        sources = []
        for result in search_results:
            # Check top-level title first, then metadata, then content header, then fallback
            title = result.get("title") or ""
            if not title:
                metadata = result.get("metadata", {})
                if isinstance(metadata, str):
                    try:
                        metadata = json.loads(metadata)
                    except (json.JSONDecodeError, TypeError):
                        metadata = {}
                title = metadata.get("title", "")
            if not title:
                # Fallback: extract title from content header
                content = result.get("content", "")
                match = re.search(r"^Title:\s*(.+)$", content, re.MULTILINE)
                if match:
                    title = match.group(1).strip()
                else:
                    # Use first line as title if it looks like one
                    first_line = content.split("\n")[0].strip()
                    if (
                        first_line
                        and len(first_line) < 100
                        and not first_line.startswith(("-", "•", "[", "##"))
                    ):
                        title = first_line
            if not title:
                title = "Untitled"

            source = {
                "id": result.get("chunk_id", ""),
                "document_id": result.get("document_id", ""),
                "title": title,
                "content_preview": result.get("content", "")[:200]
                + ("..." if len(result.get("content", "")) > 200 else ""),
                "score": result.get("score", 0.0),
                "metadata": result.get("metadata", {}),
            }
            sources.append(source)

        return sources

    async def _generate_response(
        self, question: str, context: str, temperature: float, max_tokens: int
    ) -> tuple[str, Dict[str, Any]]:
        """
        Generate a response using the LLM via OpenRouter.

        Args:
            question: The user's question
            context: Context retrieved from vector database
            temperature: LLM temperature parameter
            max_tokens: Maximum tokens for response

        Returns:
            Tuple of (answer_text, usage_info)
        """
        try:
            import httpx
            import os

            # Get OpenRouter API key from environment
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                # Fallback response if no API key is available
                fallback_answer = self._generate_fallback_response(question, context)
                return fallback_answer, {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "model": self.default_model,
                    "note": "Fallback response - OpenRouter API key not configured",
                }

            # Prepare the prompt
            if context.strip():
                prompt = f"""Based on the following context from your personal knowledge base, please answer the question. If the context doesn't contain enough information to fully answer the question, say so and provide what information you can.

Context:
{context}

Question: {question}

Answer:"""
            else:
                prompt = f"""Please answer the following question based on your general knowledge. Note that no relevant information was found in the personal knowledge base.

Question: {question}

Answer:"""

            # Call OpenRouter API
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "HTTP-Referer": "https://personal-database.local",  # Optional, for analytics
                        "X-Title": "Personal Database RAG",  # Optional, for analytics
                    },
                    json={
                        "model": self.default_model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "top_p": self.default_top_p,
                    },
                    timeout=30.0,
                )

                if response.status_code != 200:
                    logger.error(
                        f"OpenRouter API error: {response.status_code} - {response.text}"
                    )
                    # Fallback to local response
                    fallback_answer = self._generate_fallback_response(
                        question, context
                    )
                    return fallback_answer, {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "model": self.default_model,
                        "error": f"OpenRouter API error: {response.status_code}",
                    }

                result = response.json()

                # Extract answer and usage
                answer = (
                    result.get("choices", [{}])[0].get("message", {}).get("content", "")
                )
                usage = result.get("usage", {})

                # Ensure we have an answer
                if not answer.strip():
                    answer = "I apologize, but I was unable to generate a response. Please try again."

                return answer.strip(), usage

        except Exception as e:
            logger.error(f"Error generating response with OpenRouter: {e}")
            # Fallback to local response
            fallback_answer = self._generate_fallback_response(question, context)
            return fallback_answer, {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "model": self.default_model,
                "error": str(e),
            }

    def _generate_fallback_response(self, question: str, context: str) -> str:
        """
        Generate a fallback response when LLM is not available.

        Args:
            question: The user's question
            context: Context retrieved from vector database

        Returns:
            Fallback answer string
        """
        if not context.strip():
            return "I couldn't find any relevant information in your personal knowledge base to answer this question. Please try adding more documents or rephrasing your question."

        # Simple extractive answer based on context
        # In a real implementation, you might use more sophisticated techniques
        lines = context.split("\n")
        relevant_lines = []

        # Look for lines that contain question words
        question_words = set(question.lower().split())
        # Remove common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "is",
            "are",
            "was",
            "were",
            "be",
            "been",
            "have",
            "has",
            "had",
            "do",
            "does",
            "did",
            "will",
            "would",
            "could",
            "should",
            "may",
            "might",
            "must",
            "can",
        }
        question_words = question_words - stop_words

        for line in lines:
            line_lower = line.lower()
            # Count how many question words appear in this line
            matches = sum(1 for word in question_words if word in line_lower)
            if matches > 0:
                relevant_lines.append((matches, line.strip()))

        # Sort by relevance (number of matching question words)
        relevant_lines.sort(key=lambda x: x[0], reverse=True)

        if relevant_lines:
            # Take the top 3 most relevant lines
            top_lines = [line for _, line in relevant_lines[:3]]
            answer = (
                "Based on the information in your knowledge base:\n\n"
                + "\n\n".join(top_lines)
            )
            answer += "\n\nNote: This is an automated extractive answer. For more detailed responses, please configure an LLM API key."
        else:
            # Return first part of context if no specific matches
            preview = context[:500] + ("..." if len(context) > 500 else "")
            answer = f"Here's some relevant information from your knowledge base:\n\n{preview}\n\nNote: This is an automated extractive answer. For more detailed responses, please configure an LLM API key."

        return answer
