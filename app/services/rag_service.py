from app.services.chunking import TextChunker
from app.services.gemini_service import GeminiService
from app.services.qdrant_service import QdrantService

from app.services.bm25_service import (
    build_bm25,
    keyword_search
)

from app.utils.log import logger

import time
import re


class RAGService:

    def __init__(self):

        self.chunker = TextChunker()

        self.gemini = GeminiService()

        self.qdrant = QdrantService()

    def process_document(self, pages):

        total_chars = sum(
            len(page["text"])
            for page in pages
        )

        logger.info(
            f"Starting embedding for {total_chars} characters"
        )

        chunks = self.chunker.chunk_text(
            pages
        )

        build_bm25(chunks)

        logger.info(
            f"Created {len(chunks)} chunks"
        )

        if not chunks:

            logger.warning(
                "No chunks created from document"
            )

            return 0

        start = time.time()

        embeddings = (
            self.gemini.generate_embeddings(
                [
                    chunk["text"]
                    for chunk in chunks
                ]
            )
        )

        embed_time = time.time() - start

        logger.info(
            f"Batch embedded {len(chunks)} chunks in {embed_time:.1f}s"
        )

        start = time.time()

        self.qdrant.upload_chunks(
            chunks,
            embeddings
        )

        upsert_time = time.time() - start

        logger.info(
            f"Qdrant upsert time: {upsert_time:.1f}s"
        )

        logger.info(
            f"Total upload completed - {len(chunks)} chunks"
        )

        return len(chunks)

    def is_greeting(
        self,
        question
    ):

        greetings = {
            "hi",
            "hii",
            "hiii",
            "hello",
            "hey",
            "heyy",
            "hlo",
            "hola",
            "good morning",
            "good afternoon",
            "good evening",
            "good night",
            "thanks",
            "thank you",
            "bye"
        }

        return (
            question.lower().strip()
            in greetings
        )

    def is_ambiguous(
        self,
        question
    ):

        question = (
            question.lower().strip()
        )

        ambiguous_questions = [
            "timings",
            "timing",
            "opening hours",
            "hours",
            "ticket price",
            "price",
            "cost",
            "entry fee",
            "fee"
        ]

        return (
            question
            in ambiguous_questions
        )

    def chat_with_rag(

        self,

        question,

        filename=None,

        top_k=7,

        score_threshold=0.3

    ):

        start_time = time.time()

        logger.info(
            f"Question received: {question}"
        )

        if self.is_greeting(
            question
        ):

            logger.info(
                "Greeting detected"
            )

            return {
                "answer": (
                    "Hello! 👋 I'm your travel assistant. "
                    "I can help you explore destinations, monuments, "
                    "tourist attractions, hotels, restaurants and travel guides. "
                    "How can I help you today?"
                ),
                "sources": [],
                "response_time": "0 sec"
            }

        page_match = re.search(
            r"page\s+(\d+)",
            question.lower()
        )

        if page_match:

            page_number = int(
                page_match.group(1)
            )

            logger.info(
                f"Page search detected: {page_number}"
            )

            page_results = (
                self.qdrant.search_by_page(
                    page_number
                )
            )

            response_time = round(
                time.time()
                - start_time,
                2
            )

            if not page_results:

                return {
                    "answer": (
                        f"No content found on page {page_number}."
                    ),
                    "sources": [],
                    "response_time":
                    f"{response_time} sec"
                }

            page_text = "\n\n".join(
                [
                    chunk["text"]
                    for chunk in page_results
                ]
            )

            sources = []

            seen = set()

            for chunk in page_results:

                key = (
                    chunk["source"],
                    chunk["page"]
                )

                if key not in seen:

                    seen.add(key)

                    sources.append(
                        {
                            "book":
                            chunk["source"],
                            "page":
                            chunk["page"],
                            "match_percentage":
                            f"{round(chunk.get('score', 0) * 100, 2)}%"
                        }
                    )

            return {
                "answer":
                page_text[:3000],
                "sources":
                sources,
                "response_time":
                f"{response_time} sec"
            }

        if self.is_ambiguous(
            question
        ):

            return {
                "answer": (
                    "I'd be happy to help. "
                    "Could you please specify which place, attraction, "
                    "hotel, restaurant, or activity you are referring to?"
                ),
                "sources": [],
                "response_time":
                "0 sec"
            }

        question_lower = (
            question.lower()
        )

        if (
            "summary"
            in question_lower
            or
            "summarize"
            in question_lower
        ):

            logger.info(
                "Summary request detected"
            )

            relevant_chunks = (
                self.qdrant.search_similar(
                    self.gemini.generate_embedding(
                        "travel guide overview introduction contents"
                    ),
                    filename=filename,
                    top_k=7,
                    score_threshold=0.3
                )
            )

            if not relevant_chunks:

                return {
                    "answer":
                    "No document found to summarize.",
                    "sources": [],
                    "response_time":
                    "0 sec"
                }

            context = "\n\n".join(
                chunk["text"]
                for chunk in relevant_chunks
            )[:25000]

            answer = (
                self.gemini.generate_answer(
                    context,
                    "Provide a detailed summary of this document."
                )
            )

            return {
                "answer":
                answer,
                "sources": [],
                "response_time":
                f"{round(time.time()-start_time,2)} sec"
            }

        query_embedding = (
            self.gemini.generate_embedding(
                question
            )
        )

        logger.info(
            "Query embedding generated"
        )

        semantic_chunks = (
            self.qdrant.search_similar(
                query_embedding,
                filename=filename,
                top_k=top_k,
                score_threshold=score_threshold
            )
        )

        keyword_chunks = (
            keyword_search(
                question,
                top_k=top_k
            )
        )

        print(
            f"Semantic Results: {len(semantic_chunks)}"
        )

        print(
            f"Keyword Results: {len(keyword_chunks)}"
        )

        combined = []

        seen = set()

        for chunk in (
            semantic_chunks
            +
            keyword_chunks
        ):

            key = (
                chunk["text"],
                chunk["page"]
            )

            if key not in seen:

                seen.add(key)

                combined.append(
                    chunk
                )

        relevant_chunks = (
            combined[:top_k]
        )
        print("\nFIRST CHUNK:")
        print(relevant_chunks[0] if relevant_chunks else "No chunks found")


        logger.info(
            f"Retrieved {len(relevant_chunks)} hybrid chunks"
        )

        if not relevant_chunks:

            return {
                "answer": (
                    "I could not find the answer in the uploaded documents."
                ),
                "sources": [],
                "response_time":
                "0 sec"
            }

        context = "\n\n".join(
            chunk["text"]
            for chunk in relevant_chunks
        )[:25000]

        logger.info(
            f"Context size: {len(context)} characters"
        )

        response_time = round(
            time.time()
            - start_time,
            2
        )

        sources = []

        seen = set()

        for chunk in relevant_chunks:

            key = (
                chunk["source"],
                chunk["page"]
            )

            if key not in seen:

                seen.add(key)

                sources.append(
                    {
                        "book":
                        chunk["source"],
                        "page":
                        chunk["page"],
                        "debug_score":
                        chunk.get("score", "MISSING")
                    }
                )

        answer = (
            self.gemini.generate_answer(
                context,
                question
            )
        )

        logger.info(
            f"Generated answer in {response_time}s"
        )

        return {
            "answer": answer,
            "sources": sources,
            "response_time":
            f"{response_time} sec"
        }


rag_service = RAGService()