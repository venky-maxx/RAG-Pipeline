"""
Text Splitting and Chunking Strategies.
This file contains algorithms for breaking down large documents into smaller, semantically
meaningful chunks before embedding. It includes implementations for:
- Recursive Character Text Splitting: Splitting by paragraphs, sentences, and words.
- Semantic Chunking: Grouping text based on embedding similarity to maintain context.
- Token-based Splitting: Ensuring chunks fit strictly within LLM context windows.
Proper chunking is critical for effective retrieval and minimizing noise in the context.

Functions:
- chunk_by_characters(text, chunk_size, overlap): Recursive splitting.
- chunk_by_tokens(text, max_tokens): Token-aware splitting.
- semantic_chunk(text, embedding_model): Groups sentences by semantic similarity.
"""
from abc import ABC, abstractmethod
import asyncio


class BaseChunker(ABC):
    @abstractmethod
    async def chunk(self, text: str) -> list[str]:
        pass

class SlidingWindowChunking(BaseChunker):
     
    def __init__(self,chunk_size = 1000, overlap = 200):
        self.chunk_size: int = chunk_size
        self.overlap: int = overlap

    async def chunk(self, text: str) -> list[str]:
        start = 0
        chunks = []
        text_size = len(text) if text else 0

        if not text:
            return []
        
        while start < text_size:
            end = min(start + self.chunk_size, text_size)
            chunks.append(text[start:end])
            
            if end == text_size:
                break
            start += (self.chunk_size - self.overlap)

        return chunks

