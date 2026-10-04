from abc import ABC, abstractmethod

import ollama

import ollama
from config.settings import settings
from openai import AsyncOpenAI, api_key, base_url


class BaseEmbedding(ABC):

    def __init__(self):
        pass

    @abstractmethod
    async def embeddings(self, chunks: list[str]) -> list[list[float]]:
        pass


class OpenAIEmbedding(BaseEmbedding):

    def __init__(self):
        self.client = AsyncOpenAI(
            api_key=settings.OPENAI_API_KEY, 
            base_url=settings.OPENAI_API_BASE_URL
            )
        
    async def embeddings(self, chunk: list[str]) -> list[list[float]]:
        response = await self.client.embeddings.create(
            input = chunk,
              model = settings.OPENAI_EMBEDDING_MODEL)
        return response.data[0].embedding

class OllamaEmbedding(BaseEmbedding):

    def __init__(self):
        pass
        
    async def embeddings(self, chunk: list[str]) -> list[list[float]]:
        response = ollama.embeddings(model=settings.OLLAMA_EMBEDDING_MODEL, input=chunk)
        return response['embeddings'][0]


class ModelSelector:
    
    @staticmethod
    async def get_embedded(chunks: list[str]) -> list[list[float]]:
        if settings.EMBEDDING_MODEL_SOURCE == "openai":
            model = OpenAIEmbedding()
        elif settings.EMBEDDING_MODEL_SOURCE == "ollama":
            model = OllamaEmbedding()
        embedding_vectors = []
        for chunk in chunks:
            embedding = await model.embeddings(chunk)
            embedding_vectors.append(embedding)
        return embedding_vectors