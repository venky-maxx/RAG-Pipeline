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
        
    async def embeddings(self, chunks: list[str]) -> list[list[float]]:
        embeddings_list = []
        for chunk in chunks:
            response = await self.client.embeddings.create(input = chunk, model = settings.OPENAI_EMBEDDING_MODEL)
            embeddings_list.append(response.data[0].embedding)
        return embeddings_list

class OllamaEmbedding(BaseEmbedding):

    def __init__(self):
        pass
        
    async def embeddings(self, chunks: list[str]) -> list[list[float]]:
        embeddings_list = []
        for chunk in chunks:
            response = ollama.embeddings(model=settings.OLLAMA_EMBEDDING_MODEL, input=chunk)
            embeddings_list.append(response)
        return embeddings_list


class ModelSelector:
    
    def __init__(self):
        self.model_source = settings.EMBEDDING_MODEL_SOURCE.lower()
        if self.model_source == "openai":
            self.model = OpenAIEmbedding()
        elif self.model_source == "ollama":
            self.model = OllamaEmbedding()
        else:
            raise ValueError(f"Unsupported embedding model source: {self.model_source}")

    async def get_embedded(self, chunks: list[str]) -> list[list[float]]:
        return await self.model.embeddings(chunks)