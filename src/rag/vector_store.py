"""
Vector Database Interface.
This script manages the connection and interactions with the underlying vector database
(such as ChromaDB, FAISS, Qdrant, or Pinecone).
It provides standard methods for initializing the database, inserting document chunks and
their corresponding embeddings, deleting old records, and persisting the index to disk.
This abstraction ensures the core RAG logic remains decoupled from specific database vendors.

Classes:
- VectorStore: Abstract interface for vector databases.
- ChromaDBStore: Implementation using local ChromaDB.
- PineconeStore: Implementation using Pinecone cloud vector DB.

Methods:
- add_documents(documents, embeddings, metadata): Inserts records.
- search(query_embedding, top_k): Returns top_k similar documents.
- delete(document_ids): Removes documents from the index.
"""

from abc import ABC, abstractmethod
import json
import os
from config.settings import settings
from pathlib import Path

class BaseVectorStore(ABC):
    def __init__(self):
        pass
        
    @abstractmethod
    def store(self) -> str:
        pass

    

class JsonVectorStore(BaseVectorStore):
        
    #checking the existence of the vector store file and loading the existing data if present.

    def __init__(self, vector_store_path, chunks, embeddings, raw_doc_file_path):
        self.vector_store_path = vector_store_path
        self.chunks = chunks
        self.embeddings = embeddings
        self.raw_doc_file_path = raw_doc_file_path

    def store(self):
        #checking the existence of the vector store file and loading the existing data if present.
        data = {}
        if os.path.exists(settings.VECTOR_STORE_PATH):
            with open(settings.VECTOR_STORE_PATH, 'r') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError: 
                    data = {}

        #creating a record of chunks and embeddings to store in the vector store.
        record = []
        for chunk, embedding in zip(self.chunks, self.embeddings):
            record.append({"chunk": chunk, "embedding": embedding})
        
        
            file_name = self.raw_doc_file_path.name
            file_type = self.raw_doc_file_path.suffix.lower().lstrip('.')
        
            data[file_name] = {
                "file_name": file_name,
                "file_type": file_type,
                "chunks": record
            }
        
            #writing the updated data back to the vector store file.
            with open(settings.VECTOR_STORE_PATH, 'w') as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
            



   

