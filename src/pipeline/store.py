"""
SQLite Experiment Tracking.
This module manages the persistence layer for all evaluation runs. It establishes
a connection to a local SQLite database to log inputs, outputs, judge scores,
model configurations, and latency metrics for every single evaluation.

Classes:
- ExperimentTracker: Manages DB connections and table schemas.

Methods:
- log_run(config, metrics): Inserts a new experiment run.
- log_evaluation_case(run_id, query, response, score): Logs individual test case results.
- export_to_csv(): Dumps DB contents for external analysis.
"""

import asyncio
import json
from rag.document_loader import DocumentLoaderFactory
from rag.chunking import SlidingWindowChunking
from config.settings import settings
from rag.embeddings import ModelSelector
from pathlib import Path
import os
from rag.vector_store import JsonVectorStore
from config.logging_config import logger

class DocumentIngestionPipeline:

    def __init__(self):
        print("Document Ingestion Pipeline initialized!")

    async def ingest_file(self, file_path: str):
        file_path = Path(file_path)
        
        logger.info(f"Ingesting file....")
        get_content = await DocumentLoaderFactory.get_loader(file_path)

        logger.info(f"Chunking file....")
        chunker = SlidingWindowChunking(chunk_size = settings.CHUNK_SIZE, overlap = settings.CHUNK_OVERLAP)
        chunks = await chunker.chunk(text = get_content)
        
        logger.info(f"embedding file....")
        embeddings = await ModelSelector.get_embedded(chunks)

        logger.info(f"Vector store initialized....")
        vector_store = JsonVectorStore(settings.VECTOR_STORE_PATH, chunks, embeddings, file_path)
        vector_store.store()
        logger.info("Vector stored")

        
    async def ingest_directory(self, files_path: str) -> list[str]:
        files_path = Path(files_path)
        files = list(files_path.rglob("*"))  # Recursively find all files
        
        for file in files:
            if file.is_file():
                await self.ingest_file(file)
                
async def main():
    pipeline = DocumentIngestionPipeline()
    # Additional async operations can be added here
    await pipeline.ingest_directory(settings.FILES_PATH)  


if __name__ == "__main__":
    asyncio.run(main())