from abc import ABC, abstractmethod

import asyncio
import aiofiles
from pathlib import Path

# Template for our file loaders
class BaseDocumentLoader(ABC):

    def __init__(self):
        pass

    @abstractmethod
    def load(self, file_path:str) -> str:
        pass

# loads text files
class TextLoader(BaseDocumentLoader):

    async def load(self, file_path:str) -> str:
        if not file_path.exists():
            raise FileNotFoundError("File not found {file_path}")
            
        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            return await f.read()

# loads md files        
class MarkdownLoader(BaseDocumentLoader):

    async def load(self, file_path: str)-> str:
        if not file_path.exists():
            raise FileNotFoundError("File not found {file_path}")

        async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
            return await f.read()
                                 
# loads pdf files 
# loads docx files               
          
class DocumentLoaderFactory:
    """
    Factory class to create appropriate document loader instances based on file type.
    """
    @staticmethod
    async def get_loader(file_path):
        file_path = Path(file_path)
        ext = file_path.suffix.lower()

        if ext == ".txt":
            txt_ldr = TextLoader()    
            return await txt_ldr.load(file_path)  
              
        elif ext == ".md":
            md_ldr = MarkdownLoader()    
            return await md_ldr.load(file_path)         

