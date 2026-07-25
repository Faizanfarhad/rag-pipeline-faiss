from nltk.tokenize import sent_tokenize
from llama_index.core import Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os 
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import TokenTextSplitter
from tools.clean_text import clean_text
from tools.clean_markdown import clean_markdown
import pymupdf

# refrence : https://oneuptime.com/blog/post/2026-01-30-semantic-chunking/view
# not able to 

class CreateChunking:
    def __init__(self):
        super().__init__()
        """_summary_
        
        """
    
    
    def create_chunk(self,
                file_path:str,
                doc_text: str,
                chunk_size: int = 512,
                chunk_overlap: int = 50
                ):
        """_summary_
            Note: for reducing the complexity i am using the whole document to create sentences
            
        """
        ext = os.path.splitext(file_path)[1].lower()
        document = Document(
            text=doc_text,
            metadata={
                "file_path": file_path,
                "file_name": os.path.basename(file_path),
                "file_type": ext,
                "file_size": os.path.getsize(file_path),
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap
            }
        )
        
        
        splitter = TokenTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
        )
        chunks = splitter.get_nodes_from_documents([document])
        
        for i,chunk in enumerate(chunks):
            chunk.metadata['chunk_id'] = i
            chunk.metadata['char_count'] = len(chunk.text)
            chunk.metadata['word_count'] = len(chunk.text.split())
        return chunks
        


