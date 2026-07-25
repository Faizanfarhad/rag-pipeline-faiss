import faiss
import torch
import numpy as np 

def build_faiss_index(chunk_embeddings: np.ndarray) -> faiss.Index:
    """_summary_
        *Builds a FAISS index for normalized chunk embeddings.
        * This function uses IndexFlatIP, which performs inner product search.
            Since our embeddings are normalized, inner product behaves like cosine similarity.
    Args:
        chunk_embeddings (np.ndarray): stacked embeddings of chunks texts

    Raises:
        ValueError: if the chunk_embeddings is None 
        ValueError: if the chunk_embeddings is not 2D matrix.
        ValueError: if there are No embeddings founded to solve this, Create chunk embeddings before building FAISS index.

    Returns:
        faiss.Index: FAISS index containing all chunk embeddings
    """

        
    
    if chunk_embeddings is None:
        raise ValueError("chunk_embeddings cannot be None.")
    
    if len(chunk_embeddings.shape) != 2:
        raise ValueError("chunk_embeddings must be a 2D matrix.")
    
    if chunk_embeddings.dtype != np.float32:
        chunk_embeddings = chunk_embeddings.astype('float32')
        
        
    
    num_chunks, embedding_dim = chunk_embeddings.shape
    
    if num_chunks == 0:
        raise ValueError("No embeddings found. Create chunk embeddings before building FAISS index.")
    
    # Create FAISS index for inner product similarity
    index = faiss.IndexFlatIP(embedding_dim)
    
    # Add chunk embeddings to the index
    index.add(chunk_embeddings)
    
    print("FAISS index built successfully.")
    print("Number of vectors in index:", index.ntotal)
    print("Embedding dimension:", embedding_dim)
    print("Index type: IndexFlatIP")
    return index
    