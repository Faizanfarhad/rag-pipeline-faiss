from sentence_transformers import SentenceTransformer
import numpy  as np 
from llama_index.core import Document


def create_embedding(model:SentenceTransformer,chunks:Document)-> np.array:
    """_summary_

    Args:
        model (SentenceTransformer): _description_
        chunks (Document): List of Document or TextNode objects

    Returns:
        np.array: Stacked embeddings
    """
    chunk_embeddings = []
    
    for chunk in chunks:
        embeddings = model.encode(chunk.text,convert_to_numpy=True)
        chunk_embeddings.append(embeddings)
    chunk_embeddings = np.stack(chunk_embeddings,axis=0)
    
    return chunk_embeddings 
    