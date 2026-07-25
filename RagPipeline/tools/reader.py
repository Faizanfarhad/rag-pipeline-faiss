# custom_reader.py

from llama_index.core.prompts import PromptTemplate
from llama_index.core import Settings, VectorStoreIndex

# Custom prompt with citations
CUSTOM_QA_PROMPT = PromptTemplate(
    """You are a helpful assistant that answers questions based ONLY on the provided context.

Context:
{context_str}

Question: {query_str}

Instructions:
1. Answer using ONLY the information from the context above
2. Cite the source chunks using [Chunk X] where X is the chunk number
3. If the context doesn't contain the answer, say "I don't have enough information to answer this question"

Answer:"""
)

def create_reader(nodes, embed_model, llm=None, k=5):
    """
    Create a query engine from pre-chunked nodes using VectorStoreIndex.

    Parameters
    ----------
    nodes : list of TextNode
        Pre-chunked document nodes (from CreateChunking).
    embed_model : HuggingFaceEmbedding or compatible
        Embedding model for encoding queries and building the index.
    llm : optional
        LLM to use for answer generation.
    k : int
        Number of top chunks to retrieve.
    """
    if llm:
        Settings.llm = llm

    # Build a VectorStoreIndex from the nodes — llama_index handles
    # embedding and vector storage internally
    index = VectorStoreIndex(nodes, embed_model=embed_model)

    query_engine = index.as_query_engine(
        similarity_top_k=k,
        text_qa_template=CUSTOM_QA_PROMPT,
        response_mode="compact",
        node_postprocessors=[],
    )

    return query_engine

# # Usage
# query_engine = create_reader(index, k=5)
# response = query_engine.query("What is BERT?")
# print(response.response)