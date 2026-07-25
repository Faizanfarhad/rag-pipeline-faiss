from sentence_transformers import SentenceTransformer
import os
import time
import numpy as np

# NOTE : Library Hierarchy represents the flow of the data
from RagPipeline.tools.extract_doc import ExtractDocContent
from RagPipeline.tools.chunking import CreateChunking
from RagPipeline.tools.create_embeddings import create_embedding
from RagPipeline.tools.ranker import build_faiss_index
from RagPipeline.tools.retrieve_top_k import retrieve_top_k


class RagPipelineConnector:
    """
    Lightweight RAG pipeline using FAISS-only retrieval.
    No LLM — returns retrieved chunks directly for evaluation.
    LLM answer generation is optional via self.generate_answer().
    """

    def __init__(
        self,
        doc_path: str,
        top_k: int = 5,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        embd_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: str = "cpu",
    ):
        super().__init__()

        self.doc_path = doc_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k
        self.device = device if device != "auto" else "cpu"

        # Step 1: Extract document
        ext = os.path.splitext(doc_path)[1].lower()
        t0 = time.time()
        self.pdf_text = ExtractDocContent(doc_url=self.doc_path).extract_doc()
        print(f"[{time.time() - t0:.1f}s] {ext[1:]} extracted successfully — "
              f"{len(self.pdf_text.split())} words")

        # Step 2: Chunk
        t0 = time.time()
        chunk_worker = CreateChunking()
        self.chunks = chunk_worker.create_chunk(
            file_path=self.doc_path,
            doc_text=self.pdf_text,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )
        print(f"[{time.time() - t0:.1f}s] {len(self.chunks)} chunks created")

        # Step 3: Embed
        t0 = time.time()
        self.embd_model = SentenceTransformer(embd_model, device=self.device)
        self.chunks_embeddings = create_embedding(
            model=self.embd_model, chunks=self.chunks
        )
        print(f"[{time.time() - t0:.1f}s] Embeddings created — "
              f"shape={self.chunks_embeddings.shape}")

        # Step 4: FAISS index
        t0 = time.time()
        self.faiss_index = build_faiss_index(
            chunk_embeddings=self.chunks_embeddings
        )
        print(f"[{time.time() - t0:.1f}s] FAISS index built — "
              f"{self.faiss_index.ntotal} vectors")

        print(f"Pipeline ready. top_k={self.top_k}, device={self.device}")

    def query(self, question: str) -> dict:
        """
        Retrieve top-k chunks from FAISS and return them directly.
        No LLM — evaluation uses gold answers for scoring.
        """
        # Retrieve using FAISS
        retrieved = retrieve_top_k(
            question=question,
            embedding_model=self.embd_model,
            faiss_index=self.faiss_index,
            chunks_data=self.chunks,
            top_k=self.top_k,
        )

        # Build source chunks list with full text
        source_chunks = [
            {
                "id": r["chunk_id"],
                "text": r["chunk_text"],
            }
            for r in retrieved
        ]

        # For evaluation: answer is empty (scored against gold_answer)
        # Faithfulness check compares retrieved chunk text (full text now)
        return {
            "question": question,
            "answer": "",  # no LLM — evaluation uses gold_answer
            "source_chunks": source_chunks,
            "total_sources": len(source_chunks),
        }