"""
RAG Evaluation Harness
-----------------------
Evaluates retrieval quality (Recall@k, MRR, nDCG@k, Hit Rate, Context Precision)
and answer quality (faithfulness, relevance via LLM-as-judge).

Reads questions from bert_test_questions.json, runs them through the RagPipeline,
and produces evaluation_results.json.
"""

import json
import time
import math
import os
from typing import List, Dict, Any
from collections import defaultdict
from dataclasses import dataclass, field, asdict

# Lazy import — will fail gracefully if pipeline isn't ready
try:
    from RagPipeline.rag_pipeline_connector import RagPipelineConnector
    PIPELINE_AVAILABLE = True
except Exception as e:
    PIPELINE_AVAILABLE = False
    PIPELINE_ERROR = str(e)


# ═══════════════════════════════════════════════
# Data structures
# ═══════════════════════════════════════════════

@dataclass
class RetrievalMetrics:
    recall_at_k: float = 0.0
    hit_rate: float = 0.0
    mrr: float = 0.0
    ndcg_at_k: float = 0.0
    context_precision: float = 0.0
    avg_latency_ms: float = 0.0
    avg_chunks_retrieved: float = 0.0


@dataclass
class AnswerMetrics:
    faithfulness_avg: float = 0.0
    relevance_avg: float = 0.0
    hallucination_rate: float = 0.0
    em_score: float = 0.0
    f1_score: float = 0.0


@dataclass
class EvalResult:
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    answer: AnswerMetrics = field(default_factory=AnswerMetrics)
    per_question: List[Dict] = field(default_factory=list)
    cost_estimate: Dict = field(default_factory=dict)


# ═══════════════════════════════════════════════
# IR Metrics (compute without external libs)
# ═══════════════════════════════════════════════

def dcg_at_k(scores: List[int], k: int) -> float:
    """Discounted Cumulative Gain at k."""
    dcg = 0.0
    for i, score in enumerate(scores[:k]):
        dcg += (2 ** score - 1) / math.log2(i + 2)
    return dcg


def ndcg_at_k(relevance_scores: List[int], k: int) -> float:
    """Normalized DCG at k."""
    ideal = sorted(relevance_scores, reverse=True)
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(relevance_scores, k) / idcg


def recall_at_k(retrieved_ids: List[int], relevant_ids: List[int], k: int) -> float:
    """Fraction of relevant chunks retrieved in top-k."""
    if not relevant_ids:
        return 1.0  # if nothing expected, don't penalize
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(retrieved_set & relevant_set) / len(relevant_set)


def precision_at_k(retrieved_ids: List[int], relevant_ids: List[int], k: int) -> float:
    """Fraction of top-k that are relevant."""
    if k == 0:
        return 0.0
    retrieved_set = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    return len(retrieved_set & relevant_set) / k


# ═══════════════════════════════════════════════
# Answer Evaluation (LLM-as-Judge light)
# ═══════════════════════════════════════════════

def token_overlap_f1(predicted: str, reference: str) -> float:
    """Token-level F1 between predicted answer and gold answer."""
    pred_tokens = set(predicted.lower().split())
    ref_tokens = set(reference.lower().split())
    if not pred_tokens or not ref_tokens:
        return 0.0
    tp = len(pred_tokens & ref_tokens)
    precision = tp / len(pred_tokens)
    recall = tp / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def exact_match(predicted: str, reference: str) -> float:
    """Exact match (case-insensitive, trimmed)."""
    return 1.0 if predicted.strip().lower() == reference.strip().lower() else 0.0


def keyword_match_score(predicted: str, reference: str) -> float:
    """Fraction of reference keywords found in prediction."""
    ref_words = set(reference.lower().split())
    pred_words = set(predicted.lower().split())
    if not ref_words:
        return 1.0
    return len(ref_words & pred_words) / len(ref_words)


def hallucination_check(answer: str, sources: List[str]) -> float:
    """
    Rough hallucination score: 0 = fully grounded, 1 = no source support.
    Checks minimum Jaccard overlap between answer tokens and any source chunk.
    """
    if not sources:
        return 1.0
    answer_words = set(answer.lower().split())
    if not answer_words:
        return 0.0

    max_overlap = 0.0
    for src in sources:
        src_words = set(src.lower().split())
        if not src_words:
            continue
        overlap = len(answer_words & src_words) / len(answer_words)
        max_overlap = max(max_overlap, overlap)

    return 1.0 - max_overlap  # 0 = fully grounded


# ═══════════════════════════════════════════════
# Cost Estimation
# ═══════════════════════════════════════════════

def compute_cost_table() -> Dict:
    """
    Compare FAISS (local/in-memory) vs a managed vector DB (Pinecone-like pricing).
    Assumptions: $0.025/hr/pod for managed, FAISS is free (self-hosted CPU).
    """
    vectors_per_month = [100_000, 1_000_000, 10_000_000]
    dim = 384  # all-MiniLM-L6-v2
    bytes_per_vector = dim * 4  # float32

    managed_pod_cost_per_hr = 0.025  # typical managed pod
    hours_per_month = 730

    rows = []
    for n_vectors in vectors_per_month:
        # FAISS: one-time memory cost, no recurring fee
        faiss_mem_gb = (n_vectors * bytes_per_vector) / (1024 ** 3)
        faiss_monthly = 0.0  # local, only infra cost

        # Managed: pod pricing (always-on)
        managed_monthly = managed_pod_cost_per_hr * hours_per_month

        rows.append({
            "vector_count": n_vectors,
            "faiss_monthly_cost_usd": faiss_monthly,
            "faiss_memory_gb": round(faiss_mem_gb, 2),
            "managed_monthly_cost_usd": round(managed_monthly, 2),
            "comments": "FAISS is free (self-hosted); managed assumes 1 pod always-on"
        })

    return {
        "assumptions": {
            "embedding_dim": dim,
            "dtype": "float32",
            "managed_pod_cost_per_hour": managed_pod_cost_per_hr,
            "hours_per_month": hours_per_month,
            "faiss_store": "in-memory, no persistent server cost"
        },
        "comparison": rows,
        "tradeoffs": [
            "FAISS: no recurring cost, but limited to single-machine memory",
            "FAISS: no built-in persistence — must save/load index manually",
            "Managed: auto-scaling, persistence, RBAC built-in",
            "Switch to managed when: multi-node deployment needed, or >50M vectors",
        ]
    }


# ═══════════════════════════════════════════════
# Main Evaluation Runner
# ═══════════════════════════════════════════════

def run_evaluation(
    doc_path: str,
    questions_file: str = "evaluation/bert_test_questions.json",
    output_file: str = "evaluation/evaluation_results.json",
    top_k: int = 10,
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> EvalResult:
    """Run full evaluation suite."""
    result = EvalResult()

    # Load questions
    with open(questions_file, "r") as f:
        q_data = json.load(f)
    questions = q_data["questions"]

    print(f"Loaded {len(questions)} questions")
    print(f"Initializing pipeline with: {doc_path}")

    # Initialize pipeline
    pipeline = RagPipelineConnector(
        doc_path=doc_path,
        top_k=top_k,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    print("Pipeline ready. Running queries...\n")

    # Per-question metrics accumulators
    all_recall = []
    all_hits = []
    all_mrr_recip = []
    all_ndcg_raw = []
    all_precision = []
    all_latency = []
    all_chunk_counts = []
    all_faithfulness = []
    all_relevance = []
    all_em = []
    all_f1 = []

    for i, q in enumerate(questions):
        qid = q["id"]
        question = q["question"]
        relevant_ids = q["relevant_chunk_ids"]
        gold_answer = q["gold_answer"]

        print(f"[{i+1}/{len(questions)}] Q{qid}: {question}")

        # Time the query
        t_start = time.time()
        try:
            response = pipeline.query(question)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue
        t_end = time.time()
        latency_ms = (t_end - t_start) * 1000
        all_latency.append(latency_ms)

        answer = response.get("answer", "")
        source_chunks = response.get("source_chunks", [])
        chunk_count = response.get("total_sources", 0)
        all_chunk_counts.append(chunk_count)

        # Get chunk IDs from retrieval
        retrieved_chunk_ids = [c["id"] for c in source_chunks]
        source_texts = [c["text"] for c in source_chunks]

        # ── Retrieval Metrics ──
        rec = recall_at_k(retrieved_chunk_ids, relevant_ids, top_k)
        prec = precision_at_k(retrieved_chunk_ids, relevant_ids, top_k)
        hit = 1.0 if len(set(retrieved_chunk_ids) & set(relevant_ids)) > 0 else 0.0

        # MRR: reciprocal rank of first relevant
        mrr = 0.0
        for rank, cid in enumerate(retrieved_chunk_ids, start=1):
            if cid in relevant_ids:
                mrr = 1.0 / rank
                break
        all_mrr_recip.append(mrr)

        # nDCG: treat each chunk as relevant=1 or 0
        rel_scores = [1 if cid in relevant_ids else 0 for cid in retrieved_chunk_ids]
        ndcg = ndcg_at_k(rel_scores, top_k)
        all_ndcg_raw.append(ndcg)

        all_recall.append(rec)
        all_hits.append(hit)
        all_precision.append(prec)

        # ── Answer Metrics ──
        if answer and answer.strip():
            em = exact_match(answer, gold_answer)
            f1 = token_overlap_f1(answer, gold_answer)
            faithfulness = 1.0 - hallucination_check(answer, source_texts)
            relevance = keyword_match_score(answer, gold_answer)
        else:
            # No LLM answer — retrieval-only mode
            em = 0.0
            f1 = 0.0
            faithfulness = 0.0
            relevance = 0.0

        all_em.append(em)
        all_f1.append(f1)
        all_faithfulness.append(faithfulness)
        all_relevance.append(relevance)

        # Store per-question result
        result.per_question.append({
            "id": qid,
            "question": question,
            "gold_answer": gold_answer,
            "retrieved_chunk_ids": retrieved_chunk_ids[:top_k],
            "relevant_chunk_ids": relevant_ids,
            "answer": answer[:500],
            "latency_ms": round(latency_ms, 2),
            "chunks_retrieved": chunk_count,
            "recall_at_k": round(rec, 4),
            "hit": hit == 1.0,
            "mrr": round(mrr, 4),
            "ndcg_at_k": round(ndcg, 4),
            "precision_at_k": round(prec, 4),
            "exact_match": em == 1.0,
            "token_f1": round(f1, 4),
            "faithfulness": round(faithfulness, 4),
            "relevance": round(relevance, 4),
        })

        print(f"  → Recall@{top_k}: {rec:.3f} | MRR: {mrr:.3f} | "
              f"Faithfulness: {faithfulness:.3f} | Latency: {latency_ms:.0f}ms")

    # ── Aggregate Metrics ──
    n = len(all_recall)
    if n > 0:
        result.retrieval.recall_at_k = sum(all_recall) / n
        result.retrieval.hit_rate = sum(all_hits) / n
        result.retrieval.mrr = sum(all_mrr_recip) / n
        result.retrieval.ndcg_at_k = sum(all_ndcg_raw) / n
        result.retrieval.context_precision = sum(all_precision) / n
        result.retrieval.avg_latency_ms = sum(all_latency) / n
        result.retrieval.avg_chunks_retrieved = sum(all_chunk_counts) / n

        result.answer.faithfulness_avg = sum(all_faithfulness) / n
        result.answer.relevance_avg = sum(all_relevance) / n
        result.answer.hallucination_rate = sum(1 - f for f in all_faithfulness) / n
        result.answer.em_score = sum(all_em) / n
        result.answer.f1_score = sum(all_f1) / n

    # Cost estimate
    result.cost_estimate = compute_cost_table()

    # ── Write results ──
    output = {
        "summary": {
            "retrieval": asdict(result.retrieval),
            "answer": asdict(result.answer),
            "cost": result.cost_estimate,
        },
        "per_question": result.per_question,
        "config": {
            "doc_path": doc_path,
            "top_k": top_k,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "num_questions": n,
        }
    }

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"EVALUATION COMPLETE — Results saved to {output_file}")
    print(f"{'='*60}")
    print(f"Retrieval: Recall@{top_k}={result.retrieval.recall_at_k:.4f} | "
          f"MRR={result.retrieval.mrr:.4f} | nDCG={result.retrieval.ndcg_at_k:.4f}")
    print(f"Answer:    Faithfulness={result.answer.faithfulness_avg:.4f} | "
          f"Relevance={result.answer.relevance_avg:.4f} | EM={result.answer.em_score:.4f}")
    print(f"Latency:   p50={result.retrieval.avg_latency_ms:.0f}ms avg")
    print(f"{'='*60}")

    return result


# ═══════════════════════════════════════════════
# CLI Entry Point
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Evaluation Harness")
    parser.add_argument("--doc", type=str, required=True, help="Path to document (PDF/HTML/MD)")
    parser.add_argument("--questions", type=str, default="evaluation/bert_test_questions.json")
    parser.add_argument("--output", type=str, default="evaluation/evaluation_results.json")
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--chunk_size", type=int, default=512)
    parser.add_argument("--chunk_overlap", type=int, default=50)
    args = parser.parse_args()

    run_evaluation(
        doc_path=args.doc,
        questions_file=args.questions,
        output_file=args.output,
        top_k=args.top_k,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    )