import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Callable

def compute_recall_at_k(retrieved_ids: List[str], ground_truth_ids: List[str], k: int) -> int:
    top_k = set(retrieved_ids[:k])
    truth = set(ground_truth_ids)
    return int(bool(top_k & truth))

def compute_mrr(retrieved_ids: List[str], ground_truth_ids: List[str]) -> float:
    truth = set(ground_truth_ids)
    for rank, cid in enumerate(retrieved_ids, start=1):
        if cid in truth:
            return 1.0 / rank
    return 0.0

def run_evaluation(benchmark_file: Path, retriever_fn: Callable[[str, int], List[Dict[str, Any]]]) -> Dict[str, float]:
    with open(benchmark_file, "r") as f:
        benchmarks = json.load(f)

    k_values = [1, 3, 5, 10]
    metrics = {f"Recall@{k}": [] for k in k_values}
    metrics["MRR"] = []

    for item in benchmarks:
        q = item["question"]
        relevant = item["relevant_chunks"]
        retrieved_docs = retriever_fn(q, top_k=10)
        retrieved_ids = [d["chunk_id"] for d in retrieved_docs]

        for k in k_values:
            metrics[f"Recall@{k}"].append(compute_recall_at_k(retrieved_ids, relevant, k))
        metrics["MRR"].append(compute_mrr(retrieved_ids, relevant))

    return {k: float(np.mean(v)) for k, v in metrics.items()}