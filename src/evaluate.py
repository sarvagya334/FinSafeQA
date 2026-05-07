"""
evaluate.py — Sequential multi-model RAG evaluator.

Architecture:
  • Models run one at a time, ordered small → large
  • For each model: iterate all benchmark queries sequentially
  • Each result row is written to CSV immediately after scoring (no buffering)
  • Per-model CSVs + combined CSV are updated live throughout the run

Usage:
  python3 -m src.evaluate                          # all models, small → large
  python3 -m src.evaluate --models nemotron-8b llama4-scout  # subset
  python3 -m src.evaluate --dry-run                # retrieval only, no LLM
"""

import os
import csv
import json
import time
import argparse
import numpy as np
from tqdm import tqdm
from typing import Dict, List, Optional
from dotenv import load_dotenv

# Load .env BEFORE anything reads env vars
load_dotenv()

from src.rag import retrieve_and_build_prompt, rag_query, init_llm_nvidia
from src.embeddings import load_embedding_model
from src.country_indexes import load_country_indexes
from src.country_detect import detect_country_from_query


# ---------------------------------------------------------------------------
# Model registry — ordered small → large (approximate active params)
# Verify IDs at: https://build.nvidia.com/explore/discover
# ---------------------------------------------------------------------------

ALL_MODELS: Dict[str, str] = {
    "llama3-8b":        "meta/llama-3.1-8b-instruct",                   #   8B ✓         #  17B
    "llama4-maverick":  "meta/llama-4-maverick-17b-128e-instruct",       #  17B
    "gpt-oss-120b":     "openai/gpt-oss-120b", 
}

# CSV columns — fixed order for all output files
CSV_COLUMNS = [
    "model", "model_id", "benchmark_file", "type", "query", "expected", "actual",
    "correct", "exact_match", "f1_token_match", "rouge1_f1", "rouge_l_f1",
    "semantic_score", "exact_numeric", "approx_numeric",
    "answer_relevance", "is_refusal", "response_words",
    "detected_country", "latency_sec",
]


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class FinancialRAGEvaluator:
    def __init__(
        self,
        benchmark_dir: str,
        index_path: str,
        model_ids: Dict[str, str],
        out_dir: str = "eval_results",
        combined_csv: str = "eval_results_detailed.csv",
        dry_run: bool = False,
    ):
        self.benchmark_dir = benchmark_dir
        self.model_ids = model_ids
        self.out_dir = out_dir
        self.combined_csv = combined_csv
        self.dry_run = dry_run

        print("[Eval] Initializing Embeddings...")
        self.embed_model = load_embedding_model()
        self.country_indexes = load_country_indexes(index_path)
        print(f"[Eval] Loaded Market Indexes: {list(self.country_indexes.keys())}")

        if dry_run:
            print("[Eval] DRY-RUN mode: LLM calls will be skipped.")
            self.llms: Dict[str, callable] = {}
        else:
            api_key = os.environ.get("NVIDIA_API_KEY", "")
            if not api_key:
                raise ValueError("NVIDIA_API_KEY not found. Set it in your .env file.")
            print(f"[Eval] Initializing {len(model_ids)} NVIDIA LLMs (small → large)...")
            self.llms = {
                name: init_llm_nvidia(api_key=api_key, model=mid)
                for name, mid in model_ids.items()
            }
            for name, mid in model_ids.items():
                print(f"  ✓ {name:20s} → {mid}")

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        import re
        return re.findall(r"[a-z0-9]+(?:[.\-/%][a-z0-9]+)*", text.lower())

    @staticmethod
    def _f1_token_match(reference: str, hypothesis: str) -> float:
        import re
        from collections import Counter
        def toks(s): return re.findall(r"[a-z0-9]+(?:[.\-/%][a-z0-9]+)*", s.lower())
        ref, hyp = toks(reference), toks(hypothesis)
        if not ref or not hyp:
            return 0.0
        common = Counter(ref) & Counter(hyp)
        n = sum(common.values())
        if n == 0:
            return 0.0
        p, r = n / len(hyp), n / len(ref)
        return 2 * p * r / (p + r)

    @staticmethod
    def _rouge1_f1(reference: str, hypothesis: str) -> float:
        ref_t = set(reference.lower().split())
        hyp_t = set(hypothesis.lower().split())
        if not ref_t or not hyp_t:
            return 0.0
        ov = ref_t & hyp_t
        p, r = len(ov) / len(hyp_t), len(ov) / len(ref_t)
        return 0.0 if p + r == 0 else 2 * p * r / (p + r)

    @staticmethod
    def _rouge_l_f1(reference: str, hypothesis: str) -> float:
        ref = reference.lower().split()
        hyp = hypothesis.lower().split()
        if not ref or not hyp:
            return 0.0
        m, n = len(ref), len(hyp)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i-1][j-1] + 1 if ref[i-1] == hyp[j-1] else max(dp[i-1][j], dp[i][j-1])
        lcs = dp[m][n]
        p, r = lcs / n, lcs / m
        return 0.0 if p + r == 0 else 2 * p * r / (p + r)

    @staticmethod
    def _numeric_match(reference: str, hypothesis: str, tolerance: float = 0.05) -> dict:
        import re
        def extract(text):
            nums = []
            for r in re.findall(r"\d[\d,]*(?:\.\d+)?", text.replace(",", "")):
                try: nums.append(float(r))
                except ValueError: pass
            return nums
        ref_nums = extract(reference)
        hyp_nums = extract(hypothesis)
        if not ref_nums:
            return {"exact_numeric": None, "approx_numeric": None}
        exact  = any(r in hyp_nums for r in ref_nums)
        approx = any(
            any(abs(h - r) / max(abs(r), 1e-9) <= tolerance for h in hyp_nums)
            for r in ref_nums
        )
        return {"exact_numeric": exact, "approx_numeric": approx}

    def _semantic_similarity(self, a: str, b: str) -> float:
        vecs = self.embed_model.encode([a, b], normalize_embeddings=True, show_progress_bar=False)
        return float(np.dot(vecs[0], vecs[1]))

    def _answer_relevance(self, query: str, actual: str) -> float:
        return self._semantic_similarity(query, actual)

    def score_response(self, expected: str, actual: str, q_type: str, query: str) -> Dict:
        actual_lower = actual.lower().strip()
        expected_str = str(expected).lower().strip()
        detected     = detect_country_from_query(query)
        is_refusal   = any(k in actual_lower for k in
                           ["not found", "no information", "not available", "not mention"])

        if q_type == "hallucination" or expected_str == "not found":
            correct = is_refusal
            return {
                "correct": correct, "exact_match": correct,
                "f1_token_match": 1.0 if correct else 0.0,
                "rouge1_f1": None, "rouge_l_f1": None, "semantic_score": None,
                "exact_numeric": None, "approx_numeric": None,
                "answer_relevance": round(self._answer_relevance(query, actual), 4),
                "detected_country": detected, "is_refusal": is_refusal,
                "response_words": len(actual.split()),
            }

        exact_match    = expected_str in actual_lower
        f1_token       = self._f1_token_match(expected_str, actual_lower)
        rouge1         = self._rouge1_f1(expected_str, actual_lower)
        rouge_l        = self._rouge_l_f1(expected_str, actual_lower)
        semantic_score = self._semantic_similarity(expected_str, actual_lower)
        num            = self._numeric_match(expected_str, actual_lower)
        ans_rel        = self._answer_relevance(query, actual_lower)

        correct = (exact_match or f1_token >= 0.5
                   or semantic_score >= 0.75 or num.get("approx_numeric") is True)

        return {
            "correct": correct, "exact_match": exact_match,
            "f1_token_match": round(f1_token, 4),
            "rouge1_f1": round(rouge1, 4), "rouge_l_f1": round(rouge_l, 4),
            "semantic_score": round(semantic_score, 4),
            "exact_numeric": num["exact_numeric"], "approx_numeric": num["approx_numeric"],
            "answer_relevance": round(ans_rel, 4),
            "detected_country": detected, "is_refusal": is_refusal,
            "response_words": len(actual.split()),
        }

    # ------------------------------------------------------------------
    # Live CSV helpers
    # ------------------------------------------------------------------

    def _open_writer(self, path: str, append: bool = False):
        """Return (file_handle, DictWriter). Writes header if creating new file."""
        mode = "a" if append else "w"
        fh = open(path, mode, newline="", encoding="utf-8")
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        if not append or os.path.getsize(path) == 0:
            writer.writeheader()
        return fh, writer

    def _build_row(self, model_name: str, filename: str, q_type: str,
                   query: str, ground_truth: str, response: str,
                   metrics: Dict, elapsed: float) -> Dict:
        return {
            "model":            model_name,
            "model_id":         self.model_ids.get(model_name, model_name),
            "benchmark_file":   filename,
            "type":             q_type,
            "query":            query,
            "expected":         ground_truth,
            "actual":           response,
            "correct":          metrics["correct"],
            "exact_match":      metrics.get("exact_match", False),
            "f1_token_match":   metrics.get("f1_token_match"),
            "rouge1_f1":        metrics.get("rouge1_f1"),
            "rouge_l_f1":       metrics.get("rouge_l_f1"),
            "semantic_score":   metrics.get("semantic_score"),
            "exact_numeric":    metrics.get("exact_numeric"),
            "approx_numeric":   metrics.get("approx_numeric"),
            "answer_relevance": metrics.get("answer_relevance"),
            "is_refusal":       metrics.get("is_refusal", False),
            "response_words":   metrics.get("response_words"),
            "detected_country": metrics["detected_country"],
            "latency_sec":      round(elapsed, 2),
        }

    # ------------------------------------------------------------------
    # Single model call
    # ------------------------------------------------------------------

    def _call_model(self, model_name: str, query: str, retries: int = 2,
                    backoff: float = 5.0) -> str:
        if self.dry_run:
            return "[DRY-RUN] retrieval only"

        prompt = retrieve_and_build_prompt(
            query=query,
            country_indexes=self.country_indexes,
            embed_model=self.embed_model,
        )
        if prompt is None:
            return "No relevant documents found."
        if prompt == "__SUMMARIZE_ALL__":
            return rag_query(
                query=query,
                country_indexes=self.country_indexes,
                embed_model=self.embed_model,
                llm=self.llms[model_name],
            )

        llm_fn = self.llms[model_name]
        last_exc = None
        for attempt in range(1, retries + 2):
            try:
                return llm_fn(prompt).strip()
            except Exception as exc:
                err = str(exc)
                if "401" in err or "Unauthorized" in err:
                    raise RuntimeError(
                        f"401 Unauthorized for '{model_name}'. "
                        f"Check NVIDIA_API_KEY.\nOriginal: {exc}"
                    ) from exc
                last_exc = exc
                if attempt <= retries:
                    wait = backoff * attempt
                    print(f"\n  [Retry {attempt}] {model_name}: {exc}. Waiting {wait:.0f}s...")
                    time.sleep(wait)
        return f"ERROR: {last_exc}"

    # ------------------------------------------------------------------
    # Main evaluation loop — sequential, live CSV writes
    # ------------------------------------------------------------------

    def run_suite(self) -> None:
        os.makedirs(self.out_dir, exist_ok=True)
        files = sorted(f for f in os.listdir(self.benchmark_dir) if f.endswith(".json"))

        # Load all benchmark data once
        benchmarks = []
        for filename in files:
            with open(os.path.join(self.benchmark_dir, filename), encoding="utf-8") as f:
                benchmarks.append((filename, json.load(f)))

        total_queries = sum(len(d) for _, d in benchmarks)
        print(f"\n[Eval] Found {len(files)} benchmark files, {total_queries} total queries.")
        print(f"[Eval] Running {len(self.model_ids)} models sequentially (small → large).\n")

        # Combined CSV: write header once
        combined_fh, combined_writer = self._open_writer(self.combined_csv, append=False)

        try:
            for model_idx, (model_name, model_id) in enumerate(self.model_ids.items(), 1):
                model_csv = os.path.join(
                    self.out_dir,
                    f"{model_name.replace('/', '_')}_eval.csv"
                )
                model_fh, model_writer = self._open_writer(model_csv, append=False)

                correct_count = 0
                total_count   = 0

                print(f"\n{'='*60}")
                print(f"[{model_idx}/{len(self.model_ids)}] Model: {model_name}")
                print(f"  Full ID : {model_id}")
                print(f"  CSV     : {model_csv}")
                print(f"{'='*60}")

                try:
                    for filename, benchmark_data in benchmarks:
                        print(f"  ── {filename} ({len(benchmark_data)} queries)")
                        for item in tqdm(benchmark_data, desc=f"  {model_name[:18]}", leave=False):
                            query        = item["query"]
                            ground_truth = item["answer"]
                            q_type       = item.get("type", "retrieval")
                            start        = time.time()

                            try:
                                response = self._call_model(model_name, query)
                            except RuntimeError as e:
                                print(f"\n[FATAL] {e}")
                                print("[Eval] Stopping this model. Moving to next.")
                                response = f"FATAL: {e}"

                            elapsed = time.time() - start
                            metrics = self.score_response(ground_truth, response, q_type, query)
                            row     = self._build_row(
                                model_name, filename, q_type,
                                query, ground_truth, response, metrics, elapsed
                            )

                            # ---- Write immediately to both CSVs ----
                            model_writer.writerow(row)
                            model_fh.flush()
                            combined_writer.writerow(row)
                            combined_fh.flush()

                            total_count += 1
                            if metrics["correct"]:
                                correct_count += 1

                except Exception as e:
                    print(f"\n[Error] Unexpected error for model '{model_name}': {e}")
                finally:
                    model_fh.close()

                acc = correct_count / total_count if total_count else 0
                print(f"\n  ✓ {model_name} done: {correct_count}/{total_count} correct ({acc:.1%})")
                print(f"    Results → {model_csv}")

        finally:
            combined_fh.close()

        print(f"\n{'='*60}")
        print(f"All models complete. Combined CSV → '{self.combined_csv}'")
        print(f"Per-model CSVs    → '{self.out_dir}/'")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Sequential multi-model Financial RAG Evaluator")
    parser.add_argument("--dry-run", action="store_true",
                        help="Skip LLM calls; test retrieval pipeline only.")
    parser.add_argument("--models", nargs="+", default=list(ALL_MODELS.keys()),
                        choices=list(ALL_MODELS.keys()), metavar="MODEL",
                        help=f"Models to run (in order given). Default: all, small→large.")
    parser.add_argument("--bench-dir", default="src/benchmarks")
    parser.add_argument("--index-dir", default="Agentic_RAG_v2/data/processed/country_indexes")
    parser.add_argument("--out-dir",   default="eval_results",
                        help="Directory for per-model CSV files.")
    parser.add_argument("--out",       default="eval_results_detailed.csv",
                        help="Combined CSV path (all models).")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    # Preserve the small→large order from ALL_MODELS for selected models
    selected = {k: ALL_MODELS[k] for k in args.models}

    evaluator = FinancialRAGEvaluator(
        benchmark_dir=args.bench_dir,
        index_path=args.index_dir,
        model_ids=selected,
        out_dir=args.out_dir,
        combined_csv=args.out,
        dry_run=args.dry_run,
    )
    evaluator.run_suite()