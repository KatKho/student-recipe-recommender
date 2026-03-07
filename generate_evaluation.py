"""
evaluate.py – Evaluation script for the Recipe Search & Recommendation System
==============================================================================
Measures:
  • Precision@K   – fraction of top-K results that are relevant
  • Recall@K      – fraction of total relevant recipes surfaced in top-K
  • nDCG@K        – normalised Discounted Cumulative Gain (ranking quality)
  • MRR           – Mean Reciprocal Rank (how high the first hit lands)
  • Average Latency – wall-clock search time per query

Relevance is judged automatically using two complementary heuristics:
  1. KEYWORD queries  → a result is relevant if every query word appears in
     the recipe title or ingredient list (lenient substring match).
  2. INGREDIENT queries → a result is relevant when ingredient_overlap_score
     meets a configurable MIN_OVERLAP_SCORE threshold.
  3. HYBRID queries   → both criteria above are combined (union).

Usage
-----
  # Basic run (assumes data/clean_recipes.pkl already exists):
  python evaluate.py

  # Custom data path:
  python evaluate.py --data path/to/clean_recipes.pkl

  # Save results to CSV:
  python evaluate.py --csv results/eval_results.csv

  # Adjust top-K and overlap threshold:
  python evaluate.py --k 5 --min-overlap 0.3
"""

import argparse
import math
import os
import re
import time
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd

# ── project imports ──────────────────────────────────────────────────────────
from recipe_search import load_data, build_index, search, ingredient_overlap_score

# ─────────────────────────────────────────────────────────────────────────────
# Test Suite
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TestQuery:
    name: str                                   # human-readable label
    query: Optional[str] = None                 # keyword search string
    ingredients: Optional[List[str]] = None     # ingredient list
    exclude_ingredients: Optional[List[str]] = None
    alpha: float = 0.7
    beta: float = 0.3
    mode: str = "hybrid"                        # "keyword" | "ingredient" | "hybrid"

# ---------------------------------------------------------------------------
# Edit this list to add / remove test cases
# ---------------------------------------------------------------------------
TEST_QUERIES: List[TestQuery] = [
    # ── Keyword-only ──────────────────────────────────────────────────────
    TestQuery(name="fried rice",        query="fried rice",        mode="keyword",    alpha=1.0, beta=0.0),
    TestQuery(name="chocolate cake",    query="chocolate cake",    mode="keyword",    alpha=1.0, beta=0.0),
    TestQuery(name="pasta carbonara",   query="pasta carbonara",   mode="keyword",    alpha=1.0, beta=0.0),
    TestQuery(name="chicken soup",      query="chicken soup",      mode="keyword",    alpha=1.0, beta=0.0),
    TestQuery(name="banana bread",      query="banana bread",      mode="keyword",    alpha=1.0, beta=0.0),
    TestQuery(name="scrambled eggs",    query="scrambled eggs",    mode="keyword",    alpha=1.0, beta=0.0),
    TestQuery(name="tomato salad",      query="tomato salad",      mode="keyword",    alpha=1.0, beta=0.0),

    # ── Ingredient-only ───────────────────────────────────────────────────
    TestQuery(name="eggs + rice",               ingredients=["eggs", "rice"],                          mode="ingredient", alpha=0.0, beta=1.0),
    TestQuery(name="chicken + garlic + lemon",  ingredients=["chicken", "garlic", "lemon"],            mode="ingredient", alpha=0.0, beta=1.0),
    TestQuery(name="flour + butter + sugar",    ingredients=["flour", "butter", "sugar"],              mode="ingredient", alpha=0.0, beta=1.0),
    TestQuery(name="pasta + tomato + basil",    ingredients=["pasta", "tomato", "basil"],              mode="ingredient", alpha=0.0, beta=1.0),
    TestQuery(name="onion + potato + carrot",   ingredients=["onion", "potato", "carrot"],             mode="ingredient", alpha=0.0, beta=1.0),
    TestQuery(name="milk + egg + vanilla",      ingredients=["milk", "egg", "vanilla"],                mode="ingredient", alpha=0.0, beta=1.0),

    # ── Hybrid ────────────────────────────────────────────────────────────
    TestQuery(name="stir fry + chicken + garlic", query="stir fry", ingredients=["chicken", "garlic", "onion"], mode="hybrid"),
    TestQuery(name="breakfast + eggs + cheese",   query="breakfast",ingredients=["eggs", "cheese"],              mode="hybrid"),
    TestQuery(name="soup + lentil + carrot",      query="soup",     ingredients=["lentil", "carrot", "onion"],   mode="hybrid"),
    TestQuery(name="curry + rice",                query="curry",    ingredients=["rice", "coconut milk"],        mode="hybrid"),
    TestQuery(name="salad + avocado + lime",      query="salad",    ingredients=["avocado", "lime"],             mode="hybrid"),

    # ── Exclusion tests ───────────────────────────────────────────────────
    TestQuery(name="pasta (no meat)",          query="pasta",   exclude_ingredients=["beef", "pork", "chicken"], mode="keyword", alpha=1.0, beta=0.0),
    TestQuery(name="cookies (no nuts)",        query="cookies", exclude_ingredients=["walnut", "almond", "pecan"], mode="keyword", alpha=1.0, beta=0.0),
]


# ─────────────────────────────────────────────────────────────────────────────
# Relevance Heuristics
# ─────────────────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"[^a-z0-9\s]", " ", text.lower())


def _keyword_relevant(result: Dict[str, Any], query: str) -> bool:
    """True when every word in the query appears somewhere in title or ingredients."""
    if not query:
        return False
    title = _normalize(result.get("title", ""))
    ings  = _normalize(" ".join(result.get("ingredients", [])))
    combined = title + " " + ings
    return all(word in combined for word in query.lower().split())


def _ingredient_relevant(result: Dict[str, Any],
                          ingredients: List[str],
                          min_overlap: float) -> bool:
    """True when ingredient overlap score ≥ min_overlap threshold."""
    if not ingredients:
        return False
    score = ingredient_overlap_score(
        [i.lower() for i in ingredients],
        result.get("ingredients", [])
    )
    return score >= min_overlap


def is_relevant(result: Dict[str, Any],
                tq: TestQuery,
                min_overlap: float) -> bool:
    kw_hit  = _keyword_relevant(result, tq.query) if tq.query else False
    ing_hit = _ingredient_relevant(result, tq.ingredients or [], min_overlap)

    if tq.mode == "keyword":
        return kw_hit
    if tq.mode == "ingredient":
        return ing_hit
    return kw_hit or ing_hit  # hybrid: union


# ─────────────────────────────────────────────────────────────────────────────
# IR Metric Helpers
# ─────────────────────────────────────────────────────────────────────────────

def precision_at_k(relevances: List[int]) -> float:
    if not relevances:
        return 0.0
    return sum(relevances) / len(relevances)


def recall_at_k(relevances: List[int], total_relevant: int) -> float:
    if total_relevant == 0:
        return 0.0
    return sum(relevances) / total_relevant


def dcg_at_k(relevances: List[int]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def ndcg_at_k(relevances: List[int]) -> float:
    ideal = sorted(relevances, reverse=True)
    ideal_dcg = dcg_at_k(ideal)
    if ideal_dcg == 0:
        return 0.0
    return dcg_at_k(relevances) / ideal_dcg


def reciprocal_rank(relevances: List[int]) -> float:
    for i, rel in enumerate(relevances):
        if rel:
            return 1.0 / (i + 1)
    return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Per-Query Evaluation
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QueryResult:
    name: str
    mode: str
    k: int
    precision: float
    recall: float
    ndcg: float
    mrr: float
    latency_ms: float
    n_results: int
    n_relevant_in_results: int
    total_relevant_estimate: int


def evaluate_query(tq: TestQuery,
                   df: pd.DataFrame,
                   bm25,
                   k: int,
                   min_overlap: float) -> QueryResult:

    t0 = time.perf_counter()
    results = search(
        df, bm25,
        query=tq.query,
        ingredients=tq.ingredients,
        exclude_ingredients=tq.exclude_ingredients,
        alpha=tq.alpha,
        beta=tq.beta,
        top_k=k,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    relevances = [int(is_relevant(r, tq, min_overlap)) for r in results]

    # Estimate total relevant by scanning first 200 results (fast approximation)
    extended = search(
        df, bm25,
        query=tq.query,
        ingredients=tq.ingredients,
        exclude_ingredients=tq.exclude_ingredients,
        alpha=tq.alpha,
        beta=tq.beta,
        top_k=200,
    )
    total_relevant = sum(int(is_relevant(r, tq, min_overlap)) for r in extended)
    total_relevant = max(total_relevant, sum(relevances))  # at least what we found

    return QueryResult(
        name=tq.name,
        mode=tq.mode,
        k=k,
        precision=precision_at_k(relevances),
        recall=recall_at_k(relevances, total_relevant),
        ndcg=ndcg_at_k(relevances),
        mrr=reciprocal_rank(relevances),
        latency_ms=latency_ms,
        n_results=len(results),
        n_relevant_in_results=sum(relevances),
        total_relevant_estimate=total_relevant,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Report Formatting
# ─────────────────────────────────────────────────────────────────────────────

def print_header(title: str) -> None:
    width = 90
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_results_table(rows: List[QueryResult], k: int) -> None:
    hdr = f"{'Query':<35} {'Mode':<12} {'P@'+str(k):<8} {'R@'+str(k):<8} {'nDCG@'+str(k):<10} {'MRR':<8} {'ms':<8} {'Hits'}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        hits = f"{r.n_relevant_in_results}/{r.n_results}"
        print(
            f"{r.name:<35} {r.mode:<12} "
            f"{r.precision:<8.3f} {r.recall:<8.3f} "
            f"{r.ndcg:<10.3f} {r.mrr:<8.3f} "
            f"{r.latency_ms:<8.1f} {hits}"
        )


def print_summary(rows: List[QueryResult], mode_filter: Optional[str] = None) -> None:
    subset = [r for r in rows if mode_filter is None or r.mode == mode_filter]
    if not subset:
        return
    label = mode_filter.upper() if mode_filter else "ALL"
    k = subset[0].k
    print(f"\n  [{label}] Macro-averaged over {len(subset)} queries (K={k})")
    print(f"    Precision@{k}  : {np.mean([r.precision  for r in subset]):.3f}")
    print(f"    Recall@{k}     : {np.mean([r.recall     for r in subset]):.3f}")
    print(f"    nDCG@{k}       : {np.mean([r.ndcg       for r in subset]):.3f}")
    print(f"    MRR           : {np.mean([r.mrr         for r in subset]):.3f}")
    print(f"    Avg latency   : {np.mean([r.latency_ms  for r in subset]):.1f} ms")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate recipe search engine")
    parser.add_argument("--data",        default="data/clean_recipes.pkl", help="Path to clean_recipes.pkl")
    parser.add_argument("--k",           type=int,   default=10,  help="Top-K cutoff (default 10)")
    parser.add_argument("--min-overlap", type=float, default=0.25, help="Min ingredient overlap to count as relevant (default 0.25)")
    parser.add_argument("--csv",         default=None, help="Optional path to save results CSV")
    args = parser.parse_args()

    # ── Load & index ─────────────────────────────────────────────────────────
    print(f"Loading dataset from: {args.data}")
    df = load_data(args.data)
    print(f"  {len(df):,} recipes loaded")

    print("Building BM25 index …")
    t0 = time.perf_counter()
    bm25 = build_index(df)
    print(f"  Done in {(time.perf_counter()-t0)*1000:.0f} ms\n")

    # ── Run evaluation ───────────────────────────────────────────────────────
    all_results: List[QueryResult] = []

    print_header(f"PER-QUERY METRICS  (K={args.k}  |  min_overlap={args.min_overlap})")

    for tq in TEST_QUERIES:
        qr = evaluate_query(tq, df, bm25, k=args.k, min_overlap=args.min_overlap)
        all_results.append(qr)

    print_results_table(all_results, args.k)

    # ── Aggregated summaries ─────────────────────────────────────────────────
    print_header("AGGREGATED SUMMARIES")
    for mode in ("keyword", "ingredient", "hybrid"):
        print_summary(all_results, mode_filter=mode)
    print_summary(all_results, mode_filter=None)

    # ── Exclusion sanity check ───────────────────────────────────────────────
    print_header("EXCLUSION SANITY CHECK")
    excl_queries = [tq for tq in TEST_QUERIES if tq.exclude_ingredients]
    if excl_queries:
        for tq in excl_queries:
            results = search(df, bm25, query=tq.query,
                             exclude_ingredients=tq.exclude_ingredients,
                             alpha=tq.alpha, beta=tq.beta, top_k=args.k)
            violations = []
            for r in results:
                ings_text = " ".join(r.get("ingredients", [])).lower()
                for excl in (tq.exclude_ingredients or []):
                    if excl.lower() in ings_text:
                        violations.append((r["title"], excl))
            status = "✓ PASS" if not violations else f"✗ FAIL ({len(violations)} violations)"
            print(f"  {tq.name:<40}  {status}")
            for title, excl in violations[:3]:
                print(f"      → '{title}' contains excluded '{excl}'")
    else:
        print("  No exclusion test cases defined.")

    # ── Save CSV ─────────────────────────────────────────────────────────────
    if args.csv:
        os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
        rows = [
            {
                "query": r.name,
                "mode": r.mode,
                f"precision@{r.k}": round(r.precision, 4),
                f"recall@{r.k}": round(r.recall, 4),
                f"ndcg@{r.k}": round(r.ndcg, 4),
                "mrr": round(r.mrr, 4),
                "latency_ms": round(r.latency_ms, 2),
                "hits": r.n_relevant_in_results,
                "total_relevant_est": r.total_relevant_estimate,
            }
            for r in all_results
        ]
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"\n  Results saved to: {args.csv}")

    print("\n[Done]\n")


if __name__ == "__main__":
    main()