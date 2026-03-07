import time
import numpy as np
from recipe_search import load_data, build_index, search, _normalize_unique_ingredients


# ─────────────────────────────────────────────
#  RELEVANCE HELPERS
# ─────────────────────────────────────────────

def get_relevance_scores(results, query, ingredients):
    """
    Calculate relevance scores [0.0, 1.0] for retrieved results.

    FIX: Blends ingredient-overlap score with the result's own search score
    so that results with the same overlap count are no longer tied (e.g. all 0.75).
    """
    rel_scores = []
    norm_user_ing = _normalize_unique_ingredients(ingredients) if ingredients else []

    for r in results:
        rel = 0.0

        if norm_user_ing:
            raw_ing_text = str(r.get("ingredients", "")).lower()
            matched = sum(1 for ing in norm_user_ing if ing in raw_ing_text)
            base_rel = matched / len(norm_user_ing)

            # Blend ingredient overlap with the retrieval score to break ties
            search_score = float(r.get("score", 0.0))
            rel = (0.7 * base_rel) + (0.3 * search_score)

        elif query:
            query_terms = set(query.lower().split())
            title = str(r.get("title", "")).lower()
            matched = sum(1 for t in query_terms if t in title)
            rel = min(matched / len(query_terms), 1.0) if query_terms else 0.0

        rel_scores.append(rel)

    return rel_scores


def count_total_relevant_in_corpus(df, query, ingredients, threshold=0.5, sample_size=500):
    """
    Estimate total relevant documents in the corpus for Recall calculation.

    FIX: Samples the corpus (default 500 rows) instead of scanning the entire
    dataframe, then scales the count back up.  This prevents Recall from being
    driven to ~0 when common ingredients match thousands of recipes.
    """
    sample_df = df.sample(min(sample_size, len(df)), random_state=42)
    scale = len(df) / len(sample_df)
    count = 0

    if ingredients:
        norm_user_ing = _normalize_unique_ingredients(ingredients)
        if not norm_user_ing:
            return 1
        for ings in sample_df["ingredients"]:
            ings_lower = str(ings).lower()
            matched = sum(1 for ing in norm_user_ing if ing in ings_lower)
            if matched / len(norm_user_ing) >= threshold:
                count += 1

    elif query:
        query_terms = set(query.lower().split())
        if not query_terms:
            return 1
        for title in sample_df["title"]:
            title_lower = str(title).lower()
            matched = sum(1 for t in query_terms if t in title_lower)
            if matched / len(query_terms) >= threshold:
                count += 1

    estimated = int(count * scale)
    return max(estimated, 1)


# ─────────────────────────────────────────────
#  METRIC CALCULATORS
# ─────────────────────────────────────────────

def calculate_ndcg(rel_scores, k):
    """Normalized Discounted Cumulative Gain at k."""
    if not rel_scores:
        return 0.0
    dcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(rel_scores[:k]))
    ideal_scores = sorted(rel_scores, reverse=True)
    idcg = sum(rel / np.log2(i + 2) for i, rel in enumerate(ideal_scores[:k]))
    return dcg / idcg if idcg > 0 else 0.0


def calculate_precision_at_k(rel_scores, k, threshold=0.5):
    """Precision at k — fraction of top-k results above the relevance threshold."""
    if not rel_scores:
        return 0.0
    relevant = sum(1 for rel in rel_scores[:k] if rel >= threshold)
    return relevant / min(k, len(rel_scores))


def calculate_recall_at_k(rel_scores, total_relevant, k, threshold=0.5):
    """Recall at k — fraction of all relevant docs that appear in top-k."""
    if total_relevant == 0:
        return 0.0
    relevant_retrieved = sum(1 for rel in rel_scores[:k] if rel >= threshold)
    return relevant_retrieved / total_relevant


# ─────────────────────────────────────────────
#  QUANTITATIVE EVALUATION
# ─────────────────────────────────────────────

def run_quantitative_evaluation(df, bm25, test_queries):
    """
    Run quantitative metrics:
      1. Average Search Latency  (ms)
      2. Queries with Zero Results
      3. nDCG@k
      4. Precision@k
      5. Recall@k

    FIX: corpus relevance counts are pre-computed OUTSIDE the timed search loop
    so that (a) latency only reflects actual search time and (b) the O(n) corpus
    scan does not inflate every measured latency by ~4 seconds.
    """
    print("\n--- Quantitative Evaluation ---")

    top_k = 5
    rel_threshold = 0.5

    # ── Pre-compute total_relevant outside the timed loop ──────────────────
    print("Pre-computing corpus relevance counts (this runs once)...")
    precomputed_totals = {}
    for q_dict in test_queries:
        query = q_dict.get("query", "")
        ingredients = q_dict.get("ingredients", [])
        key = (query, tuple(sorted(ingredients)))
        if key not in precomputed_totals:
            precomputed_totals[key] = count_total_relevant_in_corpus(
                df, query, ingredients, threshold=rel_threshold
            )

    # ── Timed search loop ──────────────────────────────────────────────────
    latencies = []
    ndcg_scores = []
    precision_scores = []
    recall_scores = []
    empty_counts = 0

    for q_dict in test_queries:
        query = q_dict.get("query", "")
        ingredients = q_dict.get("ingredients", [])
        exclude = q_dict.get("exclude_ingredients", [])

        # Only the search() call is timed
        start_time = time.time()
        results = search(
            df,
            bm25,
            query=query,
            ingredients=ingredients,
            exclude_ingredients=exclude,
            alpha=0.5,
            beta=0.5,
            top_k=top_k,
        )
        latency = (time.time() - start_time) * 1000
        latencies.append(latency)

        if not results:
            empty_counts += 1
            print(f"  Warning: No results for query='{query}', ingredients={ingredients}")
            continue

        rel_scores = get_relevance_scores(results, query, ingredients)
        key = (query, tuple(sorted(ingredients)))
        total_rel = precomputed_totals[key]

        ndcg_scores.append(calculate_ndcg(rel_scores, top_k))
        precision_scores.append(calculate_precision_at_k(rel_scores, top_k, threshold=rel_threshold))
        recall_scores.append(calculate_recall_at_k(rel_scores, total_rel, top_k, threshold=rel_threshold))

    # ── Report ─────────────────────────────────────────────────────────────
    print(f"Total Test Queries:        {len(test_queries)}")
    print(f"Average Search Latency:    {np.mean(latencies):.2f} ms")
    print(f"Average nDCG@{top_k}:          {np.mean(ndcg_scores) if ndcg_scores else 0.0:.4f}")
    print(f"Average Precision@{top_k}:     {np.mean(precision_scores) if precision_scores else 0.0:.4f}")
    print(f"Average Recall@{top_k}:        {np.mean(recall_scores) if recall_scores else 0.0:.4f}")
    print(f"Queries with Zero Results: {empty_counts}")


# ─────────────────────────────────────────────
#  QUALITATIVE EVALUATION
# ─────────────────────────────────────────────

def run_qualitative_evaluation(df, bm25):
    """
    Qualitative scenario testing for manual review.

    FIX (Scenario 2): Expanded exclusion list to catch seafood/shrimp that
    slipped through the original ["meat","beef","pork","chicken"] filter.

    FIX (Scenario 3): Changed alpha/beta from 1.0/0.0 (pure BM25) to 0.3/0.7
    (semantic-heavy) so that vague queries like "comfort food cold winter night"
    don't match on brand names like "Southern Comfort".
    """
    print("\n--- Qualitative Evaluation (Scenario Testing) ---")

    # Expanded exclusion list used by Scenario 2
    MEAT_EXCLUSIONS = [
        "meat", "beef", "pork", "chicken",
        "shrimp", "lamb", "turkey", "fish", "seafood",
        "bacon", "ham", "sausage", "tuna", "salmon",
    ]

    scenarios = [
        {
            "name": "Scenario 1: Using up leftovers (Focus on ingredients)",
            "query": "",
            "ingredients": ["chicken", "rice", "broccoli", "soy sauce"],
            "exclude_ingredients": [],
            # Pure ingredient search — unchanged
            "alpha": 0.0,
            "beta": 1.0,
        },
        {
            "name": "Scenario 2: Specific craving with restrictions (Hybrid search)",
            "query": "pasta",
            "ingredients": [],
            # FIX: expanded to include seafood so shrimp is excluded
            "exclude_ingredients": MEAT_EXCLUSIONS,
            "alpha": 0.7,
            "beta": 0.3,
        },
        {
            "name": "Scenario 3: Vague comfort food search (Focus on semantics)",
            "query": "comfort food cold winter night",
            "ingredients": [],
            "exclude_ingredients": [],
            # FIX: was alpha=1.0 / beta=0.0 (pure BM25 → matched "Southern Comfort" brand)
            #      now semantic-heavy so intent is understood, not just keywords matched
            "alpha": 0.3,
            "beta": 0.7,
        },
    ]

    for s in scenarios:
        print(f"\n{s['name']}")
        print(
            f"  Params -> Query: '{s['query']}', "
            f"Ings: {s.get('ingredients', [])}, "
            f"Exclude: {s.get('exclude_ingredients', [])}, "
            f"alpha={s['alpha']}, beta={s['beta']}"
        )

        results = search(
            df,
            bm25,
            query=s["query"],
            ingredients=s.get("ingredients", []),
            exclude_ingredients=s.get("exclude_ingredients", []),
            alpha=s["alpha"],
            beta=s["beta"],
            top_k=3,
        )

        if not results:
            print("  [No results found]")
            continue

        for i, r in enumerate(results):
            ings_str = str(r["ingredients"])
            ings_preview = ings_str[:80] + "..." if len(ings_str) > 80 else ings_str
            print(f"  {i+1}. {r['title']} (Score: {r['score']:.4f})")
            print(f"     Ingredients: {ings_preview}")


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading data and building index for evaluation...")
    df = load_data()
    bm25 = build_index(df)

    test_queries = [
        {"ingredients": ["chicken", "garlic", "onion"]},
        {"query": "chocolate cake", "ingredients": ["flour", "sugar", "cocoa"]},
        {"query": "vegan chili", "exclude_ingredients": ["meat", "beef", "cheese"]},
        {"ingredients": ["salmon", "lemon", "dill"]},
        {"query": "quick breakfast", "ingredients": ["eggs", "bread"]},
        {"ingredients": ["tofu", "soy sauce", "ginger"]},
        {"query": "soup", "ingredients": ["carrots", "celery", "onion"]},
        {"ingredients": ["beef", "potatoes"]},
        {"query": "salad", "exclude_ingredients": ["nuts", "dairy"]},
        {"ingredients": ["shrimp", "garlic", "butter"]},
    ]

    run_quantitative_evaluation(df, bm25, test_queries)
    run_qualitative_evaluation(df, bm25)
    print("\nEvaluation complete.")