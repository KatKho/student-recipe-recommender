import pickle
import re
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer once (same as used during indexing)
_lemmatizer = WordNetLemmatizer()


def _normalize_ingredient_term(text):
    if not isinstance(text, str):
        return ""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


_INGREDIENT_SYNONYM_GROUPS = [
    {"scallion", "green onion", "spring onion"},
    {"garbanzo", "garbanzo bean", "garbanzo beans", "chickpea", "chickpeas"},
    {"coriander", "cilantro"},
    {"capsicum", "bell pepper", "sweet pepper"},
    {"aubergine", "aubergines", "eggplant", "eggplants"},
    {"courgette", "courgettes", "zucchini", "zucchinis"},
    {"rocket", "arugula"},
    {"powdered sugar", "icing sugar", "confectioners sugar", "confectioners' sugar"},
    {"maize", "corn"},
    {"cornstarch", "corn starch", "cornflour"},
    {"all purpose flour", "plain flour", "ap flour"},
    {"caster sugar", "superfine sugar"},
    {"white sugar", "granulated sugar"},
    {"bicarbonate of soda", "baking soda"},
    {"chili", "chilli", "chile"},
    {"chili flakes", "chilli flakes", "red pepper flakes", "crushed red pepper"},
    {"prawn", "prawns", "shrimp", "shrimps"},
    {"beetroot", "beet", "beets"},
    {"swede", "rutabaga"},
    {"yoghurt", "yogurt"},
    {"minced beef", "ground beef"},
    {"minced pork", "ground pork"},
    {"minced turkey", "ground turkey"},
    {"minced chicken", "ground chicken"},
    {"tinned tomato", "tinned tomatoes", "canned tomato", "canned tomatoes"},
    {"tomato ketchup", "ketchup"},
]


def _build_ingredient_alias_lookup(groups):
    lookup = {}
    for group in groups:
        normalized_group = set()
        for term in group:
            normalized_term = _normalize_ingredient_term(term)
            if normalized_term:
                normalized_group.add(normalized_term)
        for term in normalized_group:
            lookup[term] = normalized_group
    return lookup


_INGREDIENT_ALIAS_LOOKUP = _build_ingredient_alias_lookup(_INGREDIENT_SYNONYM_GROUPS)


def _expand_ingredient_aliases(term):
    normalized = _normalize_ingredient_term(term)
    if not normalized:
        return set()
    return _INGREDIENT_ALIAS_LOOKUP.get(normalized, {normalized})

############################################
# 1. Load Cleaned Dataset
############################################

def load_data(path="data/clean_recipes.pkl"):
    with open(path, "rb") as f:
        df = pickle.load(f)
    return df


def build_index(df):
    tokenized_corpus = [doc.split() for doc in df["clean_text"]]
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25


############################################
# 2. Ingredient Overlap Scoring
############################################

# Minimum number of recipe ingredients for scoring denominator.
# Prevents 1-2 ingredient recipes from dominating results.
MIN_RECIPE_INGREDIENTS = 4
MIN_RESULT_SCORE = 1e-9


def ingredient_overlap_score(user_ingredients, parsed_ingredients):
    """
    Score how well a recipe matches the user's available ingredients.

    Uses FULL INGREDIENT LINE matching: checks if each user ingredient
    (e.g. "avocado") appears as a substring in any full ingredient line
    (e.g. "1 ripe avocado, peeled"). This is far more precise than
    matching against individual tokenized words.

    Args:
        user_ingredients: List of user search terms, lowercased
        parsed_ingredients: List of full ingredient strings from recipe,
                           lowercased (e.g. ["1 ripe avocado", "2 cups flour"])
    """
    if not parsed_ingredients:
        return 0.0

    normalized_lines = [_normalize_ingredient_term(line) for line in parsed_ingredients]

    # Count how many user ingredients appear in at least one ingredient line
    matched = 0
    for user_ing in user_ingredients:
        aliases = _expand_ingredient_aliases(user_ing)
        if not aliases:
            continue
        for line in normalized_lines:
            if any(alias in line for alias in aliases):
                matched += 1
                break  # This user ingredient is found, move to next

    # Use a floor on the denominator to penalize tiny recipes
    effective_size = max(len(parsed_ingredients), MIN_RECIPE_INGREDIENTS)

    return matched / effective_size


def _normalize_unique_ingredients(ingredients):
    seen = set()
    normalized_terms = []
    for ing in ingredients or []:
        normalized = _normalize_ingredient_term(ing)
        if normalized and normalized not in seen:
            seen.add(normalized)
            normalized_terms.append(normalized)
    return normalized_terms


def _recipe_contains_any_alias(parsed_ingredients, alias_sets):
    if not parsed_ingredients:
        return False

    normalized_lines = [_normalize_ingredient_term(line) for line in parsed_ingredients]
    for aliases in alias_sets:
        for line in normalized_lines:
            if any(alias in line for alias in aliases):
                return True
    return False


############################################
# 3. Hybrid Ranking
############################################

def search(
    df,
    bm25,
    query=None,
    ingredients=None,
    exclude_ingredients=None,
    alpha=0.7,
    beta=0.3,
    top_k=10,
):
    """
    Hybrid search combining BM25 keyword relevance with ingredient overlap.
    
    Args:
        df: DataFrame of cleaned recipes
        bm25: Pre-built BM25 index
        query: Keyword search string (e.g. "fried rice")
        ingredients: List of ingredient strings (e.g. ["eggs", "rice"])
        exclude_ingredients: List of ingredients to avoid
        alpha: Weight for BM25 score (default 0.7)
        beta: Weight for ingredient overlap (default 0.3)
        top_k: Number of results to return
    
    Returns:
        List of result dicts with title, ingredients, instructions, score, etc.
    """
    normalized_user_ingredients = _normalize_unique_ingredients(ingredients)
    normalized_excluded_ingredients = _normalize_unique_ingredients(exclude_ingredients)

    if not query and not normalized_user_ingredients:
        return []

    scores = np.zeros(len(df))
    allowed_mask = np.ones(len(df), dtype=bool)

    # Keyword BM25 score (normalized)
    if query:
        query_tokens = query.lower().split()
        # Lemmatize query tokens to match lemmatized index
        query_tokens = [_lemmatizer.lemmatize(t, pos='n') for t in query_tokens]
        bm25_scores = np.array(bm25.get_scores(query_tokens))
        max_bm25 = bm25_scores.max() if bm25_scores.max() > 0 else 1.0
        bm25_scores = bm25_scores / max_bm25  # Normalize to [0, 1]
        scores += alpha * bm25_scores

    # Ingredient overlap score (matches against full ingredient lines)
    if normalized_user_ingredients:
        ingredient_scores = []
        for recipe_ings in df["parsed_ingredients"]:
            score = ingredient_overlap_score(normalized_user_ingredients, recipe_ings)
            ingredient_scores.append(score)
        scores += beta * np.array(ingredient_scores)

    # Filter out recipes containing any excluded ingredients
    if normalized_excluded_ingredients:
        excluded_alias_sets = [
            _expand_ingredient_aliases(term) for term in normalized_excluded_ingredients
        ]
        allowed_mask = np.array(
            [
                not _recipe_contains_any_alias(recipe_ings, excluded_alias_sets)
                for recipe_ings in df["parsed_ingredients"]
            ],
            dtype=bool,
        )

    ranked_indices = np.argsort(scores)[::-1]
    top_indices = [
        idx for idx in ranked_indices
        if allowed_mask[idx] and scores[idx] > MIN_RESULT_SCORE
    ][:top_k]

    results = []
    for idx in top_indices:
        row = df.iloc[idx]
        results.append({
            "title": row["title"],
            "ingredients": row["ingredients"],
            "instructions": row["instructions"],
            "score": round(float(scores[idx]), 4),
            "source": row.get("source", ""),
        })

    return results


############################################
# 4. Example Usage
############################################

if __name__ == "__main__":
    df = load_data()
    bm25 = build_index(df)

    print("\n=== Keyword Search: 'fried rice' ===\n")
    results = search(df, bm25, query="fried rice")
    for r in results[:5]:
        print(f"  [{r['score']:.4f}] {r['title']}")

    print("\n=== Ingredient Search: eggs, rice, soy sauce ===\n")
    results = search(df, bm25, ingredients=["eggs", "rice", "soy sauce"])
    for r in results[:5]:
        print(f"  [{r['score']:.4f}] {r['title']}")

    print("\n=== Hybrid Search: 'stir fry' + [chicken, garlic, onion] ===\n")
    results = search(df, bm25, query="stir fry", ingredients=["chicken", "garlic", "onion"])
    for r in results[:5]:
        print(f"  [{r['score']:.4f}] {r['title']}")
