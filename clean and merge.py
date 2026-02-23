import pandas as pd
import re
import pickle
import ast
import os
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.stem import WordNetLemmatizer

# Initialize lemmatizer once for reuse
_lemmatizer = WordNetLemmatizer()

############################################
# 1. Text Cleaning Utilities
############################################

def normalize_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text):
    tokens = text.split()
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    # Lemmatize as nouns: potatoes→potato, eggs→egg, tomatoes→tomato
    tokens = [_lemmatizer.lemmatize(t, pos='n') for t in tokens]
    return tokens


def parse_ingredients(ingredients):
    """
    Parse ingredients from various formats:
    - Python list stored as string: '["item1", "item2"]'
    - Plain comma-separated string: "item1, item2"
    - Actual list
    """
    if isinstance(ingredients, list):
        return ingredients

    if isinstance(ingredients, str):
        # Try to parse as a Python literal (e.g. '["a", "b"]')
        try:
            parsed = ast.literal_eval(ingredients)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except (ValueError, SyntaxError):
            pass
        # Fall back to comma-separated
        return [s.strip() for s in ingredients.split(",") if s.strip()]

    return []


def clean_ingredients(ingredients):
    """
    Parse and tokenize ingredient text for search indexing.
    """
    parsed = parse_ingredients(ingredients)
    combined = " ".join(parsed)
    return tokenize(normalize_text(combined))


############################################
# 2. Load GitHub Dataset (CSV)
############################################

def load_github_dataset(path):
    df = pd.read_csv(path)

    df = df.rename(columns={
        "Title": "title",
        "Ingredients": "ingredients",
        "Instructions": "instructions",
    })

    df["source"] = "github"
    return df[["title", "ingredients", "instructions", "source"]]


############################################
# 3. Load RecipeNLG Dataset (CSV, sampled)
############################################

def load_recipenlg_dataset(path, sample_size=50000):
    df = pd.read_csv(path)

    df = df.rename(columns={
        "title": "title",
        "ingredients": "ingredients",
        "directions": "instructions"
    })

    df["source"] = "recipenlg"

    # Sample to keep prototype fast
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)

    return df[["title", "ingredients", "instructions", "source"]]


############################################
# 4. Standardize + Clean
############################################

def preprocess(df):
    df = df.dropna(subset=["title", "ingredients"])

    df["clean_title"] = df["title"].apply(normalize_text)
    df["clean_ingredients"] = df["ingredients"].apply(clean_ingredients)

    # Store parsed ingredient lines as lowercased strings (for ingredient search)
    # e.g. ["1 ripe avocado, peeled", "2 cups flour"]
    df["parsed_ingredients"] = df["ingredients"].apply(
        lambda x: [s.lower().strip() for s in parse_ingredients(x)]
    )

    df["clean_text"] = (
        df["clean_title"]
        + " "
        + df["clean_ingredients"].apply(lambda x: " ".join(x))
    )

    return df.reset_index(drop=True)


############################################
# 5. Merge & Save
############################################

def main():
    os.makedirs("data", exist_ok=True)

    print("Loading GitHub dataset...")
    github_df = load_github_dataset("recipe-dataset-main/13k-recipes.csv")
    print(f"  → {len(github_df)} recipes loaded")

    print("Loading RecipeNLG dataset (sampling 50k)...")
    recipenlg_df = load_recipenlg_dataset("RecipeNLG dataset/RecipeNLG_dataset.csv")
    print(f"  → {len(recipenlg_df)} recipes loaded")

    merged_df = pd.concat([github_df, recipenlg_df], ignore_index=True)
    print(f"Merged: {len(merged_df)} recipes total")

    print("Preprocessing...")
    merged_df = preprocess(merged_df)

    with open("data/clean_recipes.pkl", "wb") as f:
        pickle.dump(merged_df, f)

    print(f"✅ Cleaned and merged dataset saved to data/clean_recipes.pkl")
    print(f"Total recipes: {len(merged_df)}")
    print(merged_df[["title", "clean_title", "source"]].head(10))


if __name__ == "__main__":
    main()
