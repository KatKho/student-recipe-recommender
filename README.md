# 🍳 Student Recipe Search Engine

A recipe search and recommendation system built for students. Search over **63,000+ recipes** by keyword or by ingredients you have on hand.

Built as a project for **INFO 376**.

## Features

- **Keyword Search** — Find recipes by name or description using BM25 ranking (e.g., "fried rice", "quick breakfast")
- **Ingredient Search** — Enter ingredients you have and get recipes that use them (e.g., "chicken, garlic, soy sauce")
- **Hybrid Scoring** — Combines keyword relevance with ingredient overlap for accurate results
- **Lemmatized Indexing** — Uses NLTK WordNet lemmatizer for better token matching (e.g., "potatoes" matches "potato")
- **63,000+ Recipes** — Merged from two datasets: a 13k GitHub recipe dataset and a 50k sample from RecipeNLG

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Python, Flask |
| Search | BM25 (rank-bm25), ingredient line matching |
| NLP | NLTK WordNetLemmatizer, scikit-learn stop words |
| Data | Pandas, Pickle |
| Frontend | HTML, CSS, JavaScript |

## Project Structure

```
├── app.py                  # Flask server and API endpoint
├── recipe_search.py        # Search engine (BM25 + ingredient scoring)
├── clean and merge.py      # Data pipeline (load, clean, merge datasets)
├── requirements.txt        # Python dependencies
├── data/
│   └── clean_recipes.pkl   # Processed dataset (generated)
├── static/
│   ├── style.css           # Dark theme UI styles
│   └── app.js              # Client-side search logic
├── templates/
│   └── index.html          # Single-page search interface
├── recipe-dataset-main/    # GitHub recipes dataset (CSV)
└── RecipeNLG dataset/      # RecipeNLG dataset (CSV)
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')"
```

### 2. Build the dataset

Place the raw datasets in the project root:
- `recipe-dataset-main/13k-recipes.csv`
- `RecipeNLG dataset/RecipeNLG_dataset.csv`

Then run:

```bash
python "clean and merge.py"
```

This will create `data/clean_recipes.pkl` with ~63,000 cleaned and merged recipes.

### 3. Run the app

```bash
python app.py
```

Open [http://localhost:5000](http://localhost:5000) in your browser.

## How It Works

### Search Algorithm

The search engine uses a **hybrid ranking** approach:

1. **Keyword Search (BM25)** — Tokenizes and lemmatizes the query, then scores all recipes using BM25 (Okapi). Scores are normalized to [0, 1].

2. **Ingredient Search (Line Matching)** — Matches user ingredients against full ingredient strings (e.g., searching "avocado" checks if it appears in "1 ripe avocado, peeled"). This is more precise than token-level matching.

3. **Combined Score** — `score = α × BM25_score + β × ingredient_score` where α=0.7 and β=0.3 by default.

### Data Pipeline

- Loads recipes from two CSV sources
- Parses ingredient lists from various string formats
- Normalizes text (lowercase, remove special characters)
- Lemmatizes tokens (potatoes → potato, eggs → egg)
- Stores both tokenized ingredients (for BM25) and parsed ingredient lines (for ingredient search)

## API

### `GET /api/search`

| Parameter | Type | Description |
|-----------|------|-------------|
| `q` | string | Keyword search query |
| `ingredients` | string | Comma-separated ingredient list |
| `alpha` | float | BM25 weight (default: 0.7) |
| `beta` | float | Ingredient weight (default: 0.3) |
| `top_k` | int | Number of results (default: 10) |

**Example:**

```
GET /api/search?q=fried+rice&top_k=5
GET /api/search?ingredients=chicken,garlic,soy+sauce&top_k=5
```

## Datasets

Download the following datasets and place them in the project root before running `clean and merge.py`:

1. **13k Recipes** — [GitHub: josephrmartinez/recipe-dataset](https://github.com/josephrmartinez/recipe-dataset)
   - Place as `recipe-dataset-main/13k-recipes.csv`
2. **RecipeNLG** (~2M recipes, sampled to 50k) — [Kaggle: RecipeNLG Dataset](https://www.kaggle.com/code/paultimothymooney/explore-recipe-nlg-dataset)
   - Place as `RecipeNLG dataset/RecipeNLG_dataset.csv`

## License

This project was built for educational purposes as part of INFO 376.
