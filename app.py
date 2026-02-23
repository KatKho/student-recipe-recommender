from flask import Flask, render_template, request, jsonify
from recipe_search import load_data, build_index, search
import ast

app = Flask(__name__)

# Load data and build BM25 index once at startup
print("Loading recipe dataset...")
df = load_data()
print(f"Building BM25 index over {len(df)} recipes...")
bm25 = build_index(df)
print("✅ Ready to serve!")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search")
def api_search():
    query = request.args.get("q", "").strip()
    ingredients_raw = request.args.get("ingredients", "").strip()
    alpha = float(request.args.get("alpha", 0.7))
    beta = float(request.args.get("beta", 0.3))
    top_k = int(request.args.get("top_k", 10))

    # Parse ingredients (comma-separated)
    ingredients = None
    if ingredients_raw:
        ingredients = [ing.strip() for ing in ingredients_raw.split(",") if ing.strip()]

    # Run search
    results = search(
        df, bm25,
        query=query if query else None,
        ingredients=ingredients,
        alpha=alpha,
        beta=beta,
        top_k=top_k
    )

    # Format ingredients for display
    for r in results:
        if isinstance(r["ingredients"], str):
            try:
                r["ingredients"] = ast.literal_eval(r["ingredients"])
            except (ValueError, SyntaxError):
                r["ingredients"] = [r["ingredients"]]
        if isinstance(r["instructions"], str):
            try:
                parsed = ast.literal_eval(r["instructions"])
                if isinstance(parsed, list):
                    r["instructions"] = parsed
            except (ValueError, SyntaxError):
                r["instructions"] = [r["instructions"]]

    return jsonify({"results": results, "count": len(results)})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
