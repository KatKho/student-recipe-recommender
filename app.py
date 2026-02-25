from flask import Flask, render_template, request, jsonify
from werkzeug.exceptions import BadRequest
from recipe_search import load_data, build_index, search
import ast

app = Flask(__name__)

MIN_TOP_K = 1
MAX_TOP_K = 20
MIN_WEIGHT = 0.0
MAX_WEIGHT = 1.0

# Load data and build BM25 index once at startup
print("Loading recipe dataset...")
df = load_data()
print(f"Building BM25 index over {len(df)} recipes...")
bm25 = build_index(df)
print("✅ Ready to serve!")


@app.route("/")
def index():
    return render_template("index.html")


def _parse_float_arg(name, default, min_value=None, max_value=None):
    raw = request.args.get(name, "")
    if raw is None or raw.strip() == "":
        value = default
    else:
        try:
            value = float(raw)
        except ValueError as exc:
            raise BadRequest(
                f"Invalid '{name}' value '{raw}'. Expected a numeric value."
            ) from exc

    if min_value is not None:
        value = max(value, min_value)
    if max_value is not None:
        value = min(value, max_value)
    return value


def _parse_int_arg(name, default, min_value=None, max_value=None):
    raw = request.args.get(name, "")
    if raw is None or raw.strip() == "":
        value = default
    else:
        try:
            value = int(raw)
        except ValueError as exc:
            raise BadRequest(
                f"Invalid '{name}' value '{raw}'. Expected an integer."
            ) from exc

    if min_value is not None:
        value = max(value, min_value)
    if max_value is not None:
        value = min(value, max_value)
    return value


@app.errorhandler(BadRequest)
def handle_bad_request(err):
    return jsonify({"error": "bad_request", "message": err.description}), 400


@app.route("/api/search")
def api_search():
    query = request.args.get("q", "").strip()
    ingredients_raw = request.args.get("ingredients", "").strip()
    exclude_ingredients_raw = request.args.get("exclude_ingredients", "").strip()
    alpha = _parse_float_arg(
        "alpha", default=0.7, min_value=MIN_WEIGHT, max_value=MAX_WEIGHT
    )
    beta = _parse_float_arg(
        "beta", default=0.3, min_value=MIN_WEIGHT, max_value=MAX_WEIGHT
    )
    top_k = _parse_int_arg(
        "top_k", default=10, min_value=MIN_TOP_K, max_value=MAX_TOP_K
    )

    if alpha == 0 and beta == 0:
        raise BadRequest("At least one of 'alpha' or 'beta' must be greater than 0.")

    # Parse ingredients (comma-separated)
    ingredients = None
    if ingredients_raw:
        ingredients = [ing.strip() for ing in ingredients_raw.split(",") if ing.strip()]

    # Parse excluded ingredients (comma-separated)
    exclude_ingredients = None
    if exclude_ingredients_raw:
        exclude_ingredients = [
            ing.strip() for ing in exclude_ingredients_raw.split(",") if ing.strip()
        ]

    # Run search
    results = search(
        df, bm25,
        query=query if query else None,
        ingredients=ingredients,
        exclude_ingredients=exclude_ingredients,
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
