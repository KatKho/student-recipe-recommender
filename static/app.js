/* ========================================
   StudentChef — Client-side Search Logic
   ======================================== */

document.addEventListener("DOMContentLoaded", () => {
    // ---- Elements ----
    const modeToggle = document.getElementById("modeToggle");
    const keywordBox = document.getElementById("keywordBox");
    const ingredientBox = document.getElementById("ingredientBox");
    const queryInput = document.getElementById("queryInput");
    const ingredientInput = document.getElementById("ingredientInput");
    const ingredientChips = document.getElementById("ingredientChips");
    const searchBtn = document.getElementById("searchBtn");
    const statusBar = document.getElementById("statusBar");
    const statusText = document.getElementById("statusText");
    const resultsSection = document.getElementById("resultsSection");
    const resultsHeader = document.getElementById("resultsHeader");
    const resultsTitle = document.getElementById("resultsTitle");
    const resultsCount = document.getElementById("resultsCount");
    const resultsGrid = document.getElementById("resultsGrid");
    const emptyState = document.getElementById("emptyState");

    let currentMode = "keyword";
    let ingredients = [];

    // ---- Mode Switching ----
    modeToggle.addEventListener("click", (e) => {
        const btn = e.target.closest(".mode-btn");
        if (!btn) return;

        document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        currentMode = btn.dataset.mode;

        updateModeUI();
    });

    function updateModeUI() {
        keywordBox.classList.toggle("hidden", currentMode === "ingredient");
        ingredientBox.classList.toggle("hidden", currentMode === "keyword");

        if (currentMode === "ingredient") {
            ingredientInput.focus();
        } else {
            queryInput.focus();
        }
    }

    // ---- Ingredient Chips ----
    ingredientInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") {
            e.preventDefault();
            addIngredient(ingredientInput.value.trim());
        }
    });

    function addIngredient(name) {
        if (!name || ingredients.includes(name.toLowerCase())) return;
        ingredients.push(name.toLowerCase());
        ingredientInput.value = "";
        renderChips();
    }

    function removeIngredient(name) {
        ingredients = ingredients.filter(i => i !== name);
        renderChips();
    }

    function renderChips() {
        ingredientChips.innerHTML = ingredients.map(ing =>
            `<span class="chip">
                ${escapeHtml(ing)}
                <button class="chip-remove" data-ing="${escapeHtml(ing)}" title="Remove">&times;</button>
            </span>`
        ).join("");

        ingredientChips.querySelectorAll(".chip-remove").forEach(btn => {
            btn.addEventListener("click", () => removeIngredient(btn.dataset.ing));
        });

        // Auto-search if in ingredient-only mode and we have ingredients
        if (currentMode === "ingredient" && ingredients.length > 0) {
            performSearch();
        }
    }

    // ---- Search ----
    searchBtn.addEventListener("click", performSearch);
    queryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter") performSearch();
    });

    async function performSearch() {
        const query = queryInput.value.trim();
        const hasQuery = query.length > 0;
        const hasIngredients = ingredients.length > 0;

        if (!hasQuery && !hasIngredients) return;

        // Show loading
        emptyState.classList.add("hidden");
        statusBar.classList.remove("hidden");
        resultsHeader.classList.add("hidden");
        resultsGrid.innerHTML = "";

        // Build URL params
        const params = new URLSearchParams();
        if (hasQuery) params.set("q", query);
        if (hasIngredients) params.set("ingredients", ingredients.join(","));
        params.set("top_k", "10");

        try {
            const res = await fetch(`/api/search?${params.toString()}`);
            const data = await res.json();

            statusBar.classList.add("hidden");
            renderResults(data.results, hasIngredients ? ingredients : []);
        } catch (err) {
            statusBar.classList.add("hidden");
            resultsGrid.innerHTML = `<p style="color: var(--text-secondary); text-align: center;">Error: ${err.message}</p>`;
        }
    }

    // ---- Render Results ----
    function renderResults(results, searchedIngredients) {
        if (!results || results.length === 0) {
            resultsGrid.innerHTML = `
                <div class="empty-state">
                    <div class="empty-illustration">😕</div>
                    <h3>No recipes found</h3>
                    <p>Try different keywords or ingredients</p>
                </div>`;
            return;
        }

        resultsHeader.classList.remove("hidden");
        resultsTitle.textContent = `Top ${results.length} Recipes`;
        resultsCount.textContent = `${results.length} results`;

        resultsGrid.innerHTML = results.map((r, i) => {
            const ingredientsList = formatIngredients(r.ingredients, searchedIngredients);
            const instructionsHtml = formatInstructions(r.instructions);
            const delay = i * 0.06;

            return `
                <div class="recipe-card" style="animation-delay: ${delay}s">
                    <div class="card-header">
                        <div>
                            <div class="card-title">${escapeHtml(r.title)}</div>
                            <span class="card-source">${escapeHtml(r.source)}</span>
                        </div>
                        <div class="card-score">⭐ ${r.score.toFixed(3)}</div>
                    </div>
                    <div class="card-section">
                        <div class="card-section-title">Ingredients</div>
                        <ul class="ingredient-list">${ingredientsList}</ul>
                    </div>
                    <div class="card-section">
                        <div class="card-section-title">Instructions</div>
                        <div class="instructions-text" id="inst-${i}" style="max-height: 80px; overflow: hidden;">
                            ${instructionsHtml}
                        </div>
                        <button class="instructions-toggle" data-target="inst-${i}" data-expanded="false">
                            Show more ▾
                        </button>
                    </div>
                </div>`;
        }).join("");

        // Instruction expand/collapse
        document.querySelectorAll(".instructions-toggle").forEach(btn => {
            btn.addEventListener("click", () => {
                const target = document.getElementById(btn.dataset.target);
                const expanded = btn.dataset.expanded === "true";
                if (expanded) {
                    target.style.maxHeight = "80px";
                    btn.textContent = "Show more ▾";
                    btn.dataset.expanded = "false";
                } else {
                    target.style.maxHeight = "none";
                    btn.textContent = "Show less ▴";
                    btn.dataset.expanded = "true";
                }
            });
        });
    }

    function formatIngredients(ings, searchedIngredients) {
        let list = ings;
        if (typeof ings === "string") {
            try { list = JSON.parse(ings.replace(/'/g, '"')); } catch { list = [ings]; }
        }
        if (!Array.isArray(list)) list = [String(list)];

        const searchSet = new Set(searchedIngredients.map(s => s.toLowerCase()));

        return list.map(ing => {
            const isMatched = searchSet.size > 0 &&
                [...searchSet].some(s => ing.toLowerCase().includes(s));
            const cls = isMatched ? "ingredient-item matched" : "ingredient-item";
            return `<li class="${cls}">${escapeHtml(ing)}</li>`;
        }).join("");
    }

    function formatInstructions(inst) {
        let steps = inst;
        if (typeof inst === "string") {
            try { steps = JSON.parse(inst.replace(/'/g, '"')); } catch { steps = [inst]; }
        }
        if (!Array.isArray(steps)) steps = [String(steps)];

        if (steps.length === 1) {
            return `<p>${escapeHtml(steps[0])}</p>`;
        }

        return `<ol>${steps.map(s => `<li>${escapeHtml(s)}</li>`).join("")}</ol>`;
    }

    function escapeHtml(str) {
        if (!str) return "";
        const div = document.createElement("div");
        div.textContent = String(str);
        return div.innerHTML;
    }
});
