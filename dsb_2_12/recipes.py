"""
Classes: DataPreparator, RatingPredictor, NutritionFacts, RecipeFinder, MenuGenerator
"""

import os
import pickle
import pandas as pd
import numpy as np


# ── Paths ─────────────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
DATA  = os.path.join(_HERE, "data")

MODEL_PATH     = os.path.join(DATA, "best_model.pkl")
RECIPES_PATH   = os.path.join(DATA, "epi_r.csv")
NUTRITION_PATH = os.path.join(DATA, "nutrition_facts.csv")


class DataPreparator:
    """Loads and prepares the Epicurious dataset."""

    def __init__(self, path: str = RECIPES_PATH):
        self.path = path
        self.df = None
        self.ingredient_cols = None

    def load(self) -> pd.DataFrame:
        self.df = pd.read_csv(self.path)
        non_ing = ['title', 'rating', 'meal', 'url']
        self.ingredient_cols = [c for c in self.df.columns if c not in non_ing]
        return self.df

    def get_features_and_target(self, target: str = 'label'):
        """Return X (ingredient matrix) and y (rating label)."""
        if self.df is None:
            self.load()
        X = self.df[self.ingredient_cols].values

        if target == 'label':
            y = self.df['rating'].apply(self._to_label).values
        elif target == 'rating':
            y = self.df['rating'].values
        else:
            y = self.df['rating'].round().astype(int).values
        return X, y, self.ingredient_cols

    @staticmethod
    def _to_label(r):
        if r <= 1:   return 'bad'
        if r <= 3:   return 'so-so'
        return 'great'


class RatingPredictor:
    """Loads trained model and predicts rating category for a list of ingredients."""

    LABELS = {'bad': 0, 'so-so': 1, 'great': 2}

    def __init__(self, model_path: str = MODEL_PATH):
        self.model_path = model_path
        self._model = None
        self._features = None
        self._classes = None

    def _load(self):
        with open(self.model_path, "rb") as f:
            payload = pickle.load(f)
        self._model    = payload['model']
        self._features = payload['features']
        self._classes  = payload['classes']

    def predict(self, ingredients: list) -> str:
        """
        Predict rating class for a list of ingredient names.
        Returns one of: 'bad', 'so-so', 'great'
        """
        if self._model is None:
            self._load()
        vec = np.zeros(len(self._features))
        for i, feat in enumerate(self._features):
            if feat in ingredients:
                vec[i] = 1
        label = self._model.predict([vec])[0]
        return label

    def predict_proba(self, ingredients: list) -> dict:
        if self._model is None:
            self._load()
        vec = np.zeros(len(self._features))
        for i, feat in enumerate(self._features):
            if feat in ingredients:
                vec[i] = 1
        probs = self._model.predict_proba([vec])[0]
        return dict(zip(self._classes, probs))


class NutritionFacts:
    """Returns nutrition facts for a list of ingredients as % of daily value."""

    DAILY = {
        'protein_g':      50,
        'fat_g':          65,
        'carbs_g':        300,
        'sodium_mg':      2300,
        'fiber_g':        25,
        'sugar_g':        50,
        'calcium_mg':     1000,
        'iron_mg':        18,
        'vitamin_c_mg':   90,
        'potassium_mg':   4700,
    }

    DISPLAY_NAMES = {
        'protein_g':    'Protein',
        'fat_g':        'Total Fat',
        'carbs_g':      'Total Carbohydrate',
        'sodium_mg':    'Sodium',
        'fiber_g':      'Dietary Fiber',
        'sugar_g':      'Total Sugars',
        'calcium_mg':   'Calcium',
        'iron_mg':      'Iron',
        'vitamin_c_mg': 'Vitamin C',
        'potassium_mg': 'Potassium',
    }

    def __init__(self, path: str = NUTRITION_PATH):
        self.path = path
        self._df = None

    def _load(self):
        self._df = pd.read_csv(self.path, index_col=0)

    def get(self, ingredients: list) -> dict:
        """
        Returns {ingredient: {nutrient_display_name: pct_daily_value}} for each ingredient.
        """
        if self._df is None:
            self._load()
        result = {}
        for ing in ingredients:
            ing_clean = ing.strip().lower().replace(' ', '_').replace('-', '_')
            if ing_clean in self._df.index:
                row = self._df.loc[ing_clean]
                facts = {}
                for col, dv in self.DAILY.items():
                    if col in row.index:
                        pct = round(row[col] / dv * 100)
                        if pct > 0:
                            facts[self.DISPLAY_NAMES[col]] = pct
                result[ing.capitalize()] = facts
            else:
                result[ing.capitalize()] = {}
        return result

    def format_output(self, ingredients: list) -> str:
        facts = self.get(ingredients)
        lines = []
        for name, data in facts.items():
            lines.append(name)
            if data:
                for nutrient, pct in data.items():
                    lines.append(f"  {nutrient} - {pct}% of Daily Value")
            else:
                lines.append("  (no data available)")
        return "\n".join(lines)


class RecipeFinder:
    """Finds the N most similar recipes to a list of ingredients."""

    def __init__(self, path: str = RECIPES_PATH):
        self.path = path
        self._df = None
        self._ing_cols = None

    def _load(self):
        self._df = pd.read_csv(self.path)
        non_ing = ['title', 'rating', 'meal', 'url']
        self._ing_cols = [c for c in self._df.columns if c not in non_ing]

    def find_similar(self, ingredients: list, n: int = 3) -> list:
        """
        Returns list of n dicts: {title, rating, url, similarity}
        Similarity = number of matching ingredients / total unique ingredients (Jaccard).
        """
        if self._df is None:
            self._load()

        query = set(i.strip().lower().replace(' ', '_').replace('-', '_')
                    for i in ingredients)
        
        scores = []

        for _, row in self._df.iterrows():
            try:
                recipe_ings = [c for c in self._ing_cols if row[c] == 1]

                if not recipe_ings:
                    continue

                intersection = len(query & set(recipe_ings))
                union = len(query | set(recipe_ings))
                jaccard = intersection / union if union > 0 else 0

                scores.append({
                    'title': row.get('title', 'Unknown'),
                    'rating': row.get('rating', 0),
                    'url': row.get('url', ''),
                    'similarity': jaccard,
                })

            except Exception:
                continue  

        scores.sort(key=lambda x: (-x['similarity'], -x['rating']))
        return scores[:n]

    def format_output(self, ingredients: list, n: int = 3) -> str:
        results = self.find_similar(ingredients, n)
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"- {r['title']}, rating: {r['rating']}, URL: {r['url']}")
        return "\n".join(lines) if lines else "No similar recipes found."


class MenuGenerator:
    """Generates a daily menu: breakfast, lunch, dinner."""

    MEAL_ORDER = ['breakfast', 'lunch', 'dinner']

    # map dataset meals → logical meals
    MEAL_MAP = {
        'breakfast': ['breakfast'],
        'lunch': ['lunch'],
        'dinner': ['dinner', 'main course', 'supper']
    }

    def __init__(self, path: str = RECIPES_PATH,
                 nutrition_path: str = NUTRITION_PATH):
        self.path = path
        self.nutrition_path = nutrition_path
        self._df = None
        self._ing_cols = None
        self._nf = NutritionFacts(nutrition_path)

    def _load(self):
        self._df = pd.read_csv(self.path)
        non_ing = ['title', 'rating', 'meal', 'url']
        self._ing_cols = [c for c in self._df.columns if c not in non_ing]

        # normalize meal column once
        self._df['meal'] = self._df['meal'].str.lower().str.strip()

    def generate(self, random_state: int = None) -> dict:
        if self._df is None:
            self._load()

        menu = {}
        rng = np.random.RandomState(random_state)

        for meal in self.MEAL_ORDER:
            allowed = self.MEAL_MAP.get(meal, [meal])

            subset = self._df[self._df['meal'].isin(allowed)].copy()

            # fallback if still empty
            if len(subset) < 5:
                subset = self._df.copy()

            # top 20% by rating
            threshold = subset['rating'].quantile(0.6)
            top = subset[subset['rating'] >= threshold]
            if top.empty:
                top = subset

            row = top.sample(1).iloc[0]

            recipe_ings = [c for c in self._ing_cols if row[c] == 1]

            menu[meal] = {
                'title': row['title'],
                'rating': row['rating'],
                'ingredients': recipe_ings,
                'url': row.get('url', ''),
                'nutrients': self._nf.get(recipe_ings),
            }

        return menu

    def format_output(self, menu: dict = None, random_state: int = 42) -> str:
        if menu is None:
            menu = self.generate(random_state=random_state)

        lines = []

        for meal in self.MEAL_ORDER:
            if meal not in menu:
                continue

            data = menu[meal]

            lines.append(meal.upper())
            lines.append("-" * 45)
            lines.append(f"{data['title']} (rating: {data['rating']})")

            lines.append("Ingredients:")
            for ing in data['ingredients']:
                lines.append(f"  - {ing}")

            lines.append("Nutrients (per serving estimate):")

            totals = {}
            for ing_facts in data['nutrients'].values():
                for nutrient, pct in ing_facts.items():
                    totals[nutrient] = totals.get(nutrient, 0) + pct

            n = max(1, len(data['ingredients']))

            for nutrient, total in list(totals.items())[:6]:
                normalized = total / n
                lines.append(f"  - {nutrient}: {round(min(normalized, 100))}%")

            lines.append(f"URL: {data['url']}")
            lines.append("")

        return "\n".join(lines)
    
        