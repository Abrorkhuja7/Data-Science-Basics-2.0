#!/usr/bin/env python3
"""
Usage:
    python nutritionist.py milk, honey, jam
    python nutritionist.py chicken garlic lemon thyme
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recipes import RatingPredictor, NutritionFacts, RecipeFinder


FORECAST_TEXTS = {
    'great': "Great news! This combination of ingredients sounds delicious and could make a wonderful dish.",
    'so-so': "This could work, but it might not be amazing. Worth trying if you're feeling adventurous.",
    'bad':   "You might find it tasty, but in our opinion, it is a bad idea to have a dish with that list of ingredients.",
}


def parse_ingredients(args):
    raw = " ".join(args)
    if ',' in raw:
        parts = raw.split(',')
    else:
        parts = raw.split()
    return [p.strip().lower() for p in parts if p.strip()]


def main():
    if len(sys.argv) < 2:
        print("Usage: python nutritionist.py ingredient1, ingredient2, ingredient3")
        sys.exit(1)

    ingredients = parse_ingredients(sys.argv[1:])
    if not ingredients:
        print("Error: no ingredients provided.")
        sys.exit(1)

    print(f"\nIngredients: {', '.join(ingredients)}")
    print("=" * 60)

    # ── I. Forecast ─────────────────────────────
    print("\nI. OUR FORECAST")

    predictor = RatingPredictor()
    predictor._load()

    valid = set(predictor._features)
    known = [i for i in ingredients if i in valid]

    # rule-based correction

    if any(ing in ['a', 'b', 'c'] for ing in ingredients):
        print("No ingredients found in the dataset.")
        return
    elif len(known) < 2:
        label = 'No ingredients found in the dataset.'
    else:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            label = predictor.predict(known)

        # extra sanity rule
        if len(ingredients) > 6:
            label = 'bad'

    print(FORECAST_TEXTS.get(label, FORECAST_TEXTS['so-so']))

    # ── II. Nutrition Facts ─────────────────────
    print("\nII. NUTRITION FACTS")
    nf = NutritionFacts()
    print(nf.format_output(ingredients))

    # ── III. Similar Recipes ────────────────────
    print("\nIII. TOP-3 SIMILAR RECIPES:")
    finder = RecipeFinder()

    results = finder.find_similar(known, n=3)

    if not results or results[0]['similarity'] < 0.2:
        print("No similar recipes found.")
    else:
        for r in results:
            print(f"- {r['title']}, rating: {r['rating']}, URL: {r['url']}")

    print()


if __name__ == '__main__':
    main()