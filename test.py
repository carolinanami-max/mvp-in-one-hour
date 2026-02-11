# Test checklist for meal_planner.py
# These are intentionally simple, high-level tests (not automated).

TESTS = [
    "Generate plan: returns success True and 7 recipes",
    "Generate plan: each recipe has exactly 5 ingredients",
    "Generate plan: all 7 themes unique",
    "Shopping list: includes every recipe ingredient",
    "Shopping list: combines duplicate ingredients with summed quantities",
    "Dietary compliance: vegan has no animal products",
    "Dietary compliance: vegetarian has no meat/fish",
    "Dietary compliance: keto avoids grains/sugar/starchy vegetables",
    "Theme requests respected when provided (e.g., Italian, Mexican)",
    "Email sending: fails gracefully if EMAIL_SENDER/PASSWORD not set",
    "Email sending: succeeds with valid Gmail app password",
    "Gradio UI: generate button streams results",
    "Gradio UI: email button returns status message",
    "LangSmith eval: dataset created once and reused",
    "LangSmith eval: evaluators return score and comment",
]

if __name__ == "__main__":
    for test in TESTS:
        print(f"- {test}")
