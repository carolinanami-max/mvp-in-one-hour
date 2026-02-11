import json
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, List

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langsmith import Client, traceable
from langsmith.evaluation import evaluate

from models import MealPlan, MealPlanDraft, ShoppingList


@dataclass
class Settings:
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    langsmith_api_key: str = os.getenv("LANGSMITH_API_KEY", "")
    langsmith_project: str = os.getenv("LANGSMITH_PROJECT", "meal-planner-mvp")
    meal_plan_model: str = os.getenv("MEAL_PLAN_MODEL", "gpt-4o-mini")
    shopping_list_model: str = os.getenv("SHOPPING_LIST_MODEL", "gpt-4o-mini")
    eval_model: str = os.getenv("EVAL_MODEL", "gpt-4o-mini")
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    email_sender: str = os.getenv("EMAIL_SENDER", "")
    email_password: str = os.getenv("EMAIL_PASSWORD", "")


def apply_env(settings: Settings) -> None:
    if settings.openai_api_key:
        os.environ["OPENAI_API_KEY"] = settings.openai_api_key
    if settings.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = settings.langsmith_api_key
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = settings.langsmith_project


def get_llm(model: str, temperature: float) -> ChatOpenAI:
    return ChatOpenAI(model=model, temperature=temperature)


MEAL_PLAN_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a professional meal planner. Generate a weekly meal plan with EXACTLY 7 themed dinners.

STRICT RULES - FOLLOW EXACTLY:
1. Each recipe MUST have EXACTLY 5 main ingredients (do NOT count salt, pepper, oil, garlic, or water)
2. ALL 7 THEMES MUST BE COMPLETELY DIFFERENT - no duplicates allowed! Use: Italian, Mexican, Japanese, Indian, Thai, Greek, American, Chinese, French, Korean, Mediterranean, Middle Eastern, etc.
3. All ingredients MUST be real, commonly available items
4. Follow the dietary preference strictly
5. Provide realistic quantities for 2 servings

Dietary Guidelines:
- Omnivore: Any ingredients allowed
- Vegetarian: No meat or fish, eggs and dairy OK
- Vegan: No animal products at all
- Keto: Low carb, high fat, no grains/sugar/starchy vegetables

Output valid JSON matching this exact structure:
{format_instructions}""",
        ),
        (
            "human",
            """Create a weekly meal plan:
- Diet: {dietary_preference}
- Theme requests: {theme_requests}
- Preferences: {additional_preferences}

IMPORTANT: Each day MUST have a DIFFERENT cuisine theme. Generate 7 unique themed dinners for Monday-Sunday.""",
        ),
    ]
)

SHOPPING_LIST_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a shopping list organizer. Consolidate ingredients from multiple recipes into a single shopping list.

RULES:
1. Combine same ingredients and sum their quantities
2. List which dishes use each ingredient
3. Use standard units (cups, tablespoons, grams, pieces)
4. Round up quantities for convenience

Output valid JSON matching this exact structure:
{format_instructions}""",
        ),
        (
            "human",
            """Consolidate the shopping list from these recipes:
{recipes_json}""",
        ),
    ]
)


def extract_json(content: str) -> str:
    if "```json" in content:
        return content.split("```json")[1].split("```")[0]
    if "```" in content:
        return content.split("```")[1].split("```")[0]
    return content


class MealPlanGenerator:
    """Generates weekly themed meal plans with a consolidated shopping list."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.meal_llm = get_llm(settings.meal_plan_model, settings.temperature)
        self.shop_llm = get_llm(settings.shopping_list_model, settings.temperature)
        self.eval_llm = get_llm(settings.eval_model, settings.temperature)
        self.meal_parser = JsonOutputParser(pydantic_object=MealPlanDraft)
        self.shopping_parser = JsonOutputParser(pydantic_object=ShoppingList)

    @traceable(name="generate_meal_plan")
    def generate(
        self,
        dietary_preference: str,
        theme_requests: str = "random",
        additional_preferences: str = "none",
    ) -> Dict[str, Any]:
        start_time = time.time()

        recipe_chain = MEAL_PLAN_PROMPT | self.meal_llm

        result = recipe_chain.invoke(
            {
                "dietary_preference": dietary_preference,
                "theme_requests": theme_requests if theme_requests else "random diverse cuisines",
                "additional_preferences": additional_preferences if additional_preferences else "none",
                "format_instructions": self.meal_parser.get_format_instructions(),
            }
        )

        try:
            content = extract_json(result.content)
            meal_plan_data = json.loads(content)
            meal_plan_draft = MealPlanDraft(**meal_plan_data)

            shopping_chain = SHOPPING_LIST_PROMPT | self.shop_llm
            shopping_result = shopping_chain.invoke(
                {
                    "recipes_json": json.dumps(meal_plan_draft.model_dump()["recipes"]),
                    "format_instructions": self.shopping_parser.get_format_instructions(),
                }
            )
            shopping_content = extract_json(shopping_result.content)
            shopping_data = json.loads(shopping_content)
            shopping_list = ShoppingList(**shopping_data).shopping_list

            meal_plan = MealPlan(
                dietary_preference=meal_plan_draft.dietary_preference,
                recipes=meal_plan_draft.recipes,
                shopping_list=shopping_list,
            )

            generation_time = time.time() - start_time

            return {
                "success": True,
                "meal_plan": meal_plan.model_dump(),
                "generation_time": round(generation_time, 2),
                "error": None,
            }

        except json.JSONDecodeError as exc:
            return {
                "success": False,
                "meal_plan": None,
                "error": f"JSON parsing error: {str(exc)}",
            }
        except Exception as exc:
            return {"success": False, "meal_plan": None, "error": str(exc)}


def format_meal_plan_email(meal_plan: Dict[str, Any]) -> str:
    lines = []
    lines.append("=" * 40)
    lines.append(f"WEEKLY MEAL PLAN ({meal_plan['dietary_preference'].upper()})")
    lines.append("=" * 40)
    lines.append("")

    lines.append("SHOPPING LIST")
    lines.append("-" * 20)
    for item in meal_plan["shopping_list"]:
        lines.append(f"- {item['total_quantity']} {item['name']}")
    lines.append("")

    lines.append("=" * 40)
    lines.append("RECIPES")
    lines.append("=" * 40)
    for recipe in meal_plan["recipes"]:
        lines.append(f"\n{recipe['day'].upper()} - {recipe['theme']}")
        lines.append(f"{recipe['dish_name']}")
        ingredients = ", ".join(
            [f"{ing['quantity']} {ing['name']}" for ing in recipe["ingredients"]]
        )
        lines.append(f"Ingredients: {ingredients}")
        lines.append(f"Instructions: {recipe['instructions']}")

    return "\n".join(lines)


def send_meal_plan_email(recipient_email: str, meal_plan: Dict[str, Any], settings: Settings) -> Dict[str, str]:
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText

    if not settings.email_sender or not settings.email_password:
        return {"success": False, "message": "Email sender or password not configured."}

    try:
        msg = MIMEMultipart()
        msg["From"] = settings.email_sender
        msg["To"] = recipient_email
        msg["Subject"] = "Your Weekly Meal Plan"

        body = format_meal_plan_email(meal_plan)
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(settings.email_sender, settings.email_password)
            server.send_message(msg)

        return {"success": True, "message": f"Email sent to {recipient_email}"}
    except Exception as exc:
        return {"success": False, "message": f"Failed to send email: {str(exc)}"}


def run_basic_validation(meal_plan_result: Dict[str, Any]) -> Dict[str, Any]:
    if not meal_plan_result.get("success"):
        return {"passed": False, "errors": [meal_plan_result.get("error")]}

    meal_plan = meal_plan_result["meal_plan"]
    errors: List[str] = []

    if len(meal_plan["recipes"]) != 7:
        errors.append(f"Expected 7 recipes, got {len(meal_plan['recipes'])}")

    themes = [r["theme"].lower() for r in meal_plan["recipes"]]
    if len(themes) != len(set(themes)):
        duplicates = [t for t in themes if themes.count(t) > 1]
        errors.append(f"Duplicate themes: {set(duplicates)}")

    for recipe in meal_plan["recipes"]:
        if len(recipe["ingredients"]) != 5:
            errors.append(
                f"{recipe['day']}: {len(recipe['ingredients'])} ingredients (expected 5)"
            )

    return {"passed": len(errors) == 0, "errors": errors}


def parse_llm_json(content: str) -> Dict[str, Any]:
    try:
        return json.loads(content.strip())
    except Exception:
        pass

    content = re.sub(r"```json\s*", "", content)
    content = re.sub(r"```\s*", "", content)

    match = re.search(r"\{[^{}]*\"score\"[^{}]*\}", content)
    if match:
        try:
            return json.loads(match.group())
        except Exception:
            pass

    if '"score": 1' in content or '"score":1' in content:
        return {"score": 1, "reason": "Passed"}
    return {"score": 0, "reason": "Could not parse evaluation"}


TEST_CASES = [
    {"dietary_preference": "omnivore", "theme_requests": "random", "additional_preferences": "none"},
    {
        "dietary_preference": "vegetarian",
        "theme_requests": "Italian, Asian, Mexican",
        "additional_preferences": "no mushrooms",
    },
    {"dietary_preference": "vegan", "theme_requests": "random", "additional_preferences": "high protein"},
    {
        "dietary_preference": "keto",
        "theme_requests": "Mediterranean, American",
        "additional_preferences": "none",
    },
]


def create_or_get_dataset(ls_client: Client, dataset_name: str = "meal-planner-eval"):
    try:
        dataset = ls_client.read_dataset(dataset_name=dataset_name)
        print(f"Using existing dataset: {dataset_name}")
    except Exception:
        dataset = ls_client.create_dataset(
            dataset_name=dataset_name, description="Meal planner test cases"
        )
        for test_case in TEST_CASES:
            ls_client.create_example(inputs=test_case, dataset_id=dataset.id)
        print(f"Created dataset: {dataset_name} with {len(TEST_CASES)} examples")
    return dataset


def target_function(generator: MealPlanGenerator, inputs: Dict[str, Any]) -> Dict[str, Any]:
    return generator.generate(
        dietary_preference=inputs["dietary_preference"],
        theme_requests=inputs.get("theme_requests", "random"),
        additional_preferences=inputs.get("additional_preferences", "none"),
    )


def run_langsmith_evaluation(generator: MealPlanGenerator):
    ls_client = Client()
    dataset = create_or_get_dataset(ls_client)
    eval_llm = generator.eval_llm

    def shopping_list_completeness(run, example) -> Dict[str, Any]:
        output = run.outputs
        if not output or not output.get("success"):
            return {
                "key": "shopping_list_completeness",
                "score": 0,
                "comment": "Generation failed",
            }

        meal_plan = output["meal_plan"]
        recipe_ingredients = [
            ing["name"] for recipe in meal_plan["recipes"] for ing in recipe["ingredients"]
        ]
        shopping_items = [item["name"] for item in meal_plan["shopping_list"]]

        prompt = f"""Check if this shopping list contains all the recipe ingredients.

RECIPE INGREDIENTS: {recipe_ingredients}

SHOPPING LIST: {shopping_items}

Does the shopping list include every ingredient? 
Respond with JSON only: {{"score": 1, "reason": "..."}} if complete, {{"score": 0, "reason": "missing: X, Y"}} if not."""

        response = eval_llm.invoke(prompt)
        result = parse_llm_json(response.content)
        return {
            "key": "shopping_list_completeness",
            "score": result.get("score", 0),
            "comment": result.get("reason", ""),
        }

    def theme_uniqueness(run, example) -> Dict[str, Any]:
        output = run.outputs
        if not output or not output.get("success"):
            return {"key": "theme_uniqueness", "score": 0, "comment": "Generation failed"}

        meal_plan = output["meal_plan"]
        themes = [r["theme"] for r in meal_plan["recipes"]]
        ingredient_counts = [len(r["ingredients"]) for r in meal_plan["recipes"]]

        prompt = f"""Evaluate this meal plan structure.

THEMES (should all be different): {themes}
INGREDIENT COUNTS (should all be 5): {ingredient_counts}

Check: Are all 7 themes unique cuisines? Does each recipe have exactly 5 ingredients?
Respond with JSON only: {{"score": 1, "reason": "..."}} if valid, {{"score": 0, "reason": "..."}} if not."""

        response = eval_llm.invoke(prompt)
        result = parse_llm_json(response.content)
        return {
            "key": "theme_uniqueness",
            "score": result.get("score", 0),
            "comment": result.get("reason", ""),
        }

    def dietary_compliance(run, example) -> Dict[str, Any]:
        output = run.outputs
        if not output or not output.get("success"):
            return {"key": "dietary_compliance", "score": 0, "comment": "Generation failed"}

        meal_plan = output["meal_plan"]
        dietary_pref = example.inputs.get("dietary_preference", "omnivore")

        all_ingredients = [
            ing["name"] for recipe in meal_plan["recipes"] for ing in recipe["ingredients"]
        ]

        prompt = f"""Check if these ingredients comply with a {dietary_pref} diet.

DIET: {dietary_pref}
- Vegan: No animal products (meat, fish, dairy, eggs, honey)
- Vegetarian: No meat or fish
- Keto: No grains, sugar, starchy vegetables, most fruits
- Omnivore: Anything allowed

INGREDIENTS: {all_ingredients}

Are all ingredients real foods that comply with the {dietary_pref} diet?
Respond with JSON only: {{"score": 1, "reason": "all compliant"}} or {{"score": 0, "reason": "violation: X"}}"""

        response = eval_llm.invoke(prompt)
        result = parse_llm_json(response.content)
        return {
            "key": "dietary_compliance",
            "score": result.get("score", 0),
            "comment": result.get("reason", ""),
        }

    results = evaluate(
        lambda inputs: target_function(generator, inputs),
        data=dataset,
        evaluators=[shopping_list_completeness, theme_uniqueness, dietary_compliance],
        experiment_prefix="meal-planner-llm-judge",
    )

    print("\nEvaluation complete! View results in LangSmith dashboard.")
    return results


def format_shopping_list(meal_plan: Dict[str, Any]) -> str:
    lines = ["## Shopping List\n"]
    for item in meal_plan["shopping_list"]:
        lines.append(f"- {item['total_quantity']} **{item['name']}**")
    return "\n".join(lines)


def format_single_recipe(recipe: Dict[str, Any]) -> str:
    ingredients = ", ".join(
        [f"{ing['quantity']} {ing['name']}" for ing in recipe["ingredients"]]
    )
    return f"**{recipe['day']}** | {recipe['theme']} | *{recipe['dish_name']}*\n{ingredients}\n"


def build_gradio_app(generator: MealPlanGenerator, settings: Settings):
    import gradio as gr

    def generate_plan_streaming(dietary_pref, theme_requests, additional_prefs):
        output = "Generating your meal plan..."
        yield output

        result = generator.generate(
            dietary_preference=dietary_pref,
            theme_requests=theme_requests if theme_requests.strip() else "random",
            additional_preferences=additional_prefs if additional_prefs.strip() else "none",
        )

        if not result["success"]:
            yield f"Error: {result['error']}"
            return

        meal_plan = result["meal_plan"]

        output = f"## Weekly {meal_plan['dietary_preference'].title()} Meal Plan\n\n"
        output += format_shopping_list(meal_plan) + "\n\n---\n\n"
        output += "## Recipes\n\n"
        yield output
        time.sleep(0.1)

        for recipe in meal_plan["recipes"]:
            output += format_single_recipe(recipe) + "\n"
            yield output
            time.sleep(0.05)

        validation = run_basic_validation(result)
        status = "All checks passed" if validation["passed"] else f"Issues: {', '.join(validation['errors'])}"
        output += f"---\n*Generated in {result['generation_time']}s | {status}*"
        yield output

    def send_email_ui(email, dietary_pref, theme_requests, additional_prefs):
        if not email or "@" not in email:
            return "Enter a valid email address."

        result = generator.generate(
            dietary_preference=dietary_pref,
            theme_requests=theme_requests if theme_requests.strip() else "random",
            additional_preferences=additional_prefs if additional_prefs.strip() else "none",
        )

        if not result["success"]:
            return f"Generation failed: {result['error']}"

        email_result = send_meal_plan_email(email, result["meal_plan"], settings)
        return email_result["message"]

    with gr.Blocks(title="Weekly Meal Planner") as app:
        gr.Markdown("# Weekly Meal Planner\nGenerate 7 themed dinners with a consolidated shopping list.")

        with gr.Row():
            with gr.Column(scale=1):
                dietary_dropdown = gr.Dropdown(
                    choices=["omnivore", "vegetarian", "vegan", "keto"],
                    value="omnivore",
                    label="Diet",
                )
                theme_input = gr.Textbox(
                    label="Themes (optional)", placeholder="Italian, Mexican...", lines=1
                )
                additional_input = gr.Textbox(
                    label="Preferences (optional)", placeholder="no nuts, quick meals...", lines=1
                )
                generate_btn = gr.Button("Generate", variant="primary")

                gr.Markdown("---")
                email_input = gr.Textbox(label="Email", placeholder="you@example.com")
                send_btn = gr.Button("Email Plan")
                email_status = gr.Textbox(label="Status", interactive=False)

            with gr.Column(scale=2):
                output_display = gr.Markdown("Click **Generate** to start.")

        generate_btn.click(
            fn=generate_plan_streaming,
            inputs=[dietary_dropdown, theme_input, additional_input],
            outputs=[output_display],
        )

        send_btn.click(
            fn=send_email_ui,
            inputs=[email_input, dietary_dropdown, theme_input, additional_input],
            outputs=[email_status],
        )

    return app


def main():
    settings = Settings()
    apply_env(settings)
    generator = MealPlanGenerator(settings)
    app = build_gradio_app(generator, settings)
    app.launch(share=True)


if __name__ == "__main__":
    main()
