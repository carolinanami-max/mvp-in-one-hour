from enum import Enum
from typing import List

from pydantic import BaseModel, Field, field_validator

INGREDIENTS_PER_RECIPE = 5


class DietaryPreference(str, Enum):
    OMNIVORE = "omnivore"
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    KETO = "keto"


class Ingredient(BaseModel):
    """Single ingredient with quantity."""

    name: str = Field(..., description="Name of the ingredient")
    quantity: str = Field(..., description="Quantity with unit (e.g., '2 cups', '500g')")


class Recipe(BaseModel):
    """A single themed dinner recipe."""

    day: str = Field(..., description="Day of the week")
    theme: str = Field(..., description="Cuisine theme (e.g., 'Italian', 'Mexican')")
    dish_name: str = Field(..., description="Name of the dish")
    ingredients: List[Ingredient] = Field(..., description="List of 5 ingredients")
    instructions: str = Field(..., description="Brief cooking instructions")

    @field_validator("ingredients")
    @classmethod
    def validate_ingredient_count(cls, v: List[Ingredient]) -> List[Ingredient]:
        if len(v) != INGREDIENTS_PER_RECIPE:
            raise ValueError(
                f"Recipe must have exactly {INGREDIENTS_PER_RECIPE} ingredients, got {len(v)}"
            )
        return v


class ShoppingItem(BaseModel):
    """Consolidated shopping list item."""

    name: str = Field(..., description="Ingredient name")
    total_quantity: str = Field(..., description="Total quantity needed")
    used_in: List[str] = Field(..., description="List of dishes using this ingredient")


class MealPlanDraft(BaseModel):
    """Meal plan without shopping list (for first LLM pass)."""

    dietary_preference: str = Field(..., description="Selected dietary preference")
    recipes: List[Recipe] = Field(..., description="List of 7 recipes for the week")

    @field_validator("recipes")
    @classmethod
    def validate_recipe_count(cls, v: List[Recipe]) -> List[Recipe]:
        if len(v) != 7:
            raise ValueError(f"Meal plan must have exactly 7 recipes, got {len(v)}")
        return v

    @field_validator("recipes")
    @classmethod
    def validate_unique_themes(cls, v: List[Recipe]) -> List[Recipe]:
        themes = [r.theme.lower() for r in v]
        if len(themes) != len(set(themes)):
            raise ValueError("All 7 themes must be unique")
        return v


class ShoppingList(BaseModel):
    """Wrapper for shopping list parsing."""

    shopping_list: List[ShoppingItem] = Field(..., description="Consolidated shopping list")


class MealPlan(BaseModel):
    """Complete weekly meal plan."""

    dietary_preference: str = Field(..., description="Selected dietary preference")
    recipes: List[Recipe] = Field(..., description="List of 7 recipes for the week")
    shopping_list: List[ShoppingItem] = Field(..., description="Consolidated shopping list")

    @field_validator("recipes")
    @classmethod
    def validate_recipe_count(cls, v: List[Recipe]) -> List[Recipe]:
        if len(v) != 7:
            raise ValueError(f"Meal plan must have exactly 7 recipes, got {len(v)}")
        return v

    @field_validator("recipes")
    @classmethod
    def validate_unique_themes(cls, v: List[Recipe]) -> List[Recipe]:
        themes = [r.theme.lower() for r in v]
        if len(themes) != len(set(themes)):
            raise ValueError("All 7 themes must be unique")
        return v
