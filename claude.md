# MVP Requirements: Weekly Themed Meal Planner

## Core Features
1. Generate 7 themed dinners (one per day, Monday-Sunday)
2. Each recipe has exactly 5 ingredients (excluding salt, pepper, oil, garlic)
3. Support basic dietary preferences (vegan, vegetarian, keto, omnivore)
4. Create consolidated shopping list with quantities
5. Display results in readable format
6. Allow user to regenerate new plan
7. Sends the user the meal plan via email easily via python integration

The AI must ask the steps
1. Ask about this week food preferences
2. Ask for themed requests, IF NOT PROVIDED IT WILL BE RANDOM



## Technical Stack
- Google Colab (Python 3.8+)
- LangChain (prompt management) 1.0
- OpenAI or Anthropic API (LLM)
- LangSmith (evaluation & monitoring)
- Pydantic (output validation)

## Quality Requirements
- Generation time < 30 seconds
- No fake/hallucinated ingredients
- All 7 themes must be different
- Shopping list includes all recipe ingredients
- API cost < $0.10 per generation

## Evaluation
- Automated checks: 5-ingredient rule, theme uniqueness, shopping list completeness
- Test cases for each dietary preference
- Track: constraint adherence, generation time, cost
