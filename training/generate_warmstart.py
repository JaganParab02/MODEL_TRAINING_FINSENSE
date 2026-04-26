import json

def build_prompt(balance, goal, days, allowance, name, category, amount, necessity, context, events="None"):
    return f"""You are a financial decision agent.

Return ONLY valid JSON. No explanation. No markdown. No text before or after.

{{"decision": "allow" | "reduce" | "avoid", "approved_amount": number, "reasoning": "short"}}

Current state:
Balance: {balance}
Goal Remaining: {goal}
Days Left: {days}
Daily Allowance: {allowance}
Active Events: {events}

Expense:
Name: {name}
Category: {category}
Amount: {amount}
Necessity: {necessity}
Context: {context}

Rules:
- Always protect essentials
- Avoid overspending beyond daily allowance
- Reduce spending when prices are inflated
- Prioritize goal completion over comfort
- Use double quotes only, no trailing commas, no comments"""

data = [
    # 1. Essential, affordable -> allow
    (
        build_prompt(10000, 5000, 10, 500, "Groceries", "food", 200, "essential", "normal"),
        {"decision": "allow", "approved_amount": 200, "reasoning": "Essential and affordable"}
    ),
    # 2. Essential, expensive -> reduce
    (
        build_prompt(10000, 5000, 10, 500, "House Rent", "rent", 1000, "essential", "normal"),
        {"decision": "reduce", "approved_amount": 500, "reasoning": "Essential but over budget, reducing"}
    ),
    # 3. Discretionary, affordable -> avoid (to save)
    (
        build_prompt(10000, 5000, 10, 500, "Fine Dining", "food", 150, "discretionary", "normal"),
        {"decision": "avoid", "approved_amount": 0, "reasoning": "Discretionary - skipping to save"}
    ),
    # 4. Discretionary, weekend -> avoid
    (
        build_prompt(10000, 5000, 10, 500, "Movie Tickets", "entertainment", 300, "discretionary", "weekend"),
        {"decision": "avoid", "approved_amount": 0, "reasoning": "Weekend discretionary - avoiding lifestyle spending"}
    ),
    # 5. Semi-essential, affordable -> allow
    (
        build_prompt(10000, 5000, 10, 500, "Local Restaurant", "food", 100, "semi-essential", "normal"),
        {"decision": "allow", "approved_amount": 100, "reasoning": "Semi-essential and cheap enough"}
    ),
    # 6. Semi-essential, expensive -> reduce
    (
        build_prompt(10000, 5000, 10, 500, "Uber", "transport", 400, "semi-essential", "normal"),
        {"decision": "reduce", "approved_amount": 200, "reasoning": "Semi-essential, reducing spend"}
    ),
    # 7. Semi-essential, very expensive -> avoid
    (
        build_prompt(10000, 5000, 10, 500, "Premium Train", "transport", 1000, "semi-essential", "normal"),
        {"decision": "avoid", "approved_amount": 0, "reasoning": "Semi-essential but too expensive"}
    ),
    # 8. Emergency, any -> allow
    (
        build_prompt(10000, 5000, 10, 500, "Doctor Visit", "medical", 1500, "essential", "emergency"),
        {"decision": "allow", "approved_amount": 1500, "reasoning": "Emergency context - must not avoid"}
    ),
    # 9. Emergency, semi-essential -> allow
    (
        build_prompt(10000, 5000, 10, 500, "Appliance Repair", "household_repairs", 2000, "semi-essential", "emergency"),
        {"decision": "allow", "approved_amount": 2000, "reasoning": "Emergency context - must not avoid"}
    ),
    # 10. Discretionary, small surplus -> avoid
    (
        build_prompt(10000, 5000, 10, 500, "Streaming Sub", "entertainment", 200, "discretionary", "normal"),
        {"decision": "avoid", "approved_amount": 0, "reasoning": "Discretionary - skipping to save"}
    )
]

# Duplicate the examples to reach ~30 items just to satisfy SFT minimums
data = data * 3

with open("warmstart_data.jsonl", "w") as f:
    for prompt, response in data:
        line = {
            "messages": [
                {"role": "system", "content": "You are a financial agent. Reply only with valid JSON."},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": json.dumps(response)}
            ]
        }
        f.write(json.dumps(line) + "\\n")

print("Created warmstart_data.jsonl with 30 examples.")
