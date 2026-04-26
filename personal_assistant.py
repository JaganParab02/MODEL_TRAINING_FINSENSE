"""
Personal Assistant — LLM Conversation Engine
=============================================
Uses the configured LLM (via OpenAI-compatible client) to generate
purchase-specific financial questions and personalized advice.

Falls back to rule-based question banks when LLM is unavailable.
"""

import os
import json
from hf_wrapper import HFWrapper as OpenAI
from dotenv import load_dotenv

load_dotenv()


def _get_client():
    """Create OpenAI client from environment config."""
    return OpenAI(
        base_url=os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/"),
        api_key=os.getenv("HF_TOKEN", "")
    )


def _get_model():
    return os.getenv("MODEL_NAME", "mistral")


SYSTEM_PROMPT = """
You are FinSense, an expert Indian personal finance advisor embedded 
inside a financial planning simulation. You help users plan major 
purchases by understanding their complete financial picture.

Your personality:
- Warm, professional, and direct
- You understand Indian household finances deeply
- You know Indian products, prices, EMI culture, and family dynamics
- You never guess numbers — you always ask
"""

QUESTION_PROMPT = """
You are a Senior Wealth Manager at a top-tier Indian financial advisory firm. 
Your job is to generate exactly 4-5 insightful, professional questions to help a client plan a major purchase.
Your tone should be authoritative but supportive. Focus on deep financial health (e.g., emergency funds, risk, total debt) rather than just surface-level budget.

Rules you must follow:
- Return questions as a JSON array of objects ONLY.
- Each object must have:
    "question": the question string
    "type": one of ["salary", "timeline", "emi", "other"]
- No preamble, no explanation, no markdown — just the JSON array
- Maximum 5 questions, minimum 3

Example for "car worth Rs.15,00,000":
[
  {"question": "To ensure we structure this purchase safely, what is your approximate monthly take-home income?", "type": "salary"},
  {"question": "Are you currently servicing any other loans or EMIs that we should factor into your debt-to-income ratio?", "type": "emi"},
  {"question": "Do you have an adequate emergency fund set aside, or will this purchase deplete your liquid savings?", "type": "other"},
  {"question": "What is your target timeline for this acquisition (e.g., 6 months, 1 year)?", "type": "timeline"}
]
"""

ADVICE_PROMPT = """
You are a Senior Wealth Manager giving a final assessment to a client.
Based on the financial data provided, write a structured, professional mini-report.

CRITICAL RULES:
1. Return PLAIN TEXT ONLY.
2. DO NOT USE JSON. DO NOT USE BRACKETS [ ], BRACES { }, or tags like "question" or "type".
3. Structure your response EXACTLY with these three headings:
   
   FINANCIAL HEALTH CHECK:
   (1-2 sentences assessing their monthly surplus vs the goal)
   
   THE REALITY CHECK:
   (1-2 sentences giving a firm but polite verdict on the feasibility, highlighting risks if the task is 'Hard')
   
   ACTIONABLE ADVICE:
   (2 specific steps they should take during their timeline)
"""

INTENT_PROMPT = """
Extract the product and price the user wants to buy from their message.
Return a JSON object with keys: "product" (string or null) and "amount" (integer or null).
- If the user mentions "Lakhs" or "L", multiply by 100,000 (e.g., "15 Lakhs" -> 1500000).
- If they say "1.5L", it's 150000.
- If they say "K", multiply by 1000.
- If they don't specify a product, set it to null.
- If they don't specify a price, set it to null.
- If they only say a price but no product, set product to "unknown".
Return ONLY the JSON object. No markdown, no explanation.
"""

SIMULATION_SUMMARY_PROMPT = """
You are a Senior Wealth Manager analyzing a client's recent savings simulation.
Your job is to provide a brief (2-3 sentences), highly actionable summary explaining WHY they succeeded or failed.
Look at the 'bad decisions' to assign credit or blame (radioactive credit assignment).

CRITICAL RULES:
1. Return PLAIN TEXT ONLY. No markdown, no JSON, no bolding.
2. If the goal was NOT ACHIEVED, point out the specific mistakes (e.g., "You failed to reach your goal primarily because you overspent your daily allowance on Discretionary items like Fine Dining, and avoided an Emergency expense which caused cascading penalties.")
3. If the goal WAS ACHIEVED, praise their discipline but highlight any close calls (e.g., "Excellent discipline! You reached your goal by consistently reducing non-essential spending, though you had a close call when prices inflated.")
4. Keep it concise, professional, and directly tied to their specific logged actions.
"""

# ── Fallback question banks ─────────────────────────────────────────────────

FALLBACK_QUESTIONS = {
    "car": [
        {"question": "What is your monthly take-home salary?", "type": "salary"},
        {"question": "Do you have any existing EMIs or loans?", "type": "emi"},
        {"question": "How many family members will use the car?", "type": "other"},
        {"question": "What is your preferred timeline - 6, 12, or 18 months?", "type": "timeline"},
    ],
    "phone": [
        {"question": "What is your monthly take-home salary?", "type": "salary"},
        {"question": "Do you prefer Android or iOS?", "type": "other"},
        {"question": "What is your preferred timeline - 1, 2, or 3 months?", "type": "timeline"},
    ],
    "default": [
        {"question": "What is your monthly take-home salary?", "type": "salary"},
        {"question": "What are your current fixed monthly expenses approximately?", "type": "other"},
        {"question": "Do you have any existing EMIs or loans?", "type": "emi"},
        {"question": "What is your preferred timeline (in months)?", "type": "timeline"},
    ]
}


def get_fallback_questions(product: str) -> list:
    """Return rule-based questions for a product category."""
    product_lower = product.lower()
    for key in FALLBACK_QUESTIONS:
        if key in product_lower:
            return FALLBACK_QUESTIONS[key]
    return FALLBACK_QUESTIONS["default"]


def _regex_extract_intent(text: str) -> dict:
    """Robust regex-based intent extraction as a fallback.
    Handles patterns like:
      'I want to buy a car worth Rs.15,00,000'
      'buy an iPhone for 80000'
      'laptop worth 1.5 lakhs'
      'bike for 1.5L'
    """
    import re
    product = None
    amount = None

    # ── Extract product name ──
    # Pattern: "buy a/an <product>" or "want a/an <product>"
    prod_match = re.search(
        r'(?:buy|purchase|get)\s+(?:a|an)\s+(.+?)\s+'
        r'(?:worth|for|at|around|priced|costing|of)',
        text, re.IGNORECASE
    )
    if not prod_match:
        # Try without article: "want car worth..."
        prod_match = re.search(
            r'(?:buy|want|planning|purchase|get)\s+(.+?)\s+'
            r'(?:worth|for|at|around|priced|costing|of)',
            text, re.IGNORECASE
        )
    if prod_match:
        raw_product = prod_match.group(1).strip().strip('"\'')
        # Clean up residual words like "to buy a" or "to get an"
        raw_product = re.sub(r'^(?:to\s+)?(?:buy|get|purchase)\s+(?:a|an)\s+', '', raw_product, flags=re.IGNORECASE)
        raw_product = re.sub(r'^(?:a|an)\s+', '', raw_product, flags=re.IGNORECASE)
        product = raw_product.strip()
    else:
        # Fallback: common product keywords
        for kw in ["car", "bike", "laptop", "phone", "iphone", "tablet",
                    "tv", "television", "watch", "camera", "scooter",
                    "house", "flat", "apartment", "motorcycle"]:
            if kw in text.lower():
                product = kw.capitalize()
                break

    # ── Extract amount ──
    # Handle "Rs.15,00,000" or "Rs 15,00,000" or "₹15,00,000"
    amt_match = re.search(
        r'(?:rs\.?|₹|inr)\s*([\d,]+(?:\.\d+)?)', text, re.IGNORECASE
    )
    if amt_match:
        raw_num = amt_match.group(1).replace(',', '')
        try:
            amount = int(float(raw_num))
        except ValueError:
            pass

    # Handle "15 lakhs" / "1.5L" / "15L"
    if amount is None:
        lakh_match = re.search(
            r'([\d.]+)\s*(?:lakhs?|lacs?|L)\b', text, re.IGNORECASE
        )
        if lakh_match:
            try:
                amount = int(float(lakh_match.group(1)) * 100_000)
            except ValueError:
                pass

    # Handle "80K" / "80 thousand"
    if amount is None:
        k_match = re.search(
            r'([\d.]+)\s*(?:K|thousand)\b', text, re.IGNORECASE
        )
        if k_match:
            try:
                amount = int(float(k_match.group(1)) * 1000)
            except ValueError:
                pass

    # Handle bare large numbers like "1500000" or "80000"
    if amount is None:
        bare_match = re.search(r'\b(\d{5,})\b', text)
        if bare_match:
            try:
                amount = int(bare_match.group(1))
            except ValueError:
                pass

    return {"product": product, "amount": amount}


def extract_intent(user_message: str) -> dict:
    """Extract product and amount from the user's initial message.
    Tries LLM first, then falls back to robust regex parsing."""

    # ── Always run regex first as a safety net ──
    regex_result = _regex_extract_intent(user_message)

    # ── Try LLM for potentially better extraction ──
    try:
        client = _get_client()
        model = _get_model()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": INTENT_PROMPT},
                {"role": "user", "content": user_message}
            ],
            temperature=0.0,
            max_tokens=100
        )
        raw = response.choices[0].message.content.strip()
        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1:
            raw = raw[start:end + 1]
            data = json.loads(raw)
            llm_product = data.get("product")
            llm_amount = data.get("amount")

            # Use LLM result only if it produced valid values
            result = {
                "product": llm_product if (llm_product and llm_product.lower() != "unknown") else regex_result["product"],
                "amount": llm_amount if llm_amount else regex_result["amount"],
            }
            # If LLM gave us at least one useful field, return it
            if result["product"] or result["amount"]:
                return result
    except Exception as e:
        print(f"[ASSISTANT LLM] Intent extraction failed, using regex fallback: {e}")

    # ── Fall back to regex result ──
    return regex_result


def generate_questions(product: str, amount: float) -> list:
    """Use LLM to generate product-specific questions. Falls back to rules."""
    try:
        client = _get_client()
        model = _get_model()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": QUESTION_PROMPT},
                {"role": "user", "content":
                 f"Generate questions for: {product} worth Rs.{amount:,.0f}"}
            ],
            temperature=0.3,
            max_tokens=500
        )
        raw = response.choices[0].message.content.strip()
        # strip markdown if present
        raw = raw.replace("```json", "").replace("```", "").strip()
        # extract JSON array
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end != -1:
            raw = raw[start:end + 1]
            questions = json.loads(raw)
            if isinstance(questions, list):
                # ENSURE CRITICAL TYPES ARE PRESENT
                types = [q.get("type") for q in questions if isinstance(q, dict)]
                if "timeline" not in types:
                    questions.append({
                        "question": "What is your preferred timeline (in months) to complete this purchase?", 
                        "type": "timeline"
                    })
                if "salary" not in types:
                    questions.insert(0, {
                        "question": "What is your approximate monthly take-home salary?", 
                        "type": "salary"
                    })
                if len(questions) >= 3:
                    return questions[:5]
        return get_fallback_questions(product)
    except Exception as e:
        print(f"[ASSISTANT LLM] Question generation failed: {e}")
        return get_fallback_questions(product)


def generate_summary_advice(product, amount, budget_result, selected_product):
    """LLM generates a warm, personalized summary of the financial plan."""
    try:
        client = _get_client()
        model = _get_model()
        prompt = f"""
The user wants to buy: {product} worth Rs.{amount:,.0f}
They selected: {selected_product['name']} at Rs.{selected_product['price']:,}
Their monthly surplus: Rs.{budget_result['monthly_surplus']:,.0f}
Task difficulty: {budget_result['task']}
Remaining balance after purchase: Rs.{budget_result['remaining_balance']:,.0f}
Timeline: {budget_result['months_needed']} months

Write a warm 2-3 sentence personalized financial summary and encouragement.
Be specific with the numbers. End with what they should focus on during 
the saving period. Do not use bullet points.
"""
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": ADVICE_PROMPT},
                {"role": "user", "content": prompt + "\n\nIMPORTANT: Return ONLY plain text. NO JSON, NO BRACKETS."}
            ],
            temperature=0.5,
            max_tokens=200
        )
        content = response.choices[0].message.content.strip()
        
        # Post-processing: Strip residual JSON if LLM fails
        if '"question":' in content or '"type":' in content:
            # Try to extract just the text values
            parts = re.findall(r'"question":\s*"([^"]+)"', content)
            if parts:
                content = " ".join(parts)
            else:
                # Last resort: remove all brackets/braces/quotes
                content = content.replace("[", "").replace("]", "").replace("{", "").replace("}", "")
                content = re.sub(r'"question":|' + r'"type":', '', content)
                content = content.replace('"', '').strip()
        
        return content
    except Exception as e:
        print(f"[ASSISTANT LLM] Advice generation failed: {e}")
        difficulty_text = {
            "easy": "solid and very achievable",
            "medium": "tight but manageable with discipline",
            "hard": "challenging but not impossible"
        }
        return (
            f"Your plan looks {difficulty_text.get(budget_result['task'], 'achievable')}. "
            f"With a monthly surplus of Rs.{budget_result['monthly_surplus']:,.0f}, "
            f"stay disciplined with your daily expenses and you'll reach your goal "
            f"in {budget_result['months_needed']} months."
        )


def generate_simulation_summary(sim_result, product):
    """LLM generates a summary of the simulation run, assigning credit/blame."""
    try:
        client = _get_client()
        model = _get_model()
        
        status = "Achieved" if sim_result["success"] else "Failed"
        bad_decisions = sim_result.get("bad_decision_list", [])
        
        # Format the bad decisions into a readable list for the prompt
        if bad_decisions:
            bd_text = "\n".join([f"- {bd['action']} on {bd['necessity']} item '{bd['expense']}' (Issue: {bd['bad_decision']})" for bd in bad_decisions])
        else:
            bd_text = "None! Flawless discipline."
            
        prompt = f"""
Simulation Product: {product['name']} (Rs.{product['price']:,})
Goal Status: {status}
Final Balance: Rs.{sim_result['final_balance']:,.0f}
Simulation Days: {sim_result['steps']}
Stress Level: {sim_result['stress_level']:.0%}

Log of Bad/Risky Decisions Made:
{bd_text}

Provide the 2-3 sentence summary based on the rules.
"""
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SIMULATION_SUMMARY_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"[ASSISTANT LLM] Simulation summary generation failed: {e}")
        if sim_result["success"]:
            return "You maintained excellent financial discipline and reached your goal successfully."
        else:
            return "The simulation failed because you overspent your daily allowance or avoided essential expenses. Try tightening your budget."
