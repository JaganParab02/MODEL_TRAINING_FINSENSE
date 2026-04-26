"""
Dynamic Questions — Product-Type-Aware Question Engine
=====================================================
Generates context-aware financial questions based on the
inferred broad product type. Keeps existing logic for known
catalog categories; extends with type-specific questions
for unknown categories.
"""

from product_type_inference import infer_product_type


# ── Core required questions (always asked) ───────────────────────────────────

CORE_QUESTIONS = [
    {"question": "What is your approximate monthly take-home salary?", "type": "salary"},
    {"question": "What is your preferred timeline for this purchase (in months)?", "type": "timeline"},
]


# ── Type-specific question banks ─────────────────────────────────────────────

TYPE_QUESTIONS = {
    "vehicle": [
        {"question": "Is this for personal use or commercial/business use?", "type": "other"},
        {"question": "Are you considering a new purchase or a pre-owned/used option?", "type": "other"},
        {"question": "Do you have any existing EMIs or loans? If yes, what is the monthly EMI amount?", "type": "emi"},
        {"question": "Have you set aside a down payment, and if so, how much?", "type": "other"},
    ],
    "electronics": [
        {"question": "How urgent is this purchase — can you wait for sales/discounts?", "type": "other"},
        {"question": "Do you have any existing EMIs or loans? If yes, what is the monthly EMI amount?", "type": "emi"},
        {"question": "Do you prefer a premium brand or are you open to budget-friendly alternatives?", "type": "other"},
    ],
    "appliance": [
        {"question": "Is this a replacement for a broken appliance or a new addition?", "type": "other"},
        {"question": "Do you have any existing EMIs or loans? If yes, what is the monthly EMI amount?", "type": "emi"},
        {"question": "Do you prefer energy-efficient/premium models or basic functional ones?", "type": "other"},
    ],
    "personal_item": [
        {"question": "Is this for a special occasion (e.g., wedding, festival) or general use?", "type": "other"},
        {"question": "Do you have any existing EMIs or loans? If yes, what is the monthly EMI amount?", "type": "emi"},
        {"question": "Do you have any existing savings allocated for this?", "type": "other"},
    ],
    "home_realestate": [
        {"question": "Are you planning to take a home loan, or will this be a full cash purchase?", "type": "other"},
        {"question": "Do you have any existing EMIs or loans? If yes, what is the monthly EMI amount?", "type": "emi"},
        {"question": "What is the urgency — is there a deadline (e.g., family need, registration)?", "type": "other"},
        {"question": "How much down payment have you saved so far?", "type": "other"},
    ],
    "service_other": [
        {"question": "Is this a one-time expense or a recurring commitment?", "type": "other"},
        {"question": "Do you have any existing EMIs or loans? If yes, what is the monthly EMI amount?", "type": "emi"},
        {"question": "Have you already started saving for this, and if so, how much?", "type": "other"},
    ],
    "unknown": [
        {"question": "Can you describe what kind of purchase this is (e.g., asset, service, luxury)?", "type": "other"},
        {"question": "Do you have any existing EMIs or loans? If yes, what is the monthly EMI amount?", "type": "emi"},
        {"question": "Is this essential or something you can delay if needed?", "type": "other"},
    ],
}


def get_dynamic_questions(product_name: str, amount: float, broad_type: str = None) -> list:
    """
    Generate product-type-aware questions.
    
    Args:
        product_name: User's product name (e.g., "bus", "fridge")
        amount: Target budget amount
        broad_type: Pre-inferred broad type (optional; will be inferred if not provided)
    
    Returns:
        List of question dicts with "question" and "type" keys.
    """
    if not broad_type:
        inference = infer_product_type(product_name)
        broad_type = inference["broad_type"]
    
    # Start with core questions
    questions = list(CORE_QUESTIONS)
    
    # Add type-specific questions
    type_qs = TYPE_QUESTIONS.get(broad_type, TYPE_QUESTIONS["unknown"])
    questions.extend(type_qs)
    
    # Ensure we have salary, timeline, and emi types
    types_present = {q["type"] for q in questions}
    if "emi" not in types_present:
        questions.append({
            "question": "Do you have any existing EMIs or loans? If yes, what is the monthly amount?",
            "type": "emi"
        })
    
    # Cap at 5 questions
    return questions[:5]
