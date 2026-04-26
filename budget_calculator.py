"""
Budget Calculator — Pure Python Financial Math
===============================================
No LLM involved. All calculations are deterministic.
Parses user answers and computes budget feasibility,
timeline, and FinSense task difficulty mapping.
"""

import re


def extract_amount(answer: str) -> float:
    """Helper to robustly extract amounts including k and lakh shorthands."""
    answer_lower = str(answer).lower().strip()
    
    # Check for lakh shorthand
    l_match = re.findall(r'(\d+(?:\.\d+)?)\s*(?:lakh|lakhs|lac|lacs|l)\b', answer_lower)
    if l_match:
        return float(l_match[0]) * 100000
        
    # Check for 'k' shorthand
    k_match = re.findall(r'(\d+(?:\.\d+)?)\s*k\b', answer_lower)
    if k_match:
        return float(k_match[0]) * 1000
        
    # Extract regular numbers
    nums = re.findall(r'[\d,]+', answer_lower.replace(' ', ''))
    if nums:
        candidates = [float(n.replace(',', '')) for n in nums]
        return max(candidates)
    
    return 0.0


def parse_salary(answer: str) -> float:
    """Extract a number from user answer like 'around 80k' or '1.5 lakh'."""
    return extract_amount(answer)


def parse_savings(answer: str) -> float:
    """Extract savings from user answer like 'I have 50k' or '1 lakh'."""
    return extract_amount(answer)


def parse_months(answer: str) -> int:
    """Extract months from answer like '6 months' or '1 year and 6 months'."""
    answer_lower = str(answer).lower()
    total_months = 0
    
    # Check for years
    year_match = re.search(r'(\d+(?:\.\d+)?)\s*year', answer_lower)
    if year_match:
        total_months += int(float(year_match.group(1)) * 12)
        
    # Check for months
    month_match = re.search(r'(\d+)\s*month', answer_lower)
    if month_match:
        total_months += int(month_match.group(1))
        
    if total_months == 0:
        # Fallback if they just typed a number like "6"
        nums = re.findall(r'\d+', answer_lower)
        if nums:
            total_months = int(nums[0])
        else:
            total_months = 6
            
    # Sanity check: cap at 50 years (600 months) for this assistant
    return min(600, max(1, total_months))


def parse_emi_amount(answer: str) -> float:
    """Extract monthly EMI amount from user answer.
    
    Handles: '15000', '15,000', 'Rs.15000', 'around 15k', '1 emi of 20000'
    Returns 0.0 if no amount found.
    """
    return extract_amount(answer)


def parse_emi_months(answer: str) -> int:
    """Extract remaining EMI tenure in months.
    
    Handles: '12 months', '2 years', '1 year and 6 months'
    Returns 0 if not found.
    """
    answer_lower = str(answer).lower()
    total_months = 0
    
    # Check for years
    year_match = re.search(r'(\d+(?:\.\d+)?)\s*year', answer_lower)
    if year_match:
        total_months += int(float(year_match.group(1)) * 12)
        
    # Check for months
    month_match = re.search(r'(\d+)\s*month', answer_lower)
    if month_match:
        total_months += int(month_match.group(1))
        
    if total_months == 0:
        nums = re.findall(r'\d+', answer_lower)
        if nums:
            val = int(nums[0])
            if val > 0:
                return min(360, val)  # cap at 30 years
        return 0
        
    return min(360, total_months)


def has_incomplete_emi(answer: str) -> bool:
    """Check if user mentioned EMI/loan but didn't provide the amount.
    
    Returns True if follow-up is needed.
    """
    answer_lower = str(answer).lower().strip()
    
    # Positive indicators that EMI exists
    emi_indicators = ['emi', 'loan', 'installment', 'instalment', 'running', 'paying']
    has_emi_mention = any(w in answer_lower for w in emi_indicators)
    
    # Negative indicators (no EMI)
    no_indicators = ['no', 'none', 'nahi', 'nope', 'not', 'nil', '0']
    is_negative = any(answer_lower.startswith(w) or answer_lower == w for w in no_indicators)
    
    if not has_emi_mention or is_negative:
        return False
    
    # Check if amount was provided
    amount = parse_emi_amount(answer_lower)
    return amount == 0.0


def estimate_fixed_expenses(salary: float, has_emi: bool, emi_amount: float = 0.0) -> float:
    """Estimate fixed expenses as percentage of salary.
    
    If exact emi_amount is provided, use it. Otherwise fall back to 15% estimate.
    """
    base = salary * 0.45  # rent, groceries, utilities ~ 45%
    if has_emi:
        if emi_amount > 0:
            base += emi_amount
        else:
            base += salary * 0.15  # EMI typically 15% additional (fallback)
    return base


def calculate_budget(salary, fixed_expenses, savings, goal_amount, months):
    """
    Core budget calculation.
    
    Returns a dict with:
    - task: 'easy' | 'medium' | 'hard'
    - monthly_surplus, total_saveable, shortfall, remaining_balance
    - months_needed, interest_if_loan, monthly_emi
    - stress_projection: 'low' | 'medium' | 'high'
    """
    monthly_surplus = salary - fixed_expenses
    if monthly_surplus <= 0:
        monthly_surplus = salary * 0.1  # floor at 10%

    total_saveable = savings + (monthly_surplus * months)
    shortfall = goal_amount - total_saveable
    remaining_balance = max(0, total_saveable - goal_amount)

    # Determine task difficulty
    if shortfall <= 0:
        if remaining_balance >= salary * 0.5:
            task = "easy"
        else:
            task = "medium"
        months_needed = months
        interest = 0
    else:
        task = "hard"
        months_needed = max(1, (goal_amount - savings) / max(1, monthly_surplus))
        interest = goal_amount * 0.085 * (months_needed / 12)

    return {
        "task": task,
        "monthly_surplus": round(monthly_surplus, 0),
        "fixed_expenses": round(fixed_expenses, 0),
        "total_saveable": round(total_saveable, 0),
        "shortfall": round(max(0, shortfall), 0),
        "remaining_balance": round(remaining_balance, 0),
        "months_needed": round(months_needed, 1),
        "interest_if_loan": round(interest, 0),
        "monthly_emi": round(goal_amount / max(1, months_needed), 0) if task == "hard" else 0,
        "stress_projection": "low" if task == "easy" else "medium" if task == "medium" else "high",
    }
