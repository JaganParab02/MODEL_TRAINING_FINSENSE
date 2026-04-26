"""
Fallback Recommender — Synthetic Option Generator
=================================================
For product categories NOT in the curated catalog, generates
realistic structured purchase options using deterministic
price bands around the user's target budget.

No LLM required. All options are deterministic.
"""

from product_type_inference import infer_product_type


# ── Naming templates per broad type ──────────────────────────────────────────

NAME_TEMPLATES = {
    "vehicle": {
        "budget": "Used {item} (Basic)",
        "stretch": "Standard {item}",
        "aspirational": "Premium {item} (Ready-to-Use)",
    },
    "electronics": {
        "budget": "Budget {item}",
        "stretch": "Mid-Range {item}",
        "aspirational": "Premium {item}",
    },
    "appliance": {
        "budget": "Basic {item}",
        "stretch": "Energy-Efficient {item}",
        "aspirational": "Premium Smart {item}",
    },
    "personal_item": {
        "budget": "Standard {item}",
        "stretch": "Quality {item}",
        "aspirational": "Premium {item}",
    },
    "home_realestate": {
        "budget": "Compact {item} Option",
        "stretch": "Standard {item}",
        "aspirational": "Premium {item}",
    },
    "service_other": {
        "budget": "Basic {item} Package",
        "stretch": "Standard {item} Package",
        "aspirational": "Premium {item} Package",
    },
    "unknown": {
        "budget": "Budget Option for {item}",
        "stretch": "Standard Option for {item}",
        "aspirational": "Premium Option for {item}",
    },
}


# ── Maintenance and ownership cost profiles ──────────────────────────────────

COST_PROFILES = {
    "vehicle": {"ownership_cost_level": "high", "maintenance_level": "high"},
    "electronics": {"ownership_cost_level": "low", "maintenance_level": "low"},
    "appliance": {"ownership_cost_level": "medium", "maintenance_level": "medium"},
    "personal_item": {"ownership_cost_level": "low", "maintenance_level": "low"},
    "home_realestate": {"ownership_cost_level": "high", "maintenance_level": "medium"},
    "service_other": {"ownership_cost_level": "low", "maintenance_level": "low"},
    "unknown": {"ownership_cost_level": "medium", "maintenance_level": "medium"},
}


# ── Price band configuration per broad type ──────────────────────────────────

PRICE_BANDS = {
    "vehicle": {"budget": (0.75, 0.90), "stretch": (1.00, 1.15), "aspirational": (1.20, 1.45)},
    "electronics": {"budget": (0.80, 0.95), "stretch": (1.00, 1.15), "aspirational": (1.20, 1.40)},
    "appliance": {"budget": (0.80, 0.95), "stretch": (1.00, 1.15), "aspirational": (1.20, 1.40)},
    "personal_item": {"budget": (0.80, 0.95), "stretch": (1.00, 1.15), "aspirational": (1.20, 1.40)},
    "home_realestate": {"budget": (0.85, 0.95), "stretch": (1.00, 1.10), "aspirational": (1.15, 1.30)},
    "service_other": {"budget": (0.80, 0.95), "stretch": (1.00, 1.15), "aspirational": (1.20, 1.40)},
    "unknown": {"budget": (0.80, 0.95), "stretch": (1.00, 1.15), "aspirational": (1.20, 1.40)},
}


# ── Summary templates ────────────────────────────────────────────────────────

SUMMARY_TEMPLATES = {
    "budget": "A cost-effective option within your comfortable savings range. Lower upfront cost means less financial stress.",
    "stretch": "Matches your target budget closely. Requires disciplined saving but is achievable within your timeline.",
    "aspirational": "Above your current target. May require extending your timeline or increasing monthly savings.",
}


def generate_fallback_options(product_name: str, target_budget: float,
                               broad_type: str = None) -> list:
    """
    Generate 3 synthetic product options (budget, stretch, aspirational)
    for an unsupported product category.
    
    Args:
        product_name: The user's product name (e.g., "bus", "tractor")
        target_budget: The user's target price
        broad_type: Pre-inferred broad type (optional)
    
    Returns:
        List of 3 option dicts with is_synthetic=True
    """
    if not broad_type:
        inference = infer_product_type(product_name)
        broad_type = inference["broad_type"]
    
    if broad_type == "unknown":
        # Still generate options with generic templates
        pass
    
    templates = NAME_TEMPLATES.get(broad_type, NAME_TEMPLATES["unknown"])
    bands = PRICE_BANDS.get(broad_type, PRICE_BANDS["unknown"])
    costs = COST_PROFILES.get(broad_type, COST_PROFILES["unknown"])
    
    # Capitalize item name for display
    item_display = product_name.strip().title()
    
    options = []
    for tier in ["budget", "stretch", "aspirational"]:
        low_mult, high_mult = bands[tier]
        # Use midpoint of band
        price = round(target_budget * ((low_mult + high_mult) / 2), -2)  # round to nearest 100
        price = max(100, price)  # floor
        
        name = templates[tier].format(item=item_display)
        
        # Adjust cost levels for budget tier
        ownership = costs["ownership_cost_level"]
        maintenance = costs["maintenance_level"]
        if tier == "budget":
            # Budget options tend to have higher maintenance
            if maintenance == "low":
                maintenance = "medium"
        
        options.append({
            "name": name,
            "category": product_name.lower(),
            "price": int(price),
            "tier": tier,
            "ownership_cost_level": ownership,
            "maintenance_level": maintenance,
            "summary": SUMMARY_TEMPLATES[tier],
            "is_synthetic": True,
        })
    
    return options


def generate_fallback_recommendation(product_name: str, saveable_budget: float,
                                      target_budget: float, broad_type: str = None) -> dict:
    """
    Full recommendation output for an unsupported category.
    Separates affordability from catalog availability.
    
    Returns dict matching the structure of product_recommender.recommend_products()
    with additional fields.
    """
    options = generate_fallback_options(product_name, target_budget, broad_type)
    
    within_budget = [o for o in options if o["price"] <= saveable_budget]
    stretch = [o for o in options if saveable_budget < o["price"] <= saveable_budget * 1.25]
    above_budget = [o for o in options if o["price"] > saveable_budget * 1.25]
    
    return {
        "category": product_name.lower(),
        "source": "synthetic_fallback",
        "within_budget": within_budget,
        "stretch": stretch,
        "above_budget": above_budget,
        "best_value": within_budget[0] if within_budget else (stretch[0] if stretch else None),
        "all_options": options,
        "affordable": saveable_budget >= target_budget * 0.8,
        "affordability_gap": max(0, target_budget - saveable_budget),
    }
