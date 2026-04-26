"""
Product Recommender — Indian Product Catalog
=============================================
Categorizes products and recommends options within, at,
and above the user's budget. No LLM needed.
"""


PRODUCT_CATALOG = {
    "car": [
        {"name": "Maruti Alto K10", "price": 374000, "type": "hatchback", "fuel": "petrol"},
        {"name": "Maruti Swift", "price": 650000, "type": "hatchback", "fuel": "petrol"},
        {"name": "Hyundai i20", "price": 750000, "type": "hatchback", "fuel": "petrol"},
        {"name": "Tata Punch", "price": 600000, "type": "micro SUV", "fuel": "petrol/EV"},
        {"name": "Tata Nexon", "price": 1000000, "type": "SUV", "fuel": "petrol/EV"},
        {"name": "Hyundai Venue", "price": 850000, "type": "SUV", "fuel": "petrol"},
        {"name": "Hyundai Creta", "price": 1500000, "type": "SUV", "fuel": "petrol/diesel"},
        {"name": "Tata Harrier", "price": 1600000, "type": "SUV", "fuel": "diesel"},
        {"name": "MG Hector", "price": 1700000, "type": "SUV", "fuel": "petrol/hybrid"},
        {"name": "Toyota Fortuner", "price": 3500000, "type": "premium SUV", "fuel": "diesel"},
    ],
    "phone": [
        {"name": "Redmi 13C", "price": 9000, "type": "budget"},
        {"name": "Realme Narzo 60", "price": 15000, "type": "mid-range"},
        {"name": "OnePlus Nord CE4", "price": 25000, "type": "mid-range"},
        {"name": "Samsung Galaxy S23 FE", "price": 45000, "type": "premium"},
        {"name": "iPhone 15", "price": 79900, "type": "flagship"},
        {"name": "Samsung Galaxy S24", "price": 89999, "type": "flagship"},
    ],
    "bike": [
        {"name": "Honda Activa 6G", "price": 74000, "type": "scooter", "fuel": "petrol"},
        {"name": "TVS Jupiter", "price": 80000, "type": "scooter", "fuel": "petrol"},
        {"name": "Bajaj Pulsar 150", "price": 120000, "type": "commuter", "fuel": "petrol"},
        {"name": "Royal Enfield Classic 350", "price": 195000, "type": "cruiser", "fuel": "petrol"},
        {"name": "Ola S1 Pro", "price": 150000, "type": "scooter", "fuel": "electric"},
    ],
    "scooter": [
        {"name": "Honda Activa 6G", "price": 74000, "type": "scooter", "fuel": "petrol"},
        {"name": "TVS Jupiter", "price": 80000, "type": "scooter", "fuel": "petrol"},
        {"name": "Ola S1 Pro", "price": 150000, "type": "scooter", "fuel": "electric"},
        {"name": "Ather 450X", "price": 140000, "type": "scooter", "fuel": "electric"},
    ],
    "laptop": [
        {"name": "Acer Aspire 3", "price": 30000, "type": "budget"},
        {"name": "HP Pavilion 14", "price": 55000, "type": "mid-range"},
        {"name": "Lenovo IdeaPad Slim 5", "price": 65000, "type": "mid-range"},
        {"name": "ASUS Vivobook Pro 15", "price": 85000, "type": "performance"},
        {"name": "MacBook Air M2", "price": 99900, "type": "premium"},
        {"name": "MacBook Pro M3", "price": 169900, "type": "flagship"},
    ],
}


def get_product_category(product: str) -> str:
    """Match user input to a product category key."""
    product_lower = product.lower()
    for key in PRODUCT_CATALOG:
        if key in product_lower:
            return key
    return None


def recommend_products(product: str, budget: float, target_budget: float = None,
                       preferences: dict = None):
    """
    Return products segmented into budget tiers.
    
    For known categories: uses curated catalog.
    For unknown categories: uses deterministic fallback generator.
    
    Args:
        product: Product name from user
        budget: Total saveable amount (what user can afford)
        target_budget: User's original target price (for fallback generation)
        preferences: Optional preference dict
    
    Returns:
        dict with keys: category, source, within_budget, stretch, above_budget, best_value,
                        affordable, affordability_gap
    """
    from product_type_inference import infer_product_type
    from fallback_recommender import generate_fallback_recommendation

    category = get_product_category(product)
    
    if not category:
        # Unknown category — use fallback recommender
        inference = infer_product_type(product)
        target = target_budget if target_budget else budget
        result = generate_fallback_recommendation(
            product, budget, target, broad_type=inference["broad_type"]
        )
        result["product_type"] = inference
        return result

    # Known category — curated catalog
    catalog = sorted(PRODUCT_CATALOG[category], key=lambda p: p["price"])
    within_budget = [p for p in catalog if p["price"] <= budget]
    stretch = [p for p in catalog if budget < p["price"] <= budget * 1.25]
    above = [p for p in catalog if p["price"] > budget * 1.25]

    target = target_budget if target_budget else budget
    
    return {
        "category": category,
        "source": "curated_catalog",
        "within_budget": within_budget[-4:],   # top 4 affordable
        "stretch": stretch[:2],                 # 2 stretch options
        "above_budget": above[:1],              # 1 aspirational
        "best_value": within_budget[-1] if within_budget else None,
        "affordable": budget >= target * 0.8,
        "affordability_gap": max(0, target - budget),
    }
