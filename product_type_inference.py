"""
Product Type Inference — Deterministic + Fallback
==================================================
Maps user purchase requests into broad product types.
Uses deterministic keyword mapping for common terms.
Returns a structured inference object.
"""


# Broad types and their keyword mappings
KEYWORD_MAP = {
    # Vehicle
    "car": {"broad_type": "vehicle", "subtype": "personal"},
    "sedan": {"broad_type": "vehicle", "subtype": "personal"},
    "suv": {"broad_type": "vehicle", "subtype": "personal"},
    "hatchback": {"broad_type": "vehicle", "subtype": "personal"},
    "bike": {"broad_type": "vehicle", "subtype": "personal"},
    "motorcycle": {"broad_type": "vehicle", "subtype": "personal"},
    "scooter": {"broad_type": "vehicle", "subtype": "personal"},
    "scooty": {"broad_type": "vehicle", "subtype": "personal"},
    "bus": {"broad_type": "vehicle", "subtype": "business_asset"},
    "truck": {"broad_type": "vehicle", "subtype": "business_asset"},
    "tractor": {"broad_type": "vehicle", "subtype": "business_asset"},
    "auto": {"broad_type": "vehicle", "subtype": "business_asset"},
    "rickshaw": {"broad_type": "vehicle", "subtype": "business_asset"},
    "van": {"broad_type": "vehicle", "subtype": "business_asset"},
    "tempo": {"broad_type": "vehicle", "subtype": "business_asset"},

    # Electronics
    "phone": {"broad_type": "electronics", "subtype": "personal"},
    "mobile": {"broad_type": "electronics", "subtype": "personal"},
    "smartphone": {"broad_type": "electronics", "subtype": "personal"},
    "iphone": {"broad_type": "electronics", "subtype": "personal"},
    "laptop": {"broad_type": "electronics", "subtype": "personal"},
    "computer": {"broad_type": "electronics", "subtype": "personal"},
    "pc": {"broad_type": "electronics", "subtype": "personal"},
    "desktop": {"broad_type": "electronics", "subtype": "personal"},
    "tablet": {"broad_type": "electronics", "subtype": "personal"},
    "ipad": {"broad_type": "electronics", "subtype": "personal"},
    "camera": {"broad_type": "electronics", "subtype": "personal"},
    "tv": {"broad_type": "electronics", "subtype": "appliance"},
    "television": {"broad_type": "electronics", "subtype": "appliance"},
    "monitor": {"broad_type": "electronics", "subtype": "personal"},
    "smartwatch": {"broad_type": "electronics", "subtype": "personal"},
    "earphones": {"broad_type": "electronics", "subtype": "personal"},
    "headphones": {"broad_type": "electronics", "subtype": "personal"},
    "speaker": {"broad_type": "electronics", "subtype": "personal"},
    "console": {"broad_type": "electronics", "subtype": "personal"},
    "playstation": {"broad_type": "electronics", "subtype": "personal"},
    "xbox": {"broad_type": "electronics", "subtype": "personal"},
    "drone": {"broad_type": "electronics", "subtype": "personal"},

    # Appliance
    "fridge": {"broad_type": "appliance", "subtype": "home"},
    "refrigerator": {"broad_type": "appliance", "subtype": "home"},
    "washing": {"broad_type": "appliance", "subtype": "home"},
    "washer": {"broad_type": "appliance", "subtype": "home"},
    "dryer": {"broad_type": "appliance", "subtype": "home"},
    "ac": {"broad_type": "appliance", "subtype": "home"},
    "air conditioner": {"broad_type": "appliance", "subtype": "home"},
    "microwave": {"broad_type": "appliance", "subtype": "home"},
    "oven": {"broad_type": "appliance", "subtype": "home"},
    "dishwasher": {"broad_type": "appliance", "subtype": "home"},
    "geyser": {"broad_type": "appliance", "subtype": "home"},
    "water purifier": {"broad_type": "appliance", "subtype": "home"},
    "inverter": {"broad_type": "appliance", "subtype": "home"},
    "mixer": {"broad_type": "appliance", "subtype": "home"},
    "grinder": {"broad_type": "appliance", "subtype": "home"},
    "vacuum": {"broad_type": "appliance", "subtype": "home"},
    "fan": {"broad_type": "appliance", "subtype": "home"},
    "cooler": {"broad_type": "appliance", "subtype": "home"},

    # Home / Real Estate
    "house": {"broad_type": "home_realestate", "subtype": "purchase"},
    "flat": {"broad_type": "home_realestate", "subtype": "purchase"},
    "apartment": {"broad_type": "home_realestate", "subtype": "purchase"},
    "plot": {"broad_type": "home_realestate", "subtype": "land"},
    "land": {"broad_type": "home_realestate", "subtype": "land"},
    "property": {"broad_type": "home_realestate", "subtype": "purchase"},
    "renovation": {"broad_type": "home_realestate", "subtype": "improvement"},
    "construction": {"broad_type": "home_realestate", "subtype": "improvement"},

    # Personal Items
    "jewelry": {"broad_type": "personal_item", "subtype": "luxury"},
    "jewellery": {"broad_type": "personal_item", "subtype": "luxury"},
    "gold": {"broad_type": "personal_item", "subtype": "luxury"},
    "ring": {"broad_type": "personal_item", "subtype": "luxury"},
    "watch": {"broad_type": "personal_item", "subtype": "luxury"},
    "furniture": {"broad_type": "personal_item", "subtype": "home"},
    "sofa": {"broad_type": "personal_item", "subtype": "home"},
    "bed": {"broad_type": "personal_item", "subtype": "home"},
    "mattress": {"broad_type": "personal_item", "subtype": "home"},
    "clothes": {"broad_type": "personal_item", "subtype": "fashion"},
    "shoes": {"broad_type": "personal_item", "subtype": "fashion"},

    # Services
    "wedding": {"broad_type": "service_other", "subtype": "event"},
    "travel": {"broad_type": "service_other", "subtype": "experience"},
    "vacation": {"broad_type": "service_other", "subtype": "experience"},
    "holiday": {"broad_type": "service_other", "subtype": "experience"},
    "education": {"broad_type": "service_other", "subtype": "investment"},
    "course": {"broad_type": "service_other", "subtype": "investment"},
    "coaching": {"broad_type": "service_other", "subtype": "investment"},
    "tuition": {"broad_type": "service_other", "subtype": "investment"},
    "gym": {"broad_type": "service_other", "subtype": "subscription"},
    "insurance": {"broad_type": "service_other", "subtype": "financial"},
}


def infer_product_type(product_name: str) -> dict:
    """
    Infer the broad product type from a user's purchase request.
    
    Returns:
        {
            "item_name": str,
            "broad_type": str,
            "subtype": str,
            "confidence": float (0.0 to 1.0)
        }
    """
    name_lower = product_name.lower().strip()
    
    # 1. Exact match
    if name_lower in KEYWORD_MAP:
        info = KEYWORD_MAP[name_lower]
        return {
            "item_name": product_name,
            "broad_type": info["broad_type"],
            "subtype": info["subtype"],
            "confidence": 1.0
        }
    
    # 2. Substring match (e.g., "washing machine" matches "washing")
    best_match = None
    best_len = 0
    for keyword, info in KEYWORD_MAP.items():
        if keyword in name_lower and len(keyword) > best_len:
            best_match = info
            best_len = len(keyword)
    
    if best_match:
        return {
            "item_name": product_name,
            "broad_type": best_match["broad_type"],
            "subtype": best_match["subtype"],
            "confidence": 0.85
        }
    
    # 3. No match — return unknown with low confidence
    return {
        "item_name": product_name,
        "broad_type": "unknown",
        "subtype": "unknown",
        "confidence": 0.0
    }


# Convenience
BROAD_TYPES = [
    "vehicle", "electronics", "appliance", "personal_item",
    "business_asset", "home_realestate", "service_other", "unknown"
]
