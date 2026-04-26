import random
from typing import Dict, Any, List
from finsense.models import Expense


class EventAgent:
    """Triggers macro events that affect the financial world."""

    def __init__(self, seed: int = 42, total_days: int = 15):
        self.rng = random.Random(seed)
        self.active_events: List[Dict[str, Any]] = []
        self.total_days = total_days
        self.event_log: List[Dict[str, Any]] = []  # Full log for demonstration

    def trigger_events(self, day: int) -> List[Dict[str, Any]]:
        """Check for new events to trigger on this day.
        
        Events use percentage-based windows so they work across all task durations.
        Events can recur throughout the full episode.
        """
        new_events = []

        # Calculate percentage through episode
        pct = day / max(1, self.total_days)

        # Fuel crisis: can occur in first half and last quarter
        if (0.2 <= pct <= 0.55) or (0.75 <= pct <= 0.9):
            if self.rng.random() < 0.15:
                intensity = self.rng.uniform(1.2, 1.8)
                event = {
                    "type": "fuel_crisis",
                    "intensity": intensity,
                    "categories": ["transport", "food"],
                    "duration": max(2, int(self.total_days * 0.1)),
                    "day_triggered": day,
                }
                new_events.append(event)

        # Inflation: can occur mid-episode
        if (0.3 <= pct <= 0.7):
            if self.rng.random() < 0.12:
                intensity = self.rng.uniform(1.1, 1.5)
                event = {
                    "type": "inflation",
                    "intensity": intensity,
                    "categories": ["food", "utility"],
                    "duration": max(3, int(self.total_days * 0.12)),
                    "day_triggered": day,
                }
                new_events.append(event)

        # Medical surge: can occur throughout
        if self.rng.random() < 0.08:
            intensity = self.rng.uniform(1.3, 2.0)
            event = {
                "type": "medical_surge",
                "intensity": intensity,
                "categories": ["medical"],
                "duration": max(2, int(self.total_days * 0.08)),
                "day_triggered": day,
            }
            new_events.append(event)

        # Festival season: spending pressure in second half
        if (0.5 <= pct <= 0.8):
            if self.rng.random() < 0.10:
                intensity = self.rng.uniform(1.15, 1.4)
                event = {
                    "type": "festival_season",
                    "intensity": intensity,
                    "categories": ["food", "entertainment"],
                    "duration": max(3, int(self.total_days * 0.1)),
                    "day_triggered": day,
                }
                new_events.append(event)

        # Tax season: expected but large expense impact
        if (0.3 <= pct <= 0.45):
            if self.rng.random() < 0.08:
                intensity = self.rng.uniform(1.2, 1.4)
                event = {
                    "type": "tax_season",
                    "intensity": intensity,
                    "categories": ["utility", "rent"],
                    "duration": max(2, int(self.total_days * 0.05)),
                    "day_triggered": day,
                }
                new_events.append(event)
                
        # Unexpected windfall (positive event: intensity < 1.0 reduces prices)
        if (0.1 <= pct <= 0.9):
            if self.rng.random() < 0.05:
                intensity = self.rng.uniform(0.7, 0.9)
                event = {
                    "type": "unexpected_windfall",
                    "intensity": intensity,
                    "categories": ["entertainment", "food", "household_repairs"],
                    "duration": max(2, int(self.total_days * 0.08)),
                    "day_triggered": day,
                }
                new_events.append(event)

        # Step 1: Decrement existing events FIRST (before adding new ones)
        for event in self.active_events:
            event["remaining_days"] -= 1

        # Step 2: Remove expired events (remaining_days < 0 means fully used)
        self.active_events = [e for e in self.active_events if e["remaining_days"] >= 0]

        # Step 3: Add new events (they start with full remaining_days, NOT decremented this turn)
        for event in new_events:
            event["remaining_days"] = event["duration"]
            self.active_events.append(event)
            self.event_log.append(event.copy())

        return new_events

    def get_active_events(self) -> List[Dict[str, Any]]:
        """Return currently active events."""
        return [e for e in self.active_events if e["remaining_days"] >= 0]

    def get_event_log(self) -> List[Dict[str, Any]]:
        """Return full event history for this episode."""
        return self.event_log.copy()


class VendorAgent:
    """Adjusts expense amounts based on active events."""

    def __init__(self):
        self.event_multipliers: Dict[str, float] = {}
        self.base_multipliers: Dict[str, float] = {}
        self.seen_events: set = set()

    def update_prices(self, expenses: List[Expense], active_events: List[Dict[str, Any]]) -> List[Expense]:
        """Adjust expense amounts based on active events."""
        # Check for new repeated shocks
        for event in active_events:
            event_id = f"{event['type']}_{event['day_triggered']}"
            if event_id not in self.seen_events:
                self.seen_events.add(event_id)
                # Count how many times we've seen this event type
                type_count = sum(1 for e in self.seen_events if e.startswith(event['type']))
                if type_count > 1:
                    # Permanent 5% increase for affected categories
                    for category in event["categories"]:
                        if category not in self.base_multipliers:
                            self.base_multipliers[category] = 1.0
                        self.base_multipliers[category] *= 1.05

        # Reset temporary multipliers
        self.event_multipliers = {}

        # Calculate multipliers from active events
        for event in active_events:
            for category in event["categories"]:
                if category not in self.event_multipliers:
                    self.event_multipliers[category] = 1.0
                self.event_multipliers[category] *= event["intensity"]

        # Apply multipliers to expenses
        adjusted_expenses = []
        for exp in expenses:
            temp_mult = self.event_multipliers.get(exp.category, 1.0)
            base_mult = self.base_multipliers.get(exp.category, 1.0)
            final_mult = temp_mult * base_mult
            adjusted_amount = round(exp.amount * final_mult, 2)
            adjusted_exp = exp.model_copy()
            adjusted_exp.amount = adjusted_amount
            adjusted_expenses.append(adjusted_exp)

        return adjusted_expenses

    def get_event_multipliers(self) -> Dict[str, float]:
        """Return current effective price multipliers by category."""
        effective = {}
        all_categories = set(self.event_multipliers.keys()).union(set(self.base_multipliers.keys()))
        for cat in all_categories:
            temp_mult = self.event_multipliers.get(cat, 1.0)
            base_mult = self.base_multipliers.get(cat, 1.0)
            effective[cat] = round(temp_mult * base_mult, 3)
        return effective