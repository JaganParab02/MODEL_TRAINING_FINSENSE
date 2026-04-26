import logging
from typing import Dict, Any, Tuple
from finsense.models import ActionModel, Expense
from finsense.expense_generator import ExpenseGenerator
from finsense.tasks import TASKS
from finsense.agents import EventAgent, VendorAgent
from finsense.memory import MemorySystem

logger = logging.getLogger(__name__)


class FinSenseEnv:
    def __init__(self):
        self.state: Dict[str, Any] = {}
        self.expense_gen = None
        self.daily_expenses = []
        self.expense_idx = 0
        self.daily_spend_total = 0.0
        self.daily_allowance = 0.0

        # Tracks future penalties caused by avoided medical essentials
        self.pending_consequences = []
        self.task_config = None

        # Episode memory buffer for delayed commits
        self.episode_memory_buffer = []

        # Multi-agent world layer
        self.event_agent = EventAgent()
        self.vendor_agent = VendorAgent()
        self.current_day = 0

        # Memory system
        self.memory = MemorySystem()
        self.episode_num = 0

        # Step-level tracking for bad decision analysis
        self.step_log = []
        self.step_count = 0
        self.max_steps_per_episode = 200

        self.process_metrics = {
            "protected_essential_count": 0,
            "discretionary_avoids": 0,
            "overspend_prevented": 0,
            "emergency_safe_actions": 0,
            "invalid_action_fallbacks": 0
        }

    def reset(self, task_id: str = "easy", seed: int = 42) -> Dict[str, Any]:
        if task_id not in TASKS:
            task_id = "easy"

        task = TASKS[task_id]
        self.expense_gen = ExpenseGenerator(seed)
        self.task_config = task

        self.state = {
            "balance": task.initial_balance,
            "goal_total": task.goal,
            "goal_remaining": task.goal,
            "days_left": task.days,
            "stress_level": 0.0,
            "risk_level": "low",
            "expected_fixed_expenses": task.expected_fixed_expenses,
            "recent_spending": [],
            "avg_daily_spend": 0.0,
            "salary": task.initial_balance,
            "task_id": task_id,
            "income_shock_active": False,
        }

        self.daily_expenses = self.expense_gen.generate_daily_expenses(num_expenses=3)
        self.expense_idx = 0
        self.daily_spend_total = 0.0
        self.pending_consequences = []
        self.step_log = []
        self.step_count = 0
        self.process_metrics = {
            "protected_essential_count": 0,
            "discretionary_avoids": 0,
            "overspend_prevented": 0,
            "emergency_safe_actions": 0,
            "invalid_action_fallbacks": 0
        }
        self.step_count = 0
        self.episode_memory_buffer = []

        # Reset agents with total_days for percentage-based event windows
        self.event_agent = EventAgent(seed, total_days=task.days)
        self.vendor_agent = VendorAgent()
        self.current_day = 0
        self.episode_num += 1

        self._recompute_daily_allowance()

        print(f"\n{'='*60}")
        print(f"  [RESET] task={task_id} | balance={self.state['balance']:.0f} | "
              f"goal={task.goal:.0f} | days={task.days}")
        print(f"  [INFO]  daily_allowance={self.daily_allowance:.0f} | "
              f"required_savings/day={task.goal / task.days:.0f}")
        print(f"{'='*60}")

        return self._get_observation()

    def _recompute_daily_allowance(self):
        """
        Daily allowance = how much the agent can spend TODAY and still hit the goal.
        """
        days_left = max(1, self.state["days_left"])
        balance = self.state["balance"]
        goal_remaining = self.state["goal_remaining"]

        spendable_total = max(0.0, balance - goal_remaining)
        self.daily_allowance = round(spendable_total / days_left, 2)

    def _get_observation(self) -> Dict[str, Any]:
        obs = dict(self.state)
        obs["recent_spending"] = obs["recent_spending"][-5:]
        days_left = max(1, self.state["days_left"])
        obs["required_savings_per_day"] = round(self.state["goal_remaining"] / days_left, 2)
        obs["goal_total"] = self.state["goal_total"]
        obs["daily_allowance"] = round(self.daily_allowance, 2)

        # Add active events for memory system
        active_events = self.event_agent.get_active_events()
        obs["active_events"] = [e["type"] for e in active_events]

        if self.expense_idx < len(self.daily_expenses):
            exp = self.daily_expenses[self.expense_idx]
            obs["current_expense"] = {
                "name": exp.name,
                "amount": exp.amount,
                "necessity_tag": exp.necessity_tag,
                "category": exp.category,
                "context": exp.context,
            }
        else:
            obs["current_expense"] = None

        return obs

    def step(self, action: ActionModel) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        self.step_count += 1
        if self.step_count >= self.max_steps_per_episode:
            return self._get_observation(), -50.0, True, {"error": "max_steps_exceeded"}

        if self.state["days_left"] <= 0:
            return self._get_observation(), 0.0, True, {"error": "Environment is already done."}

        balance = self.state["balance"]
        spend = 0.0
        necessity = "none"
        context = "normal"

        # Anti-Reward-Hacking Safeguards
        decision = action.decision
        approved_amount = max(0.0, action.approved_amount)  # bounds on approved_amount
        
        if decision not in ["allow", "reduce", "avoid"]:
            decision = "avoid"  # invalid action fallback
            self.process_metrics["invalid_action_fallbacks"] += 1

        # Capture current expense BEFORE any day-end processing
        current_expense_snapshot = None

        if self.expense_idx < len(self.daily_expenses):
            current_expense = self.daily_expenses[self.expense_idx]
            current_expense_snapshot = current_expense.model_copy()  # Snapshot for memory
            necessity = current_expense.necessity_tag
            context = current_expense.context

            if decision == "allow":
                spend = current_expense.amount
            elif decision == "reduce":
                spend = min(approved_amount, current_expense.amount)
            elif decision == "avoid":
                spend = 0.0

            # Never spend more than available balance
            spend = min(spend, max(0.0, balance))

            self.state["balance"] -= spend
            self.daily_spend_total += spend

            # Stress update & Metrics tracking
            if necessity == "essential":
                # Stress increases proportionally to how much of the essential expense was NOT paid
                unpaid_ratio = max(0.0, 1.0 - (spend / current_expense.amount))
                self.state["stress_level"] += 0.3 * unpaid_ratio
                if unpaid_ratio < 0.5:
                    self.process_metrics["protected_essential_count"] += 1
            elif necessity == "semi-essential":
                unpaid_ratio = max(0.0, 1.0 - (spend / current_expense.amount))
                self.state["stress_level"] += 0.1 * unpaid_ratio
            elif necessity == "discretionary":
                if decision == "avoid":
                    self.process_metrics["discretionary_avoids"] += 1
                    
            if decision in ["reduce", "avoid"] and current_expense.amount > self.state.get("daily_allowance", 0):
                self.process_metrics["overspend_prevented"] += 1
                
            if context == "emergency" and decision in ["allow", "reduce"]:
                self.process_metrics["emergency_safe_actions"] += 1

            # =============================================================
            # CONTEXT-BASED PENALTIES (scaled to be meaningful)
            # =============================================================
            context_penalty = 0.0

            # Emergency context: avoiding is dangerous — significant penalty
            if context == "emergency" and decision == "avoid":
                context_penalty = -10.0  # Scaled to matter in [-100, 100] range

            # Weekend context: allowing DISCRETIONARY spending is wasteful
            # Only penalize discretionary, not essential/semi-essential
            if context == "weekend" and decision == "allow" and necessity == "discretionary":
                context_penalty = -5.0  # Moderate penalty for weekend lifestyle spending

            # =============================================================
            # DELAYED CONSEQUENCE: Avoiding essential medical care
            # =============================================================
            if (decision == "avoid"
                    and necessity == "essential"
                    and current_expense.category == "medical"):
                if self.expense_gen.rng.random() < 0.5:
                    self.pending_consequences.append({
                        "days_until_trigger": 3,
                        "expense": Expense(
                            name="Delayed Health Emergency",
                            category="medical",
                            amount=3000.0,
                            necessity_tag="essential",
                            context="emergency",
                        )
                    })
                    logger.info("[CONSEQUENCE] Medical essential avoided — "
                                "health emergency scheduled in 3 days")

            # Track bad decisions for evaluation
            # Only flag genuinely harmful decisions (not just technical threshold violations)
            bad_decision = None
            if decision == "allow" and necessity == "discretionary":
                # Flag only if spend is both non-trivial (>Rs.500) AND exceeds daily allowance
                if spend > 500 and spend > self.daily_allowance:
                    bad_decision = "overspent_allowance_discretionary"
            elif decision == "avoid" and necessity == "essential":
                bad_decision = "avoided_essential"
            elif decision == "avoid" and context == "emergency" and necessity in ("essential", "semi-essential"):
                bad_decision = "avoided_emergency"

            self.step_log.append({
                "day": self.current_day,
                "expense": current_expense.name,
                "category": current_expense.category,
                "necessity": necessity,
                "context": context,
                "action": decision,
                "amount": spend,
                "bad_decision": bad_decision,
            })

        self.state["stress_level"] = min(1.0, self.state["stress_level"])

        # Track spending history
        recent = self.state["recent_spending"]
        recent.append(spend)
        self.state["recent_spending"] = recent[-5:]
        self.state["avg_daily_spend"] = sum(self.state["recent_spending"]) / max(1, len(self.state["recent_spending"]))

        # Risk level
        bal = self.state["balance"]
        fixed = self.state["expected_fixed_expenses"]
        if bal < fixed * 0.2:
            self.state["risk_level"] = "high"
        elif bal < fixed * 0.5:
            self.state["risk_level"] = "medium"
        else:
            self.state["risk_level"] = "low"

        # Per-step reward: penalize stress and risk
        stress_penalty = 0.4 * self.state["stress_level"]
        risk_penalty = {"low": 0.0, "medium": 0.1, "high": 0.3}[self.state["risk_level"]]
        
        # Independent reward function: Overspend Penalty
        # Protects against reward hacking where model spends everything early
        overspend_penalty = 0.0
        if spend > self.daily_allowance:
            overspend_penalty = (spend - self.daily_allowance) * 0.05
            
        reward = -stress_penalty - risk_penalty + context_penalty - overspend_penalty

        # Print expense line
        if self.expense_idx < len(self.daily_expenses):
            exp = self.daily_expenses[self.expense_idx]
            ctx_str = f" [{context}]" if context != "normal" else ""
            print(f"    [EXPENSE] {exp.name:20s} | {decision:6s} | "
                  f"spend={spend:>7.0f} | bal={self.state['balance']:>8.0f}{ctx_str}")

        self.expense_idx += 1

        if self.expense_idx >= len(self.daily_expenses):
            # === END OF DAY ===

            actual_savings = max(0.0, self.daily_allowance - self.daily_spend_total)

            previous_goal = self.state["goal_remaining"]
            self.state["goal_remaining"] = max(0.0, self.state["goal_remaining"] - actual_savings)
            goal_progress = previous_goal - self.state["goal_remaining"]

            # Reward for goal progress (scaled)
            reward += goal_progress * 0.01

            # Removed the duplicate reward += 50.0 here since it's handled in the `if done:` block below.
            print(f"  [DAY END] day {self.state['days_left']:>2d} | "
                  f"spent={self.daily_spend_total:>7.0f} | allowance={self.daily_allowance:>7.0f} | "
                  f"saved={actual_savings:>7.0f} | goal_left={self.state['goal_remaining']:>8.0f} | "
                  f"bal={self.state['balance']:>8.0f}")

            # Process pending consequences: countdown and trigger
            for event in self.pending_consequences:
                event["days_until_trigger"] -= 1

            ready_events = [
                e for e in self.pending_consequences
                if e["days_until_trigger"] <= 0
            ]

            self.pending_consequences = [
                e for e in self.pending_consequences
                if e["days_until_trigger"] > 0
            ]

            for event in ready_events:
                logger.info(f"[CONSEQUENCE TRIGGERED] {event['expense'].name} "
                            f"(₹{event['expense'].amount:.0f}) added to today")

            # Reset for next day
            self.expense_idx = 0
            self.daily_spend_total = 0.0
            self.state["days_left"] -= 1
            
            # Natural stress decay overnight
            self.state["stress_level"] = max(0.0, self.state["stress_level"] - 0.05)

            # Recompute allowance for the new day
            self._recompute_daily_allowance()

            # Multi-agent world layer: EventAgent triggers events
            self.current_day += 1
            new_events = self.event_agent.trigger_events(self.current_day)
            if new_events:
                for event in new_events:
                    logger.info(f"[EVENT] {event['type']} triggered (intensity: {event['intensity']:.2f}, "
                                f"duration: {event['duration']} days)")
                    print(f"  [EVENT] >> {event['type']} | intensity={event['intensity']:.2f} | "
                          f"duration={event['duration']}d")

            # Generate base expenses
            num_exps = self.expense_gen.rng.randint(2, 4)
            self.daily_expenses = self.expense_gen.generate_daily_expenses(num_exps)

            # VendorAgent adjusts prices based on active events
            active_events = self.event_agent.get_active_events()
            if active_events:
                self.daily_expenses = self.vendor_agent.update_prices(self.daily_expenses, active_events)
                multipliers = self.vendor_agent.get_event_multipliers()
                if multipliers:
                    logger.info(f"[PRICE ADJUSTMENT] {multipliers}")
                    print(f"  [PRICES] Adjustments: {multipliers}")

            # Inject triggered consequence expenses into the new day
            for event in ready_events:
                self.daily_expenses.append(event["expense"])

            # ==============================================================
            # SHOCK EVENTS: Check for financial shocks at day transition
            # ==============================================================
            self.state["income_shock_active"] = False

            if self.task_config and self.task_config.allow_shocks:
                shock = self.expense_gen.check_shock(allow_shocks=True)

                if shock == "salary_delay":
                    salary_cut = self.state["salary"] * 0.05
                    self.state["balance"] = max(0.0, self.state["balance"] - salary_cut)
                    self.state["income_shock_active"] = True
                    self._recompute_daily_allowance()
                    logger.info(f"[SHOCK] salary_delay: balance reduced by {salary_cut:.0f}")
                    print(f"  [SHOCK] salary_delay | -{salary_cut:.0f}")

                elif shock == "emergency_expense":
                    emergency = Expense(
                        name="Emergency Medical Bill",
                        category="medical",
                        amount=2500.0,
                        necessity_tag="essential",
                        context="emergency",
                    )
                    self.daily_expenses.append(emergency)
                    logger.info("[SHOCK] emergency_expense: ₹2500 medical bill injected")
                    print(f"  [SHOCK] emergency_expense | Rs.2500 medical bill")

                elif shock == "discount":
                    if len(self.daily_expenses) > 0:
                        original = self.daily_expenses[0].amount
                        self.daily_expenses[0].amount = round(original * 0.8, 2)
                        logger.info(f"[SHOCK] discount: first expense reduced "
                                    f"{original:.0f} -> {self.daily_expenses[0].amount:.0f}")
                        print(f"  [SHOCK] discount | {original:.0f} -> {self.daily_expenses[0].amount:.0f}")

        done = bool(self.state["days_left"] <= 0)

        # Episode-end sparse reward
        if done:
            if self.state["goal_remaining"] <= 0:
                reward += 50.0
            else:
                reward -= 30.0
            if self.state["stress_level"] > 0.7:
                reward -= 20.0

        # Normalize reward from an expected range of roughly [-100, 100] to [-1.0, 1.0] for RL stability
        reward = max(-100.0, min(100.0, float(reward)))
        reward_final = reward / 100.0

        if current_expense_snapshot is not None:
            active_events = self.event_agent.get_active_events()
            multiplier = self.vendor_agent.get_event_multipliers().get(current_expense_snapshot.category, 1.0)
            if active_events and multiplier != 1.0:
                event_type = active_events[0]['type']
                event_msg = f"{event_type} triggers -> {current_expense_snapshot.category}"
                original_amount = current_expense_snapshot.amount / multiplier
                adjusted_amount = current_expense_snapshot.amount
                print(f"\n[EVENT] {event_msg}")
                print(f"[PRICE] {current_expense_snapshot.name}: Rs.{original_amount:.0f} -> Rs.{adjusted_amount:.0f} (x{multiplier:.2f})")

            print(f"[DECISION] Day {self.current_day} | {current_expense_snapshot.name} | {current_expense_snapshot.context} | action={decision} | reward={reward_final:.2f}")

        # Store decision in buffer (not committed until end of episode)
        if current_expense_snapshot is not None:
            active_events = self.event_agent.get_active_events()
            event_types = [e['type'] for e in active_events] if active_events else ['none']
            event_type = event_types[0] if event_types else 'none'

            price_multiplier = self.vendor_agent.get_event_multipliers().get(
                current_expense_snapshot.category, 1.0
            )

            # Store the raw decision in the buffer
            self.episode_memory_buffer.append({
                "episode": self.episode_num,
                "day": self.current_day,
                "expense_type": current_expense_snapshot.category,
                "necessity": current_expense_snapshot.necessity_tag,
                "context": current_expense_snapshot.context,
                "event_type": event_type,
                "price_multiplier": price_multiplier,
                "balance": self.state["balance"],
                "days_left": self.state["days_left"],
                "action": decision,
                "reward_step": reward_final, # Initial step reward
                "amount": spend
            })

        # Process out-of-buffer delayed memory commit if episode is done
        if done:
            goal_achieved = self.state["goal_remaining"] <= 0
            
            for mem in self.episode_memory_buffer:
                if goal_achieved:
                    mem["outcome"] = "success"
                    # If we succeeded, significantly reward smart discipline
                    if mem["action"] == "avoid" and mem["necessity"] in ["discretionary", "semi-essential"]:
                        mem["reward_final"] = min(0.99, mem["reward_step"] + 0.3)
                    else:
                        mem["reward_final"] = mem["reward_step"]
                else:
                    mem["outcome"] = "failed"
                    # If we failed, severely penalize ONLY the actions that drained the balance unnecessarily
                    if mem["action"] in ["allow", "reduce"] and mem["amount"] > 0:
                        if mem["necessity"] == "discretionary":
                            mem["reward_final"] = max(0.01, mem["reward_step"] - 0.7)
                        elif mem["necessity"] == "semi-essential":
                            mem["reward_final"] = max(0.01, mem["reward_step"] - 0.4)
                        else:
                            # Do not penalize essential spending even in a failed episode
                            mem["reward_final"] = mem["reward_step"]
                    else:
                        # Actions like "avoiding" shouldn't be penalized just because the episode failed
                        mem["reward_final"] = mem["reward_step"]

                # Commit to DB
                self.memory.store_decision(
                    episode=mem["episode"],
                    day=mem["day"],
                    expense_type=mem["expense_type"],
                    necessity=mem["necessity"],
                    context=mem["context"],
                    event_type=mem["event_type"],
                    price_multiplier=mem["price_multiplier"],
                    balance=mem["balance"],
                    days_left=mem["days_left"],
                    action=mem["action"],
                    reward=mem["reward_final"],
                    outcome=mem["outcome"],
                )

        info = {
            "reward_components": {
                "stress_penalty": -stress_penalty if 'stress_penalty' in locals() else 0.0,
                "risk_penalty": -risk_penalty if 'risk_penalty' in locals() else 0.0,
                "context_penalty": context_penalty if 'context_penalty' in locals() else 0.0,
                "overspend_penalty": -overspend_penalty if 'overspend_penalty' in locals() else 0.0,
                "goal_progress": (goal_progress * 0.01) if 'goal_progress' in locals() else 0.0,
                "completion_bonus": 50.0 if done and self.state["goal_remaining"] <= 0 else (-30.0 if done else 0.0),
                "stress_failure_penalty": -20.0 if done and self.state["stress_level"] > 0.7 else 0.0
            },
            "metrics": self.process_metrics,
            "bad_decision_count": self.get_bad_decision_count(),
            "active_shocks": self.state.get("income_shock_active", False)
        }

        return self._get_observation(), reward_final, done, info

    def get_state(self) -> Dict[str, Any]:
        return self.state

    def get_bad_decisions(self) -> list:
        """Return list of bad decisions made during this episode."""
        return [s for s in self.step_log if s.get("bad_decision") is not None]

    def get_bad_decision_count(self) -> int:
        """Return count of bad decisions in this episode."""
        return len(self.get_bad_decisions())

    def get_process_metrics(self) -> Dict[str, int]:
        """Returns step-level process metrics for RL verification."""
        return self.process_metrics