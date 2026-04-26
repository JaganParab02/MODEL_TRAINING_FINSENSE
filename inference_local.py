"""
Local rule-based inference for testing FinSense without API credits.
Supports memory-augmented decision making for learning evaluation.

Usage: 
  python inference_local.py                     # Run all tasks
  python inference_local.py --use-memory        # Run with memory enabled
  python inference_local.py --no-memory         # Run without memory
"""
import os
import sys
from finsense.env import FinSenseEnv
from finsense.models import ActionModel
from finsense.memory import MemorySystem
from finsense.graders import grade_episode
from finsense.models import StateModel
from finsense.tasks import TASKS

# Per-episode override diagnostics (module-level for simplicity)
_override_stats = {"count": 0, "positive": 0, "negative": 0, "confidence_sum": 0.0}


def rule_based_agent(obs: dict, memory: MemorySystem = None, use_memory: bool = False, pre_train: bool = False) -> ActionModel:
    """
    Smart heuristic agent with optional memory bias override.
    
    Memory override logic:
    - Retrieve past cases matching current expense context
    - If confidence is high (>= 0.65), override rule-based decision
    - This allows the agent to learn from past episodes
    """
    expense = obs.get("current_expense")
    if not expense:
        return ActionModel(decision="avoid", approved_amount=0.0, reasoning="No expense")

    amount = float(expense["amount"])
    necessity = expense["necessity_tag"]
    context = expense.get("context", "normal")
    category = expense.get("category", "unknown")
    balance = obs.get("balance", 0)
    days_left = obs.get("days_left", 1)
    goal_remaining = obs.get("goal_remaining", 0)
    stress = obs.get("stress_level", 0)
    active_events = obs.get("active_events", [])

    daily_budget = balance / max(1, days_left)

    # === MEMORY BIAS LAYER ===
    if use_memory and memory is not None:
        event_type = active_events[0] if active_events else "none"
        bias_action, confidence = memory.get_memory_bias_with_confidence(
            expense_type=category,
            context=context,
            event_type=event_type,
            necessity=necessity,
        )

        # Retrieve count for logging
        similar = memory.retrieve_similar_cases(category, context, event_type, necessity=necessity, limit=15)
        n_cases = len(similar)
        if n_cases > 0 and bias_action:
            print(f"[MEMORY] Retrieved {n_cases} cases for {category}+{event_type}")
            print(f"         Best action: {bias_action} (confidence {confidence:.2f})")

        # Task-aware confidence threshold: hard task requires higher confidence
        task_id = obs.get("task_id", "easy")
        confidence_threshold = 0.75 if task_id == "hard" else 0.65

        if bias_action and confidence >= confidence_threshold:
            # === SAFETY GATES: Prevent memory from making dangerous overrides ===
            safe_to_override = True
            gate_reason = None

            # Gate 1: NEVER override to avoid essential expenses
            if bias_action == "avoid" and necessity == "essential":
                safe_to_override = False
                gate_reason = "essential expense"

            # Gate 2: NEVER override to avoid in emergency context
            if bias_action == "avoid" and context == "emergency" and necessity in ("essential", "semi-essential"):
                safe_to_override = False
                gate_reason = "emergency context"

            # Gate 3: Don't allow expensive discretionary over daily allowance
            daily_allowance = obs.get("daily_allowance", 0)
            if bias_action == "allow" and necessity == "discretionary" and amount > max(500, daily_allowance):
                safe_to_override = False
                gate_reason = "discretionary over budget"

            # Gate 4: Endgame protection — don't override when low balance + few days left
            if days_left <= 5 and balance < goal_remaining * 0.5:
                safe_to_override = False
                gate_reason = "endgame low-balance"

            # Gate 5 (hard only): Don't override on essential/semi-essential when balance is critical
            if task_id == "hard" and necessity in ("essential", "semi-essential") and balance < goal_remaining:
                safe_to_override = False
                gate_reason = "hard-task critical balance"

            if safe_to_override:
                print(f"[OVERRIDE] Memory overrides decision")
                _override_stats["count"] += 1
                _override_stats["confidence_sum"] += confidence
                if bias_action == "allow":
                    return ActionModel(
                        decision="allow", approved_amount=amount,
                        reasoning=f"Memory override (conf={confidence:.2f}): allow worked well before"
                    )
                elif bias_action == "reduce":
                    return ActionModel(
                        decision="reduce", approved_amount=round(amount * 0.5, 2),
                        reasoning=f"Memory override (conf={confidence:.2f}): reduce worked well before"
                    )
                elif bias_action == "avoid":
                    return ActionModel(
                        decision="avoid", approved_amount=0.0,
                        reasoning=f"Memory override (conf={confidence:.2f}): avoid worked well before"
                    )
            else:
                print(f"[GATE] Memory override blocked: {gate_reason} (was: {bias_action})")
    elif pre_train:
        # Epsilon-greedy exploration during Pre-Training phase to build a robust dataset
        import random
        if random.random() < 0.15:  # 15% chance to explore during pre-train
            random_action = random.choice(["allow", "reduce", "avoid"])
            if random_action == "allow":
                amt = amount
            elif random_action == "reduce":
                amt = round(amount * 0.5, 2)
            else:
                amt = 0.0
            return ActionModel(
                decision=random_action, 
                approved_amount=amt, 
                reasoning="Exploration (Pre-Training)"
            )

    # === CONTEXT-AWARE ADJUSTMENTS ===
    # Emergency context: ALWAYS allow essential/semi-essential
    if context == "emergency" and necessity in ("essential", "semi-essential"):
        return ActionModel(decision="allow", approved_amount=amount,
                           reasoning="Emergency context - must not avoid")

    # === RULE-BASED DECISIONS ===

    # Essential: always allow (but reduce if budget is tight)
    if necessity == "essential":
        if amount <= daily_budget * 0.5:
            return ActionModel(decision="allow", approved_amount=amount,
                               reasoning="Essential and affordable")
        else:
            reduced = min(amount, daily_budget * 0.4)
            return ActionModel(decision="reduce", approved_amount=reduced,
                               reasoning="Essential but over budget, reducing")

    # Semi-essential: allow if cheap, reduce if moderate, avoid if expensive
    if necessity == "semi-essential":
        if stress > 0.6:
            return ActionModel(decision="allow", approved_amount=amount,
                               reasoning="Allowing semi-essential to manage stress")
        if amount <= daily_budget * 0.4:  # Increased from 0.2
            return ActionModel(decision="allow", approved_amount=amount,
                               reasoning="Semi-essential and comfortably within daily budget")
        elif amount <= daily_budget * 0.7:  # Increased from 0.4
            reduced = amount * 0.5
            return ActionModel(decision="reduce", approved_amount=reduced,
                               reasoning="Semi-essential, reducing spend to stay safe")
        else:
            return ActionModel(decision="avoid", approved_amount=0.0,
                               reasoning="Semi-essential but too expensive")

    # Discretionary: mostly avoid, unless we have a large daily budget
    if necessity == "discretionary":
        # Weekend context: extra caution
        if context == "weekend":
            return ActionModel(decision="avoid", approved_amount=0.0,
                               reasoning="Weekend discretionary - avoiding lifestyle spending")

        # daily_budget is already (balance - goal_remaining) / days_left
        # So it IS the surplus. We don't need to subtract savings_per_day again.
        if amount <= daily_budget * 0.2:
            return ActionModel(decision="allow", approved_amount=amount,
                               reasoning="Discretionary but very cheap, surplus exists")
        elif amount <= daily_budget * 0.4:
            reduced = amount * 0.5
            return ActionModel(decision="reduce", approved_amount=reduced,
                               reasoning="Discretionary, reducing amount")
        else:
            return ActionModel(decision="avoid", approved_amount=0.0,
                               reasoning="Discretionary - skipping to save")

    return ActionModel(decision="avoid", approved_amount=0.0, reasoning="Fallback: avoid")


def calculate_final_score(env, task_id):
    """Calculate the final graded score for the episode."""
    s = env.state
    task = TASKS.get(task_id)

    try:
        state_data = {
            "current_day": task.days - s["days_left"],
            "total_days": task.days,
            "balance": s["balance"],
            "initial_goal": s["goal_total"],
            "current_goal_remaining": s["goal_remaining"],
            "stress_level": s["stress_level"],
            "risk_level": s["risk_level"],
            "seed": 42,
            "task_id": task_id,
            "expected_fixed_expenses": s["expected_fixed_expenses"],
            "income_shock_active": s["income_shock_active"],
            "recent_spending": s["recent_spending"],
            "user_type": "balanced",
            "current_expense_idx": 0,
            "daily_expenses": [],
            "daily_expense_idx": 0,
            "terminated": s["days_left"] <= 0,
            "truncated": False,
        }
        state_model = StateModel(**state_data)
        raw = grade_episode(state_model)
        return max(0.01, min(0.99, raw))
    except Exception:
        goal_total = s["goal_total"]
        goal_remaining = s["goal_remaining"]
        goal_progress = max(0.0, goal_total - goal_remaining)
        raw = goal_progress / max(1.0, goal_total)
        return max(0.01, min(0.99, raw))


def run_episode(task_id="easy", use_memory=False, memory=None, seed=42, pre_train=False):
    """Run a single episode and return detailed results."""
    global _override_stats
    _override_stats = {"count": 0, "positive": 0, "negative": 0, "confidence_sum": 0.0}

    env = FinSenseEnv()
    
    # Share memory system if provided
    if memory is not None:
        env.memory = memory

    obs = env.reset(task_id=task_id, seed=seed)

    step_num = 0
    all_rewards = []
    done = False

    print(f"[START] task={task_id} env=finsense-rl model=rule-based memory={'ON' if use_memory else 'OFF'} pre_train={pre_train}")

    while not done:
        step_num += 1
        action = rule_based_agent(obs, memory=memory, use_memory=use_memory, pre_train=pre_train)
        action_str = f"{action.decision}({action.approved_amount:.0f})"

        obs, reward, done, info = env.step(action)
        all_rewards.append(reward)

        done_str = str(done).lower()
        print(f"[STEP] step={step_num} action={action_str} reward={reward:.2f} done={done_str} error=null")

    success = obs.get("goal_remaining", 0) <= 0
    score = calculate_final_score(env, task_id)
    bad_decisions = env.get_bad_decision_count()
    bad_details = env.get_bad_decisions()

    # Per-episode bad decision breakdown
    if bad_decisions > 0:
        reason_counts = {}
        for bd in bad_details:
            r = bd.get("bad_decision", "unknown")
            reason_counts[r] = reason_counts.get(r, 0) + 1
        reasons_str = ", ".join(f"{k}={v}" for k, v in sorted(reason_counts.items()))
        print(f"  [BAD_DECISIONS] {bad_decisions} total: {reasons_str}")

    # Override diagnostics
    override_count = _override_stats["count"]
    avg_conf = _override_stats["confidence_sum"] / max(1, override_count)
    if use_memory and override_count > 0:
        print(f"  [OVERRIDES] count={override_count} | avg_confidence={avg_conf:.2f}")

    rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)
    process_metrics = env.get_process_metrics()
    pm_str = " | ".join(f"{k}={v}" for k, v in process_metrics.items())
    
    print(f"[END] success={str(success).lower()} steps={step_num} score={score:.2f} rewards={rewards_str}")
    print(f"  [METRICS] bad_decisions={bad_decisions} | memory_entries={memory.get_stats()['total_decisions'] if memory else 0}")
    print(f"  [PROCESS] {pm_str}")

    return {
        'total_reward': sum(all_rewards),
        'rewards': all_rewards,
        'score': score,
        'success': success,
        'steps': step_num,
        'bad_decisions': bad_decisions,
        'bad_decision_details': bad_details,
        'final_balance': obs.get('balance', 0),
        'goal_remaining': obs.get('goal_remaining', 0),
        'stress': obs.get('stress_level', 0),
        'override_count': override_count,
        'override_avg_confidence': avg_conf,
        'process_metrics': process_metrics,
    }


def run_local_inference():
    """Run episodes for all tasks showing full output."""
    print("[START] Running local rule-based agent (no API needed)\n")

    memory = MemorySystem()
    use_memory = os.getenv("USE_MEMORY", "1") == "1"

    for task_id in ["easy", "medium", "hard"]:
        result = run_episode(task_id=task_id, use_memory=use_memory, memory=memory)

        print(f"\n  [RESULT] task={task_id} | steps={result['steps']} | "
              f"total_reward={result['total_reward']:.2f} | "
              f"goal_left={result['goal_remaining']:.0f} | "
              f"final_balance={result['final_balance']:.0f} | "
              f"stress={result['stress']:.2f} | "
              f"bad_decisions={result['bad_decisions']}")
        print(f"{'='*60}\n")

    print(f"\n[MEMORY STATS] {memory.get_stats()}")
    print("[END]")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Run FinSense local rule-based agent')
    parser.add_argument('--use-memory', action='store_true', default=False)
    parser.add_argument('--no-memory', action='store_true', default=False)
    args = parser.parse_args()

    if args.no_memory:
        os.environ["USE_MEMORY"] = "0"
    elif args.use_memory:
        os.environ["USE_MEMORY"] = "1"

    run_local_inference()
