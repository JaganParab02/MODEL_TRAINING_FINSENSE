"""
Inference Script — Model-Agnostic Pipeline
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:11434/v1/")
    MODEL_NAME = os.getenv("MODEL_NAME", "mistral:latest")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.
    - Each tasks should return score in [0, 1]

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 score=1.00 rewards=0.00,0.00,1.00
"""

import os
import json
import sys
import re
from finsense.env import FinSenseEnv
from finsense.models import ActionModel, StateModel, Expense
from finsense.graders import grade_episode
from finsense.tasks import TASKS
from finsense.memory import MemorySystem
from finsense.assistant_layer import get_assistant_tip

# Import the local rule-based agent as fallback
from inference_local import rule_based_agent


# ─── Configuration ──────────────────────────────────────────────────────────

def load_config():
    """Load all configuration from environment variables."""
    return {
        "HF_TOKEN": os.getenv("HF_TOKEN", "ollama"),
        "API_BASE_URL": os.getenv("API_BASE_URL", "http://localhost:11434/v1/"),
        "MODEL_NAME": os.getenv("MODEL_NAME", "mistral"),
        "USE_MEMORY": os.getenv("USE_MEMORY", "1") == "1",
        "USE_ASSISTANT": os.getenv("USE_ASSISTANT", "1") == "1",
        "FORCE_RULE_BASED": os.getenv("FORCE_RULE_BASED", "0") == "1",
        "BENCHMARK_MODE": os.getenv("BENCHMARK_MODE", "0") == "1",
    }


# ─── Fallback ───────────────────────────────────────────────────────────────

def get_fallback_action(obs: dict, memory: MemorySystem = None, use_memory: bool = False) -> ActionModel:
    """
    Smart rule-based fallback with memory support.
    Used when API fails, LLM returns invalid output, or running without LLM.
    """
    return rule_based_agent(obs, memory=memory, use_memory=use_memory)


# ─── JSON Extraction & Parsing ──────────────────────────────────────────────

def extract_json(raw: str) -> str:
    """Sanitize raw LLM output to extract valid JSON."""
    raw = raw.strip()

    # Remove markdown code blocks
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:].strip()

    # Extract first JSON object
    match = re.search(r'\{[^{}]+\}', raw)
    if match:
        return match.group(0)

    return raw


def parse_llm_response(raw: str, obs: dict) -> ActionModel:
    """
    Parse raw LLM output into a validated ActionModel.
    Raises ValueError on complete parse failure.
    """
    print(f"[RAW LLM] {raw}")

    clean = extract_json(raw)
    print(f"[CLEANED JSON] {clean}")

    # Attempt 1: direct parse
    try:
        action_dict = json.loads(clean)
    except Exception:
        # Attempt 2: repair common issues
        repaired = clean.replace("'", '"').replace("\n", " ")
        try:
            action_dict = json.loads(repaired)
        except Exception:
            raise ValueError("JSON_PARSE_FAILED")

    # Validate decision field
    if action_dict.get("decision") not in ("allow", "reduce", "avoid"):
        print(f"[VALIDATION] Invalid decision '{action_dict.get('decision')}' → forcing 'avoid'")
        action_dict["decision"] = "avoid"

    # Validate approved_amount
    if not isinstance(action_dict.get("approved_amount"), (int, float)):
        print(f"[VALIDATION] Invalid approved_amount → forcing 0.0")
        action_dict["approved_amount"] = 0.0

    return ActionModel(**action_dict)


# ─── Safety Selector ────────────────────────────────────────────────────────

def select_final_action(fallback: ActionModel, llm: ActionModel, obs: dict) -> ActionModel:
    """
    Select the safest action between rule-based fallback and LLM suggestion.
    The fallback is trusted on essential expenses, budget violations, and emergencies.
    """
    exp = obs.get("current_expense", {}) or {}
    necessity = exp.get("necessity_tag", "")
    amount = float(exp.get("amount", 0))
    daily_allowance = float(obs.get("daily_allowance", 0))
    context = exp.get("context", "normal")

    # Never trust LLM on essential expenses
    if necessity == "essential":
        print("[SAFETY] fallback selected over LLM → essential expense protection")
        return fallback

    # Protect budget: LLM says allow but it exceeds daily allowance
    if amount > daily_allowance and llm.decision == "allow":
        print("[SAFETY] fallback selected over LLM → budget protection")
        return fallback

    # Protect emergencies: LLM says avoid during emergency
    if context == "emergency" and llm.decision == "avoid":
        print("[SAFETY] fallback selected over LLM → emergency protection")
        return fallback

    # Accept reasonable LLM actions (reduce/avoid are conservative)
    if llm.decision in ("reduce", "avoid"):
        return llm

    return fallback


# ─── Model-Agnostic Prompt ──────────────────────────────────────────────────

def build_prompt(obs: dict, memory_bias: str = None, memory_confidence: float = 0.0,
                 assistant_tip: dict = None) -> str:
    """
    Short, strict, portable prompt designed to work across different
    chat-completion models and providers.
    """
    exp = obs.get("current_expense", {}) or {}
    necessity = exp.get("necessity_tag", "discretionary")
    amount = exp.get("amount", 0)
    name = exp.get("name", "Unknown")
    category = exp.get("category", "unknown")
    context = exp.get("context", "normal")
    balance = obs.get("balance", 0)
    goal_remaining = obs.get("goal_remaining", 0)
    days_left = obs.get("days_left", 0)
    daily_allowance = obs.get("daily_allowance", 0)
    active_events = obs.get("active_events", [])
    events_str = ", ".join(active_events) if active_events else "None"

    # Assistant advice block
    assistant_block = ""
    if assistant_tip:
        assistant_block = f"""
Assistant Advice:
- Recommendation: {assistant_tip['recommendation']}
- Reason: {assistant_tip['reason']}"""

    # Memory advice block
    memory_block = ""
    if memory_bias:
        memory_block = f"""
Memory Advice:
- Suggested action: {memory_bias}
- Confidence: {memory_confidence:.2f}"""

    prompt = f"""You are a financial decision agent.

Return ONLY valid JSON. No explanation. No markdown. No text before or after.

{{"decision": "allow" | "reduce" | "avoid", "approved_amount": number, "reasoning": "short"}}

Current state:
Balance: {balance:.0f}
Goal Remaining: {goal_remaining:.0f}
Days Left: {days_left}
Daily Allowance: {daily_allowance:.0f}
Active Events: {events_str}

Expense:
Name: {name}
Category: {category}
Amount: {amount:.0f}
Necessity: {necessity}
Context: {context}{assistant_block}{memory_block}

Rules:
- Always protect essentials
- Avoid overspending beyond daily allowance
- Reduce spending when prices are inflated
- Prioritize goal completion over comfort
- Use double quotes only, no trailing commas, no comments"""

    return prompt


# ─── Scoring ────────────────────────────────────────────────────────────────

def calculate_final_score(env, task_id):
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
            "truncated": False
        }
        state_model = StateModel(**state_data)
        raw = grade_episode(state_model)
        # CRITICAL: always clamp, even if grader returns something unexpected
        return max(0.01, min(0.99, raw))
    except Exception:
        goal_total = s["goal_total"]
        goal_remaining = s["goal_remaining"]
        goal_progress = max(0.0, goal_total - goal_remaining)
        raw = goal_progress / max(1.0, goal_total)
        return max(0.01, min(0.99, raw))


# ─── Main Inference Loop ────────────────────────────────────────────────────

def run_inference(task_id="easy", config=None):
    if config is None:
        config = load_config()

    HF_TOKEN = config["HF_TOKEN"]
    API_BASE_URL = config["API_BASE_URL"]
    MODEL_NAME = config["MODEL_NAME"]
    use_memory = config["USE_MEMORY"]
    use_assistant = config["USE_ASSISTANT"]
    force_rule_based = config["FORCE_RULE_BASED"]

    # Check if LLM is available (and not forced off)
    llm_available = False
    client = None

    if not force_rule_based:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
            if MODEL_NAME:
                llm_available = True
        except Exception:
            pass
    else:
        print("  [INFO] FORCE_RULE_BASED=1 — LLM disabled")

    env = FinSenseEnv()
    memory = env.memory

    obs = env.reset(task_id=task_id)

    model_label = MODEL_NAME if llm_available else "rule-based-fallback"
    print(f"[START] task={task_id} env=finsense-rl model={model_label}")

    if not llm_available:
        print(f"  [INFO] No LLM available - using rule-based agent with memory={'ON' if use_memory else 'OFF'}")

    step_num = 0
    all_rewards = []
    done = False
    last_error = "null"
    fallback_count = 0
    parse_fail_count = 0

    try:
        while not done:
            step_num += 1
            action_str = "null"
            
            try:
                # ── Step 1: Always compute fallback action ──
                fallback_action = get_fallback_action(obs, memory=memory, use_memory=use_memory)

                # ── Step 2: Always compute assistant tip ──
                assistant_tip = None
                if use_assistant:
                    assistant_tip = get_assistant_tip(obs, memory=memory if use_memory else None)
                    print(f"[ASSISTANT] {assistant_tip['recommendation']} -> {assistant_tip['reason']}")

                # ── Step 3: Compute memory bias with confidence ──
                memory_bias = None
                memory_confidence = 0.0
                if use_memory:
                    exp = obs.get("current_expense") or {}
                    expense_type = exp.get("category", "unknown")
                    context = exp.get("context", "normal")
                    necessity = exp.get("necessity_tag", "unknown")
                    active_events = obs.get("active_events", [])
                    event_type = active_events[0] if active_events else "none"
                    memory_bias, memory_confidence = memory.get_memory_bias_with_confidence(
                        expense_type, context, event_type, necessity=necessity
                    )
                    if memory_bias:
                        print(f"[MEMORY] bias={memory_bias} confidence={memory_confidence:.2f}")

                # ── Step 4: High-confidence memory → skip LLM ──
                if use_memory and memory_bias and memory_confidence >= 0.65:
                    print(f"[LLM] skipped due to high-confidence memory (conf={memory_confidence:.2f})")
                    # Build action from memory bias
                    exp = obs.get("current_expense") or {}
                    amount = float(exp.get("amount", 0))
                    if memory_bias == "allow":
                        action = ActionModel(decision="allow", approved_amount=amount,
                                             reasoning=f"Memory override (conf={memory_confidence:.2f})")
                    elif memory_bias == "reduce":
                        action = ActionModel(decision="reduce", approved_amount=round(amount * 0.5, 2),
                                             reasoning=f"Memory override (conf={memory_confidence:.2f})")
                    else:
                        action = ActionModel(decision="avoid", approved_amount=0.0,
                                             reasoning=f"Memory override (conf={memory_confidence:.2f})")
                    action_str = f"memory:{action.decision}({action.approved_amount:.0f})"
                    print(f"[DECISION] {action_str}")
                    last_error = "null"

                # ── Step 5: LLM path with safety selector ──
                elif llm_available and client and MODEL_NAME:
                    try:
                        response = client.chat.completions.create(
                            model=MODEL_NAME,
                            messages=[
                                {"role": "system", "content": "You are a financial agent. Reply only with valid JSON."},
                                {"role": "user", "content": build_prompt(
                                    obs,
                                    memory_bias=memory_bias,
                                    memory_confidence=memory_confidence,
                                    assistant_tip=assistant_tip
                                )}
                            ],
                            max_tokens=100,
                            temperature=0.1
                        )
                        raw = response.choices[0].message.content.strip()
                        llm_action = parse_llm_response(raw, obs)

                        # Safety selector: choose between fallback and LLM
                        action = select_final_action(fallback_action, llm_action, obs)
                        if action is fallback_action and action is not llm_action:
                            fallback_count += 1
                        action_str = f"{action.decision}({action.approved_amount:.0f})"
                        print(f"[DECISION] {action_str}")
                        last_error = "null"

                    except Exception as llm_err:
                        print(f"[PARSE ERROR] {llm_err}")
                        parse_fail_count += 1
                        action = fallback_action
                        fallback_count += 1
                        action_str = f"fallback:{action.decision}({action.approved_amount:.0f})"
                        last_error = str(llm_err).replace('\n', ' ')[:200]

                # ── Step 6: Pure rule-based path ──
                else:
                    action = fallback_action
                    action_str = f"{action.decision}({action.approved_amount:.0f})"
                    last_error = "null"

            except Exception as e:
                print(f"[PARSE ERROR] {e}")
                action = get_fallback_action(obs, memory=memory, use_memory=use_memory)
                fallback_count += 1
                action_str = f"fallback:{action.decision}({action.approved_amount:.0f})"
                last_error = str(e).replace('\n', ' ')[:200]

            obs, reward, done, info = env.step(action)
            all_rewards.append(reward)

            # [STEP] line
            done_str = str(done).lower()
            print(f"[STEP] step={step_num} action={action_str} reward={reward:.2f} done={done_str} error={last_error}")

    except Exception as e:
        last_error = str(e).replace('\n', ' ')[:200]
    finally:
        success_bool = obs.get("goal_remaining", 0) <= 0
        success_str = str(success_bool).lower()
        score = calculate_final_score(env, task_id)
        rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)

        # [END] line
        print(f"[END] success={success_str} steps={step_num} score={score:.2f} rewards={rewards_str}")

    return {
        "task_id": task_id,
        "score": score,
        "success": success_bool,
        "steps": step_num,
        "total_reward": sum(all_rewards),
        "fallback_count": fallback_count,
        "parse_fail_count": parse_fail_count,
    }


if __name__ == "__main__":
    for task in ["easy", "medium", "hard"]:
        run_inference(task)