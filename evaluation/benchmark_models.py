"""
Model Benchmark Script
======================
Evaluate multiple Hugging Face / chat-completion models against the FinSense
environment using the same task setup.

Usage:
    python benchmark_models.py --task hard --episodes 5 --models mistral llama3 qwen2
    python benchmark_models.py --task easy --episodes 3 --models "meta-llama/Meta-Llama-3-8B-Instruct"

Environment variables required:
    API_BASE_URL   The API endpoint (same for all models)
    HF_TOKEN       Your API key

Output:
    - Ranked summary table printed to stdout
    - Results saved to benchmark_results.json
"""

import os
import sys
import json
import time
import argparse
from typing import List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finsense.env import FinSenseEnv
from finsense.models import ActionModel, StateModel
from finsense.graders import grade_episode
from finsense.tasks import TASKS
from finsense.memory import MemorySystem
from finsense.assistant_layer import get_assistant_tip
from inference_local import rule_based_agent


def extract_json(raw: str) -> str:
    """Sanitize raw LLM output to extract valid JSON."""
    raw = raw.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else raw
        if raw.startswith("json"):
            raw = raw[4:].strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        raw = raw[start:end+1]
    return raw


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
            "truncated": False
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


def run_benchmark_episode(client, model_name: str, task_id: str, memory: MemorySystem,
                           use_memory: bool = True) -> Dict[str, Any]:
    """
    Run a single episode with the given model and return detailed metrics.
    """
    env = FinSenseEnv()
    env.memory = memory
    obs = env.reset(task_id=task_id)

    step_num = 0
    all_rewards = []
    done = False
    parse_fail_count = 0
    fallback_count = 0
    step_latencies = []

    while not done:
        step_num += 1
        step_start = time.time()

        # Always compute fallback
        fallback_action = rule_based_agent(obs, memory=memory, use_memory=use_memory)

        # Assistant tip
        assistant_tip = get_assistant_tip(obs, memory=memory if use_memory else None)

        # Memory bias
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

        # High-confidence memory → skip LLM
        if use_memory and memory_bias and memory_confidence >= 0.65:
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
        else:
            # LLM call
            try:
                exp = obs.get("current_expense", {}) or {}
                necessity = exp.get("necessity_tag", "discretionary")
                amount = exp.get("amount", 0)
                name = exp.get("name", "Unknown")
                category = exp.get("category", "unknown")
                context_val = exp.get("context", "normal")
                balance = obs.get("balance", 0)
                goal_remaining = obs.get("goal_remaining", 0)
                days_left = obs.get("days_left", 0)
                daily_allowance = obs.get("daily_allowance", 0)
                active_events = obs.get("active_events", [])
                events_str = ", ".join(active_events) if active_events else "None"

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
Context: {context_val}

Assistant Advice:
- Recommendation: {assistant_tip['recommendation']}
- Reason: {assistant_tip['reason']}

Rules:
- Always protect essentials
- Avoid overspending beyond daily allowance
- Reduce spending when prices are inflated
- Prioritize goal completion over comfort
- Use double quotes only, no trailing commas, no comments"""

                if memory_bias:
                    prompt += f"""

Memory Advice:
- Suggested action: {memory_bias}
- Confidence: {memory_confidence:.2f}"""

                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "You are a financial agent. Reply only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ],
                    max_tokens=100,
                    temperature=0.1
                )
                raw = response.choices[0].message.content.strip()

                # Parse
                clean = extract_json(raw)
                try:
                    action_dict = json.loads(clean)
                except Exception:
                    repaired = clean.replace("'", '"').replace("\n", " ")
                    try:
                        action_dict = json.loads(repaired)
                    except Exception:
                        raise ValueError("JSON_PARSE_FAILED")

                # Validate
                if action_dict.get("decision") not in ("allow", "reduce", "avoid"):
                    action_dict["decision"] = "avoid"
                if not isinstance(action_dict.get("approved_amount"), (int, float)):
                    action_dict["approved_amount"] = 0.0

                llm_action = ActionModel(**action_dict)

                # Safety selector
                exp_data = obs.get("current_expense", {}) or {}
                nec = exp_data.get("necessity_tag", "")
                amt = float(exp_data.get("amount", 0))
                da = float(obs.get("daily_allowance", 0))
                ctx = exp_data.get("context", "normal")

                if nec == "essential":
                    action = fallback_action
                    fallback_count += 1
                elif amt > da and llm_action.decision == "allow":
                    action = fallback_action
                    fallback_count += 1
                elif ctx == "emergency" and llm_action.decision == "avoid":
                    action = fallback_action
                    fallback_count += 1
                elif llm_action.decision in ("reduce", "avoid"):
                    action = llm_action
                else:
                    action = fallback_action
                    fallback_count += 1

            except Exception:
                parse_fail_count += 1
                fallback_count += 1
                action = fallback_action

        step_latencies.append(time.time() - step_start)

        obs, reward, done, info = env.step(action)
        all_rewards.append(reward)

    success = obs.get("goal_remaining", 0) <= 0
    score = calculate_final_score(env, task_id)

    return {
        "score": score,
        "success": success,
        "total_reward": sum(all_rewards),
        "steps": step_num,
        "parse_fail_count": parse_fail_count,
        "fallback_count": fallback_count,
        "avg_latency": sum(step_latencies) / max(1, len(step_latencies)),
    }


def run_benchmark(models: List[str], task_id: str = "hard", episodes: int = 5,
                   api_base_url: str = None, hf_token: str = None) -> Dict[str, Any]:
    """
    Benchmark multiple models and return a ranked summary.
    """
    from openai import OpenAI

    api_base_url = api_base_url or os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1/")
    hf_token = hf_token or os.getenv("HF_TOKEN", "")

    results = {}

    for model_name in models:
        print(f"\n{'='*60}")
        print(f"  BENCHMARKING: {model_name}")
        print(f"  Task: {task_id} | Episodes: {episodes}")
        print(f"{'='*60}\n")

        client = OpenAI(api_key=hf_token, base_url=api_base_url)
        memory = MemorySystem()

        model_results = {
            "scores": [],
            "successes": [],
            "rewards": [],
            "parse_fails": [],
            "fallback_counts": [],
            "latencies": [],
        }

        for ep in range(1, episodes + 1):
            print(f"  Episode {ep}/{episodes}...", end=" ", flush=True)
            try:
                ep_result = run_benchmark_episode(
                    client=client,
                    model_name=model_name,
                    task_id=task_id,
                    memory=memory,
                    use_memory=True
                )
                model_results["scores"].append(ep_result["score"])
                model_results["successes"].append(ep_result["success"])
                model_results["rewards"].append(ep_result["total_reward"])
                model_results["parse_fails"].append(ep_result["parse_fail_count"])
                model_results["fallback_counts"].append(ep_result["fallback_count"])
                model_results["latencies"].append(ep_result["avg_latency"])
                print(f"score={ep_result['score']:.2f} success={ep_result['success']} "
                      f"parse_fails={ep_result['parse_fail_count']} "
                      f"fallbacks={ep_result['fallback_count']}")
            except Exception as e:
                print(f"ERROR: {e}")
                model_results["scores"].append(0.01)
                model_results["successes"].append(False)
                model_results["rewards"].append(0.0)
                model_results["parse_fails"].append(999)
                model_results["fallback_counts"].append(999)
                model_results["latencies"].append(0.0)

        # Compute aggregates
        n = len(model_results["scores"])
        results[model_name] = {
            "avg_score": sum(model_results["scores"]) / max(1, n),
            "success_rate": sum(1 for s in model_results["successes"] if s) / max(1, n),
            "avg_reward": sum(model_results["rewards"]) / max(1, n),
            "total_parse_fails": sum(model_results["parse_fails"]),
            "total_fallback_uses": sum(model_results["fallback_counts"]),
            "avg_latency_per_step": sum(model_results["latencies"]) / max(1, n),
            "episodes": n,
            "raw": model_results,
        }

    # Print ranked summary
    print(f"\n{'='*80}")
    print(f"  BENCHMARK RESULTS — Task: {task_id} | Episodes: {episodes}")
    print(f"{'='*80}")
    print(f"{'Rank':<5} {'Model':<40} {'Avg Score':<10} {'Success%':<10} "
          f"{'Avg Reward':<11} {'Parse Fails':<12} {'Fallbacks':<10} {'Latency(s)':<10}")
    print("-" * 108)

    ranked = sorted(results.items(), key=lambda x: x[1]["avg_score"], reverse=True)
    for rank, (model, data) in enumerate(ranked, 1):
        print(f"{rank:<5} {model:<40} {data['avg_score']:<10.3f} "
              f"{data['success_rate']*100:<10.1f} "
              f"{data['avg_reward']:<11.2f} "
              f"{data['total_parse_fails']:<12} "
              f"{data['total_fallback_uses']:<10} "
              f"{data['avg_latency_per_step']:<10.3f}")

    print(f"{'='*80}\n")

    # Save to file
    output_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_results.json")
    # Strip raw episode data for the JSON output to keep it clean
    save_data = {
        "task": task_id,
        "episodes": episodes,
        "models": {
            model: {k: v for k, v in data.items() if k != "raw"}
            for model, data in results.items()
        },
        "ranking": [model for model, _ in ranked],
    }
    with open(output_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Results saved to: {output_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark multiple models on FinSense")
    parser.add_argument("--task", type=str, default="hard", choices=["easy", "medium", "hard"],
                        help="Task difficulty (default: hard)")
    parser.add_argument("--episodes", type=int, default=5,
                        help="Episodes per model (default: 5)")
    parser.add_argument("--models", nargs="+", required=True,
                        help="List of model names to benchmark")
    args = parser.parse_args()

    run_benchmark(
        models=args.models,
        task_id=args.task,
        episodes=args.episodes
    )
