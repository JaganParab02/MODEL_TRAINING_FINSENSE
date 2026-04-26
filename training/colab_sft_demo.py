"""
Judge-facing Colab demo for FinSense SFT.

Flow:
1. Deterministically split warmstart_data.jsonl into train/eval.
2. Evaluate the base model on the holdout prompts.
3. Fine-tune a LoRA adapter with SFT.
4. Evaluate the SFT model on the same holdout prompts.
5. Run a seeded FinSense environment benchmark for base vs SFT.
6. Save JSON artifacts for judges.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import io
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from finsense.env import FinSenseEnv
from finsense.models import ActionModel
from inference import build_prompt, extract_json, select_final_action
from inference_local import calculate_final_score, rule_based_agent
from training.sft_warmstart import DEFAULT_MODEL, ensure_pad_token, load_dataset_splits, train_sft

SYSTEM_PROMPT = "You are a financial agent. Reply only with valid JSON."


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def print_stage(title: str) -> None:
    line = "=" * 72
    print(f"\n{line}\n{title}\n{line}")


def clear_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_inference_model(model_name: str, adapter_dir: str | None = None):
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model_kwargs: dict[str, Any] = {"torch_dtype": torch_dtype}
    if torch.cuda.is_available():
        model_kwargs["device_map"] = "auto"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ensure_pad_token(tokenizer)

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir)
    model.eval()
    return model, tokenizer


def generate_response(model, tokenizer, messages: list[dict[str, str]], max_new_tokens: int) -> str:
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(rendered, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    new_tokens = generated[0][inputs["input_ids"].shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_action_output(raw: str) -> dict[str, Any]:
    clean = extract_json(raw)
    result = {
        "raw": raw,
        "clean": clean,
        "valid_json": False,
        "schema_valid": False,
        "action": None,
        "error": None,
    }

    try:
        payload = json.loads(clean)
        result["valid_json"] = True
    except Exception:
        repaired = clean.replace("'", '"').replace("\n", " ")
        try:
            payload = json.loads(repaired)
            result["valid_json"] = True
            result["clean"] = repaired
        except Exception as exc:
            result["error"] = f"JSON_PARSE_FAILED: {exc}"
            return result

    decision = payload.get("decision")
    approved_amount = payload.get("approved_amount")
    if decision in ("allow", "reduce", "avoid") and is_number(approved_amount):
        result["schema_valid"] = True
        result["action"] = {
            "decision": decision,
            "approved_amount": float(approved_amount),
            "reasoning": payload.get("reasoning", ""),
        }
    else:
        result["error"] = "SCHEMA_INVALID"

    return result


def parse_prompt_context(user_prompt: str) -> dict[str, Any]:
    def _float(pattern: str) -> float:
        match = re.search(pattern, user_prompt)
        return float(match.group(1)) if match else 0.0

    def _text(pattern: str) -> str:
        match = re.search(pattern, user_prompt)
        return match.group(1).strip() if match else ""

    return {
        "daily_allowance": _float(r"Daily Allowance:\s*([0-9.]+)"),
        "amount": _float(r"Amount:\s*([0-9.]+)"),
        "necessity": _text(r"Necessity:\s*([A-Za-z\-]+)"),
        "context": _text(r"Context:\s*([A-Za-z_]+)"),
        "name": _text(r"Name:\s*(.+)"),
        "category": _text(r"Category:\s*([A-Za-z_]+)"),
    }


def approved_amount_is_sane(action: dict[str, Any] | None, prompt_ctx: dict[str, Any]) -> bool:
    if action is None:
        return False
    amount = prompt_ctx["amount"]
    approved = action["approved_amount"]
    decision = action["decision"]

    if approved < 0 or approved > amount + 1e-6:
        return False
    if decision == "avoid":
        return abs(approved) < 1e-6
    if decision == "allow":
        return approved > 0
    if decision == "reduce":
        return 0 < approved <= amount + 1e-6
    return False


def action_is_safe(action: dict[str, Any] | None, prompt_ctx: dict[str, Any]) -> bool:
    if action is None:
        return False

    decision = action["decision"]
    amount = prompt_ctx["amount"]
    daily_allowance = prompt_ctx["daily_allowance"]
    necessity = prompt_ctx["necessity"]
    context = prompt_ctx["context"]

    if necessity == "essential" and decision == "avoid":
        return False
    if context == "emergency" and necessity in ("essential", "semi-essential") and decision == "avoid":
        return False
    if amount > daily_allowance and decision == "allow":
        return False
    return True


def budget_violation(action: dict[str, Any] | None, prompt_ctx: dict[str, Any]) -> bool:
    if action is None:
        return False
    return prompt_ctx["amount"] > prompt_ctx["daily_allowance"] and action["decision"] == "allow"


def safe_json_load(text: str) -> dict[str, Any]:
    return json.loads(extract_json(text))


def summarize_prompt_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    total = max(1, len(cases))
    valid_json = sum(1 for case in cases if case["valid_json"])
    schema_valid = sum(1 for case in cases if case["schema_valid"])
    exact_decision = sum(1 for case in cases if case["exact_decision_match"])
    exact_action = sum(1 for case in cases if case["exact_action_match"])
    sane_amounts = sum(1 for case in cases if case["approved_amount_sane"])
    safe_actions = sum(1 for case in cases if case["safe_action"])
    budget_violations = sum(1 for case in cases if case["budget_violation"])

    return {
        "cases": len(cases),
        "valid_json_rate": round(valid_json / total, 3),
        "schema_valid_rate": round(schema_valid / total, 3),
        "exact_decision_match_rate": round(exact_decision / total, 3),
        "exact_action_match_rate": round(exact_action / total, 3),
        "approved_amount_sanity_rate": round(sane_amounts / total, 3),
        "safe_action_rate": round(safe_actions / total, 3),
        "budget_violation_count": budget_violations,
    }


def evaluate_prompt_holdout(
    model_name: str,
    adapter_dir: str | None,
    eval_records: list[dict[str, Any]],
    max_new_tokens: int,
) -> dict[str, Any]:
    model, tokenizer = load_inference_model(model_name, adapter_dir=adapter_dir)
    cases: list[dict[str, Any]] = []

    try:
        for index, record in enumerate(eval_records):
            prompt_messages = record["messages"][:-1]
            user_prompt = prompt_messages[-1]["content"]
            prompt_ctx = parse_prompt_context(user_prompt)
            target = safe_json_load(record["messages"][-1]["content"])

            raw = generate_response(model, tokenizer, prompt_messages, max_new_tokens=max_new_tokens)
            parsed = parse_action_output(raw)
            action = parsed["action"]

            exact_decision_match = bool(action and action["decision"] == target["decision"])
            exact_action_match = bool(
                action
                and exact_decision_match
                and abs(action["approved_amount"] - float(target["approved_amount"])) < 1e-6
            )

            cases.append(
                {
                    "index": index,
                    "expense_name": prompt_ctx["name"],
                    "category": prompt_ctx["category"],
                    "prompt_context": prompt_ctx,
                    "target": target,
                    "raw_output": raw,
                    "clean_output": parsed["clean"],
                    "valid_json": parsed["valid_json"],
                    "schema_valid": parsed["schema_valid"],
                    "predicted_action": action,
                    "parse_error": parsed["error"],
                    "exact_decision_match": exact_decision_match,
                    "exact_action_match": exact_action_match,
                    "approved_amount_sane": approved_amount_is_sane(action, prompt_ctx),
                    "safe_action": action_is_safe(action, prompt_ctx),
                    "budget_violation": budget_violation(action, prompt_ctx),
                }
            )
    finally:
        del model
        clear_memory()

    return {"summary": summarize_prompt_cases(cases), "cases": cases}


def summarize_env_results(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    total = max(1, len(episodes))
    return {
        "episodes": len(episodes),
        "avg_score": round(sum(ep["score"] for ep in episodes) / total, 3),
        "success_rate": round(sum(1 for ep in episodes if ep["success"]) / total, 3),
        "avg_total_reward": round(sum(ep["total_reward"] for ep in episodes) / total, 3),
        "avg_bad_decisions": round(sum(ep["bad_decisions"] for ep in episodes) / total, 3),
        "avg_fallback_count": round(sum(ep["fallback_count"] for ep in episodes) / total, 3),
        "total_fallback_count": sum(ep["fallback_count"] for ep in episodes),
        "total_parse_fail_count": sum(ep["parse_fail_count"] for ep in episodes),
    }


def run_env_episode(model, tokenizer, task_id: str, seed: int, max_new_tokens: int) -> dict[str, Any]:
    env = FinSenseEnv()
    obs = env.reset(task_id=task_id, seed=seed)
    done = False
    all_rewards: list[float] = []
    fallback_count = 0
    parse_fail_count = 0

    while not done:
        fallback_action = rule_based_agent(obs, memory=None, use_memory=False)
        prompt = build_prompt(obs)
        raw = generate_response(
            model,
            tokenizer,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            max_new_tokens=max_new_tokens,
        )
        parsed = parse_action_output(raw)

        if parsed["schema_valid"]:
            llm_action = ActionModel(**parsed["action"])
            action = select_final_action(fallback_action, llm_action, obs)
            if action is fallback_action and action is not llm_action:
                fallback_count += 1
        else:
            action = fallback_action
            fallback_count += 1
            parse_fail_count += 1

        obs, reward, done, _info = env.step(action)
        all_rewards.append(reward)

    return {
        "task_id": task_id,
        "seed": seed,
        "score": calculate_final_score(env, task_id),
        "success": obs.get("goal_remaining", 0) <= 0,
        "total_reward": round(sum(all_rewards), 3),
        "steps": len(all_rewards),
        "bad_decisions": env.get_bad_decision_count(),
        "fallback_count": fallback_count,
        "parse_fail_count": parse_fail_count,
        "final_balance": round(obs.get("balance", 0), 2),
        "goal_remaining": round(obs.get("goal_remaining", 0), 2),
        "stress": round(obs.get("stress_level", 0), 3),
        "process_metrics": env.get_process_metrics(),
    }


def evaluate_env_benchmark(
    model_name: str,
    adapter_dir: str | None,
    tasks: list[str],
    seeds: list[int],
    max_new_tokens: int,
) -> dict[str, Any]:
    model, tokenizer = load_inference_model(model_name, adapter_dir=adapter_dir)
    episodes: list[dict[str, Any]] = []

    try:
        for task_id in tasks:
            for seed in seeds:
                with contextlib.redirect_stdout(io.StringIO()):
                    episodes.append(
                        run_env_episode(
                            model=model,
                            tokenizer=tokenizer,
                            task_id=task_id,
                            seed=seed,
                            max_new_tokens=max_new_tokens,
                        )
                    )
    finally:
        del model
        clear_memory()

    per_task: dict[str, Any] = {}
    for task_id in tasks:
        task_episodes = [ep for ep in episodes if ep["task_id"] == task_id]
        per_task[task_id] = summarize_env_results(task_episodes)

    return {
        "summary": summarize_env_results(episodes),
        "per_task": per_task,
        "episodes": episodes,
    }


def print_prompt_summary_table(base_eval: dict[str, Any], sft_eval: dict[str, Any]) -> None:
    print_stage("Prompt Holdout Summary")
    headers = (
        "Model",
        "Valid JSON",
        "Schema Valid",
        "Decision Match",
        "Action Match",
        "Amount Sanity",
        "Safe Action",
        "Budget Violations",
    )
    print(
        f"{headers[0]:<16} {headers[1]:>12} {headers[2]:>13} {headers[3]:>16} "
        f"{headers[4]:>14} {headers[5]:>15} {headers[6]:>13} {headers[7]:>19}"
    )
    for label, payload in (("Base", base_eval["summary"]), ("SFT", sft_eval["summary"])):
        print(
            f"{label:<16} {payload['valid_json_rate']:>12.3f} {payload['schema_valid_rate']:>13.3f} "
            f"{payload['exact_decision_match_rate']:>16.3f} {payload['exact_action_match_rate']:>14.3f} "
            f"{payload['approved_amount_sanity_rate']:>15.3f} {payload['safe_action_rate']:>13.3f} "
            f"{payload['budget_violation_count']:>19}"
        )


def print_qualitative_examples(base_eval: dict[str, Any], sft_eval: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
    print_stage("Qualitative Before/After")
    comparisons: list[dict[str, Any]] = []
    for base_case, sft_case in zip(base_eval["cases"][:limit], sft_eval["cases"][:limit]):
        comparison = {
            "index": base_case["index"],
            "expense_name": base_case["expense_name"],
            "target": base_case["target"],
            "base_prediction": base_case["predicted_action"],
            "sft_prediction": sft_case["predicted_action"],
            "base_valid_json": base_case["valid_json"],
            "sft_valid_json": sft_case["valid_json"],
        }
        comparisons.append(comparison)

        print(f"Case {base_case['index']} | {base_case['expense_name']}")
        print(f"  Gold : {json.dumps(base_case['target'])}")
        print(f"  Base : {json.dumps(base_case['predicted_action']) if base_case['predicted_action'] else base_case['parse_error']}")
        print(f"  SFT  : {json.dumps(sft_case['predicted_action']) if sft_case['predicted_action'] else sft_case['parse_error']}")

    return comparisons


def print_env_summary_table(base_env: dict[str, Any], sft_env: dict[str, Any]) -> None:
    print_stage("Seeded Environment Benchmark")
    headers = (
        "Model",
        "Episodes",
        "Avg Score",
        "Success Rate",
        "Avg Reward",
        "Avg Bad Decisions",
        "Avg Fallbacks",
        "Parse Fails",
    )
    print(
        f"{headers[0]:<16} {headers[1]:>8} {headers[2]:>10} {headers[3]:>14} "
        f"{headers[4]:>11} {headers[5]:>19} {headers[6]:>14} {headers[7]:>12}"
    )
    for label, payload in (("Base", base_env["summary"]), ("SFT", sft_env["summary"])):
        print(
            f"{label:<16} {payload['episodes']:>8} {payload['avg_score']:>10.3f} "
            f"{payload['success_rate']:>14.3f} {payload['avg_total_reward']:>11.3f} "
            f"{payload['avg_bad_decisions']:>19.3f} {payload['avg_fallback_count']:>14.3f} "
            f"{payload['total_parse_fail_count']:>12}"
        )


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[ARTIFACT] Saved {path}")


def dependency_check() -> None:
    print_stage("Dependency Check")
    print(f"torch: {torch.__version__}")
    print(f"cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"gpu: {torch.cuda.get_device_name(0)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full FinSense Colab SFT demo.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Base model used for SFT and eval.")
    parser.add_argument("--data", default=os.path.join(os.path.dirname(__file__), "warmstart_data.jsonl"), help="Warm-start dataset path.")
    parser.add_argument("--output-dir", default="./checkpoints/sft-warmstart", help="LoRA adapter output directory.")
    parser.add_argument("--artifact-dir", default=".", help="Where to save judge JSON artifacts.")
    parser.add_argument("--epochs", type=float, default=3.0, help="Number of SFT epochs.")
    parser.add_argument("--batch-size", type=int, default=2, help="Per-device batch size.")
    parser.add_argument("--grad-accum", type=int, default=2, help="Gradient accumulation steps.")
    parser.add_argument("--learning-rate", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Maximum training sequence length.")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional total sample cap.")
    parser.add_argument("--eval-split", type=float, default=0.2, help="Holdout split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed.")
    parser.add_argument("--prompt-eval-cases", type=int, default=12, help="Holdout prompt cases to score.")
    parser.add_argument("--env-episodes", type=int, default=3, help="Episodes per task for env benchmark.")
    parser.add_argument("--max-new-tokens", type=int, default=96, help="Generation cap for each decision.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    dependency_check()

    artifact_dir = Path(args.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    splits = load_dataset_splits(
        data_path=args.data,
        eval_split=args.eval_split,
        seed=args.seed,
        max_samples=args.max_samples,
    )
    eval_count = min(args.prompt_eval_cases, len(splits["eval"]))
    eval_records = [splits["eval"][idx] for idx in range(eval_count)]
    env_seeds = [args.seed + 11 * idx for idx in range(args.env_episodes)]

    print_stage("Dataset Split")
    print(f"Train examples: {len(splits['train'])}")
    print(f"Eval examples: {len(splits['eval'])}")
    print(f"Prompt eval cases used: {len(eval_records)}")
    print(f"Env benchmark seeds: {env_seeds}")

    print_stage("Base Model Prompt Evaluation")
    base_prompt_eval = evaluate_prompt_holdout(
        model_name=args.model,
        adapter_dir=None,
        eval_records=eval_records,
        max_new_tokens=args.max_new_tokens,
    )

    print_stage("Training LoRA Adapter")
    train_metadata = train_sft(
        model_name=args.model,
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        eval_split=args.eval_split,
        seed=args.seed,
        max_samples=args.max_samples,
        dataset_splits=splits,
    )

    print_stage("SFT Prompt Evaluation")
    sft_prompt_eval = evaluate_prompt_holdout(
        model_name=args.model,
        adapter_dir=args.output_dir,
        eval_records=eval_records,
        max_new_tokens=args.max_new_tokens,
    )

    print_stage("Base Environment Benchmark")
    base_env_eval = evaluate_env_benchmark(
        model_name=args.model,
        adapter_dir=None,
        tasks=["easy", "medium"],
        seeds=env_seeds,
        max_new_tokens=args.max_new_tokens,
    )

    print_stage("SFT Environment Benchmark")
    sft_env_eval = evaluate_env_benchmark(
        model_name=args.model,
        adapter_dir=args.output_dir,
        tasks=["easy", "medium"],
        seeds=env_seeds,
        max_new_tokens=args.max_new_tokens,
    )

    print_prompt_summary_table(base_prompt_eval, sft_prompt_eval)
    qualitative_examples = print_qualitative_examples(base_prompt_eval, sft_prompt_eval)
    print_env_summary_table(base_env_eval, sft_env_eval)

    prompt_payload = {
        "model_name": args.model,
        "base": base_prompt_eval,
        "sft": sft_prompt_eval,
        "qualitative_examples": qualitative_examples,
    }
    env_payload = {
        "model_name": args.model,
        "tasks": ["easy", "medium"],
        "seeds": env_seeds,
        "base": base_env_eval,
        "sft": sft_env_eval,
    }
    summary_payload = {
        "model_name": args.model,
        "train_metadata": train_metadata,
        "prompt_summary": {
            "base": base_prompt_eval["summary"],
            "sft": sft_prompt_eval["summary"],
        },
        "env_summary": {
            "base": base_env_eval["summary"],
            "sft": sft_env_eval["summary"],
        },
    }

    save_json(artifact_dir / "judge_prompt_eval_before_after.json", prompt_payload)
    save_json(artifact_dir / "judge_env_eval_before_after.json", env_payload)
    save_json(artifact_dir / "judge_summary.json", summary_payload)


if __name__ == "__main__":
    main()
