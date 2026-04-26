"""
FinSense Personal Financial Assistant — Chat UI
================================================
A Gradio-based chat interface that guides users through purchase planning,
runs budget calculations, recommends products, and launches FinSense RL
simulations — all through a conversational interface.

Usage:
    python assistant_ui.py
    # Opens at http://localhost:7861
"""

import re
import gradio as gr

from personal_assistant import generate_questions, generate_summary_advice, extract_intent, generate_simulation_summary
from budget_calculator import (parse_salary, parse_months, parse_savings,
                                estimate_fixed_expenses, calculate_budget,
                                has_incomplete_emi, parse_emi_amount, parse_emi_months)
from product_recommender import recommend_products, get_product_category
from product_type_inference import infer_product_type
from dynamic_questions import get_dynamic_questions


# ── Session State ────────────────────────────────────────────────────────────

def initialize_state():
    return {
        "phase": "greeting",
        "product": None,
        "amount": None,
        "questions": [],
        "question_index": 0,
        "answers": {},
        "budget_result": None,
        "recommendations": None,
        "selected_product": None,
        "product_type_info": None,
        "emi_amount": 0.0,
        "emi_months": 0,
        "extended_months": 0,
    }


# ── FinSense Simulation Integration ─────────────────────────────────────────

def run_finsense_simulation(task_id, goal_amount, salary, fixed_expenses, budget_result=None):
    """
    Launch the existing FinSense RL environment with the correct task.
    Uses the trained LLM agent (inference.py) when API_BASE_URL is configured,
    falls back to rule_based_agent from inference_local.py if not.
    """
    try:
        from finsense.env import FinSenseEnv
        from finsense.models import StateModel, ActionModel
        from finsense.graders import grade_episode
        from finsense.tasks import TASKS
        from finsense.memory import MemorySystem
        from inference_local import rule_based_agent
        from inference import (load_config, get_fallback_action, build_prompt,
                               parse_llm_response, select_final_action, get_assistant_tip)

        # ── Decide whether to use LLM or rule-based ──
        config = load_config()
        llm_available = False
        client = None
        try:
            if config["API_BASE_URL"] and config["MODEL_NAME"] and not config["FORCE_RULE_BASED"]:
                from hf_wrapper import HFWrapper as OpenAI
                client = OpenAI(api_key=config["HF_TOKEN"], base_url=config["API_BASE_URL"])
                llm_available = True
        except Exception:
            pass

        def get_action(obs, memory):
            """Get action from LLM or rule-based fallback."""
            fallback = get_fallback_action(obs, memory=memory, use_memory=True)
            if not llm_available or client is None:
                return fallback
            try:
                from finsense.assistant_layer import get_assistant_tip
                assistant_tip = get_assistant_tip(obs, memory=memory)
                exp = obs.get("current_expense") or {}
                expense_type = exp.get("category", "unknown")
                context = exp.get("context", "normal")
                active_events = obs.get("active_events", [])
                event_type = active_events[0] if active_events else "none"
                necessity = exp.get("necessity_tag", "unknown")
                memory_bias, memory_confidence = memory.get_memory_bias_with_confidence(
                    expense_type, context, event_type, necessity=necessity
                )
                # Skip LLM if memory is very confident
                if memory_bias and memory_confidence >= 0.65:
                    amount = float(exp.get("amount", 0))
                    if memory_bias == "allow":
                        return ActionModel(decision="allow", approved_amount=amount, reasoning="Memory override")
                    elif memory_bias == "reduce":
                        return ActionModel(decision="reduce", approved_amount=round(amount * 0.5, 2), reasoning="Memory override")
                    else:
                        return ActionModel(decision="avoid", approved_amount=0.0, reasoning="Memory override")

                response = client.chat.completions.create(
                    model=config["MODEL_NAME"],
                    messages=[
                        {"role": "system", "content": "You are a financial agent. Reply only with valid JSON."},
                        {"role": "user", "content": build_prompt(obs, memory_bias=memory_bias,
                                                                  memory_confidence=memory_confidence,
                                                                  assistant_tip=assistant_tip)}
                    ],
                    max_tokens=100,
                    temperature=0.1
                )
                raw = response.choices[0].message.content.strip()
                llm_action = parse_llm_response(raw, obs)
                return select_final_action(fallback, llm_action, obs)
            except Exception as e:
                print(f"[SIM LLM] fallback due to: {e}")
                return fallback

        env = FinSenseEnv()
        memory = MemorySystem()
        env.memory = memory

        obs = env.reset(task_id=task_id)
        
        # ── Personalize the simulation state ──
        # The simulation runs for `task_days` days, but the user's plan spans
        # `months_needed` months. We compress the FULL timeline into the sim window.
        #
        # Key insight: The simulation is about daily expense discipline.
        # We need to set balance/goal so the RATIO matches the user's real plan.
        #
        # User's real plan:
        #   total_income = salary * months
        #   total_fixed  = fixed_expenses * months
        #   surplus      = total_income - total_fixed  (this is what they can save)
        #   goal         = goal_amount
        #
        # We compress this into task_days by maintaining the same surplus/goal ratio.
        task_days = TASKS[task_id].days
        task_defaults = TASKS[task_id]
        
        # Get months from budget result (passed via state)
        # The budget result's months_needed tells us the actual timeline
        if budget_result is None:
            budget_result = {}
        months = max(1, budget_result.get("months_needed", 6))
        
        # Total resources over the full plan
        total_income = salary * months
        total_fixed = fixed_expenses * months
        total_surplus = total_income - total_fixed
        
        # Scale everything into the simulation window proportionally
        # so the daily pressure matches the real plan's difficulty
        sim_scale = task_days / (months * 30.0)  # compress full plan into sim days
        
        sim_balance = total_income * sim_scale
        sim_goal = goal_amount * sim_scale
        sim_fixed = total_fixed * sim_scale
        
        # Safety: ensure the balance/goal ratio matches the ORIGINAL task's ratio
        # The original tasks are calibrated with specific ratios:
        #   easy:   30000/5000  = 6.0x   (comfortable)
        #   medium: 60000/15000 = 4.0x   (manageable)
        #   hard:   80000/30000 = 2.67x  (challenging)
        # When extreme timeline compression (e.g. 182 months → 30 days) degrades
        # this ratio (e.g. to 1.82x), the simulation becomes harder than intended.
        # We fix this by boosting sim_balance to preserve the original difficulty.
        original_ratio = task_defaults.initial_balance / max(1, task_defaults.goal)
        current_ratio = sim_balance / max(1, sim_goal)
        
        if task_id != "hard" and current_ratio < original_ratio:
            sim_balance = sim_goal * original_ratio
            sim_fixed = sim_balance * (fixed_expenses / max(1, salary))
        
        env.state["balance"] = round(sim_balance, 0)
        env.state["goal_total"] = round(sim_goal, 0)
        env.state["goal_remaining"] = round(sim_goal, 0)
        env.state["salary"] = round(sim_balance, 0)
        env.state["expected_fixed_expenses"] = round(sim_fixed, 0)
        
        # ── Scale expenses to match the compressed simulation ──
        # CRITICAL FIX: The expense generator produces amounts at absolute scale
        # (e.g. rent Rs.8K-25K) but our sim_balance may only be Rs.20K total.
        # We need to scale expense amounts so daily expense pressure is proportional
        # to the compressed budget, matching the original task's difficulty ratio.
        #
        # Use SPENDABLE budget (balance - goal) since expenses compete with the
        # discretionary portion, not the savings target.
        # Original easy task: spendable = 30000 - 5000 = 25000 over 15 days
        # Our sim:            spendable = sim_balance - sim_goal over task_days
        original_spendable = (task_defaults.initial_balance - task_defaults.goal) / task_days
        sim_spendable = max(1.0, (sim_balance - sim_goal)) / task_days
        expense_scale = sim_spendable / max(1.0, original_spendable)
        # Clamp to avoid extreme scaling (keep between 0.1x and 5x)
        expense_scale = max(0.1, min(5.0, expense_scale))
        
        # Scale the already-generated first batch of daily expenses
        for exp in env.daily_expenses:
            exp.amount = round(exp.amount * expense_scale, 2)
        
        # Monkey-patch the expense generator so ALL future daily expenses
        # are also scaled proportionally throughout the simulation
        original_generate = env.expense_gen.generate_daily_expenses
        def scaled_generate(num_expenses=3):
            expenses = original_generate(num_expenses)
            for exp in expenses:
                exp.amount = round(exp.amount * expense_scale, 2)
            return expenses
        env.expense_gen.generate_daily_expenses = scaled_generate
        
        # Recompute daily allowance with new values
        env._recompute_daily_allowance()
        
        # Re-sync observation for the agent
        obs = env._get_observation()

        total_reward = 0
        steps = 0
        done = False
        step_log = []

        while not done and steps < 200:
            action = get_action(obs, memory)
            obs, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            step_log.append({
                "step": steps,
                "action": action.decision,
                "reward": round(reward, 3),
            })

        # Score using the existing grading system
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
            score = max(0.01, min(0.99, grade_episode(state_model)))
        except Exception:
            goal_total = s["goal_total"]
            goal_remaining = s["goal_remaining"]
            progress = max(0.0, goal_total - goal_remaining)
            score = max(0.01, min(0.99, progress / max(1.0, goal_total)))

        return {
            "success": s.get("goal_remaining", 1) <= 0,
            "score": score,
            "total_reward": round(total_reward, 2),
            "steps": steps,
            "final_balance": round(s.get("balance", 0), 0),
            "stress_level": round(s.get("stress_level", 0), 2),
            "bad_decisions": env.get_bad_decision_count(),
            "bad_decision_list": env.get_bad_decisions(),
            "step_log": step_log[-5:],
        }
    except Exception as e:
        return {"error": str(e), "success": False}


# ── Formatting Helpers ───────────────────────────────────────────────────────

def format_budget_result(result, product_name, selected_price):
    """Format the budget assessment as a readable block."""
    task_labels = {
        "easy": "✅ Comfortable",
        "medium": "⚠️ Manageable - stay disciplined",
        "hard": "🚨 Challenging - requires replanning",
    }
    
    lines = [
        f"### Financial Assessment for {product_name}",
        "",
        "| Metric | Assessment |",
        "|--------|------------|",
        f"| **Monthly Surplus** | Rs. {result['monthly_surplus']:,.0f} |",
        f"| **Saveable ({result['months_needed']} mo)** | Rs. {result['total_saveable']:,.0f} |",
        f"| **Product Price** | Rs. {selected_price:,} |",
        f"| **Remaining Balance**| Rs. {result['remaining_balance']:,.0f} |",
        "",
        f"**Difficulty:** {task_labels[result['task']]}",
        f"**Stress Projection:** {result['stress_projection'].upper()}",
    ]
    if result["task"] == "hard":
        lines += [
            "",
            "### 📉 Reality Check (Revised Plan)",
            f"- **Revised timeline:** {result['months_needed']} months",
            f"- **Estimated Loan Interest:** ~Rs.{result['interest_if_loan']:,.0f}",
            f"- **Monthly EMI:** Rs.{result['monthly_emi']:,.0f}",
            "> **Warning:** Your savings will be near zero. High financial stress expected. Proceed with caution.",
        ]
    return "\n".join(lines)


def format_simulation_result(sim_result, selected_product, summary_text=None):
    """Format the simulation result as a readable block."""
    if "error" in sim_result:
        return f"Simulation error: {sim_result['error']}"

    status = "Goal Achieved!" if sim_result["success"] else "Goal Not Reached"
    emoji = "🎉" if sim_result["success"] else "📊"

    score = sim_result["score"]
    if score >= 0.8:
        grade = "A — Excellent"
    elif score >= 0.6:
        grade = "B — Good"
    elif score >= 0.4:
        grade = "C — Needs Improvement"
    else:
        grade = "D — Challenging"

    lines = [
        f"## {emoji} Simulation Complete - {status}",
        "",
        f"**Product:** {selected_product['name']} (Rs.{selected_product['price']:,})",
        "",
        f"| Metric | Result |",
        f"|--------|--------|",
        f"| Financial Discipline Score | **{score:.2f}** / 1.00 (Grade: {grade}) |",
        f"| Simulation Days | {sim_result['steps']} |",
        f"| Final Balance | Rs.{sim_result['final_balance']:,.0f} |",
        f"| Stress Level | {sim_result['stress_level']:.2f} |",
        f"| Risky Decisions | {sim_result['bad_decisions']} |",
        "",
    ]

    if sim_result["success"]:
        lines.append("✅ **Your disciplined approach paid off!** The simulation shows you can reach your savings goal with careful daily expense management.")
    else:
        lines.append("⚠️ **The simulation suggests this goal is very challenging** with your current financial parameters. Consider extending your timeline, increasing income, or choosing a more affordable option.")

    if summary_text:
        lines.append("")
        lines.append(f"### 🤖 AI Agent Summary")
        lines.append(f">{summary_text}")

    lines.append("\n*Type anything to plan another purchase!*")

    return "\n".join(lines)


# ── Chat Engine ──────────────────────────────────────────────────────────────

def chat(user_message, history, state):
    """Main chat function — handles all conversation phases."""
    if state is None:
        state = initialize_state()

    history = history or []
    response = ""

    # ── PHASE: GREETING ──────────────────────────────────────────────────
    if state["phase"] == "greeting":
        
        # If we already have a partial state (e.g. got product but waiting for price)
        if state["product"] and not state["amount"]:
            # Use the robust parser instead of basic regex
            from budget_calculator import extract_amount
            val = extract_amount(user_message)
            if val > 0:
                state["amount"] = val
        elif state["amount"] and not state["product"]:
            state["product"] = user_message.strip()
        else:
            # First interaction: use LLM parser
            intent = extract_intent(user_message)
            if intent["product"] and intent["product"].lower() != "unknown":
                state["product"] = intent["product"]
            if intent["amount"]:
                state["amount"] = intent["amount"]

        # Check what's missing and prompt the user gracefully
        if state["product"] and not state["amount"]:
            response = f"Great! Planning for a **{state['product']}** is exciting. What is your approximate budget for it?"
        elif state["amount"] and not state["product"]:
            response = f"I see a budget of **Rs. {state['amount']:,.0f}**. What product are you planning to buy with this?"
        elif not state["product"] and not state["amount"]:
            response = (
                "I'd be happy to help you plan your purchase! "
                "Please tell me what you'd like to buy and the approximate price.\n\n"
                "For example:\n"
                "- *I want to buy a car worth 15 Lakhs*\n"
                "- *I want to buy an iPhone worth Rs.80,000*"
            )
        else:
            state["phase"] = "questioning"
            
            # Infer product type for type-aware questions
            pt_info = infer_product_type(state["product"])
            state["product_type_info"] = pt_info
            
            # Use type-aware questions for unknown categories, LLM for known
            cat = get_product_category(state["product"])
            if cat:
                questions = generate_questions(state["product"], state["amount"])
            else:
                questions = get_dynamic_questions(
                    state["product"], state["amount"], pt_info["broad_type"]
                )
            state["questions"] = questions
            state["question_index"] = 0

            catalog_note = ""
            if not cat:
                catalog_note = (
                    f"\n\n*I don't have a fixed product catalog for "
                    f"**{state['product']}**, but I can still analyze your "
                    f"affordability and generate realistic options based on "
                    f"your financial profile.*"
                )

            response = (
                f"Great! Planning for a **{state['product']}** worth "
                f"**Rs.{state['amount']:,.0f}**. Let me ask you a few questions "
                f"to understand your financial situation better.{catalog_note}\n\n"
                f"**Question 1 of {len(questions)}:**\n{questions[0]['question']}"
            )

    # ── PHASE: QUESTIONING ───────────────────────────────────────────────
    elif state["phase"] == "questioning":
        current_q_obj = state["questions"][state["question_index"]]
        q_type = current_q_obj["type"]
        state["answers"][q_type] = user_message
        state["question_index"] += 1

        if state["question_index"] < len(state["questions"]):
            next_q_obj = state["questions"][state["question_index"]]
            next_q_text = next_q_obj["question"]
            q_num = state["question_index"] + 1
            total = len(state["questions"])
            
            # Check if the previous answer was an incomplete EMI mention
            if q_type == "emi" and has_incomplete_emi(user_message):
                state["phase"] = "emi_followup"
                state["question_index"] -= 1  # stay on this question
                response = (
                    "I see you have existing EMIs/loans. To calculate your budget accurately, "
                    "I need a bit more detail:\n\n"
                    "**What is your monthly EMI amount?** (e.g., Rs.15,000)"
                )
            else:
                response = (
                    f"Got it!\n\n"
                    f"**Question {q_num} of {total}:**\n{next_q_text}"
                )
        else:
            # All questions answered — run budget calculation
            state["phase"] = "recommending"

            # Find answers by structured type
            salary_answer = state["answers"].get("salary", "50000")
            timeline_answer = state["answers"].get("timeline", "6 months")
            emi_answer = state["answers"].get("emi", "no").lower()
            
            all_answers_text = " ".join(state["answers"].values()).lower()

            salary = parse_salary(salary_answer) or 50000.0
            months = parse_months(timeline_answer)
            has_emi = any(w in emi_answer or w in all_answers_text for w in ["emi", "loan", "yes", "running"])
            
            # Use exact EMI amount if collected during follow-up
            emi_amount = state.get("emi_amount", 0.0)
            if emi_amount == 0.0 and has_emi:
                emi_amount = parse_emi_amount(emi_answer)

            fixed = estimate_fixed_expenses(salary, has_emi, emi_amount)
            savings = salary * 0.1  # assume 10% existing savings

            result = calculate_budget(salary, fixed, savings, state["amount"], months)
            state["budget_result"] = result

            recs = recommend_products(state["product"], result["total_saveable"],
                                      target_budget=state["amount"])
            state["recommendations"] = recs
            
            source = recs.get("source", "unknown")
            is_synthetic = source == "synthetic_fallback"

            # Build response
            response_parts = [
                "Thanks for sharing! Here's what I've worked out:\n",
                f"**Your monthly surplus:** Rs.{result['monthly_surplus']:,.0f}",
                f"**Saveable in {months} months:** Rs.{result['total_saveable']:,.0f}",
            ]
            
            if emi_amount > 0:
                response_parts.append(f"**EMI deducted:** Rs.{emi_amount:,.0f}/month")
            
            response_parts.append("")
            
            # Affordability assessment
            gap = recs.get("affordability_gap", 0)
            
            # ── Extended timeline calculation for the EXACT desired product ──
            # If the product is out of budget, tell the user how long it would
            # ACTUALLY take to save for it — empowering rather than restricting.
            extended_months_needed = 0
            if gap > 0 and result["monthly_surplus"] > 0:
                extended_months_needed = int(
                    (state["amount"] - savings) / result["monthly_surplus"]
                ) + 1  # +1 to round up
                # Ensure it's at least more than current timeline
                if extended_months_needed <= months:
                    extended_months_needed = months + 1

            if gap > 0:
                response_parts.append(
                    f"> **Note:** Your target of Rs.{state['amount']:,.0f} is above your "
                    f"achievable budget in {months} months by Rs.{gap:,.0f}."
                )
                # ── KEY IMPROVEMENT: Show the path to the exact product ──
                if extended_months_needed > 0:
                    years_part = extended_months_needed // 12
                    months_part = extended_months_needed % 12
                    if years_part > 0 and months_part > 0:
                        time_str = f"{years_part} year{'s' if years_part > 1 else ''} and {months_part} month{'s' if months_part > 1 else ''}"
                    elif years_part > 0:
                        time_str = f"{years_part} year{'s' if years_part > 1 else ''}"
                    else:
                        time_str = f"{extended_months_needed} months"

                    response_parts.append("")
                    response_parts.append(
                        f"### 🎯 Want the exact {state['product']} (Rs.{state['amount']:,.0f})?"
                    )
                    response_parts.append(
                        f"If you extend your timeline to **{time_str}** (~{extended_months_needed} months), "
                        f"you can save enough to buy your exact desired product!"
                    )
                    response_parts.append(
                        f"- **Monthly savings:** Rs.{result['monthly_surplus']:,.0f}"
                    )
                    response_parts.append(
                        f"- **Total saved in {extended_months_needed} months:** "
                        f"Rs.{savings + (result['monthly_surplus'] * extended_months_needed):,.0f}"
                    )
                    response_parts.append(
                        f"\n> 💡 **Type `extend`** to go with the extended {time_str} plan "
                        f"for your **{state['product']}** (Rs.{state['amount']:,.0f}), "
                        f"OR pick a more affordable option below."
                    )
                    # Store extended timeline info in state for later use
                    state["extended_months"] = extended_months_needed
            
            if is_synthetic:
                response_parts.append(
                    f"\n> *I don't have a fixed catalog for **{state['product']}**, "
                    f"so I've generated realistic options based on your financial profile.*"
                )

            if recs["within_budget"]:
                label = "Options within your budget" if not is_synthetic else "Budget-Friendly Options"
                response_parts.append(f"### {label}")
                for i, p in enumerate(recs["within_budget"], 1):
                    line = f"**{i}.** {p['name']} - Rs.{p['price']:,}"
                    if p.get("is_synthetic"):
                        line += f" *(estimated)*"
                    response_parts.append(line)
            
            if recs["stretch"]:
                response_parts.append("\n### Stretch options (slightly above)")
                for p in recs["stretch"]:
                    line = f"- {p['name']} - Rs.{p['price']:,}"
                    if p.get("is_synthetic"):
                        line += f" *(estimated)*"
                    response_parts.append(line)

            if not recs["within_budget"] and not recs["stretch"]:
                if is_synthetic:
                    response_parts.append(
                        f"\n> Your current budget doesn't cover even the basic "
                        f"option for a **{state['product']}**. Consider extending "
                        f"your timeline or increasing monthly savings."
                    )
                else:
                    response_parts.append(
                        "\n> **No options found** within your current budget and timeline."
                    )
                if recs.get("above_budget"):
                    response_parts.append("### Aspirational options (requires more saving)")
                    for p in recs["above_budget"][:2]:
                        line = f"- {p['name']} - Rs.{p['price']:,}"
                        if p.get("is_synthetic"):
                            line += f" *(estimated)*"
                        response_parts.append(line)
            
            # Show all synthetic options if nothing fit budget
            if is_synthetic and not recs["within_budget"] and recs.get("all_options"):
                response_parts.append("\n### All Generated Options")
                for i, p in enumerate(recs["all_options"], 1):
                    response_parts.append(
                        f"**{i}.** {p['name']} - Rs.{p['price']:,} *({p['tier']})*"
                    )

            if result["task"] == "hard":
                response_parts.append(
                    f"\n> **Note:** Your original target of Rs.{state['amount']:,.0f} "
                    f"exceeds your {months}-month budget by "
                    f"Rs.{result['shortfall']:,.0f}."
                )

            if gap > 0 and extended_months_needed > 0:
                response_parts.append(
                    "\n**Which product would you like to go with?** "
                    "Type the name/number, or type **`extend`** to save for your original choice with an extended timeline."
                )
            else:
                response_parts.append(
                    "\n**Which product would you like to go with?** "
                    "Type the name/number, or type **`original`** to go with your original choice."
                )

            response = "\n".join(response_parts)

    # ── PHASE: EMI FOLLOW-UP ─────────────────────────────────────────────
    elif state["phase"] == "emi_followup":
        emi_amount = parse_emi_amount(user_message)
        if emi_amount > 0:
            state["emi_amount"] = emi_amount
            state["answers"]["emi"] = f"yes, Rs.{emi_amount:,.0f}/month"
            state["phase"] = "emi_followup_months"
            response = (
                f"Got it — **Rs.{emi_amount:,.0f}/month** in EMIs.\n\n"
                f"**How many months are remaining on this EMI?** "
                f"(e.g., 12 months, 2 years)"
            )
        else:
            response = (
                "I couldn't extract the EMI amount. Please enter just the "
                "monthly amount, e.g., **15000** or **Rs.20,000**."
            )

    elif state["phase"] == "emi_followup_months":
        emi_months = parse_emi_months(user_message)
        state["emi_months"] = emi_months
        state["phase"] = "questioning"
        state["question_index"] += 1  # move past the EMI question
        
        if state["question_index"] < len(state["questions"]):
            next_q = state["questions"][state["question_index"]]
            q_num = state["question_index"] + 1
            total = len(state["questions"])
            months_note = f" ({emi_months} months remaining)" if emi_months > 0 else ""
            response = (
                f"Noted{months_note}. Let's continue.\n\n"
                f"**Question {q_num} of {total}:**\n{next_q['question']}"
            )
        else:
            # Force recalculation
            state["phase"] = "questioning"
            # Trigger the "all questions answered" path by simulating the next step
            # We'll let the next user message trigger the recommending phase
            response = "Thanks! Type **done** to see your financial assessment."

    # ── PHASE: RECOMMENDING (user picks product) ─────────────────────────
    elif state["phase"] == "recommending":
        msg_lower = user_message.lower().strip()

        # ── Handle "extend" / "original" — user wants the exact product ──
        if msg_lower in ("extend", "extend timeline", "original", "same product", "original choice"):
            selected = {"name": state["product"], "price": int(state["amount"])}
            state["selected_product"] = selected

            # Recalculate budget — with extended timeline if needed, or current timeline
            salary_answer = state["answers"].get("salary", "50000")
            salary = parse_salary(salary_answer) or 50000.0

            emi_answer = state["answers"].get("emi", "no").lower()
            all_answers_text = " ".join(state["answers"].values()).lower()
            has_emi = any(w in emi_answer or w in all_answers_text for w in ["emi", "loan", "yes", "running"])

            emi_amount = state.get("emi_amount", 0.0)
            if emi_amount == 0.0 and has_emi:
                emi_amount = parse_emi_amount(emi_answer)

            fixed = estimate_fixed_expenses(salary, has_emi, emi_amount)
            savings = salary * 0.1

            if state.get("extended_months"):
                # Budget gap exists — use extended timeline
                use_months = state["extended_months"]
                state["answers"]["timeline"] = f"{use_months} months"
            else:
                # Original product already within budget — use current timeline
                timeline_answer = state["answers"].get("timeline", "6 months")
                use_months = parse_months(timeline_answer)

            result = calculate_budget(salary, fixed, savings, state["amount"], use_months)
            state["budget_result"] = result

            assessment = format_budget_result(result, state["product"], int(state["amount"]))

            advice = generate_summary_advice(
                state["product"], state["amount"],
                result, selected
            )

            state["phase"] = "confirmed"

            if state.get("extended_months"):
                extended_months = state["extended_months"]
                years_part = extended_months // 12
                months_part = extended_months % 12
                if years_part > 0 and months_part > 0:
                    time_str = f"{years_part} year{'s' if years_part > 1 else ''} and {months_part} month{'s' if months_part > 1 else ''}"
                elif years_part > 0:
                    time_str = f"{years_part} year{'s' if years_part > 1 else ''}"
                else:
                    time_str = f"{extended_months} months"
                response = (
                    f"Great choice! 🎯 Let's plan for your **{state['product']}** "
                    f"(Rs.{state['amount']:,.0f}) with an extended timeline of **{time_str}**.\n\n"
                    f"{assessment}\n\n{advice}\n\n"
                    f"Ready to start your saving simulation? Type **start** "
                    f"to launch the FinSense RL environment with your goal."
                )
            else:
                response = (
                    f"Great choice! 🎯 Let's plan for your **{state['product']}** "
                    f"(Rs.{state['amount']:,.0f}).\n\n"
                    f"{assessment}\n\n{advice}\n\n"
                    f"Ready to start your saving simulation? Type **start** "
                    f"to launch the FinSense RL environment with your goal."
                )

        else:
            # ── Normal product selection flow ──
            selected = None
            recs = state["recommendations"]
            all_options = (
                recs["within_budget"] +
                recs["stretch"] +
                recs.get("above_budget", [])
            )
            # Also include synthetic all_options if available
            synthetic_all = recs.get("all_options", [])
            combined = all_options if all_options else synthetic_all

            # 1. Match by exact name first
            for p in combined:
                if p["name"].lower() in msg_lower:
                    selected = p
                    break

            # 2. Match by standalone number or short index selection
            if not selected:
                # Look for standalone digits (e.g., "1", "option 2")
                nums = re.findall(r'\b\d+\b', user_message)
                if nums:
                    # Pick the first valid digit that could be an index
                    for num_str in nums:
                        idx = int(num_str) - 1
                        pick_from = recs["within_budget"] if recs["within_budget"] else combined
                        if 0 <= idx < len(pick_from):
                            selected = pick_from[idx]
                            break
                    
                    # If still not selected, check if it was a custom price
                    if not selected:
                        price_val = float(nums[0].replace(',', ''))
                        if price_val > 10000:
                            selected = {"name": "Custom selection", "price": int(price_val)}

            # 3. Match by loose words as a final fallback
            if not selected:
                for p in combined:
                    if any(word.lower() in msg_lower
                           for word in p["name"].split() if len(word) > 2):
                        selected = p
                        break

            if not selected:
                response = (
                    "I didn't catch that. Please type the product name, "
                    "its number from the list, or **`extend`** to save for your original choice."
                )
            else:
                state["selected_product"] = selected
                result = state["budget_result"]

                # Recalculate for the selected price
                if selected["price"] != state["amount"]:
                    salary_answer = state["answers"].get("salary", "50000")
                    timeline_answer = state["answers"].get("timeline", "6")
                    
                    salary = parse_salary(salary_answer) or 50000.0
                    months = parse_months(timeline_answer)
                    
                    emi_answer = state["answers"].get("emi", "no").lower()
                    has_emi = "emi" in emi_answer or "loan" in emi_answer
                    
                    # Extraction logic for savings: look for "savings" or "funds" in answers
                    raw_savings = "0"
                    for key, val in state["answers"].items():
                        if any(x in key.lower() for x in ["savings", "funds", "allocated", "initial"]):
                            raw_savings = val
                            break
                    
                    user_savings = parse_savings(raw_savings)
                    # Fallback to 10% of salary if no savings found
                    if user_savings <= 0:
                        user_savings = salary * 0.1

                    fixed = estimate_fixed_expenses(salary, has_emi,
                                                     state.get("emi_amount", 0.0))
                    result = calculate_budget(
                        salary, fixed, user_savings, selected["price"], months
                    )
                    state["budget_result"] = result

                assessment = format_budget_result(
                    result, selected["name"], selected["price"]
                )

                if result["task"] == "hard":
                    state["phase"] = "confirmed_hard"
                    response = (
                        f"{assessment}\n\n"
                        f"This is going to be a tough journey financially. "
                        f"Would you still like to proceed with **{selected['name']}** "
                        f"on this extended timeline? (yes / no)"
                    )
                else:
                    state["phase"] = "confirmed"
                    advice = generate_summary_advice(
                        state["product"], state["amount"],
                        result, selected
                    )
                    response = (
                        f"{assessment}\n\n{advice}\n\n"
                        f"Ready to start your saving simulation? Type **start** "
                        f"to launch the FinSense RL environment with your goal."
                    )

    # ── PHASE: HARD TASK CONFIRMATION ────────────────────────────────────
    elif state["phase"] == "confirmed_hard":
        if "yes" in user_message.lower():
            state["phase"] = "confirmed"
            result = state["budget_result"]
            advice = generate_summary_advice(
                state["product"], state["amount"],
                result, state["selected_product"]
            )
            response = (
                f"{advice}\n\n"
                f"This will be a **Hard** simulation - high stress, "
                f"tight balance, and very little room for error. "
                f"Type **start** to begin."
            )
        else:
            state["phase"] = "recommending"
            response = (
                "No problem! Let's look at more affordable options. "
                "Which product from the list would you prefer?"
            )

    # ── PHASE: CONFIRMED — launch simulation ─────────────────────────────
    elif state["phase"] == "confirmed":
        if "start" in user_message.lower():
            state["phase"] = "simulating"
            result = state["budget_result"]
            selected = state["selected_product"]

            sim_result = run_finsense_simulation(
                task_id=result["task"],
                goal_amount=selected["price"],
                salary=result["monthly_surplus"] + result["fixed_expenses"],
                fixed_expenses=result["fixed_expenses"],
                budget_result=result
            )

            summary_text = generate_simulation_summary(sim_result, selected)
            response = format_simulation_result(sim_result, selected, summary_text=summary_text)
        else:
            response = "Type **start** when you're ready to begin the simulation!"

    # ── PHASE: SIMULATING — done, restart ────────────────────────────────
    elif state["phase"] == "simulating":
        state = initialize_state()
        response = (
            "Let's plan another purchase! Tell me what you'd like to buy "
            "and the approximate price."
        )

    else:
        state = initialize_state()
        response = (
            "Welcome back! Tell me what you'd like to buy "
            "and the approximate price."
        )

    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": response})
    return history, state


# ── Gradio UI ────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Global ── */
.gradio-container {
    max-width: 1100px !important;
    margin: 0 auto !important;
    font-family: 'Inter', system-ui, -apple-system, sans-serif !important;
}

/* ── Header ── */
#app-header {
    text-align: center;
    padding: 28px 0 8px;
    background: linear-gradient(135deg, #065f46 0%, #047857 50%, #059669 100%);
    border-radius: 16px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}
#app-header::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(255,255,255,0.06) 0%, transparent 60%);
    animation: shimmer 8s ease-in-out infinite;
}
@keyframes shimmer {
    0%, 100% { transform: translate(0, 0); }
    50% { transform: translate(10%, 5%); }
}
#app-header h1 {
    font-size: 32px;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
    letter-spacing: -0.5px;
    position: relative;
    z-index: 1;
}
#app-header p {
    color: #a7f3d0;
    font-size: 14px;
    margin: 6px 0 16px;
    font-weight: 400;
    position: relative;
    z-index: 1;
}

/* ── Chatbot ── */
.chatbot-container {
    border-radius: 16px !important;
    border: 1px solid #e5e7eb !important;
    box-shadow: 0 4px 24px rgba(0,0,0,0.06) !important;
}

/* ── Sidebar ── */
.sidebar-card {
    background: linear-gradient(135deg, #f0fdf4 0%, #ecfdf5 50%, #f0f9ff 100%);
    border: 1px solid #bbf7d0;
    border-radius: 16px;
    padding: 24px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04);
}
.sidebar-card h3 {
    font-size: 15px;
    font-weight: 700;
    color: #065f46;
    margin: 0 0 16px;
    letter-spacing: -0.3px;
}
.sidebar-step {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    margin-bottom: 14px;
}
.step-num {
    width: 26px;
    height: 26px;
    background: #059669;
    color: white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 12px;
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 1px;
}
.step-text {
    font-size: 13px;
    color: #374151;
    line-height: 1.5;
}
.difficulty-section {
    margin-top: 20px;
    padding-top: 16px;
    border-top: 1px solid #d1fae5;
}
.difficulty-item {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 8px;
    font-size: 12px;
    color: #374151;
}
.diff-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    flex-shrink: 0;
}
.diff-easy { background: #16a34a; }
.diff-medium { background: #ea580c; }
.diff-hard { background: #dc2626; }

/* ── Input ── */
.input-row {
    margin-top: 8px;
}
.input-row .textbox {
    border-radius: 12px !important;
}

/* ── Send button ── */
.send-btn {
    border-radius: 12px !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    transition: all 0.2s ease !important;
}
.send-btn:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(5, 150, 105, 0.3) !important;
}

/* ── Category badges ── */
.category-badges {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 12px;
}
.badge {
    background: white;
    border: 1px solid #d1fae5;
    border-radius: 20px;
    padding: 4px 12px;
    font-size: 11px;
    color: #047857;
    font-weight: 500;
}
"""


WELCOME_MESSAGE = (
    "Hello! I'm your **FinSense Financial Assistant** 🏦\n\n"
    "I help you plan major purchases by analyzing your finances, "
    "recommending products within your budget, and running a savings simulation.\n\n"
    "**Tell me what you'd like to buy and your budget.** For example:\n"
    "- *I want to buy a car worth Rs.15,00,000*\n"
    "- *I want to buy an iPhone worth Rs.80,000*\n"
    "- *I want to buy a bike worth Rs.1,50,000*\n"
    "- *I want to buy a laptop worth Rs.70,000*"
)


def build_ui():
    with gr.Blocks(
        title="FinSense Personal Assistant",
    ) as demo:

        # ── Header ──
        gr.HTML("""
        <div id="app-header">
            <h1>FinSense Personal Assistant</h1>
            <p>Plan your major purchases with AI-powered financial simulation</p>
        </div>
        """)

        state = gr.State(initialize_state())

        with gr.Row():
            # ── Chat Panel ──
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="",
                    height=520,
                    show_label=False,
                    value=[{"role": "assistant", "content": WELCOME_MESSAGE}],
                )

                with gr.Row(elem_classes="input-row"):
                    msg_input = gr.Textbox(
                        placeholder="Type your message here...",
                        show_label=False,
                        scale=5,
                        container=False,
                        autofocus=True,
                    )
                    send_btn = gr.Button(
                        "Send",
                        variant="primary",
                        scale=1,
                        elem_classes="send-btn",
                    )

            # ── Sidebar ──
            with gr.Column(scale=1):
                gr.HTML("""
                <div class="sidebar-card">
                    <h3>How it works</h3>

                    <div class="sidebar-step">
                        <div class="step-num">1</div>
                        <div class="step-text">Tell me what you want to buy</div>
                    </div>
                    <div class="sidebar-step">
                        <div class="step-num">2</div>
                        <div class="step-text">Answer 4-5 finance questions</div>
                    </div>
                    <div class="sidebar-step">
                        <div class="step-num">3</div>
                        <div class="step-text">See product recommendations</div>
                    </div>
                    <div class="sidebar-step">
                        <div class="step-num">4</div>
                        <div class="step-text">Pick your product</div>
                    </div>
                    <div class="sidebar-step">
                        <div class="step-num">5</div>
                        <div class="step-text">Run FinSense simulation</div>
                    </div>
                    <div class="sidebar-step">
                        <div class="step-num">6</div>
                        <div class="step-text">See your financial score</div>
                    </div>

                    <div class="difficulty-section">
                        <h3>Task difficulty</h3>
                        <div class="difficulty-item">
                            <div class="diff-dot diff-easy"></div>
                            <span><b>Easy</b> - comfortable budget</span>
                        </div>
                        <div class="difficulty-item">
                            <div class="diff-dot diff-medium"></div>
                            <span><b>Medium</b> - tight savings</span>
                        </div>
                        <div class="difficulty-item">
                            <div class="diff-dot diff-hard"></div>
                            <span><b>Hard</b> - needs replanning</span>
                        </div>
                    </div>

                    <div class="category-badges">
                        <span class="badge">Cars</span>
                        <span class="badge">Phones</span>
                        <span class="badge">Bikes</span>
                        <span class="badge">Laptops</span>
                        <span class="badge">Scooters</span>
                    </div>
                </div>
                """)

        # ── Event handlers ──
        def respond(message, history, state):
            if not message or not message.strip():
                return history, state, ""
            history, state = chat(message, history, state)
            return history, state, ""

        send_btn.click(
            respond,
            inputs=[msg_input, chatbot, state],
            outputs=[chatbot, state, msg_input],
        )
        msg_input.submit(
            respond,
            inputs=[msg_input, chatbot, state],
            outputs=[chatbot, state, msg_input],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="localhost",
        server_port=7861,
        share=False,
        theme=gr.themes.Soft(
            primary_hue="emerald",
            secondary_hue="slate",
            font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
        ),
        css=CUSTOM_CSS,
    )
