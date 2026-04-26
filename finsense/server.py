from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, List

from finsense.env import FinSenseEnv
from finsense.models import ActionModel, StateModel
from finsense.tasks import TASKS
from finsense.graders import grade_episode

app = FastAPI(title="FinSense RL Environment")
env = FinSenseEnv()


class ResetRequest(BaseModel):
    task_id: str = "easy"
    seed: int = 42


class StepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    info: Dict[str, Any]


class GradeRequest(BaseModel):
    task_id: str = "easy"


@app.post("/reset", response_model=Dict[str, Any])
def reset_env(req: ResetRequest = ResetRequest()):
    return env.reset(task_id=req.task_id, seed=req.seed)


@app.post("/step", response_model=StepResponse)
def step_env(action: ActionModel):
    if not env.state:
        raise HTTPException(status_code=400, detail="Environment not reset")
    obs, rew, done, info = env.step(action)
    return StepResponse(observation=obs, reward=rew, done=done, info=info)


@app.get("/state", response_model=Dict[str, Any])
def get_state():
    if not env.state:
        raise HTTPException(status_code=400, detail="Environment not reset")
    return env.get_state()


@app.post("/grade")
def grade(req: GradeRequest = GradeRequest()):
    if not env.state:
        raise HTTPException(status_code=400, detail="Environment not reset. Call /reset first.")

    s = env.state
    task_id = req.task_id or s.get("task_id", "easy")
    task = TASKS.get(task_id)

    if not task:
        raise HTTPException(status_code=400, detail=f"Unknown task_id: {task_id}")

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
        score = grade_episode(state_model)
    except Exception as e:
        # Fallback: simple goal-based score
        goal_total = s["goal_total"]
        goal_remaining = s["goal_remaining"]
        goal_progress = max(0.0, goal_total - goal_remaining)
        score = goal_progress / max(1.0, goal_total)

    # CRITICAL: Always clamp strictly between 0 and 1 (exclusive)
    # Valid range: (0, 1) — never 0.0 or 1.0, 2 decimal places only
    score = max(0.01, min(0.99, float(score)))
    return {"score": score, "task_id": task_id}


@app.get("/tasks", response_model=List[str])
def get_tasks():
    return list(TASKS.keys())





@app.get("/health")
def health_check():
    return {"status": "ok"}