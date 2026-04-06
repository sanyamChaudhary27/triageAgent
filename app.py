"""
FastAPI server exposing the CustomerSupportTriageEnv as an HTTP API.
The hackathon platform calls /reset (POST) and /step (POST) to interact with the environment.
"""
import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional

from env import (
    CustomerSupportTriageEnv,
    Action,
    ActionType,
    Severity,
    Team,
)

app = FastAPI(title="Customer Support Triage OpenEnv", version="1.0.0")

# Global environment state (one session)
_env: Optional[CustomerSupportTriageEnv] = None


class ResetRequest(BaseModel):
    task_id: Optional[str] = "ticket-classification-easy"


class StepRequest(BaseModel):
    action: dict


@app.get("/")
def root():
    return {"status": "ok", "name": "customer-support-triage", "version": "1.0.0"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/reset")
def reset(request: ResetRequest = None):
    global _env
    task_id = (request.task_id if request and request.task_id else None) or "ticket-classification-easy"
    _env = CustomerSupportTriageEnv(task_id=task_id)
    result = _env.reset()
    return {
        "observation": result.observation.model_dump(),
        "info": result.info,
    }


@app.post("/step")
def step(request: StepRequest):
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")

    try:
        action_data = request.action
        action = Action(
            action_type=ActionType(action_data.get("action_type", "classify")),
            severity=Severity(action_data["severity"]) if action_data.get("severity") else None,
            assigned_team=Team(action_data["assigned_team"]) if action_data.get("assigned_team") else None,
            response_text=action_data.get("response_text"),
            reason=action_data.get("reason"),
        )
        result = _env.step(action)
        return {
            "observation": result.observation.model_dump(),
            "reward": result.reward.value,
            "done": result.done,
            "info": result.info,
        }
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e))


@app.get("/state")
def state():
    global _env
    if _env is None:
        raise HTTPException(status_code=400, detail="Environment not initialised. Call /reset first.")
    return _env.state()


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)
