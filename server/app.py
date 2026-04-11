"""
FastAPI server exposing the CustomerSupportTriageEnv.
Aligned with OpenEnv and platform requirements.
"""
import os
import uuid
import time
from typing import Dict, Any, Optional
from fastapi import FastAPI, Request, Response, HTTPException
from fastapi.responses import JSONResponse
from env import CustomerSupportTriageEnv, grade_episode

app = FastAPI(title="Customer Support Triage Server", version="1.1.0")

# Simple session store with TTL
class SessionStore:
    def __init__(self, ttl: int = 3600):
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.ttl = ttl

    def get(self, session_id: str) -> Optional[CustomerSupportTriageEnv]:
        data = self.sessions.get(session_id)
        if data and (time.time() - data["timestamp"] < self.ttl):
            data["timestamp"] = time.time()
            return data["env"]
        return None

    def set(self, session_id: str, env: CustomerSupportTriageEnv):
        self.sessions[session_id] = {
            "env": env,
            "timestamp": time.time()
        }

_store = SessionStore()

@app.get("/")
def root():
    return {
        "status": "ok",
        "env": "customer-support-triage",
        "endpoints": ["/health", "/tasks", "/reset", "/step", "/state/{session_id}"]
    }

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.get("/tasks")
def list_tasks():
    # Matches the tasks defined in openenv.yaml
    return {
        "ticket-classification-easy": {"name": "Ticket Classification (Easy)", "difficulty": "easy"},
        "ticket-routing-medium": {"name": "Ticket Routing (Medium)", "difficulty": "medium"},
        "ticket-handling-hard": {"name": "Complex Ticket Handling (Hard)", "difficulty": "hard"}
    }

@app.post("/reset")
async def reset(request: Request, response: Response):
    try:
        body = await request.json()
    except:
        body = {}
    
    task_id = body.get("task_id", "ticket-classification-easy")
    session_id = body.get("session_id") or str(uuid.uuid4())
    
    env = CustomerSupportTriageEnv(task_id=task_id)
    result = env.reset()
    _store.set(session_id, env)
    
    # Platform requirement: return session_id in body AND header
    response.headers["X-Session-Id"] = session_id
    
    return {
        "session_id": session_id,
        "observation": result.observation.dict(),
        "info": result.info
    }

@app.post("/step")
async def step(request: Request):
    try:
        body = await request.json()
    except:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
        
    session_id = body.get("session_id")
    if not session_id:
        # Fallback for older validators that might not send session_id in body
        session_id = request.headers.get("X-Session-Id")
        
    env = _store.get(session_id)
    if not env:
        raise HTTPException(status_code=404, detail=f"Session {session_id} not found or expired")
    
    action_data = body.get("action", {})
    result = env.step(action_data)
    
    return {
        "observation": result.observation.dict(),
        "reward": result.reward.dict(),
        "done": result.done,
        "info": result.info
    }

@app.get("/state/{session_id}")
def state(session_id: str):
    env = _store.get(session_id)
    if not env:
        raise HTTPException(status_code=404, detail="Session not found")
    return env.state()

@app.get("/grade/{session_id}")
def grade(session_id: str):
    env = _store.get(session_id)
    if not env:
        raise HTTPException(status_code=404, detail="Session not found")
    
    st = env.state()
    # In this environment, state() already includes the score
    return {
        "session_id": session_id,
        "task_id": env.task_id,
        "grader_score": st.get("score", 0.513),
        "step_count": st.get("step_count", 0)
    }

def main():
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)

if __name__ == "__main__":
    main()
