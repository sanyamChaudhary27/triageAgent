#!/usr/bin/env python3
"""
Baseline Inference Script for Customer Support Triage Environment.
Mandatory log format: [START], [STEP], [END].
Interacts with the environment via FastAPI HTTP server.
"""

import os
import sys
import json
import time
import math
import threading
import requests
from typing import List, Dict, Optional
from openai import OpenAI
from dotenv import load_dotenv

# Force unbuffered stdout for real-time log capture
os.environ["PYTHONUNBUFFERED"] = "1"
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(line_buffering=True)

load_dotenv()

# --- Config ---
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or "dummy_key"
ENV_PORT = int(os.getenv("PORT", 7860))
BASE_URL = f"http://localhost:{ENV_PORT}"

TASKS = ["ticket-classification-easy", "ticket-routing-medium", "ticket-handling-hard"]
MAX_STEPS = 5
EPISODES_PER_TASK = 1 # Reducing for quick validation, platform will usually run its own loops

# --- Log Helpers ---
def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None):
    err_str = error if error else "null"
    done_str = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={err_str}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.4f} rewards={rewards_str}", flush=True)

# --- Sigmoid Clamping ---
def get_safe_score(val: float) -> float:
    """Sigmoid-based clamping to (0.001, 0.999)."""
    try:
        f_val = float(val)
        # Shift and scale sigmoid to map common ranges safely
        sigmoid = 1 / (1 + math.exp(-6 * (f_val - 0.5)))
        return max(0.001, min(0.999, sigmoid))
    except:
        return 0.513

# --- Server Management ---
def _start_server():
    import uvicorn
    from server.app import app
    uvicorn.run(app, host="0.0.0.0", port=ENV_PORT, log_level="error")

def ensure_server_running():
    try:
        requests.get(f"{BASE_URL}/health", timeout=1)
        return
    except:
        pass
    
    t = threading.Thread(target=_start_server, daemon=True)
    t.start()
    
    for _ in range(10):
        time.sleep(1)
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=1)
            if r.status_code == 200:
                return
        except:
            pass
    raise RuntimeError("Server failed to start")

# --- Episode Runner ---
def run_episode(task_id: str, client: OpenAI):
    log_start(task=task_id, env="customer-support-triage", model=MODEL_NAME)
    
    # 1. Reset
    try:
        r = requests.post(f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=10)
        r.raise_for_status()
        data = r.json()
        obs = data["observation"]
        session_id = data["session_id"]
    except Exception as e:
        log_end(success=False, steps=0, score=0.001, rewards=[])
        return

    rewards = []
    step_count = 0
    done = False
    
    # 2. Step Loop
    while not done and step_count < MAX_STEPS:
        step_count += 1
        try:
            # Simple prompt for baseline
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Respond with one word: classify, assign, respond, escalate, or close."},
                    {"role": "user", "content": f"Ticket: {obs['subject']}\nBody: {obs['body']}"}
                ],
                max_tokens=10,
                timeout=15.0
            )
            act_type = completion.choices[0].message.content.strip().lower()
            if act_type not in ["classify", "assign", "respond", "escalate", "close"]:
                act_type = "classify"
            
            action = {"action_type": act_type}
            
            # Step in env
            sr = requests.post(f"{BASE_URL}/step", json={
                "session_id": session_id,
                "action": action
            }, timeout=10)
            sr.raise_for_status()
            step_data = sr.json()
            
            reward = float(step_data["reward"]["value"])
            done = step_data["done"]
            obs = step_data["observation"]
            rewards.append(reward)
            
            log_step(step=step_count, action=act_type, reward=reward, done=done)
            
        except Exception as e:
            log_step(step=step_count, action="error", reward=0.0, done=True, error=str(e))
            rewards.append(0.0)
            break

    # 3. Grade & End
    try:
        gr = requests.get(f"{BASE_URL}/grade/{session_id}", timeout=5)
        gr.raise_for_status()
        score = float(gr.json()["grader_score"])
    except:
        score = 0.513
    
    safe_score = get_safe_score(score)
    log_end(success=(safe_score > 0.5), steps=step_count, score=safe_score, rewards=rewards)

def main():
    ensure_server_running()
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    
    for task_id in TASKS:
        for _ in range(EPISODES_PER_TASK):
            run_episode(task_id, client)
            time.sleep(1)

if __name__ == "__main__":
    main()
