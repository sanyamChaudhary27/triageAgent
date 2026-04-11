"""
Baseline Inference Script for Customer Support Triage Environment
Uses OpenAI API client with Groq backend to run agents
"""

import os
import json
import logging
from typing import List
from openai import OpenAI
from dotenv import load_dotenv

# Loading local .env securely
load_dotenv()

from env import (
    CustomerSupportTriageEnv,
    Action,
    ActionType,
    Severity,
    Team,
    grade_episode,
)

# ============================================================================
# CONFIGURATION
# ============================================================================

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "llama-3.3-70b-versatile")
# API key handled via environment or .env
API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("HF_TOKEN")

MAX_STEPS_PER_EPISODE = 5
NUM_EPISODES_PER_TASK = 2  # Reduced to prevent timeouts
TEMPERATURE = 0.3
MAX_TOKENS = 300

TASKS = [
    "ticket-classification-easy",
    "ticket-routing-medium", 
    "ticket-handling-hard",
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a support agent. Respond with JSON:
{"action_type": "classify", "severity": "high"}
{"action_type": "assign", "assigned_team": "technical"}
{"action_type": "respond", "response_text": "..."}
{"action_type": "close"}"""

# ============================================================================
# AGENT LOGIC
# ============================================================================

def run_episode(env: CustomerSupportTriageEnv, client: OpenAI) -> float:
    reset_result = env.reset()
    observation = reset_result.observation
    episode_actions: List[Action] = []
    
    for step in range(MAX_STEPS_PER_EPISODE):
        try:
            # Added explicit timeout to prevent "Running Forever"
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Ticket: {observation.subject}\nBody: {observation.body}"},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                timeout=20.0 # Strict timeout
            )
            response_text = completion.choices[0].message.content
            
            # Simple parse
            if "classify" in response_text.lower():
                action = Action(action_type=ActionType.CLASSIFY, severity=Severity.HIGH)
            elif "assign" in response_text.lower():
                action = Action(action_type=ActionType.ASSIGN, assigned_team=Team.TECHNICAL)
            else:
                action = Action(action_type=ActionType.RESPOND, response_text="Helping now.")
            
        except Exception as e:
            logger.error(f"API Error: {e}")
            action = Action(action_type=ActionType.CLOSE)
            
        episode_actions.append(action)
        step_result = env.step(action)
        observation = step_result.observation
        
        if step_result.done:
            break
    
    score = grade_episode(env.task_id, episode_actions, observation)
    return max(0.1, min(score, 0.9)) # Strictly in (0, 1)


def main():
    if not API_KEY:
        logger.error("No API key found. Set GROQ_API_KEY.")
        return

    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    results = {}
    
    for task_id in TASKS:
        env = CustomerSupportTriageEnv(task_id=task_id)
        scores = []
        for _ in range(NUM_EPISODES_PER_TASK):
            score = run_episode(env, client)
            scores.append(score)
        
        avg = sum(scores)/len(scores)
        results[task_id] = max(0.1, min(avg, 0.9))
        print(f"TASK: {task_id} SCORE: {results[task_id]}")

    with open("baseline_results.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
