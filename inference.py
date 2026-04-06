"""
Baseline Inference Script for Customer Support Triage Environment
Uses OpenAI API client to run agents against all 3 tasks
Produces reproducible baseline scores
"""

import os
import json
import logging
from typing import List, Tuple
from enum import Enum

from openai import OpenAI

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
API_KEY = os.getenv("GROQ_API_KEY") or os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")

MAX_STEPS_PER_EPISODE = 10
NUM_EPISODES_PER_TASK = 3  # Run 3 episodes per task to get average score
TEMPERATURE = 0.7
MAX_TOKENS = 500

# Task definitions
TASKS = [
    "ticket-classification-easy",
    "ticket-routing-medium", 
    "ticket-handling-hard",
]

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# SYSTEM PROMPT FOR THE AGENT
# ============================================================================

SYSTEM_PROMPT = """You are an expert customer support agent. Your job is to triage incoming support tickets by:

1. CLASSIFYING the severity (critical, high, medium, low, info)
2. ASSIGNING to the correct team (billing, technical, product, sales, general)
3. RESPONDING with empathy and clarity
4. ESCALATING complex issues

You must respond with a JSON action object. Valid actions:

{
  "action_type": "classify",
  "severity": "critical" | "high" | "medium" | "low" | "info"
}

{
  "action_type": "assign",
  "assigned_team": "billing" | "technical" | "product" | "sales" | "general"
}

{
  "action_type": "respond",
  "response_text": "Your empathetic response to the customer..."
}

{
  "action_type": "escalate",
  "reason": "Why this needs escalation..."
}

{
  "action_type": "close",
}

Always respond with valid JSON only, no additional text."""


# ============================================================================
# AGENT LOGIC
# ============================================================================

def build_user_message(observation) -> str:
    """Format observation into user prompt"""
    
    sentiment_label = "happy" if observation.customer_sentiment > 0.6 else "neutral" if observation.customer_sentiment > 0.3 else "angry"
    
    msg = f"""
TICKET #{observation.ticket_id}
Customer: {observation.customer_id} ({observation.metadata.customer_tier.value} tier)
Subject: {observation.subject}
Body: {observation.body}

Customer Sentiment: {sentiment_label}
SLA Response Time: {observation.metadata.sla_minutes} minutes
Is Repeat Customer: {observation.metadata.is_repeat_customer}
Previous Tickets: {observation.metadata.previous_tickets_count}

Step {observation.step_number + 1} of {MAX_STEPS_PER_EPISODE}

Please take the next appropriate action.
"""
    return msg


def parse_action(response_text: str) -> Action | None:
    """Parse JSON action from model response"""
    
    try:
        # Try to extract JSON from response
        lines = response_text.strip().split('\n')
        
        for line in lines:
            if line.strip().startswith('{'):
                # Found a JSON line, try to parse it
                json_str = line.strip()
                data = json.loads(json_str)
                
                action_type = ActionType(data.get("action_type", "").lower())
                
                action = Action(
                    action_type=action_type,
                    severity=Severity(data["severity"]) if "severity" in data and data["severity"] else None,
                    assigned_team=Team(data["assigned_team"]) if "assigned_team" in data and data["assigned_team"] else None,
                    response_text=data.get("response_text"),
                    reason=data.get("reason"),
                )
                return action
        
        # If we get here, no valid JSON found
        logger.warning(f"Could not parse action from: {response_text}")
        return None
        
    except (json.JSONDecodeError, ValueError, KeyError) as e:
        logger.warning(f"Error parsing action: {e}")
        return None


def run_episode(env: CustomerSupportTriageEnv, client: OpenAI) -> float:
    """Run one episode and return the episode score"""
    
    # Reset environment
    reset_result = env.reset()
    observation = reset_result.observation
    episode_actions: List[Action] = []
    
    logger.info(f"Started episode for task: {env.task_id}")
    logger.info(f"Ticket: {observation.subject}")
    
    # Run episode
    for step in range(MAX_STEPS_PER_EPISODE):
        # Build prompt
        user_message = build_user_message(observation)
        
        # Call model
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            response_text = completion.choices[0].message.content
        except Exception as e:
            logger.error(f"API error: {e}")
            return 0.0
        
        # Parse action
        action = parse_action(response_text)
        if not action:
            logger.warning("Could not parse action, defaulting to noop equivalent")
            action = Action(action_type=ActionType.CLASSIFY, severity=Severity.MEDIUM)
        
        episode_actions.append(action)
        logger.info(f"Step {step + 1}: {action.action_type.value}")
        
        # Step environment
        step_result = env.step(action)
        observation = step_result.observation
        
        logger.info(f"  Reward: {step_result.reward.value:+.2f}")
        
        if step_result.done:
            logger.info("Episode completed (done=True)")
            break
    
    # Grade the episode
    episode_score = grade_episode(env.task_id, episode_actions, observation)
    logger.info(f"Episode score: {episode_score:.3f}")
    
    return episode_score


def main():
    """Run baseline inference on all tasks"""
    
    # Initialize client
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)
    
    logger.info(f"Model: {MODEL_NAME}")
    logger.info(f"API Base: {API_BASE_URL}")
    logger.info(f"Running {len(TASKS)} tasks x {NUM_EPISODES_PER_TASK} episodes each")
    logger.info("=" * 80)
    
    results = {}
    
    for task_id in TASKS:
        logger.info(f"\n{'='*80}")
        logger.info(f"TASK: {task_id}")
        logger.info(f"{'='*80}\n")
        
        # Create environment for this task
        env = CustomerSupportTriageEnv(task_id=task_id)
        
        # Run multiple episodes
        episode_scores = []
        for ep in range(NUM_EPISODES_PER_TASK):
            logger.info(f"\n--- Episode {ep + 1}/{NUM_EPISODES_PER_TASK} ---")
            score = run_episode(env, client)
            episode_scores.append(score)
            logger.info(f"Score: {score:.3f}\n")
        
        # Compute statistics
        avg_score = sum(episode_scores) / len(episode_scores) if episode_scores else 0.0
        max_score = max(episode_scores) if episode_scores else 0.0
        min_score = min(episode_scores) if episode_scores else 0.0
        
        results[task_id] = {
            "episodes": episode_scores,
            "average": avg_score,
            "max": max_score,
            "min": min_score,
        }
        
        logger.info(f"Task {task_id} Results:")
        logger.info(f"  Average: {avg_score:.3f}")
        logger.info(f"  Max:     {max_score:.3f}")
        logger.info(f"  Min:     {min_score:.3f}")
    
    # Summary
    logger.info(f"\n{'='*80}")
    logger.info("BASELINE RESULTS SUMMARY")
    logger.info(f"{'='*80}")
    
    overall_avg = sum(r["average"] for r in results.values()) / len(results)
    
    for task_id, task_results in results.items():
        logger.info(f"{task_id}:  {task_results['average']:.3f}")
    
    logger.info(f"\nOverall Average: {overall_avg:.3f}")
    
    # Save results to JSON
    with open("baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info("\nResults saved to baseline_results.json")


if __name__ == "__main__":
    main()
