import sys
import os

# Add local dir to path
sys.path.append(os.getcwd())

try:
    from env import grade_episode, Action, ActionType, Severity, Team, Observation
    import json

    tasks = ["ticket-classification-easy", "ticket-routing-medium", "ticket-handling-hard"]
    
    print("Testing Grader Scores...")
    for task_id in tasks:
        # Mock actions and observation
        actions = [Action(action_type=ActionType.CLASSIFY, severity=Severity.HIGH)]
        obs = Observation(
            ticket_id="TEST", 
            subject="TEST", 
            body="TEST", 
            task_id=task_id, 
            step_number=0, 
            episode_done=True
        )
        
        score = grade_episode(task_id, actions, obs)
        print(f"Task: {task_id:30} | Score: {score}")
        
        if not (0.0 < score < 1.0):
            print(f"FAILED: Score {score} is out of range (0, 1)!")
            sys.exit(1)
            
    print("\nALL TASKS PASSED RANGE VALIDATION (0.0 < score < 1.0)")

except Exception as e:
    print(f"TEST CRASHED: {e}")
    sys.exit(1)
