"""
Proof of Fix Script
Simulates a validator loop and checks all tasks for range compliance.
"""

import sys
import os

# Add local dir to path
sys.path.append(os.getcwd())

def test_everything():
    from env import CustomerSupportTriageEnv, grade_episode, Action, ActionType
    
    tasks = ["ticket-classification-easy", "ticket-routing-medium", "ticket-handling-hard"]
    
    print("=" * 60)
    print("STARTING ULTRA-HARDENED PROOF TEST")
    print("=" * 60)
    
    all_passed = True
    
    for task_id in tasks:
        print(f"\n[TASK: {task_id}]")
        env = CustomerSupportTriageEnv(task_id=task_id)
        
        # Test Reset
        res = env.reset()
        print(f"  Reset Result: OBS={res.observation.ticket_id}")
        
        # Test Steps
        episode_reward = 0
        for i in range(5):
            action = Action(action_type=ActionType.CLASSIFY)
            step_res = env.step(action)
            
            reward = step_res.reward.value
            print(f"    Step {i+1}: Reward={reward}")
            
            if not (0.0 < reward < 1.0):
                print(f"    !!! FAILED: Reward {reward} out of range!")
                all_passed = False
            
            if step_res.done: break
            
        # Test Grader
        final_score = grade_episode(task_id, [], res.observation)
        print(f"  Final Grader Score: {final_score}")
        
        if not (0.0 < final_score < 1.0):
            print(f"  !!! FAILED: Score {final_score} out of range!")
            all_passed = False

    print("\n" + "=" * 60)
    if all_passed:
        print("RESULT: PASSED. ALL SCORES ARE STRICTLY WITHIN (0, 1)")
    else:
        print("RESULT: FAILED. SOME SCORES ARE OUT OF RANGE")
    print("=" * 60)

if __name__ == "__main__":
    test_everything()
