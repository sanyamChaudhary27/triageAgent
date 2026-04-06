#!/usr/bin/env python3
"""
Pre-submission validation script
Tests all components before deploying to competition
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def test_imports():
    """Test all imports work"""
    logger.info("Testing imports...")
    try:
        from env import (
            CustomerSupportTriageEnv,
            Action,
            ActionType,
            Severity,
            Team,
            Observation,
            Reward,
            grade_episode,
        )
        from openai import OpenAI
        logger.info("[OK] All imports successful")
        return True
    except ImportError as e:
        logger.error(f"[FAIL] Import failed: {e}")
        return False


def test_env_initialization():
    """Test environment can be created"""
    logger.info("Testing environment initialization...")
    try:
        from env import CustomerSupportTriageEnv
        
        for task_id in [
            "ticket-classification-easy",
            "ticket-routing-medium",
            "ticket-handling-hard",
        ]:
            env = CustomerSupportTriageEnv(task_id=task_id)
            logger.info(f"  [OK] Created env for {task_id}")
        
        return True
    except Exception as e:
        logger.error(f"[FAIL] Environment initialization failed: {e}")
        return False


def test_reset():
    """Test reset() function"""
    logger.info("Testing reset()...")
    try:
        from env import CustomerSupportTriageEnv
        
        env = CustomerSupportTriageEnv(task_id="ticket-classification-easy")
        reset_result = env.reset()
        
        assert reset_result.observation is not None
        assert reset_result.observation.ticket_id is not None
        assert reset_result.observation.subject is not None
        assert reset_result.info is not None
        
        logger.info("[OK] reset() works correctly")
        return True
    except Exception as e:
        logger.error(f"[FAIL] reset() failed: {e}")
        return False


def test_step():
    """Test step() function"""
    logger.info("Testing step()...")
    try:
        from env import CustomerSupportTriageEnv, Action, ActionType, Severity
        
        env = CustomerSupportTriageEnv(task_id="ticket-classification-easy")
        reset_result = env.reset()
        
        action = Action(
            action_type=ActionType.CLASSIFY,
            severity=Severity.HIGH
        )
        step_result = env.step(action)
        
        assert step_result.observation is not None
        assert step_result.reward is not None
        assert isinstance(step_result.reward.value, float)
        assert -1.0 <= step_result.reward.value <= 1.0
        assert step_result.done is not None
        assert step_result.info is not None
        
        logger.info("[OK] step() works correctly")
        return True
    except Exception as e:
        logger.error(f"[FAIL] step() failed: {e}")
        return False


def test_state():
    """Test state() function"""
    logger.info("Testing state()...")
    try:
        from env import CustomerSupportTriageEnv, Action, ActionType, Severity
        
        env = CustomerSupportTriageEnv(task_id="ticket-classification-easy")
        reset_result = env.reset()
        
        action = Action(
            action_type=ActionType.CLASSIFY,
            severity=Severity.HIGH
        )
        env.step(action)
        
        state = env.state()
        
        assert state is not None
        assert "current_observation" in state
        assert "step_count" in state
        assert "episode_actions" in state
        assert "episode_reward" in state
        
        logger.info("[OK] state() works correctly")
        return True
    except Exception as e:
        logger.error(f"[FAIL] state() failed: {e}")
        return False


def test_graders():
    """Test grader functions"""
    logger.info("Testing graders...")
    try:
        from env import (
            CustomerSupportTriageEnv,
            Action,
            ActionType,
            Severity,
            Team,
            grade_episode,
        )
        
        env = CustomerSupportTriageEnv(task_id="ticket-classification-easy")
        reset_result = env.reset()
        obs = reset_result.observation
        
        actions = [
            Action(action_type=ActionType.CLASSIFY, severity=Severity.HIGH),
            Action(action_type=ActionType.ASSIGN, assigned_team=Team.TECHNICAL),
        ]
        
        score = grade_episode("ticket-classification-easy", actions, obs)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        
        logger.info(f"[OK] Graders work correctly (score: {score:.3f})")
        return True
    except Exception as e:
        logger.error(f"[FAIL] Grader test failed: {e}")
        return False


def test_files_exist():
    """Check all required files exist"""
    logger.info("Checking required files...")
    
    required_files = [
        "env.py",
        "inference.py",
        "openenv.yaml",
        "requirements.txt",
        "Dockerfile",
        "README.md",
    ]
    
    missing = []
    for fname in required_files:
        if not Path(fname).exists():
            missing.append(fname)
    
    if missing:
        logger.error(f"[FAIL] Missing files: {', '.join(missing)}")
        return False
    
    logger.info(f"[OK] All required files present")
    return True


def test_openenv_yaml():
    """Validate openenv.yaml structure"""
    logger.info("Validating openenv.yaml...")
    
    try:
        import yaml
        
        with open("openenv.yaml", "r") as f:
            config = yaml.safe_load(f)
        
        required_fields = ["name", "version", "description", "tasks", "observation_space", "action_space"]
        missing = [f for f in required_fields if f not in config]
        
        if missing:
            logger.error(f"[FAIL] Missing fields in openenv.yaml: {', '.join(missing)}")
            return False
        
        # Check tasks
        if not isinstance(config.get("tasks"), list) or len(config["tasks"]) < 3:
            logger.error("[FAIL] Must have at least 3 tasks defined")
            return False
        
        logger.info(f"[OK] openenv.yaml valid ({len(config['tasks'])} tasks)")
        return True
    except Exception as e:
        logger.error(f"[FAIL] openenv.yaml validation failed: {e}")
        return False


def main():
    """Run all tests"""
    
    print("=" * 70)
    print("CUSTOMER SUPPORT TRIAGE ENV - PRE-SUBMISSION VALIDATION")
    print("=" * 70)
    print()
    
    tests = [
        ("File Structure", test_files_exist),
        ("OpenEnv YAML", test_openenv_yaml),
        ("Imports", test_imports),
        ("Environment Initialization", test_env_initialization),
        ("reset() Function", test_reset),
        ("step() Function", test_step),
        ("state() Function", test_state),
        ("Grader Functions", test_graders),
    ]
    
    results = []
    for test_name, test_func in tests:
        result = test_func()
        results.append((test_name, result))
        print()
    
    # Summary
    print("=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for test_name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"[{status:4}] | {test_name}")
    
    print("-" * 70)
    print(f"Results: {passed}/{total} tests passed")
    print()
    
    if passed == total:
        print("All validation tests passed! Ready for submission.")
        return 0
    else:
        print("Some tests failed. Fix issues before submitting.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
