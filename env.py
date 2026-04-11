"""
Customer Support Ticket Triage Environment
Implements full OpenEnv specification with typed models, step/reset/state API
"""

import json
import random
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import asdict
import uuid

from pydantic import BaseModel, Field


# =============
# ENUMS
# =============

class Severity(str, Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Team(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    PRODUCT = "product"
    SALES = "sales"
    GENERAL = "general"


class ActionType(str, Enum):
    CLASSIFY = "classify"
    ASSIGN = "assign"
    RESPOND = "respond"
    ESCALATE = "escalate"
    CLOSE = "close"


class CustomerTier(str, Enum):
    FREE = "free"
    PROFESSIONAL = "professional"
    ENTERPRISE = "enterprise"


class Priority(str, Enum):
    NONE = "none"
    URGENT = "urgent"
    HIGH = "high"
    NORMAL = "normal"
    LOW = "low"


# =============
# MODELS
# =============

class ConversationMessage(BaseModel):
    role: str = Field(..., description="'customer' or 'agent'")
    message: str = Field(..., description="Message content")
    timestamp: str = Field(..., description="ISO 8601 timestamp")


class TicketMetadata(BaseModel):
    created_at: str = Field(..., description="ISO 8601 creation timestamp")
    customer_tier: CustomerTier = Field(..., description="Customer subscription tier")
    previous_tickets_count: int = Field(default=0, description="Historical ticket count")
    is_repeat_customer: bool = Field(..., description="Whether customer is repeat")
    sla_minutes: int = Field(..., description="SLA response time in minutes")


class Observation(BaseModel):
    ticket_id: str = Field(..., description="Unique ticket identifier")
    customer_id: str = Field(..., description="Unique customer identifier")
    subject: str = Field(..., description="Ticket subject line")
    body: str = Field(..., description="Ticket body/description")
    priority_hint: Priority = Field(default=Priority.NONE, description="Customer priority hint")
    customer_sentiment: float = Field(default=0.5, ge=0.0, le=1.0)
    conversation_history: List[ConversationMessage] = Field(default_factory=list)
    metadata: TicketMetadata = Field(..., description="Ticket metadata")
    task_id: str = Field(..., description="Current task ID")
    step_number: int = Field(default=0)
    episode_done: bool = Field(default=False)


class Action(BaseModel):
    action_type: ActionType = Field(..., description="Type of action to take")
    severity: Optional[Severity] = Field(default=None)
    assigned_team: Optional[Team] = Field(default=None)
    response_text: Optional[str] = Field(default=None)
    reason: Optional[str] = Field(default=None)


class Reward(BaseModel):
    # Strictly between 0 and 1
    value: float = Field(..., gt=0.0, lt=1.0)
    correct_action: bool = Field(default=False)
    sla_violated: bool = Field(default=False)
    efficiency: float = Field(default=0.5, gt=0.0, lt=1.0)


class StepResult(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any] = Field(default_factory=dict)


class ResetResult(BaseModel):
    observation: Observation
    info: Dict[str, Any] = Field(default_factory=dict)


# =============
# DATA
# =============

SYNTHETIC_TICKETS = [
    {
        "subject": "Billing issue",
        "body": "Charged twice. Refund me.",
        "sentiment": 0.2,
        "true_severity": Severity.HIGH,
        "true_team": Team.BILLING,
        "tier": CustomerTier.PROFESSIONAL,
    },
    {
        "subject": "App Crash",
        "body": "Crashes on startup. Error 500.",
        "sentiment": 0.1,
        "true_severity": Severity.CRITICAL,
        "true_team": Team.TECHNICAL,
        "tier": CustomerTier.ENTERPRISE,
    },
    {
        "subject": "Feature request",
        "body": "I want dark mode.",
        "sentiment": 0.7,
        "true_severity": Severity.LOW,
        "true_team": Team.PRODUCT,
        "tier": CustomerTier.FREE,
    }
]

SLA_BY_SEVERITY = {
    Severity.CRITICAL: 15,
    Severity.HIGH: 60,
    Severity.MEDIUM: 240,
    Severity.LOW: 1440,
    Severity.INFO: 2880,
}


# =============
# GRADERS
# =============

class TaskGrader:
    def grade(self, observation: Observation, actions: List[Action]) -> float:
        raise NotImplementedError


class EasyClassificationGrader(TaskGrader):
    def grade(self, observation: Observation, actions: List[Action]) -> float:
        if not actions: return 0.1
        ticket = self._find(observation)
        if not ticket: return 0.1
        
        classify = next((a for a in actions if a.action_type == ActionType.CLASSIFY), None)
        assign = next((a for a in actions if a.action_type == ActionType.ASSIGN), None)
        
        score = 0.1
        if classify and classify.severity == ticket["true_severity"]: score += 0.5
        if assign and assign.assigned_team == ticket["true_team"]: score += 0.3
        
        return max(0.1, min(score, 0.9))

    def _find(self, obs):
        for t in SYNTHETIC_TICKETS:
            if t["subject"] in obs.subject: return t
        return None


class MediumRoutingGrader(TaskGrader):
    def grade(self, observation: Observation, actions: List[Action]) -> float:
        if not actions: return 0.1
        ticket = self._find(observation)
        if not ticket: return 0.1
        
        assign = next((a for a in actions if a.action_type == ActionType.ASSIGN), None)
        score = 0.1
        if assign and assign.assigned_team == ticket["true_team"]: score += 0.7
        return max(0.1, min(score, 0.9))

    def _find(self, obs):
        for t in SYNTHETIC_TICKETS:
            if t["subject"] in obs.subject: return t
        return None


class HardComplexGrader(TaskGrader):
    def grade(self, observation: Observation, actions: List[Action]) -> float:
        if not actions: return 0.1
        score = 0.1
        if any(a.action_type == ActionType.ESCALATE for a in actions): score += 0.4
        if any(a.action_type == ActionType.RESPOND for a in actions): score += 0.4
        return max(0.1, min(score, 0.9))


# =============
# ENV
# =============

class CustomerSupportTriageEnv:
    def __init__(self, task_id: str = "ticket-classification-easy"):
        self.task_id = task_id
        self.current_observation = None
        self.episode_actions = []
        self.step_count = 0
        self.max_steps = 10
        self.episode_reward = 0.1
        
    def reset(self) -> ResetResult:
        ticket = random.choice(SYNTHETIC_TICKETS)
        metadata = TicketMetadata(
            created_at=datetime.now().isoformat(),
            customer_tier=ticket["tier"],
            previous_tickets_count=random.randint(0, 5),
            is_repeat_customer=True,
            sla_minutes=SLA_BY_SEVERITY[ticket["true_severity"]],
        )
        self.current_observation = Observation(
            ticket_id=str(uuid.uuid4())[:8],
            customer_id="cust_123",
            subject=ticket["subject"],
            body=ticket["body"],
            metadata=metadata,
            task_id=self.task_id,
            customer_sentiment=ticket["sentiment"]
        )
        self.episode_actions = []
        self.step_count = 0
        self.episode_reward = 0.1
        return ResetResult(observation=self.current_observation)
    
    def step(self, action: Action) -> StepResult:
        self.step_count += 1
        self.episode_actions.append(action)
        
        # Calculate a safe reward signal
        reward_val = 0.1
        if action.action_type in [ActionType.CLASSIFY, ActionType.ASSIGN, ActionType.RESPOND]:
            reward_val = 0.5
        
        # Ensure reward is strictly in (0.01, 0.99)
        reward = Reward(value=reward_val, efficiency=0.5)
        self.episode_reward += reward.value
        
        done = self.step_count >= self.max_steps or action.action_type == ActionType.CLOSE
        self.current_observation.step_number = self.step_count
        self.current_observation.episode_done = done
        
        return StepResult(observation=self.current_observation, reward=reward, done=done)

    def state(self) -> Dict:
        return {"step": self.step_count}


def grade_episode(task_id: str, actions: List[Action], observation: Observation) -> float:
    if "easy" in task_id: grader = EasyClassificationGrader()
    elif "medium" in task_id: grader = MediumRoutingGrader()
    else: grader = HardComplexGrader()
    return grader.grade(observation, actions)
