"""
Customer Support Ticket Triage Environment
Implements full OpenEnv specification with typed models, step/reset/state API
"""

import json
import random
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field, asdict
import uuid

from pydantic import BaseModel, Field


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

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


# ============================================================================
# PYDANTIC MODELS (OpenEnv Spec Compliance)
# ============================================================================

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
    """OpenEnv Observation model for Customer Support Triage"""
    
    ticket_id: str = Field(..., description="Unique ticket identifier")
    customer_id: str = Field(..., description="Unique customer identifier")
    subject: str = Field(..., description="Ticket subject line")
    body: str = Field(..., description="Ticket body/description")
    priority_hint: Priority = Field(default=Priority.NONE, description="Customer priority hint")
    customer_sentiment: float = Field(default=0.5, ge=0.0, le=1.0, description="Sentiment 0=angry, 1=happy")
    conversation_history: List[ConversationMessage] = Field(default_factory=list, description="Previous messages")
    metadata: TicketMetadata = Field(..., description="Ticket metadata")
    
    # Task identifier (used for curriculum learning)
    task_id: str = Field(..., description="Current task: ticket-classification-easy, routing-medium, or handling-hard")
    
    # State tracking
    step_number: int = Field(default=0, description="Current step in episode")
    episode_done: bool = Field(default=False, description="Whether episode is complete")


class Action(BaseModel):
    """OpenEnv Action model for Customer Support Triage"""
    
    action_type: ActionType = Field(..., description="Type of action to take")
    severity: Optional[Severity] = Field(default=None, description="Classification (required for CLASSIFY)")
    assigned_team: Optional[Team] = Field(default=None, description="Team assignment (required for ASSIGN)")
    response_text: Optional[str] = Field(default=None, description="Auto-response (required for RESPOND)")
    reason: Optional[str] = Field(default=None, description="Escalation reason (required for ESCALATE)")
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v is not None}


class Reward(BaseModel):
    """OpenEnv Reward model - composite score"""
    
    value: float = Field(..., ge=-0.99, le=0.99, description="Step reward in (-1, 1)")
    correct_action: bool = Field(default=False, description="Was action correct?")
    sla_violated: bool = Field(default=False, description="Did action violate SLA?")
    efficiency: float = Field(default=0.0, ge=0.0, le=1.0, description="Action efficiency score")
    customer_satisfaction: float = Field(default=0.5, ge=0.0, le=1.0, description="Satisfaction signal")


class StepResult(BaseModel):
    """Result of environment.step()"""
    
    observation: Observation
    reward: Reward
    done: bool = Field(..., description="Episode termination flag")
    info: Dict[str, Any] = Field(default_factory=dict, description="Additional info")
    

class ResetResult(BaseModel):
    """Result of environment.reset()"""
    
    observation: Observation
    info: Dict[str, Any] = Field(default_factory=dict, description="Additional info")


# ============================================================================
# TICKET DATASET (Synthetic Data)
# ============================================================================

SYNTHETIC_TICKETS = [
    # EASY TASK: Clear classifications
    {
        "subject": "Billing issue with subscription",
        "body": "I was charged twice for my subscription this month. Please refund.",
        "sentiment": 0.2,
        "true_severity": Severity.HIGH,
        "true_team": Team.BILLING,
        "tier": CustomerTier.PROFESSIONAL,
    },
    {
        "subject": "App not responding on mobile",
        "body": "The app crashes every time I try to upload a file. Error: RuntimeException",
        "sentiment": 0.1,
        "true_severity": Severity.CRITICAL,
        "true_team": Team.TECHNICAL,
        "tier": CustomerTier.ENTERPRISE,
    },
    {
        "subject": "Feature request: Dark mode",
        "body": "Would love to see dark mode in the next release. Thanks!",
        "sentiment": 0.7,
        "true_severity": Severity.LOW,
        "true_team": Team.PRODUCT,
        "tier": CustomerTier.FREE,
    },
    {
        "subject": "Can't login to my account",
        "body": "I forgot my password and the reset email isn't arriving. Urgent!",
        "sentiment": 0.3,
        "true_severity": Severity.HIGH,
        "true_team": Team.TECHNICAL,
        "tier": CustomerTier.PROFESSIONAL,
    },
    {
        "subject": "Pricing question",
        "body": "What are the enterprise pricing options?",
        "sentiment": 0.6,
        "true_severity": Severity.LOW,
        "true_team": Team.SALES,
        "tier": CustomerTier.PROFESSIONAL,
    },
    {
        "subject": "Delete my account",
        "body": "I want to close my account and delete all my data immediately. Please confirm once done.",
        "sentiment": 0.2,
        "true_severity": Severity.MEDIUM,
        "true_team": Team.GENERAL,
        "tier": CustomerTier.FREE,
    },
    {
        "subject": "How to export data?",
        "body": "Is there a way to export all my project data to a PDF or Excel file?",
        "sentiment": 0.5,
        "true_severity": Severity.INFO,
        "true_team": Team.PRODUCT,
        "tier": CustomerTier.FREE,
    },
    # MEDIUM TASK: Contextual decisions
    {
        "subject": "Inconsistent export formatting",
        "body": "CSV exports are showing different column orders. Critical for my workflow.",
        "sentiment": 0.4,
        "true_severity": Severity.MEDIUM,
        "true_team": Team.TECHNICAL,
        "tier": CustomerTier.ENTERPRISE,
    },
    {
        "subject": "Integration with Slack broken",
        "body": "Slack sync stopped working 2 hours ago. We rely on this for team notifications.",
        "sentiment": 0.2,
        "true_severity": Severity.CRITICAL,
        "true_team": Team.TECHNICAL,
        "tier": CustomerTier.ENTERPRISE,
    },
    {
        "subject": "Delayed delivery of physical goods",
        "body": "My order #9982 was supposed to arrive yesterday. Tracking shows it hasn't even left the warehouse.",
        "sentiment": 0.1,
        "true_severity": Severity.HIGH,
        "true_team": Team.BILLING,
        "tier": CustomerTier.PROFESSIONAL,
    },
    {
        "subject": "Bulk discount inquiry",
        "body": "We are looking to purchase 500 licenses for our university. Who should we talk to for a bulk discount?",
        "sentiment": 0.8,
        "true_severity": Severity.MEDIUM,
        "true_team": Team.SALES,
        "tier": CustomerTier.ENTERPRISE,
    },
    {
        "subject": "API returning 500 errors intermittently",
        "body": "Our production system is seeing 5% failure rate on the /v1/search endpoint since this morning.",
        "sentiment": 0.2,
        "true_severity": Severity.CRITICAL,
        "true_team": Team.TECHNICAL,
        "tier": CustomerTier.ENTERPRISE,
    },
    {
        "subject": "Update credit card on file",
        "body": "Our company card was lost. I need to update the billing method before the next cycle starts tomorrow.",
        "sentiment": 0.4,
        "true_severity": Severity.HIGH,
        "true_team": Team.BILLING,
        "tier": CustomerTier.PROFESSIONAL,
    },
    # HARD TASK: Multi-part, contextual
    {
        "subject": "Multiple issues: Performance + API + billing",
        "body": "1) Dashboard is slow. 2) API rate limit too low. 3) Billing shows duplicate charge from 3 months ago. Need all fixed ASAP.",
        "sentiment": 0.15,
        "true_severity": Severity.CRITICAL,
        "true_team": Team.TECHNICAL,
        "tier": CustomerTier.ENTERPRISE,
    },
    {
        "subject": "Security Breach Concern + Access Revocation",
        "body": "We noticed suspicious login attempts from an unknown IP. We need to audit all logs and revoke access for user 'dev_team_1' immediately. This is a top priority.",
        "sentiment": 0.05,
        "true_severity": Severity.CRITICAL,
        "true_team": Team.TECHNICAL,
        "tier": CustomerTier.ENTERPRISE,
    },
    {
        "subject": "Contract Renewal + Feature Roadmap + Bug",
        "body": "Our contract is up for renewal next month. We also want to discuss the upcoming roadmap for AI features. Lastly, the current dashboard has a bug where charts don't load.",
        "sentiment": 0.5,
        "true_severity": Severity.MEDIUM,
        "true_team": Team.SALES,
        "tier": CustomerTier.ENTERPRISE,
    },
    {
        "subject": "App crashing on Start + Data Loss",
        "body": "After the last update, the app won't open. I'm worried about my unsaved work from yesterday. Please help!",
        "sentiment": 0.1,
        "true_severity": Severity.CRITICAL,
        "true_team": Team.TECHNICAL,
        "tier": CustomerTier.PROFESSIONAL,
    },
    {
        "subject": "Incorrect tax in invoice",
        "body": "Our latest invoice #4421 has the wrong tax rate for our region. We previously discussed this with support, but it happened again.",
        "sentiment": 0.3,
        "true_severity": Severity.MEDIUM,
        "true_team": Team.BILLING,
        "tier": CustomerTier.ENTERPRISE,
    },
    {
        "subject": "Slow response time during peak hours",
        "body": "The system becomes almost unusable between 9 AM and 11 AM EST. We have an SLA that guarantees 99.9% uptime, and this is unacceptable.",
        "sentiment": 0.2,
        "true_severity": Severity.HIGH,
        "true_team": Team.TECHNICAL,
        "tier": CustomerTier.ENTERPRISE,
    },
    {
        "subject": "Question about GDPR compliance",
        "body": "Where can I find your latest DPA and GDPR compliance documentation? Our legal team requires it for the upcoming audit.",
        "sentiment": 0.6,
        "true_severity": Severity.INFO,
        "true_team": Team.GENERAL,
        "tier": CustomerTier.ENTERPRISE,
    },
]

SLA_BY_SEVERITY = {
    Severity.CRITICAL: 15,  # 15 minutes
    Severity.HIGH: 60,
    Severity.MEDIUM: 240,
    Severity.LOW: 1440,
    Severity.INFO: 2880,
}


# ============================================================================
# GRADERS (Task Evaluation Functions)
# ============================================================================

class TaskGrader:
    """Base grader class"""
    
    def grade(self, observation: Observation, actions: List[Action]) -> float:
        """Return score in [0.0, 1.0]"""
        raise NotImplementedError


class EasyClassificationGrader(TaskGrader):
    """Grade simple classification tasks"""
    
    def grade(self, observation: Observation, actions: List[Action]) -> float:
        """
        Score based on:
        - Correct severity classification: +0.6
        - Correct team assignment: +0.4
        """
        if not actions:
            return 0.01
        
        # Get the true severity/team from our synthetic data
        ticket_data = self._find_ticket_data(observation.subject, observation.body)
        if not ticket_data:
            return 0.01
        
        classify_action = next((a for a in actions if a.action_type == ActionType.CLASSIFY), None)
        assign_action = next((a for a in actions if a.action_type == ActionType.ASSIGN), None)
        
        score = 0.0
        
        if classify_action and classify_action.severity == ticket_data["true_severity"]:
            score += 0.6
        
        if assign_action and assign_action.assigned_team == ticket_data["true_team"]:
            score += 0.4
        
        return max(0.01, min(score, 0.99))
    
    @staticmethod
    def _find_ticket_data(subject: str, body: str) -> Optional[Dict]:
        for ticket in SYNTHETIC_TICKETS:
            if ticket["subject"] == subject and ticket["body"] == body:
                return ticket
        return None


class MediumRoutingGrader(TaskGrader):
    """Grade routing + prioritization under constraints"""
    
    def grade(self, observation: Observation, actions: List[Action]) -> float:
        """
        Score based on:
        - Correct team assignment: +0.5
        - Respects SLA: +0.3
        - Efficient (no unnecessary actions): +0.2
        """
        if not actions:
            return 0.01
        
        ticket_data = self._find_ticket_data(observation.subject, observation.body)
        if not ticket_data:
            return 0.01
        
        assign_action = next((a for a in actions if a.action_type == ActionType.ASSIGN), None)
        
        score = 0.0
        
        # Correct assignment
        if assign_action and assign_action.assigned_team == ticket_data["true_team"]:
            score += 0.5
        
        # SLA compliance (did we respond in time?)
        if len(actions) <= 2:  # Quick response = good
            score += 0.3
        
        # Efficiency (avoid spam)
        if len(actions) <= 3:
            score += 0.2
        
        return max(0.01, min(score, 0.99))
    
    @staticmethod
    def _find_ticket_data(subject: str, body: str) -> Optional[Dict]:
        for ticket in SYNTHETIC_TICKETS:
            if ticket["subject"] == subject and ticket["body"] == body:
                return ticket
        return None


class HardComplexGrader(TaskGrader):
    """Grade complex multi-part ticket handling"""
    
    def grade(self, observation: Observation, actions: List[Action]) -> float:
        """
        Score based on:
        - Identifies critical issue first: +0.4
        - Escalates appropriately: +0.3
        - Provides thoughtful response: +0.3
        """
        if not actions:
            return 0.01
        
        score = 0.0
        
        # Check if first action targets the critical issue
        if "Multiple issues" in observation.subject or "ASAP" in observation.body:
            first_action = actions[0] if actions else None
            if first_action and first_action.action_type == ActionType.CLASSIFY:
                if first_action.severity == Severity.CRITICAL:
                    score += 0.4
        
        # Check for escalation
        escalate_action = next((a for a in actions if a.action_type == ActionType.ESCALATE), None)
        if escalate_action:
            score += 0.3
        
        # Check for meaningful response
        respond_action = next((a for a in actions if a.action_type == ActionType.RESPOND), None)
        if respond_action and respond_action.response_text and len(respond_action.response_text) > 20:
            score += 0.3
        
        return max(0.01, min(score, 0.99))


# ============================================================================
# MAIN ENVIRONMENT CLASS
# ============================================================================

class CustomerSupportTriageEnv:
    """
    OpenEnv-compliant Customer Support Ticket Triage Environment
    
    Implements:
    - reset() -> ResetResult
    - step(action: Action) -> StepResult
    - state() -> Dict
    """
    
    def __init__(self, task_id: str = "ticket-classification-easy"):
        self.task_id = task_id
        self.current_observation: Optional[Observation] = None
        self.episode_actions: List[Action] = []
        self.step_count = 0
        self.max_steps = 10
        self.grader = self._get_grader()
        self.episode_reward = 0.0
        
    def reset(self) -> ResetResult:
        """Reset environment and return initial observation"""
        
        # Sample a random ticket
        ticket_template = random.choice(SYNTHETIC_TICKETS)
        
        # Create metadata
        now = datetime.now()
        metadata = TicketMetadata(
            created_at=now.isoformat(),
            customer_tier=CustomerTier(ticket_template["tier"].value),
            previous_tickets_count=random.randint(0, 10),
            is_repeat_customer=random.random() > 0.5,
            sla_minutes=SLA_BY_SEVERITY[ticket_template["true_severity"]],
        )
        
        # Create observation
        self.current_observation = Observation(
            ticket_id=str(uuid.uuid4())[:8],
            customer_id=f"cust_{random.randint(1000, 9999)}",
            subject=ticket_template["subject"],
            body=ticket_template["body"],
            priority_hint=Priority.NONE,
            customer_sentiment=ticket_template["sentiment"],
            conversation_history=[],
            metadata=metadata,
            task_id=self.task_id,
            step_number=0,
            episode_done=False,
        )
        
        self.episode_actions = []
        self.step_count = 0
        self.episode_reward = 0.0
        
        return ResetResult(
            observation=self.current_observation,
            info={"episode": "started", "task": self.task_id}
        )
    
    def step(self, action: Action) -> StepResult:
        """Execute one action and return new observation + reward"""
        
        if self.current_observation is None:
            raise RuntimeError("Must call reset() before step()")
        
        self.step_count += 1
        self.episode_actions.append(action)
        
        # Compute reward for this action
        reward = self._compute_reward(action)
        self.episode_reward += reward.value
        
        # Check if episode is done
        done = (
            self.step_count >= self.max_steps or
            action.action_type == ActionType.CLOSE
        )
        
        # Update observation for next step
        self.current_observation.step_number = self.step_count
        self.current_observation.episode_done = done
        
        return StepResult(
            observation=self.current_observation,
            reward=reward,
            done=done,
            info={
                "step": self.step_count,
                "action_type": action.action_type.value,
                "cumulative_reward": self.episode_reward,
            }
        )
    
    def state(self) -> Dict[str, Any]:
        """Return current environment state"""
        
        return {
            "current_observation": self.current_observation.dict() if self.current_observation else None,
            "step_count": self.step_count,
            "episode_actions": [a.dict() for a in self.episode_actions],
            "episode_reward": self.episode_reward,
            "task_id": self.task_id,
        }
    
    def close(self) -> None:
        """Cleanup"""
        pass
    
    # ========================================================================
    # PRIVATE METHODS
    # ========================================================================
    
    def _compute_reward(self, action: Action) -> Reward:
        """Compute step reward based on action"""
        
        base_reward = 0.0
        correct = False
        
        # Get ground truth for this ticket
        ticket_data = self._find_ticket_data(
            self.current_observation.subject,
            self.current_observation.body
        )
        
        if not ticket_data:
            return Reward(value=0.0)
        
        # Reward correct actions
        if action.action_type == ActionType.CLASSIFY:
            if action.severity == ticket_data["true_severity"]:
                base_reward = 0.3
                correct = True
            else:
                base_reward = -0.1
        
        elif action.action_type == ActionType.ASSIGN:
            if action.assigned_team == ticket_data["true_team"]:
                base_reward = 0.3
                correct = True
            else:
                base_reward = -0.15
        
        elif action.action_type == ActionType.RESPOND:
            # Reward thoughtful responses
            if action.response_text and len(action.response_text) > 30:
                base_reward = 0.2
                correct = True
            else:
                base_reward = -0.1
        
        elif action.action_type == ActionType.ESCALATE:
            # Escalation is good for complex/critical issues
            if ticket_data["true_severity"] in [Severity.CRITICAL, Severity.HIGH]:
                base_reward = 0.2
                correct = True
            else:
                base_reward = -0.05
        
        elif action.action_type == ActionType.CLOSE:
            # Closing is a final action - grade whole episode
            if len(self.episode_actions) >= 2:
                base_reward = 0.25
                correct = True
            else:
                base_reward = -0.2
        
        # SLA compliance bonus/penalty
        sla_violated = self.step_count > (self.current_observation.metadata.sla_minutes / 10)
        if sla_violated:
            base_reward -= 0.1
        
        return Reward(
            value=max(-0.99, min(0.99, base_reward)),
            correct_action=correct,
            sla_violated=sla_violated,
            efficiency=1.0 / (1.0 + self.step_count),
            customer_satisfaction=self.current_observation.customer_sentiment,
        )
    
    def _find_ticket_data(self, subject: str, body: str) -> Optional[Dict]:
        """Find ticket in synthetic data"""
        for ticket in SYNTHETIC_TICKETS:
            if ticket["subject"] == subject and ticket["body"] == body:
                return ticket
        return None
    
    def _get_grader(self) -> TaskGrader:
        """Get the appropriate grader for this task"""
        if self.task_id == "ticket-classification-easy":
            return EasyClassificationGrader()
        elif self.task_id == "ticket-routing-medium":
            return MediumRoutingGrader()
        elif self.task_id == "ticket-handling-hard":
            return HardComplexGrader()
        else:
            return EasyClassificationGrader()


# ============================================================================
# CONVENIENCE FUNCTION FOR GRADING FULL EPISODES
# ============================================================================

def grade_episode(task_id: str, actions: List[Action], observation: Observation) -> float:
    """Grade a complete episode"""
    grader = {
        "ticket-classification-easy": EasyClassificationGrader(),
        "ticket-routing-medium": MediumRoutingGrader(),
        "ticket-handling-hard": HardComplexGrader(),
    }.get(task_id, EasyClassificationGrader())
    
    return grader.grade(observation, actions)
