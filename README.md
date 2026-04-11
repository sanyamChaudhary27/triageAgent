---
title: Customer Support Triage
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Customer Support Ticket Triage Environment

## 🎯 Overview

A **real-world OpenEnv environment** for training and evaluating AI agents on customer support ticket triage. Agents learn to classify ticket severity, assign to appropriate teams, and route complex issues while managing SLA constraints.

This environment models a task that **millions of support teams worldwide perform daily**, making it immediately valuable for agent evaluation and training.

### Why This Matters

- **Real-world utility**: Companies spend billions annually on support operations
- **Agent challenge**: Requires multi-step reasoning, contextual understanding, and constraint satisfaction
- **Reward shaping**: Rich signal over trajectories, not just binary success/failure
- **Curriculum learning**: Natural progression from classification → routing → complex handling

---

## 📋 Environment Specification

### Task Definitions

#### Task 1: Ticket Classification (Easy) ⭐

**Objective**: Correctly classify ticket severity and assign to team.

| Property              | Value                   |
| --------------------- | ----------------------- |
| **Difficulty**        | Easy                    |
| **Expected Baseline** | ~65-75%                 |
| **Evaluation**        | Classification accuracy |
| **Episode Length**    | 2-4 steps               |

**Success Criteria**:

- Correctly classify severity (5 classes: critical, high, medium, low, info)
- Assign to correct team (5 teams: billing, technical, product, sales, general)
- Respond within SLA window

**Example Ticket**:

```
Subject: App not responding on mobile
Body: The app crashes every time I try to upload a file. Error: RuntimeException
→ True Labels: CRITICAL severity, TECHNICAL team
```

---

#### Task 2: Ticket Routing & Prioritization (Medium) ⭐⭐

**Objective**: Route tickets intelligently under resource constraints.

| Property              | Value                             |
| --------------------- | --------------------------------- |
| **Difficulty**        | Medium                            |
| **Expected Baseline** | ~45-60%                           |
| **Evaluation**        | Routing accuracy + SLA compliance |
| **Episode Length**    | 3-6 steps                         |

**Success Criteria**:

- Route to correct team
- Respect SLA response time (varies: CRITICAL=15min, HIGH=60min, etc.)
- Prioritize efficiently
- Avoid unnecessary actions

**Complexity Factors**:

- Multiple ticket features (tier, history, sentiment)
- SLA windows create time-based constraints
- Repeat customers get priority
- Enterprise customers have tighter SLAs

---

#### Task 3: Complex Ticket Handling (Hard) ⭐⭐⭐

**Objective**: Handle multi-part tickets requiring escalation and context awareness.

| Property              | Value                |
| --------------------- | -------------------- |
| **Difficulty**        | Hard                 |
| **Expected Baseline** | ~25-40%              |
| **Evaluation**        | Full episode grading |
| **Episode Length**    | 4-10 steps           |

**Success Criteria**:

- Identify critical issues first
- Escalate appropriately when needed
- Provide contextual, empathetic responses
- Manage conversation history

**Example Ticket**:

```
Subject: Multiple issues: Performance + API + billing
Body: 1) Dashboard is slow. 2) API rate limit too low.
      3) Billing shows duplicate charge. Need all fixed ASAP.
→ Requires: ESCALATE, multiple CLASSIFY actions, thoughtful RESPOND
```

---

## 🔧 Observation Space

```python
class Observation(BaseModel):
    ticket_id: str                          # Unique ticket ID
    customer_id: str                        # Customer identifier
    subject: str                            # Email subject
    body: str                               # Full ticket description
    priority_hint: Priority                 # Customer-provided priority
    customer_sentiment: float               # NLP sentiment [0=angry, 1=happy]
    conversation_history: List[Message]    # Previous messages in thread
    metadata: TicketMetadata               # Creation time, tier, SLA, etc.
    task_id: str                            # Current task identifier
    step_number: int                        # Current step in episode
    episode_done: bool                      # Episode termination flag
```

**Size**: ~500-2000 tokens per observation (depending on conversation history)

---

## 🎬 Action Space

Valid actions (mutually exclusive per step):

```python
class Action(BaseModel):
    action_type: ActionType  # classify | assign | respond | escalate | close

    # For CLASSIFY:
    severity: Severity       # critical, high, medium, low, info

    # For ASSIGN:
    assigned_team: Team      # billing, technical, product, sales, general

    # For RESPOND:
    response_text: str       # Custom response to customer

    # For ESCALATE:
    reason: str              # Why escalation is needed

    # For CLOSE:
    # (no additional fields)
```

**Example Actions**:

```json
{"action_type": "classify", "severity": "critical"}
{"action_type": "assign", "assigned_team": "technical"}
{"action_type": "respond", "response_text": "We're on it! Here's..."}
{"action_type": "escalate", "reason": "Needs senior engineer..."}
{"action_type": "close"}
```

---

## 🏆 Reward Function

Episode reward = sum of step rewards

**Step Reward Components**:

- **Correct classification**: +0.3
- **Correct assignment**: +0.3
- **Thoughtful response** (>30 chars): +0.2
- **Appropriate escalation**: +0.2
- **SLA violation penalty**: -0.1
- **Incorrect classification**: -0.1
- **Team mismatch**: -0.15

**Episode Properties**:

- Rich reward signal (not sparse)
- Rewards partial progress
- Penalizes inefficiency
- Captures SLA constraints

---

## 📦 API Reference

### Environment Initialization

```python
from env import CustomerSupportTriageEnv

# Easy classification task
env = CustomerSupportTriageEnv(task_id="ticket-classification-easy")

# Medium routing task
env = CustomerSupportTriageEnv(task_id="ticket-routing-medium")

# Hard complex handling
env = CustomerSupportTriageEnv(task_id="ticket-handling-hard")
```

### Reset

```python
reset_result = env.reset()
observation = reset_result.observation
info = reset_result.info

# observation is fully typed Observation model
# info contains: {"episode": "started", "task": "..."}
```

### Step

```python
from env import Action, ActionType, Severity

action = Action(
    action_type=ActionType.CLASSIFY,
    severity=Severity.CRITICAL
)

step_result = env.step(action)
observation = step_result.observation
reward = step_result.reward
done = step_result.done
info = step_result.info
```

### State

```python
state = env.state()
# Returns dict with current observation, actions, cumulative reward, etc.
```

---

## 🚀 Setup & Usage

### Local Development

```bash
# Clone repo
git clone <repo-url>
cd customer-support-triage

# Install dependencies
pip install -r requirements.txt

# Run a single episode
python -c "
from env import CustomerSupportTriageEnv, Action, ActionType, Severity

env = CustomerSupportTriageEnv(task_id='ticket-classification-easy')
obs = env.reset().observation

print(f'Ticket: {obs.subject}')
print(f'Body: {obs.body}')

# Take an action
action = Action(action_type=ActionType.CLASSIFY, severity=Severity.CRITICAL)
step_result = env.step(action)
print(f'Reward: {step_result.reward.value}')
"
```

### Run Baseline Inference

```bash
# Set API credentials
export OPENAI_API_KEY="your-key"
export MODEL_NAME="gpt-4"
export API_BASE_URL="https://api.openai.com/v1"

# Run baseline on all 3 tasks
python inference.py

# Results saved to baseline_results.json
```

### Docker

```bash
# Build image
docker build -t customer-support-triage:latest .

# Run with environment variables
docker run \
  -e OPENAI_API_KEY="your-key" \
  -e MODEL_NAME="gpt-4" \
  customer-support-triage:latest
```

### Hugging Face Spaces

```bash
# Create Space at https://huggingface.co/new-space
# Upload files:
#   - env.py
#   - inference.py
#   - openenv.yaml
#   - requirements.txt
#   - Dockerfile
#   - README.md

# Space auto-deploys from Dockerfile
# Access at: https://huggingface.co/spaces/sanyamChaudhary27/customer-support-triage
```

---

## 📊 Baseline Performance

**Model**: Llama 2 7B Chat  
**Setup**: 2 vCPU, 8GB RAM  
**Runtime**: ~12 minutes (3 tasks × 3 episodes)

| Task                    | Episodes | Avg Score | Max  | Min  |
| ----------------------- | -------- | --------- | ---- | ---- |
| **Easy Classification** | 3        | 0.68      | 0.81 | 0.52 |
| **Medium Routing**      | 3        | 0.54      | 0.67 | 0.41 |
| **Hard Handling**       | 3        | 0.35      | 0.48 | 0.19 |
| **Overall Average**     | 9        | **0.52**  | —    | —    |

### Interpretation

- **Easy task**: Model masters basic classification
- **Medium task**: More challenging with constraints and routing logic
- **Hard task**: Multi-step reasoning with escalation needed

---

## 🏗️ Project Structure

```
customer-support-triage/
├── env.py                    # Main environment (OpenEnv spec compliant)
├── inference.py              # Baseline inference script
├── openenv.yaml             # OpenEnv metadata
├── requirements.txt         # Python dependencies
├── Dockerfile               # Container definition
├── README.md               # This file
└── baseline_results.json   # Sample baseline scores
```

---

## ✅ OpenEnv Compliance

This environment fully implements the OpenEnv specification:

- ✅ **Typed Models**: All observations, actions, rewards use Pydantic
- ✅ **API**: `reset()`, `step()`, `state()` with correct signatures
- ✅ **Metadata**: Complete `openenv.yaml` with task definitions
- ✅ **Graders**: Deterministic graders for all 3 tasks (0.0-1.0 scale)
- ✅ **Validation**: Can pass `openenv validate` command
- ✅ **Docker**: Working `Dockerfile` with health checks
- ✅ **Inference**: Baseline script using OpenAI client

Run validation:

```bash
pip install openenv
openenv validate .
```

---

## 🎓 Future Extensions

Possible enhancements for future iterations:

1. **Dynamic task generator**: Generate synthetic tickets programmatically
2. **Real dataset integration**: Integrate with actual support ticket datasets
3. **Multi-agent scenarios**: Multiple agents competing for queue capacity
4. **Conversation continuation**: Multi-turn conversations with model
5. **A/B testing framework**: Built-in AB test harness
6. **Reward learning**: Learn reward function from human feedback

---

## 📝 Citation

If you use this environment in research, please cite:

```bibtex
@software{support_triage_2026,
  title={Customer Support Ticket Triage Environment},
  author={Sanyam Chaudhary},
  year={2026},
  url={https://huggingface.co/spaces/sanyamChaudhary27/customer-support-triage}
}
```

---

## 📄 License

MIT License - See LICENSE file

---

## 🤝 Contributing

Contributions welcome! Submit issues and PRs to improve:

- Task definitions
- Reward shaping
- Synthetic data diversity
- Grader accuracy

---

## ❓ FAQ

**Q: How long does an episode take?**  
A: 3-10 steps depending on task. With API calls, ~30-60 seconds per episode.

**Q: Can I use my own model?**  
A: Yes! Any OpenAI-compatible API works. Set `API_BASE_URL` and `MODEL_NAME`. For Groq, use `https://api.groq.com/openai/v1`.

**Q: What's the minimum hardware?**  
A: 2 vCPU + 8GB RAM for inference. Environment itself is lightweight.

**Q: How many tasks can I add?**  
A: Easily extensible. Add new task definitions to `openenv.yaml` and graders in `env.py`.

**Q: Can I modify the environment?**  
A: Absolutely! It's designed for customization. Maintain OpenEnv spec compliance.

---
