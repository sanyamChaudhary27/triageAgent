# Deployment Guide: Customer Support Triage Environment

## 🚀 Deploying to Hugging Face Spaces

### Step 1: Create a HuggingFace Account (if needed)
1. Go to https://huggingface.co/join
2. Create account with email
3. Verify email
4. Create access token at https://huggingface.co/settings/tokens

### Step 2: Create a New Space

1. Go to https://huggingface.co/spaces
2. Click "New Space"
3. Fill in details:
   - **Space name**: `customer-support-triage` (or your preference)
   - **License**: MIT
   - **Space SDK**: Docker
   - **Visibility**: Public (for competition)
4. Click "Create Space"

### Step 3: Upload Project Files

You now have a Git repo for your Space. Clone and push:

```bash
# Clone the space repo
git clone https://huggingface.co/spaces/your-username/customer-support-triage
cd customer-support-triage

# Copy all your files here
cp ../env.py .
cp ../inference.py .
cp ../openenv.yaml .
cp ../requirements.txt .
cp ../Dockerfile .
cp ../README.md .

# Push to HuggingFace
git add .
git commit -m "Initial commit: Customer Support Triage Environment"
git push origin main
```

### Step 4: Configure Environment Variables

1. Go to your Space settings: https://huggingface.co/spaces/your-username/customer-support-triage/settings
2. Under "Repository secrets", add:
   - `HF_TOKEN`: Your Hugging Face API token
   - `API_BASE_URL`: (optional) Default is https://router.huggingface.co/v1
   - `MODEL_NAME`: (optional) Default is meta-llama/Llama-2-7b-chat-hf

3. The Space will auto-restart and deploy

### Step 5: Monitor Deployment

1. Click "Logs" tab to see build output
2. Wait for "Building..." to become "Running"
3. Once deployed, you'll see a URL like:
   - https://huggingface.co/spaces/your-username/customer-support-triage

### Step 6: Test Your Deployment

```bash
# Test from command line
curl https://huggingface.co/spaces/your-username/customer-support-triage

# Or use Python
python -c "
import requests
response = requests.get('https://huggingface.co/spaces/your-username/customer-support-triage')
print(f'Status: {response.status_code}')
"
```

---

## 📋 Troubleshooting

### Space won't build
**Error**: Docker build fails
**Fix**:
- Check Docker file for syntax errors
- Ensure all required files are in the directory
- Check that requirements.txt has all dependencies

### Inference script hangs
**Error**: Space times out or hangs
**Fix**:
- Set a timeout in your inference script
- Limit episodes per run (currently 3 per task)
- Add logging to see progress

### API key not found
**Error**: "OPENAI_API_KEY not found"
**Fix**:
- Add secret to Space settings
- Use correct secret name in code
- Restart Space after adding secrets

### Model fails to load
**Error**: "Model not found" or "Invalid model name"
**Fix**:
- Verify MODEL_NAME is correct
- Check that model exists on Hugging Face
- Ensure you have permission to access model

---

## 🔄 Iterating During Development

### Local Testing
```bash
# Test locally before pushing
python validate.py        # Run tests
python -c "
from env import CustomerSupportTriageEnv
env = CustomerSupportTriageEnv()
obs = env.reset().observation
print(obs.subject)
"
```

### Docker Testing
```bash
# Build and test Docker image locally
docker build -t my-env:latest .
docker run -it \
  -e OPENAI_API_KEY="sk-..." \
  -e MODEL_NAME="gpt-4" \
  my-env:latest
```

### Push Updates
```bash
git add .
git commit -m "Update: better reward shaping"
git push origin main
# Space auto-rebuilds
```

---

## 📊 Monitoring Your Space

### Check Hardware Usage
1. Go to Settings → Hardware → Space Hardware
2. Default: **2 vCPU, 16GB RAM**
3. For inference, 2vCPU + 8GB is minimum

### View Logs
1. Click "Logs" tab
2. See build output and runtime logs
3. Useful for debugging API errors

### Monitor Inference Runs
Space will show:
- Last run timestamp
- Execution status (success/failed)
- Runtime duration

---

## 🎯 Pre-Submission Checklist

Before submitting to the competition, verify:

- ✅ Space is public
- ✅ Dockerfile builds without errors
- ✅ All environment variables are set
- ✅ `inference.py` runs without errors
- ✅ Baseline scores are generated
- ✅ README clearly documents everything
- ✅ OpenEnv spec validation passes
- ✅ At least 3 tasks defined with graders

### Validation Script
```bash
# Run locally to verify compliance
python validate.py

# Output should be:
# ✓ PASS | File Structure
# ✓ PASS | OpenEnv YAML
# ✓ PASS | Imports
# ✓ PASS | Environment Initialization
# ✓ PASS | reset() Function
# ✓ PASS | step() Function
# ✓ PASS | state() Function
# ✓ PASS | Grader Functions
```

---

## 📝 Submission Instructions

When you're ready to submit:

1. **Copy Space URL**
   - Format: `https://huggingface.co/spaces/username/customer-support-triage`

2. **Prepare submission document** with:
   - Space URL
   - Environment description
   - Key design decisions
   - Baseline score summary
   - Link to GitHub repo (optional)

3. **Submit to competition**
   - Go to Scaler School submission portal
   - Fill in environment details
   - Paste Space URL
   - Submit!

---

## 🆘 Getting Help

If you encounter issues:

1. **Check Logs**: Space → Logs tab
2. **Check Dependencies**: Ensure all in requirements.txt
3. **Local Test**: Run `python validate.py` locally
4. **Docker Test**: Build and run Docker image locally
5. **Ask on forums**: Scaler School Discord/forums

---

## 📚 Additional Resources

- OpenEnv Spec: https://huggingface.co/docs/openenv/
- Docker Docs: https://docs.docker.com/
- HF Spaces Docs: https://huggingface.co/docs/hub/spaces
- Competition Details: Check Scaler School portal

---

**You're ready to deploy! Good luck! 🚀**
