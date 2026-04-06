FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY requirements.txt .
COPY env.py .
COPY inference.py .
COPY openenv.yaml .
COPY README.md .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default API configuration (can be overridden)
ENV API_BASE_URL="https://api.groq.com/openai/v1"
ENV MODEL_NAME="llama-3.3-70b-versatile"

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from env import CustomerSupportTriageEnv; env = CustomerSupportTriageEnv(); env.reset(); print('OK')"

# Start the OpenEnv environment API server
EXPOSE 7860
CMD openenv serve . --host 0.0.0.0 --port 7860
