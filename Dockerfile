FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY uv.lock .
COPY env.py .
COPY inference.py .
COPY server/ server/
COPY openenv.yaml .
COPY README.md .

# Install Python dependencies and the project
RUN pip install --no-cache-dir .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default API configuration (can be overridden)
ENV API_BASE_URL="https://api.groq.com/openai/v1"
ENV MODEL_NAME="llama-3.3-70b-versatile"

# Health check endpoint
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from env import CustomerSupportTriageEnv; env = CustomerSupportTriageEnv(); env.reset(); print('OK')"

# Start the environment server via the installed script
EXPOSE 7860
CMD ["server"]
