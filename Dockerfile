# Dockerfile
FROM python:3.10-slim-bookworm

# Set environment variables for Python and uv
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install CA certificates and basic tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Install uv globally
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# Copy dependency files first to leverage Docker layer caching
COPY pyproject.toml uv.lock ./

# Install production dependencies only using uv
RUN uv sync --frozen --no-dev

# Copy the rest of the application code
COPY src/ /app/src/
COPY configs/ /app/configs/

# Expose port for SageMaker / API Gateway integration
EXPOSE 8080

# Command to run the FastAPI server (SageMaker expects port 8080)
CMD ["/app/.venv/bin/uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8080"]