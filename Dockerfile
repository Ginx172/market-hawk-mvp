# Multi-stage build for Market Hawk MVP
FROM python:3.12-slim AS builder
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 MH_LOG_LEVEL=INFO
COPY . .
RUN useradd -m -u 1000 trader && chown -R trader:trader /app && mkdir -p /app/logs /app/data/cache
USER trader
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 CMD python -c "import sys; sys.exit(0)"
CMD ["python", "-m", "trading.paper_trader", "--status"]