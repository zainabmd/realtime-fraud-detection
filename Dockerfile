# One image, four roles (producer / consumer / api / dashboard) selected by the command.
FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

# librdkafka is bundled in the confluent-kafka wheel; no system packages needed.
COPY requirements.txt .
RUN pip install -r requirements.txt confluent-kafka==2.6.1

COPY src/ src/
COPY dashboard/ dashboard/
COPY models/ models/
COPY .streamlit/ .streamlit/

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 fraud && mkdir -p /app/data && chown -R fraud /app
USER fraud

HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD python -c "import src.config" || exit 1
CMD ["python", "-m", "src.consumer"]
