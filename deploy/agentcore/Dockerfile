FROM python:3.12-slim

ARG RUNTIME_REQUIREMENTS=deploy/agentcore/requirements.txt
ARG WHEEL_FILE

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OPENINFERENCE_HIDE_INPUTS=true \
    OPENINFERENCE_HIDE_OUTPUTS=true

WORKDIR /app
COPY ${RUNTIME_REQUIREMENTS} /tmp/requirements.txt
RUN pip install --no-cache-dir --requirement /tmp/requirements.txt
COPY ${WHEEL_FILE} /tmp/
RUN pip install --no-cache-dir --no-deps /tmp/*.whl

EXPOSE 8080
CMD ["opentelemetry-instrument", "uvicorn", "agentops_demo.agentcore.server:create_app_from_environment", "--factory", "--host", "0.0.0.0", "--port", "8080"]
