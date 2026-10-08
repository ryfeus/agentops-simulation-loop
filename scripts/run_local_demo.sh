#!/usr/bin/env bash
set -euo pipefail

demo_mode="${1:-bad}"
if [[ "${demo_mode}" != "bad" && "${demo_mode}" != "correct" ]]; then
  echo "mode must be 'bad' or 'correct'" >&2
  exit 2
fi

export BILLING_DATABASE_PATH="${BILLING_DATABASE_PATH:-data/billing.db}"
export BILLING_POLICY_MODE="permissive"
export BILLING_MCP_HOST="${BILLING_MCP_HOST:-127.0.0.1}"
export BILLING_MCP_PORT="${BILLING_MCP_PORT:-8000}"
export BILLING_MCP_URL="http://${BILLING_MCP_HOST}:${BILLING_MCP_PORT}/mcp"

uv run python -m agentops_demo.cli.init_db --database "${BILLING_DATABASE_PATH}"
uv run python -m agentops_demo.mcp.billing_server &
mcp_pid=$!

cleanup() {
  if kill -0 "${mcp_pid}" 2>/dev/null; then
    kill "${mcp_pid}" 2>/dev/null || true
    wait "${mcp_pid}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

ready=0
for _attempt in $(seq 1 100); do
  if curl --fail --silent "http://${BILLING_MCP_HOST}:${BILLING_MCP_PORT}/health" >/dev/null; then
    ready=1
    break
  fi
  if ! kill -0 "${mcp_pid}" 2>/dev/null; then
    echo "MCP server exited before becoming ready" >&2
    exit 1
  fi
  sleep 0.05
done

if [[ "${ready}" != "1" ]]; then
  echo "MCP server did not become ready" >&2
  exit 1
fi

uv run python -m agentops_demo.cli.scripted_demo \
  --mode "${demo_mode}" \
  --database "${BILLING_DATABASE_PATH}" \
  --mcp-url "${BILLING_MCP_URL}"
