#!/usr/bin/env bash
set -u

mkdir -p /logs/verifier
if ! python /tests/verify.py; then
  printf '%s\n' '{"reward":0.0,"no_refund":0.0,"must_escalate":0.0}' \
    > /logs/verifier/reward.json
  printf '%s\n' '{"error":"verifier process failed"}' \
    > /logs/verifier/diagnostics.json
fi
