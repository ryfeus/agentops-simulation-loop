#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -eq 0 ]]; then
  echo "usage: $0 COMMAND [ARG ...]" >&2
  exit 2
fi

profile_value="${AWS_PROFILE:-default}"
expected_account_id="${EXPECTED_AWS_ACCOUNT_ID:-}"
if [[ ! "$expected_account_id" =~ ^[0-9]{12}$ ]]; then
  echo "EXPECTED_AWS_ACCOUNT_ID must be set to exactly 12 digits" >&2
  exit 1
fi

export TF_VAR_expected_aws_account_id="$expected_account_id"
export AWS_PROFILE="$profile_value"
export AWS_SDK_LOAD_CONFIG=1
export AWS_REGION=us-west-2
export AWS_DEFAULT_REGION=us-west-2

if ! identity_json="$(aws sts get-caller-identity --profile "$profile_value" --output json)"; then
  echo "AWS credentials are unavailable; run: aws sso login --profile $profile_value" >&2
  exit 1
fi

account_id="$(jq -r '.Account' <<<"$identity_json")"
if [[ "$account_id" != "$expected_account_id" ]]; then
  echo "refusing AWS operation in account $account_id; expected $expected_account_id" >&2
  exit 1
fi

# Export short-lived SSO credentials only into this process and its child.
eval "$(aws configure export-credentials --profile "$profile_value" --format env)"
exec "$@"
