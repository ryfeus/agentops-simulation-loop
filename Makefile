.PHONY: test test-unit test-integration test-harbor smoke-real lint web-install web-check \
	validate-scenarios check demo-db demo-reset mcp chat chat-cli demo harbor-runtime \
	harbor-runtime-check harbor-seed harbor-seed-check harbor-configs \
	harbor-build-agent harbor-oracle harbor-bad harbor-correct harbor-e2e \
	agentcore-runtime agentcore-runtime-check agentcore-image-check rl-smoke-config-check rl-harbor-config-check rl-harbor-env-check rl-harbor-baseline-config-check rl-harbor-baseline-suite-check tf-fmt tf-validate aws-login \
	aws-plan aws-foundation aws-dsql-bootstrap aws-dsql-reset aws-dsql-show \
	aws-agentcore-config aws-agentcore-image aws-agentcore-deploy \
	aws-agentcore-smoke aws-chat aws-observability-bootstrap aws-observability-check \
	aws-observability-status aws-observability-restore aws-observability-smoke aws-evaluator-build aws-evaluation-plan \
	aws-evaluation-deploy aws-evaluation-status aws-evaluation-trigger \
	aws-evaluation-results aws-evaluation-find-failure aws-phase4-demo aws-up aws-reset aws-down aws-taskify-failure \
	taskify-fixture taskify-fixture-harbor \
	benchmark-billing benchmark-billing-render benchmark-billing-noop benchmark-billing-oracle \
	benchmark-billing-generated benchmark-billing-generated-check benchmark-billing-bedrock \
	benchmark-billing-phase10 benchmark-billing-phase10-check benchmark-billing-phase10-stats \
	benchmark-billing-phase10-render-smoke benchmark-billing-phase10-render benchmark-billing-phase10-oracle \
	benchmark-billing-phase10-calibrate benchmark-billing-phase10-acceptance \
	aws-harbor-ec2-billing \
	aws-harbor-ec2-tfvars aws-harbor-ec2-plan aws-harbor-ec2-foundation \
	harbor-ec2-config-check aws-harbor-ec2-preflight aws-harbor-ec2-oracle \
	aws-harbor-ec2-correct aws-harbor-ec2-bad aws-harbor-ec2-parity \
	aws-harbor-ec2-scale aws-harbor-ec2-status aws-harbor-ec2-cleanup aws-phase6-demo \
	aws-rl-smoke-plan aws-rl-smoke-up aws-rl-smoke-status aws-rl-smoke-run \
	aws-rl-smoke-evidence aws-rl-smoke-down aws-rl-harbor-smoke-run aws-rl-harbor-smoke-evidence aws-rl-harbor-baseline-run aws-rl-harbor-baseline-evidence aws-rl-harbor-overfit-run aws-rl-harbor-overfit-evidence aws-rl-harbor-overfit-fetch-artifacts aws-rl-harbor-overfit-eval-run aws-rl-harbor-overfit-eval-evidence \
	rl-harbor-overfit-config-check rl-harbor-overfit-source-check rl-harbor-overfit-eval-config-check \
	rl-harbor-generalization-config-check rl-harbor-generalization-selection-check rl-harbor-generalization-report \
	aws-rl-harbor-generalization-train aws-rl-harbor-generalization-train-evidence \
	aws-rl-harbor-generalization-final-eval aws-rl-harbor-generalization-final-evidence \
	aws-e2e-demo-preflight aws-e2e-demo aws-e2e-demo-resume aws-e2e-demo-replay \
	e2e-demo-show e2e-demo-clean


MODE ?= bad
HARBOR_TASK := benchmarks/disputed-refund
HARBOR_JOBS := .harbor/jobs
HARBOR_MANIFEST := .harbor/package/manifest.json
HARBOR_AGENT := agentops_demo.harbor.agent:LangGraphBillingAgent
TF_DIR := infra/terraform
AWS_RUN := scripts/with_aws_credentials.sh
AGENTCORE_TFVARS := $(abspath .agentcore/deployment.auto.tfvars.json)
MODEL_ID ?= us.anthropic.claude-sonnet-4-6
MODEL_PROVIDER ?= bedrock
LOCAL_CHAT_CANDIDATE = $(if $(AGENTCORE_CANDIDATE),$(AGENTCORE_CANDIDATE),scripted-correct)
AWS_CHAT_CANDIDATE = $(if $(AGENTCORE_CANDIDATE),$(AGENTCORE_CANDIDATE),bedrock)
AGENTCORE_EVAL_CANDIDATE ?= scripted-bad
AGENTCORE_E2E_CANDIDATE ?= bedrock
HARBOR_EC2_ROOT := .harbor-ec2
HARBOR_EC2_OUTPUTS := $(HARBOR_EC2_ROOT)/terraform-outputs.json
HARBOR_EC2_TASK_SOURCE ?= taskify
HARBOR_EC2_SCENARIO ?= .taskify/disputed-refund-000000000000
RL_SMOKE_TF_DIR := infra/terraform-rl-smoke
RL_SMOKE_ROOT := .rl-smoke
RL_SMOKE_OUTPUTS := $(RL_SMOKE_ROOT)/terraform-outputs.json
PHASE6_DEMO_RUN_ID ?= phase6-demo-$(shell date +%Y%m%d%H%M%S)
DEMO_RUN_ID ?= demo-$(shell date +%Y%m%d%H%M%S)
BENCHMARK_CANDIDATE ?= correct
BENCHMARK_OUTPUT ?=

test:
	uv run --group harbor --extra aws --extra dsql --extra web pytest \
		-m "not real_model and not aws_dsql and not aws_agentcore and not aws_observability and not aws_evaluation"

test-unit:
	uv run --extra aws --extra dsql --extra web pytest tests/unit

test-integration:
	uv run --extra aws --extra dsql --extra web pytest tests/integration

test-harbor:
	uv run --group harbor pytest tests/harbor

smoke-real:
	RUN_REAL_MODEL_TESTS=1 uv run --extra bedrock pytest -m real_model tests/smoke

lint:
	uv run ruff check .
	uv run ruff format --check .

validate-scenarios:
	uv run python -m agentops_demo.validation.scenario scenarios/

demo-db:
	uv run python -m agentops_demo.cli.init_db

demo-reset: demo-db

mcp:
	uv run python -m agentops_demo.mcp.billing_server

web-install:
	pnpm --dir web install --frozen-lockfile

web-check: web-install
	pnpm --dir web lint
	pnpm --dir web typecheck
	pnpm --dir web build

chat: web-install
	uv run --extra web --extra bedrock python -m scripts.run_web_chat \
		--candidate "$(LOCAL_CHAT_CANDIDATE)"

chat-cli:
	uv run --extra bedrock python -m agentops_demo.cli.chat \
		--candidate "$(LOCAL_CHAT_CANDIDATE)"

demo:
	bash scripts/run_local_demo.sh "$(MODE)"

harbor-runtime:
	uv export --frozen --no-dev --no-emit-project --extra bedrock --no-hashes --no-header \
		--output-file $(HARBOR_TASK)/environment/requirements.txt

harbor-runtime-check:
	@temporary=$$(mktemp); \
	uv export --frozen --no-dev --no-emit-project --extra bedrock --no-hashes --no-header \
		--output-file $$temporary >/dev/null; \
	cmp -s $$temporary $(HARBOR_TASK)/environment/requirements.txt; result=$$?; \
	rm -f $$temporary; \
	test $$result -eq 0 || { echo "Harbor runtime requirements are stale" >&2; exit 1; }

harbor-seed:
	uv run python -m scripts.generate_harbor_seed

harbor-seed-check:
	uv run python -m scripts.generate_harbor_seed --check

harbor-configs:
	uv run python -m scripts.generate_harbor_configs

harbor-build-agent:
	uv run python -m scripts.build_harbor_agent

harbor-oracle: harbor-seed-check harbor-runtime-check
	rm -rf $(HARBOR_JOBS)/oracle
	uv run --group harbor harbor run -p $(HARBOR_TASK) -a oracle -m oracle \
		--env docker --n-attempts 1 --n-concurrent 1 --yes \
		--job-name oracle --jobs-dir $(HARBOR_JOBS)
	uv run python -m scripts.assert_harbor_result \
		--job $(HARBOR_JOBS)/oracle --expected-reward 1

harbor-bad: harbor-build-agent harbor-seed-check harbor-runtime-check
	rm -rf $(HARBOR_JOBS)/bad
	@set -e; wheel=$$(uv run python -m scripts.build_harbor_agent --print-wheel); \
	uv run --group harbor harbor run -p $(HARBOR_TASK) \
		-a $(HARBOR_AGENT) -m scripted/bad --env docker \
		--ak agent_config_path=$(abspath .harbor/configs/bad.json) \
		--ak package_path=$$wheel \
		--n-attempts 1 --n-concurrent 1 --yes \
		--job-name bad --jobs-dir $(HARBOR_JOBS)
	uv run python -m scripts.assert_harbor_result \
		--job $(HARBOR_JOBS)/bad --expected-reward 0 \
		--expected-tools get_invoice refund_invoice \
		--provenance-manifest $(HARBOR_MANIFEST) --candidate bad

harbor-correct: harbor-build-agent harbor-seed-check harbor-runtime-check
	rm -rf $(HARBOR_JOBS)/correct
	@set -e; wheel=$$(uv run python -m scripts.build_harbor_agent --print-wheel); \
	uv run --group harbor harbor run -p $(HARBOR_TASK) \
		-a $(HARBOR_AGENT) -m scripted/correct --env docker \
		--ak agent_config_path=$(abspath .harbor/configs/correct.json) \
		--ak package_path=$$wheel \
		--n-attempts 1 --n-concurrent 1 --yes \
		--job-name correct --jobs-dir $(HARBOR_JOBS)
	uv run python -m scripts.assert_harbor_result \
		--job $(HARBOR_JOBS)/correct --expected-reward 1 \
		--expected-tools get_invoice escalate_dispute \
		--provenance-manifest $(HARBOR_MANIFEST) --candidate correct

harbor-e2e: harbor-oracle harbor-bad harbor-correct

check: lint web-check test validate-scenarios harbor-seed-check harbor-runtime-check \
	agentcore-runtime-check

agentcore-runtime:
	uv export --frozen --no-dev --no-group harbor --extra bedrock --extra dsql \
		--extra aws --no-emit-project --no-hashes --no-header \
		--output-file deploy/agentcore/requirements.txt

agentcore-runtime-check:
	uv run --extra aws --extra dsql python -c \
		"from scripts.agentcore_package import verify_runtime_requirements; verify_runtime_requirements()"

agentcore-image-check: agentcore-runtime-check
	uv run python -m scripts.agentcore_package --build-only

tf-fmt:
	terraform fmt -recursive $(TF_DIR)
	terraform fmt -recursive $(RL_SMOKE_TF_DIR)

tf-validate:
	terraform -chdir=$(TF_DIR) init -backend=false
	terraform -chdir=$(TF_DIR) validate
	terraform fmt -check -recursive $(TF_DIR)
	terraform -chdir=$(RL_SMOKE_TF_DIR) init -backend=false
	terraform -chdir=$(RL_SMOKE_TF_DIR) validate
	terraform fmt -check -recursive $(RL_SMOKE_TF_DIR)

aws-login:
	aws sso login --profile "$${AWS_PROFILE:-default}"

rl-smoke-config-check:
	bash scripts/rl_smoke_config_check.sh

rl-harbor-config-check:
	bash scripts/rl_harbor_config_check.sh

rl-harbor-env-check: benchmark-billing-generated-check
	@set -e; stage="$$(mktemp -d "$${TMPDIR:-/tmp}/phase8b-execution.XXXXXX")"; trap 'rm -rf "$$stage"' EXIT; \
	uv run python rl/phase8b/execution_task.py --base benchmarks/billing/tasks/paid-refund-direct --output "$$stage/paid-refund-direct" >/dev/null; \
	wheel="$$(uv run python -c 'from agentops_demo.harbor.provenance import build_clean_wheel; print(build_clean_wheel(__import__("pathlib").Path(".rl-smoke/phase8b/local-wheel"))[0])')"; \
	PHASE8B_TRIAL_ROOT="$$PWD/.rl-smoke/phase8b/local-trials" PHASE8B_WHEEL_PATH="$$wheel" TRL_EXPERIMENTAL_SILENCE=1 uv run --project rl --locked --extra harbor \
		python rl/phase8b/env_smoke.py --task "$$stage/paid-refund-direct" \
		--summary .rl-smoke/phase8b/local-harbor-env-preflight.json

rl-harbor-baseline-config-check:
	PYTHONPATH="$$PWD/src:$$PWD" TRL_EXPERIMENTAL_SILENCE=1 uv run --project rl --locked --extra harbor \
		python -m rl.phase8c.config_check

rl-harbor-baseline-suite-check: benchmark-billing-generated-check
	@set -e; mkdir -p .rl-smoke/phase8c; stage="$$(mktemp -d "$$PWD/.rl-smoke/phase8c/suite.XXXXXX")"; trap 'rm -rf "$$stage"' EXIT; \
	TRL_EXPERIMENTAL_SILENCE=1 uv run --group harbor python -m rl.phase8c.suite_check \
		--stage "$$stage" --cache-root .rl-smoke/phase8c/oracle-cache

rl-harbor-overfit-config-check:
	PYTHONPATH="$$PWD/src:$$PWD" TRL_EXPERIMENTAL_SILENCE=1 uv run --project rl --locked --extra harbor \
		python -m rl.phase9.config_check

rl-harbor-overfit-source-check:
	@test -n "$(PHASE9B_SOURCE_RUN)" || { echo "PHASE9B_SOURCE_RUN is required" >&2; exit 2; }
	PYTHONPATH="$$PWD/src:$$PWD" uv run --project rl --locked --extra harbor \
		python -m scripts.rl_harbor_overfit source-check --run-id "$(PHASE9B_SOURCE_RUN)"

rl-harbor-overfit-eval-config-check:
	@if [ -n "$(PHASE9B_SOURCE_RUN)" ]; then $(MAKE) --no-print-directory rl-harbor-overfit-source-check PHASE9B_SOURCE_RUN="$(PHASE9B_SOURCE_RUN)"; fi
	PYTHONPATH="$$PWD/src:$$PWD" TRL_EXPERIMENTAL_SILENCE=1 uv run --project rl --locked --extra harbor \
		python -m rl.phase9b.config_check

aws-rl-smoke-plan:
	@set -e; tf_args="$$( $(AWS_RUN) uv run --extra aws python -m scripts.rl_smoke terraform-args)"; \
	$(AWS_RUN) terraform -chdir=$(RL_SMOKE_TF_DIR) init; \
	$(AWS_RUN) terraform -chdir=$(RL_SMOKE_TF_DIR) plan $$tf_args

aws-rl-smoke-up:
	@set -e; tf_args="$$( $(AWS_RUN) uv run --extra aws python -m scripts.rl_smoke terraform-args)"; \
	mkdir -p $(RL_SMOKE_ROOT); \
	$(AWS_RUN) terraform -chdir=$(RL_SMOKE_TF_DIR) init; \
	$(AWS_RUN) terraform -chdir=$(RL_SMOKE_TF_DIR) apply -auto-approve $$tf_args; \
	$(AWS_RUN) terraform -chdir=$(RL_SMOKE_TF_DIR) output -json > $(RL_SMOKE_OUTPUTS)

aws-rl-smoke-status:
	$(AWS_RUN) uv run --extra aws python -m scripts.rl_smoke status --outputs $(RL_SMOKE_OUTPUTS)

aws-rl-smoke-run:
	$(AWS_RUN) uv run --extra aws python -m scripts.rl_smoke run --outputs $(RL_SMOKE_OUTPUTS) \
		$(if $(RL_SMOKE_RUN_ID),--run-id $(RL_SMOKE_RUN_ID))

aws-rl-smoke-evidence:
	$(AWS_RUN) uv run --extra aws python -m scripts.rl_smoke evidence \
		$(if $(RL_SMOKE_RUN_ID),--run-id $(RL_SMOKE_RUN_ID))

aws-rl-smoke-down:
	@set -e; tf_args="$$( $(AWS_RUN) uv run --extra aws python -m scripts.rl_smoke terraform-args)"; \
	$(AWS_RUN) terraform -chdir=$(RL_SMOKE_TF_DIR) init; \
	$(AWS_RUN) terraform -chdir=$(RL_SMOKE_TF_DIR) destroy -auto-approve $$tf_args

aws-rl-harbor-smoke-run: benchmark-billing-generated-check
	$(AWS_RUN) uv run --extra aws python -m scripts.rl_harbor_smoke run --outputs $(RL_SMOKE_OUTPUTS) \
		$(if $(RL_SMOKE_RUN_ID),--run-id $(RL_SMOKE_RUN_ID))

aws-rl-harbor-smoke-evidence:
	$(AWS_RUN) uv run --extra aws python -m scripts.rl_harbor_smoke evidence \
		$(if $(RL_SMOKE_RUN_ID),--run-id $(RL_SMOKE_RUN_ID))

aws-rl-harbor-baseline-run:
	@test -z "$(PHASE8C_ATTEMPTS)" || { echo "PHASE8C_ATTEMPTS is retired; use PHASE8C_ATTEMPTS_PER_PASS=4 and PHASE8C_PASSES" >&2; exit 2; }
	$(AWS_RUN) uv run --extra aws python -m scripts.rl_harbor_baseline run --outputs $(RL_SMOKE_OUTPUTS) \
		$(if $(RL_SMOKE_RUN_ID),--run-id $(RL_SMOKE_RUN_ID))

aws-rl-harbor-baseline-evidence:
	$(AWS_RUN) uv run --extra aws python -m scripts.rl_harbor_baseline evidence \
		$(if $(RL_SMOKE_RUN_ID),--run-id $(RL_SMOKE_RUN_ID))

aws-rl-harbor-overfit-run:
	$(AWS_RUN) uv run --extra aws python -m scripts.rl_harbor_overfit run --outputs $(RL_SMOKE_OUTPUTS) \
		$(if $(RL_SMOKE_RUN_ID),--run-id $(RL_SMOKE_RUN_ID))

aws-rl-harbor-overfit-evidence:
	$(AWS_RUN) uv run --extra aws python -m scripts.rl_harbor_overfit evidence \
		$(if $(RL_SMOKE_RUN_ID),--run-id $(RL_SMOKE_RUN_ID))

aws-rl-harbor-overfit-fetch-artifacts:
	$(AWS_RUN) uv run --extra aws python -m scripts.rl_harbor_overfit fetch-artifacts --outputs $(RL_SMOKE_OUTPUTS) \
		$(if $(RL_SMOKE_RUN_ID),--run-id $(RL_SMOKE_RUN_ID))

aws-rl-harbor-overfit-eval-run:
	$(AWS_RUN) uv run --extra aws python -m scripts.rl_harbor_overfit_eval run --outputs $(RL_SMOKE_OUTPUTS) \
		$(if $(RL_SMOKE_RUN_ID),--run-id $(RL_SMOKE_RUN_ID))

aws-rl-harbor-overfit-eval-evidence:
	$(AWS_RUN) uv run --extra aws python -m scripts.rl_harbor_overfit_eval evidence \
		$(if $(RL_SMOKE_RUN_ID),--run-id $(RL_SMOKE_RUN_ID))

rl-harbor-generalization-config-check:
	PYTHONPATH="$$PWD/src:$$PWD" TRL_EXPERIMENTAL_SILENCE=1 uv run --project rl --locked --extra harbor \
		python -m scripts.rl_harbor_generalization config-check

aws-rl-harbor-generalization-train:
	$(AWS_RUN) uv run --extra aws python -m scripts.rl_harbor_generalization train \
		$(if $(PHASE11_RUN_ID),--run-id $(PHASE11_RUN_ID))

aws-rl-harbor-generalization-train-evidence:
	$(AWS_RUN) uv run --extra aws python -m scripts.rl_harbor_generalization train-evidence \
		$(if $(PHASE11_RUN_ID),--run-id $(PHASE11_RUN_ID))

rl-harbor-generalization-selection-check:
	PYTHONPATH="$$PWD/src:$$PWD" uv run python -m scripts.rl_harbor_generalization selection-check \
		$(if $(PHASE11_RUN_ID),--run-id $(PHASE11_RUN_ID))

aws-rl-harbor-generalization-final-eval:
	$(AWS_RUN) uv run --extra aws python -m scripts.rl_harbor_generalization final-eval \
		$(if $(PHASE11_RUN_ID),--run-id $(PHASE11_RUN_ID))

aws-rl-harbor-generalization-final-evidence:
	$(AWS_RUN) uv run --extra aws python -m scripts.rl_harbor_generalization final-evidence \
		$(if $(PHASE11_RUN_ID),--run-id $(PHASE11_RUN_ID))

rl-harbor-generalization-report:
	PYTHONPATH="$$PWD/src:$$PWD" uv run python -m scripts.rl_harbor_generalization report \
		$(if $(PHASE11_RUN_ID),--run-id $(PHASE11_RUN_ID))

aws-plan:
	$(AWS_RUN) terraform -chdir=$(TF_DIR) init
	@set -e; var_files="$$(uv run python -m scripts.terraform_deployment)"; \
	set -- $$var_files; \
	$(AWS_RUN) terraform -chdir=$(TF_DIR) plan "$$@"

aws-foundation:
	$(AWS_RUN) terraform -chdir=$(TF_DIR) init
	@set -e; var_files="$$(uv run python -m scripts.terraform_deployment)"; \
	set -- $$var_files; \
	$(AWS_RUN) terraform -chdir=$(TF_DIR) apply "$$@" -auto-approve

aws-dsql-bootstrap:
	$(AWS_RUN) uv run --extra dsql --extra aws python -m scripts.bootstrap_dsql

aws-dsql-reset:
	$(AWS_RUN) uv run --extra dsql --extra aws python -m scripts.reset_dsql_demo

aws-dsql-show:
	$(AWS_RUN) uv run --extra dsql --extra aws python -m scripts.dsql_admin show

aws-agentcore-config:
	uv run python -m scripts.generate_agentcore_config \
		--model-provider "$(MODEL_PROVIDER)" --model-id "$(MODEL_ID)"

aws-agentcore-image: agentcore-runtime-check
	$(AWS_RUN) uv run --extra aws python -m scripts.agentcore_package
	uv run python -m scripts.generate_agentcore_tfvars

aws-agentcore-deploy:
	@set -e; var_files="$$(uv run python -m scripts.terraform_deployment)"; \
	set -- $$var_files; \
	$(AWS_RUN) terraform -chdir=$(TF_DIR) apply "$$@" -auto-approve
	$(AWS_RUN) terraform -chdir=$(TF_DIR) output

aws-agentcore-smoke:
	$(AWS_RUN) uv run --extra aws python -m scripts.invoke_agentcore
	$(AWS_RUN) uv run --extra dsql --extra aws python -m scripts.dsql_admin \
		show --output .agentcore/smoke/dsql-state.json
	uv run python -m scripts.assert_agentcore_smoke

aws-chat:
	$(AWS_RUN) uv run --extra aws python -m scripts.chat_agentcore \
		--candidate "$(AWS_CHAT_CANDIDATE)"

aws-observability-bootstrap:
	@test "$${AGENTOPS_ALLOW_ACCOUNT_WIDE_OBSERVABILITY_CHANGE:-}" = yes || { echo "set AGENTOPS_ALLOW_ACCOUNT_WIDE_OBSERVABILITY_CHANGE=yes" >&2; exit 2; }
	@test -n "$(TRANSACTION_SEARCH_INDEXING_PERCENTAGE)" || { echo "set TRANSACTION_SEARCH_INDEXING_PERCENTAGE explicitly" >&2; exit 2; }
	$(AWS_RUN) uv run --extra aws python -m scripts.ensure_transaction_search \
		--apply --indexing-percentage "$(TRANSACTION_SEARCH_INDEXING_PERCENTAGE)"

aws-observability-status:
	$(AWS_RUN) uv run --extra aws python -m scripts.ensure_transaction_search --status

aws-observability-restore:
	@test -n "$(OBSERVABILITY_RECEIPT)" || { echo "set OBSERVABILITY_RECEIPT" >&2; exit 2; }
	$(AWS_RUN) uv run --extra aws python -m scripts.ensure_transaction_search --restore "$(OBSERVABILITY_RECEIPT)"

aws-observability-check:
	@test -n "$(TRANSACTION_SEARCH_INDEXING_PERCENTAGE)" || { echo "set TRANSACTION_SEARCH_INDEXING_PERCENTAGE explicitly" >&2; exit 2; }
	$(AWS_RUN) uv run --extra aws python -m scripts.ensure_transaction_search --check \
		--indexing-percentage "$(TRANSACTION_SEARCH_INDEXING_PERCENTAGE)"
	$(AWS_RUN) uv run --extra aws python -m scripts.observability_status

aws-observability-smoke:
	$(AWS_RUN) uv run --extra aws python -m scripts.observability_smoke

aws-evaluator-build:
	uv run python -m scripts.build_evaluator_lambda

aws-evaluation-plan:
	uv run python -m scripts.evaluation_deployment
	@set -e; var_files="$$(uv run python -m scripts.terraform_deployment)"; \
	set -- $$var_files; \
	$(AWS_RUN) terraform -chdir=$(TF_DIR) plan "$$@"

aws-evaluation-deploy:
	uv run python -m scripts.evaluation_deployment --validate
	@set -e; var_files="$$(uv run python -m scripts.terraform_deployment)"; \
	set -- $$var_files; \
	$(AWS_RUN) terraform -chdir=$(TF_DIR) apply "$$@" -auto-approve
	uv run python -m scripts.evaluation_deployment --hydrate

aws-evaluation-status:
	$(AWS_RUN) uv run --extra aws python -m scripts.evaluation_status

aws-evaluation-trigger:
	$(AWS_RUN) uv run --extra aws --extra dsql python -m scripts.trigger_online_failure \
		--candidate "$(AGENTCORE_EVAL_CANDIDATE)"

aws-evaluation-results:
	$(AWS_RUN) uv run --extra aws python -m scripts.read_online_evaluation_results

aws-evaluation-find-failure:
	@test -n "$(SESSION_ID)" || { echo "set SESSION_ID to the interactive AgentCore session" >&2; exit 2; }
	@test -n "$(TRACE_ID)" || { echo "set TRACE_ID to the failed interactive turn trace" >&2; exit 2; }
	$(AWS_RUN) uv run --extra aws python -m scripts.capture_agentcore_failure \
		--session-id "$(SESSION_ID)" --trace-id "$(TRACE_ID)"

aws-phase4-demo: aws-observability-check aws-evaluation-status aws-evaluation-trigger

aws-taskify-failure:
	$(AWS_RUN) uv run --extra aws python -m scripts.taskify create \
		--failure .agentcore/evaluation/failure.json

taskify-fixture:
	rm -rf .taskify/disputed-refund-000000000000
	uv run python -m scripts.taskify create --offline \
		--failure tests/fixtures/taskify/failure.json \
		--trace tests/fixtures/taskify/failing_trace.json
	uv run python -m scripts.taskify validate-scenario \
		--scenario .taskify/disputed-refund-000000000000/scenario.yaml

taskify-fixture-harbor: taskify-fixture
	uv run python -m scripts.taskify validate-benchmark \
		--scenario .taskify/disputed-refund-000000000000/scenario.yaml

benchmark-billing:
	uv run --group harbor python -m scripts.benchmark_billing

benchmark-billing-render:
	uv run python -m scripts.benchmark_billing --render-only $(if $(BENCHMARK_OUTPUT),--output "$(BENCHMARK_OUTPUT)")

benchmark-billing-noop:
	uv run --group harbor python -m scripts.benchmark_billing --candidate noop

benchmark-billing-oracle:
	uv run --group harbor python -m scripts.benchmark_billing --oracle

benchmark-billing-generated:
	uv run python -m scripts.benchmark_billing_generated

benchmark-billing-generated-check:
	uv run python -m scripts.benchmark_billing_generated --check

benchmark-billing-phase10:
	uv run python -m scripts.benchmark_billing_phase10 generate

benchmark-billing-phase10-check:
	uv run python -m scripts.benchmark_billing_phase10 check

benchmark-billing-phase10-stats:
	uv run python -m scripts.benchmark_billing_phase10 stats

benchmark-billing-phase10-render-smoke: benchmark-billing-phase10-check
	uv run python -m scripts.benchmark_billing_phase10 render-smoke

benchmark-billing-phase10-render: benchmark-billing-phase10-check
	uv run python -m scripts.benchmark_billing_phase10 render

benchmark-billing-phase10-oracle: benchmark-billing-phase10-render
	uv run --group harbor python -m scripts.benchmark_billing_phase10 oracle

benchmark-billing-phase10-calibrate: benchmark-billing-phase10-render
	uv run --group harbor python -m scripts.benchmark_billing_phase10 calibrate

benchmark-billing-phase10-acceptance: benchmark-billing-phase10-render
	uv run --group harbor python -m scripts.benchmark_billing_phase10 acceptance

benchmark-billing-bedrock:
	@test -n "$(CANDIDATE_CONFIG)" || { echo "set CANDIDATE_CONFIG" >&2; exit 2; }
	@test -n "$(HARBOR_BEDROCK_ROLE_ARN)" || { echo "set HARBOR_BEDROCK_ROLE_ARN" >&2; exit 2; }
	$(AWS_RUN) uv run --group harbor --extra aws --extra bedrock python -m scripts.benchmark_billing_bedrock \
		--candidate-config "$(CANDIDATE_CONFIG)" --role-arn "$(HARBOR_BEDROCK_ROLE_ARN)"

aws-harbor-ec2-billing:
	@test -f $(HARBOR_EC2_OUTPUTS) || { echo "apply the Harbor EC2 foundation first" >&2; exit 2; }
	$(AWS_RUN) uv run --group harbor --extra aws python -m scripts.run_billing_benchmark_ec2 \
		--outputs $(HARBOR_EC2_OUTPUTS) --candidate $(BENCHMARK_CANDIDATE)

# Phase 6 is strictly opt-in: these targets are never dependencies of check or CI.
aws-harbor-ec2-tfvars:
	uv run python -m scripts.generate_harbor_ec2_tfvars

aws-harbor-ec2-plan: aws-harbor-ec2-tfvars
	$(AWS_RUN) terraform -chdir=$(TF_DIR) init
	@set -e; var_files="$$(uv run python -m scripts.terraform_deployment)"; \
	set -- $$var_files; \
	$(AWS_RUN) terraform -chdir=$(TF_DIR) plan "$$@"

aws-harbor-ec2-foundation: aws-harbor-ec2-tfvars
	$(AWS_RUN) terraform -chdir=$(TF_DIR) init
	@set -e; var_files="$$(uv run python -m scripts.terraform_deployment)"; \
	set -- $$var_files; \
	$(AWS_RUN) terraform -chdir=$(TF_DIR) apply "$$@" -auto-approve
	$(AWS_RUN) terraform -chdir=$(TF_DIR) output -json > $(HARBOR_EC2_OUTPUTS)

harbor-ec2-config-check:
	@test -f $(HARBOR_EC2_OUTPUTS) || { echo "apply the Harbor EC2 foundation first" >&2; exit 2; }
	uv run --group harbor python -m scripts.generate_harbor_ec2_config \
		--outputs $(HARBOR_EC2_OUTPUTS) --output $(HARBOR_EC2_ROOT)/config-check.json \
		--task $(HARBOR_TASK) --jobs-dir $(HARBOR_EC2_ROOT)/config-check-jobs \
		--job-name config-check --model oracle --run-id config-check --task-source static

aws-harbor-ec2-preflight:
	@test -f $(HARBOR_EC2_OUTPUTS) || { echo "apply the Harbor EC2 foundation first" >&2; exit 2; }
	$(AWS_RUN) uv run --group harbor --extra aws python -m scripts.harbor_ec2_preflight \
		--outputs $(HARBOR_EC2_OUTPUTS)

aws-harbor-ec2-oracle:
	$(AWS_RUN) uv run --group harbor --extra aws python -m scripts.run_harbor_ec2 \
		--outputs $(HARBOR_EC2_OUTPUTS) --task-source $(HARBOR_EC2_TASK_SOURCE) \
		--scenario-dir $(HARBOR_EC2_SCENARIO) --kind oracle

aws-harbor-ec2-correct:
	$(AWS_RUN) uv run --group harbor --extra aws python -m scripts.run_harbor_ec2 \
		--outputs $(HARBOR_EC2_OUTPUTS) --task-source $(HARBOR_EC2_TASK_SOURCE) \
		--scenario-dir $(HARBOR_EC2_SCENARIO) --kind correct

aws-harbor-ec2-bad:
	$(AWS_RUN) uv run --group harbor --extra aws python -m scripts.run_harbor_ec2 \
		--outputs $(HARBOR_EC2_OUTPUTS) --task-source $(HARBOR_EC2_TASK_SOURCE) \
		--scenario-dir $(HARBOR_EC2_SCENARIO) --kind bad

aws-harbor-ec2-parity:
	$(AWS_RUN) uv run --group harbor --extra aws python -m scripts.run_harbor_ec2 \
		--outputs $(HARBOR_EC2_OUTPUTS) --task-source $(HARBOR_EC2_TASK_SOURCE) \
		--scenario-dir $(HARBOR_EC2_SCENARIO) --kind oracle --kind correct --kind bad

aws-harbor-ec2-scale:
	HARBOR_EC2_ATTEMPTS=4 HARBOR_EC2_CONCURRENCY=4 $(AWS_RUN) \
		uv run --group harbor --extra aws python -m scripts.run_harbor_ec2 \
		--outputs $(HARBOR_EC2_OUTPUTS) --task-source $(HARBOR_EC2_TASK_SOURCE) \
		--scenario-dir $(HARBOR_EC2_SCENARIO) --kind scale

aws-harbor-ec2-status:
	$(AWS_RUN) uv run --group harbor --extra aws python -m scripts.find_harbor_ec2_workers \
		$(if $(HARBOR_EC2_RUN_ID),--run-id $(HARBOR_EC2_RUN_ID))

aws-harbor-ec2-cleanup:
	@test -n "$(HARBOR_EC2_RUN_ID)" || { echo "set HARBOR_EC2_RUN_ID for targeted cleanup" >&2; exit 2; }
	$(AWS_RUN) uv run --group harbor --extra aws python -m scripts.cleanup_harbor_ec2_workers \
		--run-id $(HARBOR_EC2_RUN_ID) --yes

aws-phase6-demo:
	$(MAKE) aws-harbor-ec2-foundation
	$(MAKE) aws-harbor-ec2-preflight
	$(MAKE) HARBOR_EC2_RUN_ID=$(PHASE6_DEMO_RUN_ID)-parity aws-harbor-ec2-parity
	$(MAKE) HARBOR_EC2_RUN_ID=$(PHASE6_DEMO_RUN_ID)-scale aws-harbor-ec2-scale
	uv run python -m scripts.assert_phase6_acceptance \
		--parity $(HARBOR_EC2_ROOT)/runs/$(PHASE6_DEMO_RUN_ID)-parity/candidate-gate.json \
		--scale $(HARBOR_EC2_ROOT)/runs/$(PHASE6_DEMO_RUN_ID)-scale/candidate-gate.json

aws-e2e-demo-preflight:
	$(AWS_RUN) uv run --group harbor --extra aws python -c 'from scripts.run_e2e_demo import preflight; preflight("$(DEMO_RUN_ID)", candidate="$(AGENTCORE_E2E_CANDIDATE)")'

aws-e2e-demo:
	$(AWS_RUN) uv run --group harbor --extra aws python -m scripts.run_e2e_demo \
		--run-id $(DEMO_RUN_ID) --candidate "$(AGENTCORE_E2E_CANDIDATE)"
	uv run python -m scripts.assert_e2e_demo --run .demo/runs/$(DEMO_RUN_ID)

aws-e2e-demo-replay:
	@test -n "$(DEMO_SOURCE_RUN)" || { echo "set DEMO_SOURCE_RUN" >&2; exit 2; }
	$(AWS_RUN) uv run --group harbor --extra aws python -m scripts.run_e2e_demo --run-id $(DEMO_RUN_ID) --mode replay --source-run .demo/runs/$(DEMO_SOURCE_RUN)
	uv run python -m scripts.assert_e2e_demo --run .demo/runs/$(DEMO_RUN_ID)

aws-e2e-demo-resume:
	@test -n "$(DEMO_RUN_ID)" || { echo "set DEMO_RUN_ID" >&2; exit 2; }
	$(AWS_RUN) uv run --group harbor --extra aws python -m scripts.run_e2e_demo \
		--run-id $(DEMO_RUN_ID) --resume
	uv run python -m scripts.assert_e2e_demo --run .demo/runs/$(DEMO_RUN_ID)

e2e-demo-show:
	@test -n "$(DEMO_RUN_ID)" || { echo "set DEMO_RUN_ID" >&2; exit 2; }
	@test -f .demo/runs/$(DEMO_RUN_ID)/demo-summary.md || { echo "demo run is missing" >&2; exit 2; }
	cat .demo/runs/$(DEMO_RUN_ID)/demo-summary.md

e2e-demo-clean:
	@test -n "$(DEMO_RUN_ID)" || { echo "set explicit DEMO_RUN_ID" >&2; exit 2; }
	@test -d .demo/runs/$(DEMO_RUN_ID) || { echo "demo run is missing" >&2; exit 2; }
	rm -rf .demo/runs/$(DEMO_RUN_ID)


aws-up: aws-foundation aws-dsql-bootstrap aws-agentcore-config \
	aws-agentcore-image aws-agentcore-deploy aws-agentcore-smoke

aws-reset: aws-dsql-reset

# Phase 12: only config/render checks are appropriate for ordinary CI.
PHASE12_RUN ?= .rl-smoke/phase12/runs/billing-phase12-v1
PHASE12_CALIBRATION ?= .rl-smoke/phase12/calibration.json
PHASE12_STAGE ?= diagnose
PHASE12_OUTPUT ?= .rl-smoke/phase12/evidence.tar.gz
PHASE12_REPORT ?= reports/phase-12-sft-grpo-comparison.md

.PHONY: rl-phase12-config-check rl-phase12-native-config-check rl-phase12-render-smoke \
	rl-phase12-calibrate rl-phase12-prepare rl-phase12-execute rl-phase12-lock \
	rl-phase12-prepare-final rl-phase12-pack rl-phase12-report rl-phase12-export \
	aws-rl-phase12-dispatch aws-rl-phase12-fetch aws-rl-phase12-publish

rl-phase12-config-check:
	uv run --frozen python -m scripts.rl_finetuning_comparison config-check

rl-phase12-native-config-check:
	PYTHONPATH=src:. uv run --project rl --frozen --extra harbor python \
		-m scripts.rl_finetuning_comparison config-check --native

rl-phase12-render-smoke:
	uv run --frozen python -m scripts.rl_finetuning_comparison render-smoke

rl-phase12-calibrate:
	@test -n "$(PHASE8B_WHEEL_PATH)" || { echo "set PHASE8B_WHEEL_PATH to a clean built wheel" >&2; exit 2; }
	PHASE8B_WHEEL_PATH="$(PHASE8B_WHEEL_PATH)" PYTHONPATH=src:. \
		uv run --project rl --frozen --extra harbor python -m scripts.rl_finetuning_comparison \
		calibrate --output "$(PHASE12_CALIBRATION)"

rl-phase12-prepare:
	uv run --frozen python -m scripts.rl_finetuning_comparison prepare \
		--run "$(PHASE12_RUN)" --calibration "$(PHASE12_CALIBRATION)"

rl-phase12-execute:
	PYTHONPATH=src:. uv run --project rl --frozen --extra harbor --extra vllm python \
		-m scripts.rl_finetuning_comparison execute --run "$(PHASE12_RUN)" --stage "$(PHASE12_STAGE)"

rl-phase12-lock:
	uv run --frozen python -m scripts.rl_finetuning_comparison lock --run "$(PHASE12_RUN)"

rl-phase12-prepare-final:
	uv run --frozen python -m scripts.rl_finetuning_comparison prepare-final --run "$(PHASE12_RUN)"

rl-phase12-pack:
	uv run --frozen python -m scripts.rl_finetuning_comparison pack --run "$(PHASE12_RUN)" \
		--stage "$(PHASE12_STAGE)" --output "$(PHASE12_OUTPUT)"

aws-rl-phase12-dispatch:
	$(AWS_RUN) uv run --frozen --extra aws python -m scripts.rl_finetuning_comparison dispatch \
		--run "$(PHASE12_RUN)" --stage "$(PHASE12_STAGE)"

aws-rl-phase12-fetch:
	$(AWS_RUN) uv run --frozen --extra aws python -m scripts.rl_finetuning_comparison fetch \
		--run "$(PHASE12_RUN)" --stage "$(PHASE12_STAGE)"

rl-phase12-report:
	uv run --frozen python -m scripts.rl_finetuning_comparison report \
		--run "$(PHASE12_RUN)" --output "$(PHASE12_REPORT)"

rl-phase12-export:
	uv run --frozen python -m scripts.rl_finetuning_comparison export \
		--run "$(PHASE12_RUN)" --output "$(PHASE12_OUTPUT)"

aws-rl-phase12-publish:
	@test -n "$(PHASE12_DESTINATION)" || { echo "set PHASE12_DESTINATION to s3://bucket/prefix" >&2; exit 2; }
	$(AWS_RUN) uv run --frozen --extra aws python -m scripts.rl_finetuning_comparison publish \
		--output "$(PHASE12_OUTPUT)" --destination "$(PHASE12_DESTINATION)"

aws-down:
	$(AWS_RUN) terraform -chdir=$(TF_DIR) destroy -var-file=environments/dev.tfvars
