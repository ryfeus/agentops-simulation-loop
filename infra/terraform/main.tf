data "aws_caller_identity" "current" {}
data "aws_partition" "current" {}

locals {
  name_prefix = "${var.project_name}-${var.environment}"
  common_tags = {
    Project     = var.project_name
    Environment = var.environment
    ManagedBy   = "terraform"
  }
  runtime_enabled = var.agent_image_uri != ""
  runtime_name    = "agentops_demo_dev"
}

check "runtime_configuration" {
  assert {
    condition     = !local.runtime_enabled || var.agent_config_json != ""
    error_message = "agent_config_json must be supplied when agent_image_uri is set."
  }
}

check "harbor_ec2_configuration" {
  assert {
    condition = (
      !var.harbor_ec2_enabled ||
      (var.harbor_controller_cidr != "" && var.harbor_ssh_public_key != "")
    )
    error_message = "Harbor EC2 requires an explicit controller CIDR and public SSH key."
  }
}

check "harbor_bedrock_configuration" {
  assert {
    condition = (
      !var.harbor_bedrock_enabled ||
      (var.harbor_bedrock_trusted_principal_arn != "" && length(var.harbor_bedrock_model_resource_arns) > 0)
    )
    error_message = "Harbor Bedrock requires an explicit trusted principal and model resource ARNs."
  }
}

module "dsql" {
  source = "./modules/dsql"
  tags   = local.common_tags
}

module "ecr" {
  source          = "./modules/ecr"
  repository_name = "${local.name_prefix}-agentcore"
  tags            = local.common_tags
}

module "agentcore_role" {
  source = "./modules/agentcore_role"

  role_name          = "${local.name_prefix}-agentcore-runtime"
  aws_region         = var.aws_region
  account_id         = data.aws_caller_identity.current.account_id
  partition          = data.aws_partition.current.partition
  dsql_cluster_arn   = module.dsql.arn
  ecr_repository_arn = module.ecr.repository_arn
  runtime_name       = local.runtime_name
  tags               = local.common_tags
}

module "agentcore_runtime" {
  count  = local.runtime_enabled ? 1 : 0
  source = "./modules/agentcore_runtime"

  runtime_name  = local.runtime_name
  role_arn      = module.agentcore_role.arn
  container_uri = var.agent_image_uri
  environment_variables = {
    AGENT_CONFIG_JSON                  = var.agent_config_json
    AWS_REGION                         = var.aws_region
    AWS_DEFAULT_REGION                 = var.aws_region
    BILLING_REPOSITORY_BACKEND         = "dsql"
    BILLING_POLICY_MODE                = "permissive"
    BILLING_MCP_HOST                   = "127.0.0.1"
    BILLING_MCP_PORT                   = "8000"
    DSQL_ENDPOINT                      = module.dsql.endpoint
    DSQL_REGION                        = var.aws_region
    DSQL_USER                          = "billing_runtime"
    DSQL_DATABASE                      = "postgres"
    DSQL_MAX_RETRIES                   = "4"
    AGENT_OBSERVABILITY_ENABLED        = "true"
    OTEL_PYTHON_DISTRO                 = "aws_distro"
    OTEL_PYTHON_CONFIGURATOR           = "aws_configurator"
    UNIFIED_TRACES_DESTINATION_ENABLED = "true"
    AGENTOPS_TRACE_CONTENT_ENABLED     = tostring(var.synthetic_demo_trace_content_capture)
    AGENTOPS_WORLD_SNAPSHOT_ENABLED    = tostring(var.synthetic_demo_trace_content_capture)
    AGENTOPS_WORLD_SNAPSHOT_REQUIRED   = tostring(var.synthetic_demo_trace_content_capture)
    AGENTOPS_WORLD_SNAPSHOT_MAX_BYTES  = "32768"
    OPENINFERENCE_HIDE_INPUTS          = tostring(!var.synthetic_demo_trace_content_capture)
    OPENINFERENCE_HIDE_OUTPUTS         = tostring(!var.synthetic_demo_trace_content_capture)
  }
  tags = local.common_tags
}

module "observability" {
  source = "./modules/observability"

  account_id        = data.aws_caller_identity.current.account_id
  aws_region        = var.aws_region
  partition         = data.aws_partition.current.partition
  runtime_log_group = local.runtime_enabled ? local.runtime_log_group : ""
}

locals {
  evaluation_enabled = var.agentcore_evaluation_enabled && local.runtime_enabled
  evaluator_zip      = "${path.root}/../../.agentcore/evaluation/dispute-policy.zip"
  runtime_id         = try(module.agentcore_runtime[0].runtime_id, "")
  runtime_service    = "${local.runtime_name}.DEFAULT"
  runtime_log_group  = "/aws/bedrock-agentcore/runtimes/${local.runtime_id}-DEFAULT"
}

check "evaluation_configuration" {
  assert {
    condition     = !var.agentcore_evaluation_enabled || local.runtime_enabled
    error_message = "AgentCore evaluation requires a deployed runtime."
  }
  assert {
    condition = (
      !var.agentcore_evaluation_enabled ||
      (var.evaluator_package_sha256 != "" && try(filesha256(local.evaluator_zip), "") == var.evaluator_package_sha256)
    )
    error_message = "Evaluator package is missing or does not match evaluator_package_sha256."
  }
}

module "evaluator_lambda" {
  count  = local.evaluation_enabled ? 1 : 0
  source = "./modules/evaluator_lambda"

  function_name = "${local.name_prefix}-dispute-policy-evaluator"
  package_path  = local.evaluator_zip
  package_hash  = local.evaluation_enabled ? filebase64sha256(local.evaluator_zip) : ""
  partition     = data.aws_partition.current.partition
  tags          = local.common_tags
}

module "evaluation_role" {
  count  = local.evaluation_enabled ? 1 : 0
  source = "./modules/evaluation_role"

  role_name            = "${local.name_prefix}-agentcore-evaluation"
  account_id           = data.aws_caller_identity.current.account_id
  aws_region           = var.aws_region
  partition            = data.aws_partition.current.partition
  evaluator_lambda_arn = module.evaluator_lambda[0].function_arn
  tags                 = local.common_tags
}

module "agentcore_evaluation" {
  count  = local.evaluation_enabled ? 1 : 0
  source = "./modules/agentcore_evaluation"

  evaluator_lambda_arn = module.evaluator_lambda[0].function_arn
  evaluation_role_arn  = module.evaluation_role[0].arn
  source_log_group     = local.runtime_log_group
  service_name         = local.runtime_service
  tags                 = local.common_tags
}

module "harbor_ec2" {
  count  = var.harbor_ec2_enabled ? 1 : 0
  source = "./modules/harbor_ec2"

  project_name    = var.project_name
  environment     = var.environment
  controller_cidr = var.harbor_controller_cidr
  ssh_public_key  = var.harbor_ssh_public_key
  ami_id          = var.harbor_ami_id
  tags            = local.common_tags
}

module "harbor_bedrock" {
  count  = var.harbor_bedrock_enabled ? 1 : 0
  source = "./modules/harbor_bedrock"

  role_name             = "${local.name_prefix}-harbor-bedrock-invoke"
  trusted_principal_arn = var.harbor_bedrock_trusted_principal_arn
  model_resource_arns   = var.harbor_bedrock_model_resource_arns
  tags                  = local.common_tags
}
