output "aws_region" {
  value = var.aws_region
}

output "dsql_cluster_id" {
  value = module.dsql.identifier
}

output "dsql_cluster_arn" {
  value = module.dsql.arn
}

output "dsql_endpoint" {
  value = module.dsql.endpoint
}

output "ecr_repository_name" {
  value = module.ecr.repository_name
}

output "ecr_repository_url" {
  value = module.ecr.repository_url
}

output "agentcore_runtime_role_arn" {
  value = module.agentcore_role.arn
}

output "agentcore_runtime_arn" {
  value = try(module.agentcore_runtime[0].runtime_arn, null)
}

output "agentcore_runtime_version" {
  value = try(module.agentcore_runtime[0].runtime_version, null)
}

output "agentcore_endpoint_arn" {
  value = try(module.agentcore_runtime[0].endpoint_arn, null)
}

output "agentcore_runtime_id" {
  value = try(module.agentcore_runtime[0].runtime_id, null)
}

output "agentcore_runtime_service_name" {
  value = local.runtime_enabled ? local.runtime_service : null
}

output "agentcore_trace_log_group" {
  value = local.runtime_enabled ? local.runtime_log_group : null
}

output "dispute_policy_lambda_arn" {
  value = try(module.evaluator_lambda[0].function_arn, null)
}

output "dispute_policy_evaluator_id" {
  value = try(module.agentcore_evaluation[0].evaluator_id, null)
}

output "dispute_policy_evaluator_arn" {
  value = try(module.agentcore_evaluation[0].evaluator_arn, null)
}

output "online_evaluation_config_id" {
  value = try(module.agentcore_evaluation[0].online_config_id, null)
}

output "online_evaluation_config_arn" {
  value = try(module.agentcore_evaluation[0].online_config_arn, null)
}

output "online_evaluation_execution_status" {
  value = try(module.agentcore_evaluation[0].execution_status, null)
}

output "harbor_vpc_id" {
  value = try(module.harbor_ec2[0].vpc_id, null)
}

output "harbor_subnet_id" {
  value = try(module.harbor_ec2[0].subnet_id, null)
}

output "harbor_security_group_id" {
  value = try(module.harbor_ec2[0].security_group_id, null)
}

output "harbor_key_name" {
  value = try(module.harbor_ec2[0].key_name, null)
}

output "harbor_ami_id" {
  value = try(module.harbor_ec2[0].ami_id, null)
}

output "harbor_bedrock_invoke_role_arn" {
  value = try(module.harbor_bedrock[0].role_arn, null)
}
