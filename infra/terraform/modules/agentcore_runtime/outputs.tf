output "runtime_arn" {
  value = awscc_bedrockagentcore_runtime.this.agent_runtime_arn
}

output "runtime_version" {
  value = awscc_bedrockagentcore_runtime.this.agent_runtime_version
}

output "runtime_id" {
  value = awscc_bedrockagentcore_runtime.this.agent_runtime_id
}

output "runtime_name" {
  value = awscc_bedrockagentcore_runtime.this.agent_runtime_name
}

output "endpoint_arn" {
  description = "Service-created DEFAULT endpoint belonging to this runtime."
  value       = "${awscc_bedrockagentcore_runtime.this.agent_runtime_arn}/runtime-endpoint/DEFAULT"
}
