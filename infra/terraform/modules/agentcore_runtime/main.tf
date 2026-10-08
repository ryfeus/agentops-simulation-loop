resource "awscc_bedrockagentcore_runtime" "this" {
  agent_runtime_name     = var.runtime_name
  description            = "AgentOps Phase 3 billing agent"
  role_arn               = var.role_arn
  protocol_configuration = "HTTP"
  network_configuration = {
    network_mode = "PUBLIC"
  }
  lifecycle_configuration = {
    idle_runtime_session_timeout = 3600
    max_lifetime                 = 28800
  }
  environment_variables = var.environment_variables
  agent_runtime_artifact = {
    container_configuration = {
      container_uri = var.container_uri
    }
  }
  tags = var.tags
}
