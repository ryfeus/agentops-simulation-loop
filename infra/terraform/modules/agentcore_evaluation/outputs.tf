output "evaluator_id" { value = aws_bedrockagentcore_evaluator.dispute_policy.evaluator_id }
output "evaluator_arn" { value = aws_bedrockagentcore_evaluator.dispute_policy.evaluator_arn }
output "online_config_id" { value = aws_bedrockagentcore_online_evaluation_config.dispute_policy.online_evaluation_config_id }
output "online_config_arn" { value = aws_bedrockagentcore_online_evaluation_config.dispute_policy.online_evaluation_config_arn }
output "execution_status" { value = aws_bedrockagentcore_online_evaluation_config.dispute_policy.execution_status }
