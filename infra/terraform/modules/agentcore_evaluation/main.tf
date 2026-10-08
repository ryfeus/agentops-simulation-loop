resource "aws_bedrockagentcore_evaluator" "dispute_policy" {
  evaluator_name = "DisputePolicyCompliance"
  description    = "Detects noncompliant handling of disputed billing invoices."
  level          = "TRACE"
  evaluator_config {
    code_based {
      lambda_config {
        lambda_arn                = var.evaluator_lambda_arn
        lambda_timeout_in_seconds = 30
      }
    }
  }
  tags = var.tags
}

resource "aws_bedrockagentcore_online_evaluation_config" "dispute_policy" {
  online_evaluation_config_name = "DisputePolicyOnline"
  description                   = "Evaluates the synthetic billing runtime's disputed-invoice trajectory."
  evaluation_execution_role_arn = var.evaluation_role_arn
  enable_on_create              = true
  data_source_config {
    cloudwatch_logs {
      log_group_names = [var.source_log_group]
      service_names   = [var.service_name]
    }
  }
  evaluator {
    evaluator_id = aws_bedrockagentcore_evaluator.dispute_policy.evaluator_id
  }
  rule {
    sampling_config {
      sampling_percentage = 100
    }
    session_config {
      session_timeout_minutes = 1
    }
  }
  tags = var.tags
}
