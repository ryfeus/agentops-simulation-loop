data "aws_iam_policy_document" "trust" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["bedrock-agentcore.amazonaws.com"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [var.account_id]
    }
    condition {
      test     = "ArnLike"
      variable = "aws:SourceArn"
      values   = ["arn:${var.partition}:bedrock-agentcore:${var.aws_region}:${var.account_id}:*"]
    }
  }
}

resource "aws_iam_role" "this" {
  name               = var.role_name
  assume_role_policy = data.aws_iam_policy_document.trust.json
  tags               = var.tags
}

data "aws_iam_policy_document" "runtime" {
  statement {
    sid       = "DSQLConnect"
    actions   = ["dsql:DbConnect"]
    resources = [var.dsql_cluster_arn]
  }

  statement {
    sid = "BedrockInvoke"
    actions = [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream",
    ]
    resources = [
      "arn:${var.partition}:bedrock:*::foundation-model/*",
      "arn:${var.partition}:bedrock:*:${var.account_id}:inference-profile/*",
      "arn:${var.partition}:bedrock:*:${var.account_id}:application-inference-profile/*",
    ]
  }

  statement {
    sid       = "BedrockDiscovery"
    actions   = ["bedrock:ListInferenceProfiles"]
    resources = ["*"]
  }

  statement {
    sid       = "ECRAuthorization"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid = "ECRPull"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
    ]
    resources = [var.ecr_repository_arn]
  }

  statement {
    sid = "RuntimeLogGroups"
    actions = [
      "logs:CreateLogGroup",
      "logs:DescribeLogGroups",
    ]
    resources = ["arn:${var.partition}:logs:${var.aws_region}:${var.account_id}:log-group:*"]
  }

  statement {
    sid = "RuntimeLogStreams"
    actions = [
      "logs:CreateLogStream",
      "logs:DescribeLogStreams",
      "logs:PutLogEvents",
    ]
    resources = ["arn:${var.partition}:logs:${var.aws_region}:${var.account_id}:log-group:/aws/bedrock-agentcore/runtimes/*"]
  }

  statement {
    sid       = "RuntimeMetrics"
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"]
  }

  statement {
    sid = "RuntimeTracing"
    actions = [
      "xray:GetSamplingRules",
      "xray:GetSamplingTargets",
      "xray:PutTelemetryRecords",
      "xray:PutTraceSegments",
    ]
    resources = ["*"]
  }

  statement {
    sid       = "UnifiedTracePolicy"
    actions   = ["logs:PutResourcePolicy"]
    resources = ["arn:${var.partition}:logs:${var.aws_region}:${var.account_id}:log-group:/aws/bedrock-agentcore/runtimes/${var.runtime_name}-*"]
  }
}

resource "aws_iam_role_policy" "runtime" {
  name   = "${var.role_name}-permissions"
  role   = aws_iam_role.this.id
  policy = data.aws_iam_policy_document.runtime.json
}
