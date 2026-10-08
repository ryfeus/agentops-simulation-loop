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
      test     = "StringEquals"
      variable = "aws:ResourceAccount"
      values   = [var.account_id]
    }
    condition {
      test     = "ArnLike"
      variable = "aws:SourceArn"
      values = [
        "arn:${var.partition}:bedrock-agentcore:${var.aws_region}:${var.account_id}:evaluator/*",
        "arn:${var.partition}:bedrock-agentcore:${var.aws_region}:${var.account_id}:online-evaluation-config/*",
      ]
    }
  }
}

resource "aws_iam_role" "this" {
  name               = var.role_name
  assume_role_policy = data.aws_iam_policy_document.trust.json
  tags               = var.tags
}

data "aws_iam_policy_document" "permissions" {
  statement {
    sid = "CloudWatchTraceRead"
    actions = [
      "logs:DescribeLogGroups",
      "logs:GetQueryResults",
      "logs:StartQuery",
    ]
    resources = ["*"]
  }
  statement {
    sid = "CloudWatchEvaluationWrite"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]
    resources = ["arn:${var.partition}:logs:${var.aws_region}:${var.account_id}:log-group:/aws/bedrock-agentcore/evaluations/*"]
  }
  statement {
    sid       = "CloudWatchIndexPolicy"
    actions   = ["logs:DescribeIndexPolicies", "logs:PutIndexPolicy"]
    resources = ["*"]
  }
  statement {
    sid       = "EvaluatorLambdaInvoke"
    actions   = ["lambda:GetFunction", "lambda:InvokeFunction"]
    resources = [var.evaluator_lambda_arn]
  }
}

resource "aws_iam_role_policy" "permissions" {
  name   = "${var.role_name}-permissions"
  role   = aws_iam_role.this.id
  policy = data.aws_iam_policy_document.permissions.json
}
