data "aws_iam_policy_document" "transaction_search" {
  statement {
    sid     = "TransactionSearchXRayAccess"
    actions = ["logs:PutLogEvents"]
    resources = concat(
      [
        "arn:${var.partition}:logs:${var.aws_region}:${var.account_id}:log-group:aws/spans:*",
        "arn:${var.partition}:logs:${var.aws_region}:${var.account_id}:log-group:/aws/application-signals/data:*",
      ],
      var.runtime_log_group == "" ? [] : [
        "arn:${var.partition}:logs:${var.aws_region}:${var.account_id}:log-group:${var.runtime_log_group}:*",
      ],
    )
    principals {
      type        = "Service"
      identifiers = ["xray.amazonaws.com"]
    }
    condition {
      test     = "ArnLike"
      variable = "aws:SourceArn"
      values   = ["arn:${var.partition}:xray:${var.aws_region}:${var.account_id}:*"]
    }
    condition {
      test     = "StringEquals"
      variable = "aws:SourceAccount"
      values   = [var.account_id]
    }
  }
}

resource "aws_cloudwatch_log_resource_policy" "transaction_search" {
  policy_name     = "agentops-demo-transaction-search"
  policy_document = data.aws_iam_policy_document.transaction_search.json
}
