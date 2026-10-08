provider "aws" {
  allowed_account_ids = [var.expected_aws_account_id]
  region              = var.aws_region

  default_tags {
    tags = local.common_tags
  }
}

provider "awscc" {
  region = var.aws_region
}
