resource "aws_dsql_cluster" "this" {
  deletion_protection_enabled = false
  force_destroy               = true
  tags                        = var.tags
}

locals {
  endpoint = "${aws_dsql_cluster.this.identifier}.dsql.${data.aws_region.current.region}.on.aws"
}

data "aws_region" "current" {}
