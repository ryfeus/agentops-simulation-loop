output "identifier" {
  value = aws_dsql_cluster.this.identifier
}

output "arn" {
  value = aws_dsql_cluster.this.arn
}

output "endpoint" {
  value = local.endpoint
}
