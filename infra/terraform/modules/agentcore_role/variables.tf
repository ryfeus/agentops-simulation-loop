variable "role_name" { type = string }
variable "aws_region" { type = string }
variable "account_id" { type = string }
variable "partition" { type = string }
variable "dsql_cluster_arn" { type = string }
variable "ecr_repository_arn" { type = string }
variable "tags" {
  type    = map(string)
  default = {}
}
variable "runtime_name" { type = string }
