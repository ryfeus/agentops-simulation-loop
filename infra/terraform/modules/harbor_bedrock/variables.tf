variable "role_name" { type = string }
variable "trusted_principal_arn" { type = string }
variable "model_resource_arns" { type = list(string) }
variable "tags" { type = map(string) }
