variable "runtime_name" { type = string }
variable "role_arn" { type = string }
variable "container_uri" { type = string }
variable "environment_variables" { type = map(string) }
variable "tags" {
  type    = map(string)
  default = {}
}
