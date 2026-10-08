variable "project_name" {
  type = string
}

variable "environment" {
  type = string
}

variable "controller_cidr" {
  type = string
}

variable "ssh_public_key" {
  type      = string
  sensitive = true
}

variable "ami_id" {
  type = string
}

variable "tags" {
  type = map(string)
}
