variable "aws_region" {
  type    = string
  default = "us-west-2"
  validation {
    condition     = var.aws_region == "us-west-2"
    error_message = "Phase 8A is intentionally pinned to us-west-2."
  }
}

variable "instance_type" {
  type    = string
  default = "g6.2xlarge"
}

variable "ami_id" {
  type    = string
  default = ""
  validation {
    condition     = var.ami_id == "" || can(regex("^ami-[0-9a-f]+$", var.ami_id))
    error_message = "ami_id must be blank or an EC2 AMI ID."
  }
}

variable "availability_zone" {
  type    = string
  default = ""
}

variable "root_volume_size" {
  type    = number
  default = 100
  validation {
    condition     = var.root_volume_size >= 100
    error_message = "root_volume_size must be at least 100 GiB."
  }
}

variable "expected_aws_account_id" {
  description = "Operator-selected AWS account; mandatory even for direct Terraform usage."
  type        = string
  validation {
    condition     = can(regex("^[0-9]{12}$", var.expected_aws_account_id))
    error_message = "Set TF_VAR_expected_aws_account_id to your intended twelve-digit AWS account."
  }
}
