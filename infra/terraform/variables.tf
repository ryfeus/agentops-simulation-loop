variable "aws_region" {
  description = "AWS region for every Phase 3 resource."
  type        = string
  default     = "us-west-2"

  validation {
    condition     = var.aws_region == "us-west-2"
    error_message = "Phase 3 is intentionally pinned to us-west-2."
  }
}

variable "project_name" {
  type    = string
  default = "agentops-demo"
}

variable "environment" {
  type    = string
  default = "dev"
}

variable "agent_image_uri" {
  description = "Immutable ECR image URI; blank creates foundation resources only."
  type        = string
  default     = ""

  validation {
    condition = (
      var.agent_image_uri == "" ||
      can(regex("^[0-9]+\\.dkr\\.ecr\\.us-west-2\\.amazonaws\\.com/.+@sha256:[0-9a-f]{64}$", var.agent_image_uri))
    )
    error_message = "agent_image_uri must be blank or an immutable us-west-2 ECR digest URI."
  }
}

variable "agent_config_json" {
  description = "Canonical serialized AgentConfig passed to the runtime."
  type        = string
  default     = ""
  sensitive   = true
}

variable "agentcore_evaluation_enabled" {
  description = "Enable the Phase 4 deterministic online evaluation resources."
  type        = bool
  default     = false
}

variable "evaluator_package_sha256" {
  description = "Hex SHA-256 of the reproducible dispute-policy Lambda package."
  type        = string
  default     = ""

  validation {
    condition     = var.evaluator_package_sha256 == "" || can(regex("^[0-9a-f]{64}$", var.evaluator_package_sha256))
    error_message = "evaluator_package_sha256 must be blank or a lowercase SHA-256 digest."
  }
}

variable "harbor_ec2_enabled" {
  description = "Enable the opt-in Phase 6 Harbor EC2 network and key foundation."
  type        = bool
  default     = false
}

variable "harbor_controller_cidr" {
  description = "Single controller CIDR allowed to SSH to ephemeral Harbor workers."
  type        = string
  default     = ""

  validation {
    condition = (
      var.harbor_controller_cidr == "" ||
      (can(cidrnetmask(var.harbor_controller_cidr)) && var.harbor_controller_cidr != "0.0.0.0/0")
    )
    error_message = "harbor_controller_cidr must be a non-wildcard CIDR when supplied."
  }
}

variable "harbor_ssh_public_key" {
  description = "Operator-owned SSH public key registered for Harbor EC2 workers."
  type        = string
  default     = ""
  sensitive   = true
}

variable "harbor_ami_id" {
  description = "Optional Ubuntu 24.04 x86_64 AMI override; blank resolves Canonical's SSM parameter."
  type        = string
  default     = ""

  validation {
    condition     = var.harbor_ami_id == "" || can(regex("^ami-[0-9a-f]+$", var.harbor_ami_id))
    error_message = "harbor_ami_id must be blank or an explicit EC2 AMI ID."
  }
}

variable "harbor_bedrock_enabled" {
  description = "Create the opt-in Harbor controller-assumable Bedrock inference role."
  type        = bool
  default     = false
}

variable "harbor_bedrock_trusted_principal_arn" {
  description = "Exact controller role ARN allowed to assume the Bedrock-only role."
  type        = string
  default     = ""
}

variable "harbor_bedrock_model_resource_arns" {
  description = "Explicit Bedrock model or inference-profile ARNs allowed to Harbor."
  type        = list(string)
  default     = []
}

variable "expected_aws_account_id" {
  description = "Operator-selected AWS account; mandatory even for direct Terraform usage."
  type        = string
  validation {
    condition     = can(regex("^[0-9]{12}$", var.expected_aws_account_id))
    error_message = "Set TF_VAR_expected_aws_account_id to your intended twelve-digit AWS account."
  }
}

variable "synthetic_demo_trace_content_capture" {
  description = "Opt in to full synthetic billing trace content and world snapshots; never use real customer data."
  type        = bool
  default     = false
}
