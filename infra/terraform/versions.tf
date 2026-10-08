terraform {
  required_version = ">= 1.6.0, < 2.0.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.39"
    }
    awscc = {
      source  = "hashicorp/awscc"
      version = "~> 1.78"
    }
  }
}
