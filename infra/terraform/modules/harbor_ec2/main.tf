data "aws_ssm_parameter" "ubuntu_2404" {
  count = var.ami_id == "" ? 1 : 0
  name  = "/aws/service/canonical/ubuntu/server/24.04/stable/current/amd64/hvm/ebs-gp3/ami-id"
}

locals {
  ami_id   = var.ami_id != "" ? var.ami_id : data.aws_ssm_parameter.ubuntu_2404[0].value
  key_name = "${var.project_name}-${var.environment}-harbor"
  tags     = merge(var.tags, { Component = "harbor-ec2-foundation", Phase = "6" })
}

resource "aws_vpc" "this" {
  cidr_block           = "10.77.0.0/24"
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags                 = merge(local.tags, { Name = "${local.key_name}-vpc" })
}

resource "aws_internet_gateway" "this" {
  vpc_id = aws_vpc.this.id
  tags   = merge(local.tags, { Name = "${local.key_name}-igw" })
}

resource "aws_subnet" "public" {
  vpc_id                  = aws_vpc.this.id
  cidr_block              = "10.77.0.0/24"
  availability_zone       = data.aws_availability_zones.available.names[0]
  map_public_ip_on_launch = true
  tags                    = merge(local.tags, { Name = "${local.key_name}-public" })
}

data "aws_availability_zones" "available" {
  state = "available"
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.this.id
  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.this.id
  }
  tags = merge(local.tags, { Name = "${local.key_name}-public" })
}

resource "aws_route_table_association" "public" {
  subnet_id      = aws_subnet.public.id
  route_table_id = aws_route_table.public.id
}

resource "aws_security_group" "workers" {
  name_prefix = "${local.key_name}-"
  description = "SSH from the explicit Harbor controller CIDR only"
  vpc_id      = aws_vpc.this.id
  ingress {
    description = "Harbor controller SSH"
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = [var.controller_cidr]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  tags = merge(local.tags, { Name = "${local.key_name}-workers" })
}

resource "aws_key_pair" "harbor" {
  key_name   = local.key_name
  public_key = var.ssh_public_key
  tags       = local.tags
}
