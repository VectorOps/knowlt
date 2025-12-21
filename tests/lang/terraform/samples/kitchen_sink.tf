############################################
# Terraform HCL kitchen sink example
############################################

terraform {
  required_version = ">= 1.0"

  backend "s3" {
    bucket = "my-terraform-state-bucket"
    key    = "path/to/state.tfstate"
    region = "us-west-2"
  }
}

provider "aws" {
  region = "us-west-2"
}

variable "default_tags" {
  type = map(string)
  default = {
    Project = "Example"
    Owner   = "InfraTeam"
  }
  description = "Default tags applied to resources"
}

locals {
  env = "production"
  owners = ["alice", "bob"]
}

module "vpc" {
  source = "./modules/vpc"
  cidr_block = "10.0.0.0/16"
}

resource "aws_security_group" "sg" {
  name        = "example-sg"
  description = "Security group with dynamic ingress rules"
  vpc_id      = "vpc-123456"

  dynamic "ingress" {
    for_each = var.default_tags
    content {
      from_port   = 80
      to_port     = 80
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]

      dynamic "ipv6_cidr_blocks" {
        for_each = ["::/0"]
        content {
          cidr_blocks = ["::/0"]
        }
      }
    }
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(
    var.default_tags,
    { Name = "example-sg-${local.env}" }
  )
}

data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"]
  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*"]
  }
}

output "sg_id" {
  value = aws_security_group.sg.id
}

locals {
  nested = {
    inner = {
      list = [1, 2, 3]
    }
  }
}
