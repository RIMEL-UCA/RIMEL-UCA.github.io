provider "aws" {
  region = "us-east-2"
}

terraform {
  backend "s3" {
    bucket  = "app-bnge-tf-state"
    key     = "app-bnge.tfstate"
    region  = "us-east-2"
    encrypt = true

  }
}

locals {
  prefix = "${var.prefix}-${terraform.workspace}"
  common_tags = {
    Environment = terraform.workspace
    Project     = var.project
    ManageBy    = "Terraform"
    Owner       = "Vivek Gupta"

  }
}