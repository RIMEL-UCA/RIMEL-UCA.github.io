terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
    }
  }
  required_version = ">= 1.0.0, < 2.0.0"

  backend "s3" {
    region  = "us-east-1"
    encrypt = true
  }
}

provider "aws" {
  region = "us-east-1"
}

module "personal_website" {
  source      = "./module"
  domain_name = "derekbrown.io"
}
