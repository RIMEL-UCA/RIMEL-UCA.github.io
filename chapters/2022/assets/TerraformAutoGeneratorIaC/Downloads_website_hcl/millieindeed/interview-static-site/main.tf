terraform {
  backend "s3" {
    region  = "eu-west-1"
    bucket  = "will2bill-terraform-state"
    key     = "interview-static-site/terraform.tfstate"
    encrypt = true
  }
}


provider "aws" {
  profile = "default"
  region  = "eu-west-1"
}