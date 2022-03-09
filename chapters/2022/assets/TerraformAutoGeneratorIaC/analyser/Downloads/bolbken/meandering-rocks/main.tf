terraform {
  backend "s3" {
    bucket     = "meandering-rocks-configuration"
    key        = "terraform/common/.tfstate"
    region     = "us-east-1"
    kms_key_id = "effdbfaf-9a81-48ce-ac1f-0ca69f79871b"
  }
}

provider "aws" {
  region = "us-east-1"
}

resource "aws_s3_bucket" "configuration" {
  bucket = "meandering-rocks-configuration"
}
