terraform {
  required_version = ">= 0.13.3"

  backend "s3" {
    bucket  = "mtl-terraform-state"
    key     = "www"
    region  = "us-east-1"
    encrypt = true
    profile = "mtl-root"
  }
}
