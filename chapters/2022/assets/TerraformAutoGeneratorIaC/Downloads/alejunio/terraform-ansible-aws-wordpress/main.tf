
##############################################################################
# main
##############################################################################


// Provider
provider "aws" {
  region     = "us-east-1"
  access_key = var.access-key
  secret_key = var.secret-key
}

// Chave Publica SSH
resource "aws_key_pair" "acesso" {
  key_name   = "chave-acesso"
  public_key = file("ssh/id_rsa.pub")
}