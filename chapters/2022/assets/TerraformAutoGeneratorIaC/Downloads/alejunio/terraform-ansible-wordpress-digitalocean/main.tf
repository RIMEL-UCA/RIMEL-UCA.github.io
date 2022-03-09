
##############################################################################
# Configuracoes basicas
##############################################################################

// Provider
provider "digitalocean" {
  token = var.do_token
}

// Chave SSH
resource "digitalocean_ssh_key" "acesso" {
  name       = "SSH"
  public_key = file("ssh/id_rsa.pub")
}
