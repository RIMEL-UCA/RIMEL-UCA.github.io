
##############################################################################
# Definicao do Provider - Digital Ocean
##############################################################################

// Provider
provider "vultr" {
  api_key = var.vultr_token
  rate_limit = 700
  retry_limit = 3
}

// Chave SSH - Inserir chave Publica
resource "vultr_ssh_key" "my_ssh_key" {
  name = "my-ssh-key"
  ssh_key = "ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAACAQCzruV1ch7oGu5sCv4n6p4zjcEE1Ndcvh8HvWWeZmy9Bt1c+1Xq0a05IxjeRzoWNC6bLWaQfcspuNIWE58W4lh956I7K+8TyD8KJuOcCpRc79LRwUHTqNmi23FA3vgLS4H4FcbFEqBeOuBACraYT9biIk2cw7WW6Yj1PWIzkVGr9sWN6k/C40NFjjsmI6sSc4u61njO0ImphPVGunMRk07OhN/Py1ETb1E1aa+Q0Wn019JH+DwElMhn6IHkvVUgBc7B6z9Kc4uKAPJ15YBcPcs79D8mLwgii6xr0zK8NlEUyVbJcNlgiFyXOIgbnZifwT+KtkSSHkmUSGZUrVaOgxsT5CeHUyVLRzh/fqFlCINYTG0yRXn+bIVeI1IWtAJzO7K+NIqA6ukBbkXJQpXoleY/HhR64A3lqy018pq+knJFb6P1szKt9Opi/mbpfOiPH1p+eZMiZJJa9uAA+mKY0Kz7pTlkTJKM9yHqo/s3zPtAY34FLqyqB1iJ0WCKHxvuTmK3y0p0+fudPoU9CAWWO0DT/XKgLEiY0qwiCkL8186+vZ0/GelcXKoXUuaa6vpyteQS9eckl+SLxP1QmYbDYdVeiarF4dk7t/mKWK8u36fwt7exOYHLMxog67WTLNKKB5s+A2VF4uE3C4t7CV+xw2w05dovZ4sxu/k0vIva6ufVBw=="
}