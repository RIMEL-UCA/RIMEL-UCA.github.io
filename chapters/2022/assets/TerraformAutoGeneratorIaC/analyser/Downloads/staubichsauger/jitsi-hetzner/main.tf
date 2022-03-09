provider "hcloud" {
	token = var.hcloud_token
}

resource "random_string" "name" { 
	length = 6
	special = false
	upper = false
}

locals {                        
	name = "${var.name}-${random_string.name.result}"
}
