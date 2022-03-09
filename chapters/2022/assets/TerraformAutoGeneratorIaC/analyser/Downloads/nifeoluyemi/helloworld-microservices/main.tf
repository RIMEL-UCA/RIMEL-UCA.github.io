
### Replace the placeholder and comment out the next 3 lines
# variable "ibmcloud_api_key" { default = "<IBMCLOUD_API_KEY>" }
# variable "ibmcloud_username" { default = "<IBMCLOUD_USERNAME>" }
# variable "ibmcloud_password" { default = "<IBMCLOUD_PASSWORD>" }

provider "ibm" {
  ibmcloud_api_key = var.ibmcloud_api_key
  region = "us-south"
  version = "= 1.2.0"
}

terraform {
  backend "local" {
    path = "terraform.tfstate"
  }
}

resource "ibm_container_cluster" "helloworld_kube" {
  name            = "helloworld-kubernetes"
  datacenter      = "hou02"
  machine_type    = "free"
  hardware        = "shared"

  default_pool_size = 1
  public_service_endpoint = true
}


resource "null_resource" "deploy_k8s_resources" {
  depends_on = [ibm_container_cluster.helloworld_kube]

  triggers = {
    build_number = "${timestamp()}"
  }
  
  provisioner "local-exec" {
    command = "chmod +x ${path.module}/script/deploy_resources.sh; ${path.module}/script/deploy_resources.sh ${ibm_container_cluster.helloworld_kube.id} ${var.ibmcloud_username} ${var.ibmcloud_password}"
  }
}
