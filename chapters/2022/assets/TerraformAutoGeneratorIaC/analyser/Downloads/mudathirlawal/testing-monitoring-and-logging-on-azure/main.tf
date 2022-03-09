terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "=2.66.0"
    }
  }
}
terraform {  
  backend "azurerm" {
    storage_account_name = "tstate1096"
    container_name       = "tstate"
    key                  = "terraform.tfstate"
  }
}
provider "azurerm" {
  tenant_id       = var.tenant_id
  client_id       = var.client_id
  client_secret   = var.client_secret
  subscription_id = var.subscription_id  
  features {}
}
module "resource_group" {
  source               = "../../modules/resource_group"
  resource_group       = var.resource_group
  location             = var.location
}
module "network" {
  source               = "../../modules/network"
  address_space        = var.address_space
  location             = var.location
  virtual_network_name = var.virtual_network_name
  application_type     = var.application_type
  resource_type        = "NET"
}  
module "nsg-test" {
  source           = "../../modules/networksecuritygroup"
  location         = var.location
  application_type = var.application_type
  resource_type    = "NSG"
  resource_group   = module.resource_group.resource_group_name
  subnet_id        = module.network.subnet_id_test
  address_prefix_test = var.address_prefix_test
}
module "appservice" {
  source           = "../../modules/appservice"
  location         = var.location
  application_type = var.application_type
  resource_type    = "AppService"
  resource_group   = module.resource_group.resource_group_name
}
module "publicip" {
  source           = "../../modules/publicip"
  location         = var.location
  application_type = var.application_type
  resource_type    = "publicip"
  resource_group   = module.resource_group.resource_group_name
}
module "vm" {
  resource_type    = "vm"
  source           = "../../modules/vm"
  location         = "${var.location}"
  subnet_id        = "${module.network.subnet_id_test}"  
  resource_group   = "${module.resource_group.resource_group_name}"
  application_type = "${var.application_type}"
  vm_admin_username = "${var.vm_admin_username}"
  public_ip_address_id = "${module.publicip.public_ip_address_id}"
}
