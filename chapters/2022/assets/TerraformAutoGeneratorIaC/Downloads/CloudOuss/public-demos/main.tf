provider "azurerm" {
  version = "~>2.55.0"
  features {}
}

terraform {
  backend "azurerm" {}
}

variable "resource_group_name" {
  default = "tf-demo-rg"
  description = "The name of the resource group"
}

variable "resource_group_location" {
  description = "The location of the resource group"
  default = "canadacentral"
}

variable "app_service_plan_name" {
  default = "tf-demo-asp"
  description = "The name of the app service plan"
}

variable "app_service_name_prefix" {
  default = "tf-demo-web"
  description = "The beginning part of your App Service host name"
}

resource "random_integer" "app_service_name_suffix" {
  min = 1000
  max = 9999
}

resource "azurerm_resource_group" "tf_demo" {
  name     = "${var.resource_group_name}"
  location = "${var.resource_group_location}"
}

resource "azurerm_app_service_plan" "tf_demo" {
  name                = "${var.app_service_plan_name}"
  location            = "${azurerm_resource_group.tf_demo.location}"
  resource_group_name = "${azurerm_resource_group.tf_demo.name}"
  kind                = "Linux"
  reserved            = true

  sku {
    tier = "Basic"
    size = "B1"
  }
}

resource "azurerm_app_service" "app_service" {
  name                = "${var.app_service_name_prefix}-${random_integer.app_service_name_suffix.result}"
  location            = "${azurerm_resource_group.tf_demo.location}"
  resource_group_name = "${azurerm_resource_group.tf_demo.name}"
  app_service_plan_id = "${azurerm_app_service_plan.tf_demo.id}"

  site_config {
    linux_fx_version = "DOTNETCORE|3.1"
    app_command_line = "dotnet Tailspin.SpaceGame.Web.dll"
  }
}

output "appservice_name" {
  value       = "${azurerm_app_service.app_service.name}"
  description = "The App Service name for the dev environment"
}
output "website_hostname" {
  value       = "${azurerm_app_service.app_service.default_site_hostname}"
  description = "The hostname of the website in the dev environment"
}
