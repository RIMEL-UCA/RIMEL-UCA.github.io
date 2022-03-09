terraform {
  required_providers {
    azurerm = {
      source = "hashicorp/azurerm"
      version = ">= 2.26"
    }
  }
}

provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "example" {
  name     = "${var.prefix}-demo-rg"
  location = "${var.region}"
}

resource "azurerm_storage_account" "storage_account" {
  name                     = "${var.prefix}894"
  resource_group_name      = azurerm_resource_group.example.name
  location                 = azurerm_resource_group.example.location
  account_tier             = "Standard"
  account_kind             = "StorageV2"
  account_replication_type = "GRS"
  enable_https_traffic_only = true

  static_website {
    index_document = "index.html"
    error_404_document= "error.html"
    }


  tags = {
    environment = "dev"
  }
}

resource "azurerm_storage_blob" "index_page" {
  name                   = "index.html"
  storage_account_name   = azurerm_storage_account.storage_account.name
  storage_container_name = "$web"
  type                   = "Block"
  content_type           = "text/html"
  source_content         = "<h1>This is static content coming from the Terraform</h1>"
}

resource "azurerm_storage_blob" "error_page" {
  name                   = "error.html"
  storage_account_name   = azurerm_storage_account.storage_account.name
  storage_container_name = "$web"
  type                   = "Block"
  content_type           = "text/html"
  source_content         = "<h1>This is error page</h1>"
}
