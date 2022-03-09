resource "azurerm_resource_group" "rg" {
  name     = var.env
  location = var.location
}

resource "azurerm_app_service_plan" "serviceplan" {
  name                = "${var.name}-plan"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name

  sku {
    tier = "Basic"
    size = "B1"
  }
}

resource "azurerm_app_service" "appservice" {
  name                = "${var.name}-bff-${var.env}"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  app_service_plan_id = azurerm_app_service_plan.serviceplan.id

  site_config {
    always_on = true
    java_version = "11"
    java_container = "JAVA"
    java_container_version = "11"
    app_command_line = "java -jar /home/site/wwwroot/*.jar" 
  }

  app_settings = {}
}

resource "azurerm_storage_account" "staticwebapp" {
  name                      = var.name
  resource_group_name       = azurerm_resource_group.rg.name
  location                  = azurerm_resource_group.rg.location
  account_tier              = "Standard"
  account_replication_type  = "LRS"
  enable_https_traffic_only = true

  static_website {
    index_document     = "index.html"
    error_404_document = "index.html"
  }
}