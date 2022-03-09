terraform {
   backend "azurerm" {
    storage_account_name = "jsoliveirastorageaccount"
    container_name       = "terraform"
    key                  = "tamanna.0.terraform.tfstate"
    # set the storage access key in the ARM_ACCESS_KEY (environment variable)
  }
}

# spin up a cluster for dev
module "aks01_cluster_dev" {
  source = "./modules/kubernetes"
  vm_size = "Standard_DS2_v2"
  location = "westeurope"
  cluster_name = "aks01"
  environment = "dev"
  node_count = 1
}

# spin up a cluster for prod
module "aks01_cluster_prod" {
  source = "./modules/kubernetes"
  vm_size = "Standard_DS2_v2"
  location = "westeurope"
  cluster_name = "aks01"
  environment = "prod"
  node_count = 2
}

# spin up a dns zone for dev
module "azure_dns_dev" {
  source = "./modules/dnszones"
  domain = "azure.tamanna.com"
  environment = "dev"
  a_records = [{
    ttl = 3600
    name = "aks01"
    records = [ module.aks01_cluster_dev.ingress_ip ]
  }]
}

# spin up a dns zone for prod
module "azure_dns_prod" {
  source = "./modules/dnszones"
  domain = "azure.tamanna.com"
  environment = "prod"
  a_records = [{
    ttl = 3600
    name = "aks01"
    records = [ module.aks01_cluster_prod.ingress_ip ]
  }]
}

# install FluxCD in dev
module "fluxcd_install_dev" {
  source                 = "./modules/fluxcd"
  kubernetes = {
    host                   = module.aks01_cluster_dev.kube_config.host
    client_key             = module.aks01_cluster_dev.kube_config.client_key
    client_certificate     = module.aks01_cluster_dev.kube_config.client_certificate
    cluster_ca_certificate = module.aks01_cluster_dev.kube_config.cluster_ca_certificate
  }
  config = {
    repository = "https://github.com/jsoliveir/challenge-devops-master"
    repository_path = "kubernetes/aks01-dev"
    repository_branch = "develop"
    namespace = "flux-system"
  }
}

# install FluxCD in prod
module "fluxcd_install_prod" {
  source                 = "./modules/fluxcd"
  kubernetes = {
    host                   = module.aks01_cluster_prod.kube_config.host
    client_key             = module.aks01_cluster_prod.kube_config.client_key
    client_certificate     = module.aks01_cluster_prod.kube_config.client_certificate
    cluster_ca_certificate = module.aks01_cluster_prod.kube_config.cluster_ca_certificate
  }
  config = {
    repository = "https://github.com/jsoliveir/challenge-devops-master"
    repository_path = "kubernetes/aks01-prod"
    repository_branch = "master"
    namespace = "flux-system"
  }
}

# To be continued...
## TODO: +1 AKS cluster for dev (in a different region) 
## TODO: +2 AKS clusters for prod (in different regions) 
## TODO: +2 virtual networks for the GWs
## TODO: +1 APPGATEWAY for the 2 dev clusters 
## TODO: +1 APPGATEWAY for the 2 prod clusters 
## TODO: +1 dnszone for dev 
## TODO: +1 dnszone for prod
## TODO: +1 frontdor for prod (depending on the customers region)
## TODO: make the AKS clusters private accesible thru a private network
# ...
