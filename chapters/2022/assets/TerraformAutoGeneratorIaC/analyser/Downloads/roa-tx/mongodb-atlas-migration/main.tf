# 0. configure provider(s) and backend
terraform {

  required_providers {
    aws = {
      source                                      = "hashicorp/aws"
    }
    
    mongodbatlas = {
      source                                      = "mongodb/mongodbatlas"
    }
      
    random = {
      source                                      = "hashicorp/random"
    }
  }

  backend "s3" {
    bucket                                        = "test-bucket-eco"
    key                                           = "mongodb-atlas/terraform.tfstate"
    region                                        = "us-east-1"
    encrypt                                       = true
  }
}


# 1. create mongodb atlas project
resource "mongodbatlas_project" "mongodb_project" {
  org_id                                          = var.mongodb_org_id
  name                                            = "${var.org_identifier}-${var.environment}-${var.mongodb_project_name}"
}


# 2. create mongodb atlas project maintenance window
resource "mongodbatlas_maintenance_window" "mongodb_project_maintenance_window" {
  depends_on                                      = [mongodbatlas_project.mongodb_project]
  project_id                                      = mongodbatlas_project.mongodb_project.id
  day_of_week                                     = var.day_of_week
  hour_of_day                                     = var.hour_of_day
}
  
  
# 3 create admin user's credential and storage for the admin user's credentials
# a. create random password to be used as database password
resource "random_password" "random_password" {
  depends_on                                      =  [mongodbatlas_maintenance_window.mongodb_project_maintenance_window]
  length                                          =  var.random_password_length
  special                                         =  var.random_password_true
  lower                                           =  var.random_password_true
  upper                                           =  var.random_password_true
  number                                          =  var.random_password_true
  override_special                                =  var.random_password_override_special
}

# b. create uuid, to be appended to the secret name (uuid full string) to ensure uniqueness
resource "random_uuid" "secret_random_uuid" { }

# c. create aws secret manager's secret
resource "aws_secretsmanager_secret" "secret" {
  depends_on                                      = [random_password.random_password, random_uuid.secret_random_uuid]
  name                                            = "${var.org_identifier}-${var.environment}-${var.aws_secretsmanager_secret_name}-${random_uuid.secret_random_uuid.result}"
  description                                     = var.secret_description
  recovery_window_in_days                         = var.recovery_window_in_days
  
  tags = {
    Name                                          = "${var.org_identifier}-${var.environment}-${var.aws_secretsmanager_secret_name}-${random_uuid.secret_random_uuid.result}"
    Creator                                       =  var.creator
    "aws-migration-project-id"                    =  var.aws_migration_project_id
    Environment                                   =  var.environment
    ManagedBy                                     =  var.creator
    Owner                                         =  var.owner
    Team                                          =  var.team
  }
}

# d. create local variables for referencing purpose
locals {
  depends_on                                      = [aws_secretsmanager_secret.secret]
  credentials = {
    username                                      = var.username
    password                                      = random_password.random_password.result
  }
}

# e. create aws secret version - stores the credential (username and password) in secret manager's secret
resource "aws_secretsmanager_secret_version" "secret_version" {
  depends_on                                      = [aws_secretsmanager_secret.secret]
  secret_id                                       = aws_secretsmanager_secret.secret.id
  secret_string                                   = jsonencode(local.credentials)
}


# 4. create mongodb atlas admin database user and assign the credentials stored in secret manager to the user
resource "mongodbatlas_database_user" "mongodb_admin_database_user" {
  depends_on                                      = [aws_secretsmanager_secret_version.secret_version]
  username                                        = local.credentials.username
  password                                        = local.credentials.password
  project_id                                      = mongodbatlas_project.mongodb_project.id
  auth_database_name                              = local.credentials.username

  roles {
    role_name                                     = var.mongodb_admin_role_name
    database_name                                 = var.mongodb_admin_database_name
  }

  
  labels {
    key                                           = "Name"
    value                                         = "${var.org_identifier}-${var.environment}-read-write-admin-db"
  }
  
  labels {
    key                                           = "Creator"
    value                                         = var.creator
  }
  
  labels {
    key                                           = "aws-migration-project-id"
    value                                         = var.aws_migration_project_id
  }
  
  labels {
    key                                           = "Environment"
    value                                         = var.environment
  }
  
  labels {
    key                                           = "ManagedBy"
    value                                         = var.creator
  }
  
  labels {
    key                                           = "Owner"
    value                                         = var.owner
  }
  
  labels {
    key                                           = "Team"
    value                                         = var.team
  }
}


# 5. create three regional mongodb atlas clusters
#    note 1: each cluster is located in a different region
#    note 2: the 3 nodes (within any regional cluster) are located in thesame region i.e. single-region clusters
resource "mongodbatlas_cluster" "mongodb_cluster_regionals" {
  depends_on                                      = [mongodbatlas_database_user.mongodb_admin_database_user]
  count                                           = length(var.regional_cluster_names)
  project_id                                      = mongodbatlas_project.mongodb_project.id
  name                                            = "${var.org_identifier}-${var.environment}-${var.regional_cluster_names[count.index]}"
  cluster_type                                    = var.cluster_type
  cloud_backup                                    = var.cloud_backup
  auto_scaling_disk_gb_enabled                    = var.auto_scaling_disk_gb_enabled
  auto_scaling_compute_enabled                    = var.auto_scaling_compute_enabled
  mongo_db_major_version                          = var.mongo_db_major_version
  disk_size_gb                                    = var.disk_size_gb
  provider_name                                   = var.provider_name
  provider_instance_size_name                     = var.provider_instance_size_name
  provider_volume_type                            = var.provider_volume_type
  
  replication_specs {
    num_shards                                    = var.num_shards
    
    regions_config {
      region_name                                 = var.provider_region_names[count.index]
      electable_nodes                             = var.electable_nodes
      priority                                    = var.priority
      read_only_nodes                             = var.read_only_nodes
      analytics_nodes                             = var.analytics_nodes
    }
  }
  
  advanced_configuration {
    javascript_enabled                            = var.javascript_enabled
    minimum_enabled_tls_protocol                  = var.minimum_enabled_tls_protocol
 }
 
  
 labels {
    key                                           = "Name"
    value                                         = "${var.org_identifier}-${var.environment}-${var.regional_cluster_names[count.index]}"
  }
  
  labels {
    key                                           = "Creator"
    value                                         = var.creator
  }
  
  labels {
    key                                           = "aws-migration-project-id"
    value                                         = var.aws_migration_project_id
  }
  
  labels {
    key                                           = "Environment"
    value                                         = var.environment
  }
  
  labels {
    key                                           = "ManagedBy"
    value                                         = var.creator
  }
  
  labels {
    key                                           = "Owner"
    value                                         = var.owner
  }
  
  labels {
    key                                           = "Team"
    value                                         = var.team
  }
}


# 6. create one central mongodb atlas cluster
#    note 1: the 3 nodes (within the central cluster) are located in different regions i.e. multi-region cluster
resource "mongodbatlas_cluster" "mongodb_cluster_central" {
  depends_on                                      = [mongodbatlas_database_user.mongodb_admin_database_user]
  project_id                                      = mongodbatlas_project.mongodb_project.id
  name                                            = "${var.org_identifier}-${var.environment}-${var.central_cluster_name}"
  cluster_type                                    = var.cluster_type
  cloud_backup                                    = var.cloud_backup
  auto_scaling_disk_gb_enabled                    = var.auto_scaling_disk_gb_enabled
  auto_scaling_compute_enabled                    = var.central_auto_scaling_compute_enabled
  auto_scaling_compute_scale_down_enabled         = var.central_auto_scaling_compute_scale_down_enabled
  mongo_db_major_version                          = var.mongo_db_major_version
  disk_size_gb                                    = var.disk_size_gb
  provider_name                                   = var.provider_name
  provider_instance_size_name                     = var.central_provider_instance_size_name
  provider_volume_type                            = var.provider_volume_type
  provider_auto_scaling_compute_min_instance_size = var.central_provider_auto_scaling_compute_min_instance_size
  provider_auto_scaling_compute_max_instance_size = var.central_provider_auto_scaling_compute_max_instance_size
  
  replication_specs {
    num_shards                                    = var.num_shards
    
    regions_config {
      region_name                                 = var.central_one_provider_region_name
      electable_nodes                             = var.central_one_electable_nodes
      priority                                    = var.central_one_priority
      read_only_nodes                             = var.central_one_read_only_nodes
    }
    
    regions_config {
      region_name                                 = var.central_two_provider_region_name
      electable_nodes                             = var.central_two_electable_nodes
      priority                                    = var.central_two_priority
      read_only_nodes                             = var.central_two_read_only_nodes
    }
    
    regions_config {
      region_name                                 = var.central_three_provider_region_name
      electable_nodes                             = var.central_three_electable_nodes
      priority                                    = var.central_three_priority
      read_only_nodes                             = var.central_three_read_only_nodes
    }
  }
  
  advanced_configuration {
    javascript_enabled                            = var.javascript_enabled
    minimum_enabled_tls_protocol                  = var.minimum_enabled_tls_protocol
  }
  
  
  labels {
    key                                           = "Name"
    value                                         = "${var.org_identifier}-${var.environment}-${var.central_cluster_name}"
  }
  
  labels {
    key                                           = "Creator"
    value                                         = var.creator
  }
  
  labels {
    key                                           = "aws-migration-project-id"
    value                                         = var.aws_migration_project_id
  }
  
  labels {
    key                                           = "Environment"
    value                                         = var.environment
  }
  
  labels {
    key                                           = "ManagedBy"
    value                                         = var.creator
  }
  
  labels {
    key                                           = "Owner"
    value                                         = var.owner
  }
  
  labels {
    key                                           = "Team"
    value                                         = var.team
  }
}
