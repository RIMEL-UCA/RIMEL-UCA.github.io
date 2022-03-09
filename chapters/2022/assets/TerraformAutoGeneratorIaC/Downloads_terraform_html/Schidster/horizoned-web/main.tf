terraform {
  required_version = ">=0.12.24"

  backend "s3" {
    bucket         = "terraform-web-state"
    key            = "terraform.tfstate"
    region         = "ap-south-1"
    dynamodb_table = "terraform-web-state-lock"
    encrypt        = true
  }
}

provider "aws" {
  version                 = "~> 2.58.0"
  shared_credentials_file = "$HOME/.aws/credentials"
  profile                 = var.aws_profile
  region                  = var.aws_region
}

module "bootstrap" {
  source               = "./bootstrap"
  name_of_s3_bucket    = var.state_bucket_name
  dynamo_db_table_name = var.state_lock_table_name
}

resource "aws_ecr_repository" "web" {
  name                 = "horizoned-web"
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  lifecycle {
    prevent_destroy = true
  }

  tags = {
    BuiltBy = "Terraform"
  }
}

resource "aws_ecs_cluster" "hercules" {
  name               = "hercules"
  capacity_providers = ["FARGATE"]

  tags = {
    BuiltBy = "Terraform"
  }
}

resource "aws_ecs_task_definition" "web" {
  family                   = "web"
  container_definitions    = file("task-definition.json")
  execution_role_arn       = aws_iam_role.web_task_execution.arn
  network_mode             = "awsvpc"
  cpu                      = "256"
  memory                   = "512"
  requires_compatibilities = ["FARGATE"]
}

resource "aws_ecs_service" "web_service" {
  name                = "web-service"
  cluster             = aws_ecs_cluster.hercules.id
  task_definition     = aws_ecs_task_definition.web.arn
  desired_count       = var.task_count
  launch_type         = "FARGATE"
  scheduling_strategy = "REPLICA"
  platform_version    = "LATEST"

  network_configuration {
    subnets          = [aws_subnet.public.id]
    security_groups  = [aws_security_group.web.id]
    assign_public_ip = true
  }
}

resource "aws_cloudwatch_log_group" "web" {
  name              = "horizoned-web"
  retention_in_days = 3

  tags = {
    BuiltBy = "Terraform"
  }
}
