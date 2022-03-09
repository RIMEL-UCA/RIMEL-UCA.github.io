provider "aws" {
    region = "us-east-1"
}

variable "name" {
    description = "Name the instance on deploy"
}

resource "aws_instance" "admin_frontend" {
    ami = "ami-042e8287309f5df03"
    instance_type = "t2.micro"
    key_name = "admin"

    tags = {
        Name = var.name
    }
}