variable "aws_access_key_id" {
  type        = "string"
  default     = "XXXXXXXXXX"
  description = "AWS access key"
}

variable "aws_secret_access_key" {
  type        = "string"
  default     = "XXXXXXXXXX"
  description = "AWS secret access key"
}

variable "mongo_url" {
  type        = "string"
  default     = "mongodb://mongo"
  description = "Mongodb url"
}

provider "docker" {
  host = "unix:///var/run/docker.sock"

  registry_auth {
    address = "registry.hub.docker.com"

    config_file = "~/.docker/config.json"
  }
}

data "docker_registry_image" "antennae" {
  name = "antmounds/antennae"
}

resource "docker_network" "antnet" {
  name = "antnet"
}

resource "docker_image" "antennae" {
  name          = "${data.docker_registry_image.antennae.name}"
  pull_triggers = ["${data.docker_registry_image.antennae.sha256_digest}"]
}

resource "docker_container" "app" {
  name              = "ant"
  image             = "${docker_image.antennae.latest}"   #"${docker_image.antennae.name}"
  must_run          = true
  network_mode      = "bridge"
  networks          = ["${docker_network.antnet.name}"]
  publish_all_ports = true
  restart           = "always"

  env = ["AWS_ACCESS_KEY_ID=${var.aws_access_key_id}",
    "AWS_SECRET_ACCESS_KEY=${var.aws_secret_access_key}",
    "MONGO_URL=${var.mongo_url}",
  ]
}

resource "docker_image" "mongo" {
  name = "mongo"
}

resource "docker_container" "db" {
  name              = "mongo"
  image             = "${docker_image.mongo.latest}"
  must_run          = true
  network_mode      = "bridge"
  networks          = ["${docker_network.antnet.name}"]
  publish_all_ports = false
  restart           = "always"
}

output "Image version" {
  value = "${data.docker_registry_image.antennae.sha256_digest}"
}
