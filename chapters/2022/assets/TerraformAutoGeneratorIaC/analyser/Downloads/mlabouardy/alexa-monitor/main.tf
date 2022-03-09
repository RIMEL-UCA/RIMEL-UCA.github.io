provider "aws" {
  region = "${var.region}"
  secret_key = "${var.secret_key}"
  access_key = "${var.access_key}"
}

resource "aws_security_group" "default" {
  name = "nodes-sgp"
  description = "Allow SSH from everywhere"

  ingress {
    from_port = 22
    to_port = 22
    protocol = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port = 0
    to_port = 0
    protocol = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "default" {
  count = "${length(var.hostnames)}"
  ami = "${lookup(var.amis, var.region)}"
  instance_type = "${var.instance_type}"
  key_name = "${var.key_name}"
  security_groups = ["${aws_security_group.default.name}"]

  tags {
    Name = "${element(var.hostnames,count.index)}"
  }
}
