provider "aws" {

  region  = "us-east-1"

}

resource "aws_instance" "ec2" {
  ami                     = "ami-0817d428a6fb68645"
  instance_type           = "t2.micro"
  key_name = "key-venkat"
  vpc_security_group_ids  = ["${aws_security_group.security.id}"]

  user_data = <<-EOF
          #!/bin/bash
          sudo apt update
          sudo apt install git
        
          
          curl -sL https://deb.nodesource.com/setup_10.x -o nodesource_setup.sh
          sudo bash nodesource_setup.sh
          sudo apt-get install nodejs -y
          sudo npm install forever -g
          sudo apt install -y mongodb
          sudo apt install python3-pip -y

        
          git clone https://github.com/venkateshwaran-git/Task_Manager_React_App.git
          cd Task_Manager_React_App
          forever start -c "npm run start" .
          sleep 2
          cd ..
          
          git clone https://github.com/venkateshwaran-git/Task_Manager_PythonApp.git
          cd Task_Manager_PythonApp
          sudo pip3 install -r requirements.txt
          python3 client1.py
          cd ..
          
          
      EOF

}

resource "aws_security_group" "security" {
  name        = "security"
  description = "security group"

  ingress {
 protocol = "tcp"
from_port = 3000    
to_port = 3000
cidr_blocks = ["0.0.0.0/0"]
 }

  ingress {
 protocol = "tcp"
from_port = 5000    
to_port = 5000
cidr_blocks = ["0.0.0.0/0"]
 }

ingress {
 protocol = "-1"
from_port = 0
to_port = 0
cidr_blocks = ["0.0.0.0/0"]
 }

 ingress {
 protocol = "tcp"
from_port = 22
to_port = 22
cidr_blocks = ["0.0.0.0/0"]
 }
egress {
 from_port = 0
to_port = 0
protocol = "-1"
 cidr_blocks = ["0.0.0.0/0"]
 }
}