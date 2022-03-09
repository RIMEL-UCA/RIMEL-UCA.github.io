# root / main

# ^ general
module "VPC" {
  source               = "./VPC"
  cidr_block           = var.vpc_cidr_block
  enable_dns_hostnames = true
  enable_dns_support   = true
  name_tag             = "hq_vpc"
}

module "INTERNET_GATEWAY" {
  source   = "./INTERNET_GATEWAY"
  vpc_id   = module.VPC.vpc_id
  name_tag = "hq_internet_gateway"
}

module "PUBLIC_SUBNET" {
  source                  = "./SUBNET"
  vpc_id                  = module.VPC.vpc_id
  availability_zone       = "eu-west-2a"
  cidr_block              = "10.0.2.0/24"
  map_public_ip_on_launch = true
  tags = {
    Name = "hq_general_public_subnet"
  }
}

module "PUBLIC_RT" {
  source           = "./RT"
  vpc_id           = module.VPC.vpc_id
  gateway_id       = module.INTERNET_GATEWAY.id
  route_cidr_block = "0.0.0.0/0"
  name_tag         = "hq_public_rt"
}

module "PUBLIC_RT_ASSOCIATION" {
  source         = "./RT_A"
  subnet_id      = module.PUBLIC_SUBNET.id
  route_table_id = module.PUBLIC_RT.id
}

module "EIP_NGW" {
  source         = "./EIP"
  eip_depends_on = module.INTERNET_GATEWAY
  name_tag       = "hq_eip_for_nat_gateway"
}

module "NAT_GATEWAY_MAIN" {
  source                 = "./NAT_GATEWAY"
  allocation_id          = module.EIP_NGW.id
  subnet_id              = module.PUBLIC_SUBNET.id
  nat_gateway_depends_on = [module.PUBLIC_SUBNET]
  name_tag               = "hq_nat_main"
}

module "SSH_SG_PUBLIC" {
  source      = "./SG"
  description = "Allow SSH"
  vpc_id      = module.VPC.vpc_id
  name_tag    = "hq_ssh_sg"
}

module "SSH_ING_PUBLIC_SG_RULE" {
  source            = "./SG_RULE"
  cidr_blocks       = ["0.0.0.0/0"]
  type              = "ingress"
  from_port         = 22
  to_port           = 22
  protocol          = "tcp"
  security_group_id = module.SSH_SG_PUBLIC.id
  name_tag          = "hq_allow_ssh_from_public"
}

module "JENKINS_ING_PUBLIC_SG_RULE" {
  source            = "./SG_RULE"
  cidr_blocks       = ["0.0.0.0/0"]
  type              = "ingress"
  from_port         = 8080
  to_port           = 8080
  protocol          = "tcp"
  security_group_id = module.SSH_SG_PUBLIC.id
  name_tag          = "hq_allow_jenkins_from_public"
}

module "ALL_EG_PUBLIC_SG_RULE" {
  source            = "./SG_RULE"
  cidr_blocks       = ["0.0.0.0/0"]
  type              = "egress"
  from_port         = 0
  to_port           = 65535
  protocol          = "all"
  security_group_id = module.SSH_SG_PUBLIC.id
  name_tag          = "hq_allow_ssh_to_public"
}

module "SSH_SG_PRIVATE" {
  source      = "./SG"
  description = "Allow SSH Access to VM"
  vpc_id      = module.VPC.vpc_id
  name_tag    = "hq_ssh_sg"
}

module "SSH_ING_PRIVATE_SG_RULE" {
  source            = "./SG_RULE"
  cidr_blocks       = [var.vpc_cidr_block]
  type              = "ingress"
  from_port         = 22
  to_port           = 22
  protocol          = "tcp"
  security_group_id = module.SSH_SG_PRIVATE.id
  name_tag          = "hq_allow_ssh_from_private"
}

module "ALL_EG_PRIVATE_SG_RULE" {
  source            = "./SG_RULE"
  cidr_blocks       = ["0.0.0.0/0"]
  type              = "egress"
  from_port         = 0
  to_port           = 65535
  protocol          = "all"
  security_group_id = module.SSH_SG_PRIVATE.id
  name_tag          = "hq_all_eg_private_sg_rule"
}

module "MAJOR_KEY" {
  source          = "./KEY_PAIR"
  key_name        = var.key_name
  public_key_path = var.public_key_path
}

module "INSTANCE_JENKINS" {
  source                 = "./INSTANCE"
  ami                    = var.ec2_ami
  instance_type          = var.ec2_instance_type
  subnet_id              = module.PUBLIC_SUBNET.id
  volume_size            = 100
  vpc_security_group_ids = module.SSH_SG_PUBLIC.id
  key_name               = module.MAJOR_KEY.key_pair_id
  name_tag               = "hq_jenkins_vm"
}

# ^ bastian host for private resources
module "INSTANCE_BASTION" {
  source                 = "./INSTANCE"
  subnet_id              = module.PUBLIC_SUBNET.id
  vpc_security_group_ids = module.SSH_SG_PUBLIC.id
  ami                    = var.ec2_ami
  instance_type          = var.ec2_instance_type
  key_name               = module.MAJOR_KEY.key_pair_id
  volume_size            = 8
  name_tag               = "hq_bastion_vm"
}

# ^ private resources - testVM
module "TEST_PRIVATE_SUBNET" {
  source                  = "./SUBNET"
  vpc_id                  = module.VPC.vpc_id
  availability_zone       = "eu-west-2a"
  cidr_block              = "10.0.1.0/24"
  map_public_ip_on_launch = false
  tags = {
    Name = "hq_test_private_subnet"
  }
}

module "INSTANCE_TEST" {
  source                 = "./INSTANCE"
  ami                    = var.ec2_ami
  instance_type          = var.ec2_instance_type
  key_name               = module.MAJOR_KEY.key_pair_id
  subnet_id              = module.TEST_PRIVATE_SUBNET.id
  volume_size            = 50
  vpc_security_group_ids = module.SSH_SG_PRIVATE.id
  name_tag               = "Testing VM"
}

# ^ resources needed for private subnet (test server)
module "PRIVATE_RT" {
  source           = "./RT"
  vpc_id           = module.VPC.vpc_id
  gateway_id       = module.NAT_GATEWAY_MAIN.id
  route_cidr_block = "0.0.0.0/0"
  name_tag         = "hq_private_rt"
}

module "PRIVATE_RT_ASSOCIATION" {
  source         = "./RT_A"
  route_table_id = module.PRIVATE_RT.id
  subnet_id      = module.TEST_PRIVATE_SUBNET.id
}

# ^ RDSs
module "RDS_PRIVATE_SUBNET_A" {
 source                  = "./SUBNET"
 vpc_id                  = module.VPC.vpc_id
 availability_zone       = "eu-west-2a"
 cidr_block              = "10.0.33.0/24"
 map_public_ip_on_launch = false
 tags = {
   Name = "hq_test_private_subnet_a"
 }
}

module "RDS_PRIVATE_SUBNET_B" {
 source                  = "./SUBNET"
 vpc_id                  = module.VPC.vpc_id
 availability_zone       = "eu-west-2b"
 cidr_block              = "10.0.55.0/24"
 map_public_ip_on_launch = false
 tags = {
   Name = "hq_rds_private_subnet_b"
 }
}

module "RDS_SUBNET_GROUP" {
 source     = "./SUBNET_GROUP"
 name       = "rds_subnet_group"
 subnet_ids = [module.RDS_PRIVATE_SUBNET_B.id, module.RDS_PRIVATE_SUBNET_A.id]
 name_tag   = "hq_rds_subnet_group"
}

module "RDS_SG_PRIVATE" {
 source = "./SG"
 description = "Allow RDS Accesses"
 vpc_id = module.VPC.vpc_id
 name_tag = "hq_rds_sg"
}

module "RDS_ING_PRIVATE_SG_RULE" {
 source = "./SG_RULE"
 cidr_blocks = [var.vpc_cidr_block]
 type = "ingress"
 from_port = 3306
 to_port = 3306
 protocol = "all"
 security_group_id = module.RDS_SG_PRIVATE.id
 name_tag = "hq_rds_ing_private_sg_rule"
}

module "RDS_EG_PRIVATE_SG_RULE" {
 source = "./SG_RULE"
 cidr_blocks = ["0.0.0.0/0"]
 type = "egress"
 from_port = 0
 to_port = 65535
 protocol = "all"
 security_group_id = module.RDS_SG_PRIVATE.id
 name_tag = "hq_rds_eg_private_sg_rule"
}

module "TEST_RDS" {
 source = "./DB_INSTANCE"
 allocated_storage = 10
 apply_immediately = true
 db_subnet_group_name = module.RDS_SUBNET_GROUP.subnet_group_name
 engine = "mysql"
 engine_version = "5.7"
 instance_class = "db.t2.micro"
 name = "testdb"
 username = var.test_db_username
 password = var.test_db_password
 skip_final_snapshot = true
 vpc_security_group_ids = [module.RDS_SG_PRIVATE.id]
  identifier             = "hqtestrds"
 name_tag = "hq_test_rds"
}

module "PROD_RDS" {
 source = "./DB_INSTANCE"
 allocated_storage = 10
 apply_immediately = true
 db_subnet_group_name = module.RDS_SUBNET_GROUP.subnet_group_name
 engine = "mysql"
 engine_version = "5.7"
 instance_class = "db.t2.micro"
 name = "proddb"
 username = var.prod_db_username
 password = var.prod_db_password
 skip_final_snapshot = true
 vpc_security_group_ids = [module.RDS_SG_PRIVATE.id]
  identifier             = "hqprodrds"
 name_tag = "hq_prod_rds"
}


## ^ deployment resources (eks)
#module "EKS_PUBLIC_SUBNET_A" {
#  source                  = "./SUBNET"
#  vpc_id                  = module.VPC.vpc_id
#  availability_zone       = "eu-west-2a"
#  cidr_block              = "10.0.22.0/24"
#  map_public_ip_on_launch = true
#  tags = {
#    Name = "hq_eks_public_subnet_a"
#    # tags for kubernetes
#    "kubernetes.io/cluster/eks" = "shared"
#    "kubernetes.io/role/elb"    = 1
#  }
#}
#
#module "EKS_PUBLIC_SUBNET_B" {
#  source                  = "./SUBNET"
#  vpc_id                  = module.VPC.vpc_id
#  availability_zone       = "eu-west-2b"
#  cidr_block              = "10.0.44.0/24"
#  map_public_ip_on_launch = true
#  tags = {
#    Name = "hq_eks_public_subnet_b"
#    # tags for kubernetes load balancer
#    "kubernetes.io/cluster/eks" = "shared"
#    "kubernetes.io/role/elb"    = 1
#  }
#}
#
#module "EKS_RT_ASSOCIATION_A" {
#  source         = "./RT_A"
#  subnet_id      = module.EKS_PUBLIC_SUBNET_A.id
#  route_table_id = module.PUBLIC_RT.id
#}
#
#module "EKS_RT_ASSOCIATION_B" {
#  source         = "./RT_A"
#  subnet_id      = module.EKS_PUBLIC_SUBNET_B.id
#  route_table_id = module.PUBLIC_RT.id
#}
#
#module "EKS_INGRESS_PUBLIC_SG_RULE" {
#  source            = "./SG_RULE"
#  cidr_blocks       = ["0.0.0.0/0"]
#  type              = "ingress"
#  from_port         = 443
#  to_port           = 443
#  protocol          = "tcp"
#  security_group_id = module.SSH_SG_PUBLIC.id
#  name_tag          = "eks_ingress_public_sg_rule"
#}
#
#module "EKS_WEB_INGRESS_PUBLIC_SG_RULE" {
#  source            = "./SG_RULE"
#  cidr_blocks       = ["0.0.0.0/0"]
#  type              = "ingress"
#  from_port         = 80
#  to_port           = 80
#  protocol          = "tcp"
#  security_group_id = module.SSH_SG_PUBLIC.id
#  name_tag          = "eks_web_ingress_public_sg_rule"
#}
#
## Roles and policies for eks cluster and eks node group
#module "EKS_ROLES_POLICIES" {
#  source = "./POLICIES"
#}
#
#module "EKS_CLUSTER" {
#  source = "./EKS_CLUSTER"
#  name = "eks_cluster"
#  role_arn = module.EKS_ROLES_POLICIES.cluster_arn
#  subnet_ids = [module.EKS_PUBLIC_SUBNET_A.id,module.EKS_PUBLIC_SUBNET_B.id]
#  endpoint_public_access = true
#  endpoint_private_access = true
#  depends_on_a = module.EKS_ROLES_POLICIES.cluster_policy_attachment_a
#  depends_on_b = module.EKS_ROLES_POLICIES.cluster_policy_attachment_b
#  name_tag = "eks_cluster"
#}
#
#module "EKS_NODE_GROUP" {
#  source = "./EKS_NODE_GROUP"
#  node_group_name = "eks_nodes"
#  ami_type = "AL2_x86_64"
#  instance_type = var.ec2_instance_type
#  cluster_name = module.EKS_CLUSTER.cluster_name
#  node_role_arn = module.EKS_ROLES_POLICIES.nodes_arn
#  subnet_ids = [module.EKS_PUBLIC_SUBNET_A.id,module.EKS_PUBLIC_SUBNET_B.id]
#  desired_size = 2
#  max_size = 4
#  min_size = 2
#  depends_on_a = module.EKS_ROLES_POLICIES.ng_policy_attachment_a
#  depends_on_b = module.EKS_ROLES_POLICIES.ng_policy_attachment_b
#  depends_on_c = module.EKS_ROLES_POLICIES.ng_policy_attachment_c
#  name_tag = "eks_node_group"
#}
