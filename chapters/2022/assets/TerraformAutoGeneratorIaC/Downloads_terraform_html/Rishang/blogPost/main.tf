
resource "aws_db_instance" "rds_instance" {
  
  identifier             = var.rds_identifier
  allocated_storage      = var.rds_db_storage["min_storage"]
  max_allocated_storage  = var.rds_db_storage["max_storage"]
  storage_type           = "gp2"
  storage_encrypted      = false
  engine                 = var.database_type["name"]
  engine_version         = var.database_type["engine_version"]
  port                   = var.database_type["port"]
  parameter_group_name   = var.database_type["parameter_group_name"]
  instance_class         = "db.t2.micro"
  name                   = var.rds_db_name
  username               = var.rds_db_username
  password               = var.rds_db_password
  db_subnet_group_name   = var.subnet_group_name
  vpc_security_group_ids = var.security_group_names
  multi_az               = false
  publicly_accessible    = false
  skip_final_snapshot    = true
  tags                   = var.tags
}
