//This is a simple terrafrom reasouce that will create dynamodb table called Rides with pirtation/Primary key named RideId
resource "aws_dynamodb_table" "basic_dynamodb_table" {
  name           = "Rides"
  billing_mode   = "PROVISIONED"
  read_capacity  = 5
  write_capacity = 5
  hash_key       = "RideId"

  attribute {
    name = "RideId"
    type = "S"
  }

  tags = {
    Name        = "${var.env}-${var.partner}-${var.app}"
    Environment = var.env
    Partner = var.partner
  }
}