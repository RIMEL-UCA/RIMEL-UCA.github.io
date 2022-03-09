
module "aws_s3" {
  source  = "Lunik/s3/aws"
  version = "0.1.0"

  providers = {
    aws = aws.france
  }

  bucket_name = "website-${var.website_hostname}"

  bucket_tags = {
    usage = "website-hosting"
    owner = var.website_owner
    hostname = var.website_hostname
  }

  bucket_policy = {
    Id = ""
    Version = "2012-10-17"
    Statement = [
      {
        Sid = ""
        Action = [
          "s3:GetObject"
        ]
        Effect = "Allow",
        Resource = [
          "arn:aws:s3:::website-${var.website_hostname}",
          "arn:aws:s3:::website-${var.website_hostname}/*"
        ]
        Principal = {
          AWS = module.aws_cloudfront.cloudfront_oai.iam_arn
        }
      }
    ]
  }
}

module "aws_cloudfront" {
  source  = "Lunik/cloudfront/aws"
  version = "0.2.0"

  providers = {
    aws = aws.france
  }

  domain_name = var.website_hostname

  aws_route53_zone_name = var.aws_route53_zone_name

  certificate_tags = {
    usage = "website-hosting"
    owner = var.website_owner
    hostname = var.website_hostname
  }

  aws_s3_bucket = module.aws_s3.bucket
}

module "aws_route53" {
  source  = "Lunik/route53/aws"
  version = "0.1.0"

  providers = {
    aws = aws.france
  }

  zone_name = var.aws_route53_zone_name

  records_alias_A = [{
    name = var.website_hostname
    alias = {
      name = module.aws_cloudfront.cloudfront.domain_name
      zone_id = module.aws_cloudfront.cloudfront.hosted_zone_id
    }
  }]

  records_alias_AAAA = [{
    name = var.website_hostname
    alias = {
      name = module.aws_cloudfront.cloudfront.domain_name
      zone_id = module.aws_cloudfront.cloudfront.hosted_zone_id
    }
  }]
}