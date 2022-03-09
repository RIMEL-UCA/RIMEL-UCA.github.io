provider "aws" {
  alias = "euwst1"
}

provider "aws" {
  alias = "useas1"
}

locals {
  s3_origin_id = "S3-website-${var.bucket_name}"

  tags_common = {
    description      = "Website hosting"
    terraform_module = "terraform-aws-website-hosting"
    name             = "website-${var.bucket_name}"
  }
}

resource "aws_s3_bucket" "this" {
  bucket = var.bucket_name
  acl    = var.website == true ? "public-read" : var.acl

  versioning {
    enabled = var.versioning
  }

  dynamic "lifecycle_rule" {
    for_each = var.lifecycle_rule
    content {
      enabled = length(var.lifecycle_rule) >= 1 ? true : false

      dynamic "transition" {
        for_each = lifecycle_rule.value.transitions
        content {
          storage_class = transition.key
          days          = transition.value
        }
      }
    }
  }

  website {
    index_document = "index.html"
    error_document = "error.html"
  }
  tags = merge(var.tags, local.tags_common, { "service" = "s3" })
}

resource "aws_acm_certificate" "this" {
  provider                  = aws.useas1
  domain_name               = var.common_name
  subject_alternative_names = var.dns_names
  validation_method         = "DNS"

  tags = merge(var.tags, local.tags_common, { "service" = "s3" })

  lifecycle {
    create_before_destroy = true
  }
}

data "aws_iam_policy_document" "this" {
  statement {
    actions   = ["s3:GetObject"]
    resources = ["${aws_s3_bucket.this.arn}/*"]

    principals {
      type        = "AWS"
      identifiers = [aws_cloudfront_origin_access_identity.this.iam_arn]
    }
  }
}

resource "aws_s3_bucket_policy" "this" {
  bucket = aws_s3_bucket.this.id
  policy = data.aws_iam_policy_document.this.json
}

resource "aws_cloudfront_origin_access_identity" "this" {
  comment = var.bucket_name
}

resource "aws_cloudfront_distribution" "this" {
  origin {
    domain_name = aws_s3_bucket.this.bucket_regional_domain_name
    origin_id   = local.s3_origin_id

    s3_origin_config {
      origin_access_identity = aws_cloudfront_origin_access_identity.this.cloudfront_access_identity_path
    }
  }

  enabled             = true
  is_ipv6_enabled     = true
  comment             = format("website %s", var.bucket_name)
  default_root_object = "index.html"

  aliases = concat([var.common_name], var.dns_names)

  default_cache_behavior {
    allowed_methods  = ["GET", "HEAD", "OPTIONS"]
    cached_methods   = ["GET", "HEAD"]
    target_origin_id = local.s3_origin_id

    forwarded_values {
      query_string = false

      cookies {
        forward = "none"
      }
    }

    viewer_protocol_policy = "redirect-to-https"
    min_ttl                = 0
    default_ttl            = 3600
    max_ttl                = 86400
    compress               = true
  }

  price_class = "PriceClass_100"

  restrictions {
    geo_restriction {
      restriction_type = "none"
    }
  }

  tags = merge(var.tags, local.tags_common, { "service" = "s3" })

  viewer_certificate {
    cloudfront_default_certificate = false
    acm_certificate_arn            = aws_acm_certificate.this.arn
    ssl_support_method             = "sni-only"

    minimum_protocol_version = "TLSv1.2_2019"
  }
}
