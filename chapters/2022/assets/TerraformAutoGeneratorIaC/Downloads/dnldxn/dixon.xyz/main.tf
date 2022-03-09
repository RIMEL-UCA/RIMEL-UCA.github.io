terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 3.27"
    }
    cloudflare = {
      source = "cloudflare/cloudflare"
      version = "~> 3.0"
    }
  }

  required_version = ">= 0.14.9"
}

provider "aws" {
  profile = "default"
  region  = "us-west-2"
}

# Create some variables
locals {
  domain_name = "dixon.xyz"
  cloudflare_zone_id = "7cb6e61582b82a97ff6e72f48b608988"
}

# Create main bucket to hold website contents
resource "aws_s3_bucket" "root" {
    bucket = local.domain_name
    acl    = "public-read"

    website {
      index_document = "index.html"
      error_document = "error.html"
    }

    # Bucket policy to grant CloudFlare's servers s3:GetObject permission
    # See URL for latest server list: https://support.cloudflare.com/hc/en-us/articles/360037983412-Configuring-an-Amazon-Web-Services-static-site-to-use-Cloudflare
    policy = <<EOF
      {
        "Version": "2012-10-17",
        "Statement": [
          {
            "Sid": "PublicReadGetObject",
            "Effect": "Allow",
            "Principal": "*",
            "Action": "s3:GetObject",
            "Resource": "arn:aws:s3:::${local.domain_name}/*",
            "Condition": {
              "IpAddress": {
                "aws:SourceIp": [
                  "2400:cb00::/32",
                  "2606:4700::/32",
                  "2803:f800::/32",
                  "2405:b500::/32",
                  "2405:8100::/32",
                  "2a06:98c0::/29",
                  "2c0f:f248::/32",
                  "173.245.48.0/20",
                  "103.21.244.0/22",
                  "103.22.200.0/22",
                  "103.31.4.0/22",
                  "141.101.64.0/18",
                  "108.162.192.0/18",
                  "190.93.240.0/20",
                  "188.114.96.0/20",
                  "197.234.240.0/22",
                  "198.41.128.0/17",
                  "162.158.0.0/15",
                  "172.64.0.0/13",
                  "131.0.72.0/22",
                  "104.16.0.0/13",
                  "104.24.0.0/14"
                ]
              }
            }
          }
        ]
      }
    EOF
}

# Create Cloudflare records
resource "cloudflare_record" "dixon_xyz" {
  zone_id = local.cloudflare_zone_id
  name    = "dixon.xyz"
  value   = "${local.domain_name}.s3-website-us-west-2.amazonaws.com"
  type    = "CNAME"
  ttl     = "1"
  proxied = true
}

resource "cloudflare_record" "www" {
  zone_id = local.cloudflare_zone_id
  name    = "www"
  value   = "${local.domain_name}.s3-website-us-west-2.amazonaws.com"
  type    = "CNAME"
  ttl     = "1"
  proxied = true
}

resource "cloudflare_record" "daniel" {
  zone_id = local.cloudflare_zone_id
  name    = "daniel"
  value   = "${local.domain_name}.s3-website-us-west-2.amazonaws.com"
  type    = "CNAME"
  ttl     = "1"
  proxied = true
}

resource "cloudflare_record" "www_daniel" {
  zone_id = local.cloudflare_zone_id
  name    = "www.daniel"
  value   = "${local.domain_name}.s3-website-us-west-2.amazonaws.com"
  type    = "CNAME"
  ttl     = "1"
  proxied = true
}

# Create a page rule to redirect all requests to https://domain.xyz
resource "cloudflare_page_rule" "www" {
  zone_id = local.cloudflare_zone_id
  target = "http://*${local.domain_name}/"
  priority = 1

  actions {
    forwarding_url {
      url = "https://${local.domain_name}"
      status_code = 301
    }
  }
}

# 
resource "cloudflare_zone_settings_override" "zone_settings_override" {
  zone_id = local.cloudflare_zone_id
  settings {
    always_online = "on"  # Keep your website online for visitors when your origin server is unavailable
    browser_check = "on"  # Evaluate HTTP headers from your visitors browser for threats. If a threat is found a block page will be delivered
    ssl = "flexible"  # Encrypts traffic between the browser and Cloudflare
    brotli = "on"  # Speed up page load times for your visitor's HTTPS traffic by applying Brotli compression
    security_level = "high"  # Adjust website's Security Level to determine which visitors will receive a challenge page
    minify {
      css = "on"
      js = "on"
      html = "on"
    }
  }
}
