terraform {
  required_version = ">= 1.6"
  required_providers {
    cloudflare = {
      source  = "cloudflare/cloudflare"
      version = "~> 4.52"
    }
  }
  backend "s3" {}
}

provider "cloudflare" {
  api_token = var.cloudflare_api_token
}

locals {
  name = var.project
  tags = ["asculto", "stage1", var.environment]
}

# ---------------------------------------------------------------- object storage
resource "cloudflare_r2_bucket" "audio" {
  account_id = var.account_id
  name       = "${local.name}-audio"
  location   = var.r2_location
}

resource "cloudflare_r2_bucket" "reports" {
  account_id = var.account_id
  name       = "${local.name}-reports"
  location   = var.r2_location
}

resource "cloudflare_r2_bucket" "models" {
  account_id = var.account_id
  name       = "${local.name}-models"
  location   = var.r2_location
}

# ---------------------------------------------------------------- queue + quotas
resource "cloudflare_queue" "jobs" {
  account_id = var.account_id
  name       = "${local.name}-jobs"
}

resource "cloudflare_workers_kv_namespace" "quotas" {
  account_id = var.account_id
  title      = "${local.name}-quotas"
}

# ---------------------------------------------------------------- frontend (Pages)
resource "cloudflare_pages_project" "web" {
  account_id        = var.account_id
  name              = "${local.name}-web"
  production_branch = "main"

  build_config {
    build_command   = "npm run build"
    destination_dir = "dist"
    root_dir        = "frontend"
  }
}

# ---------------------------------------------------------------- DNS + domain
resource "cloudflare_record" "app" {
  count   = var.domain != "" ? 1 : 0
  zone_id = var.zone_id
  name    = var.domain
  type    = "CNAME"
  content = "${local.name}-web.pages.dev"
  proxied = true
  comment = "Asculto web (Pages)"
}

resource "cloudflare_record" "api" {
  count   = var.domain != "" ? 1 : 0
  zone_id = var.zone_id
  name    = "api.${var.domain}"
  type    = "CNAME"
  content = "${local.name}-api.${var.workers_subdomain}.workers.dev"
  proxied = true
  comment = "Asculto API gateway (Worker)"
}
