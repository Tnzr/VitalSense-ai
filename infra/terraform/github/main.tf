terraform {
  required_version = ">= 1.6"
  required_providers {
    github = {
      source  = "integrations/github"
      version = "~> 6.0"
    }
  }
  backend "s3" {}
}

provider "github" {
  token = var.github_token
  owner = var.github_owner
}

resource "github_repository" "repo" {
  name                   = var.repo_name
  description            = "Asculto — auscultation benchmark, inference-as-a-service & educational game"
  visibility             = "private"
  has_issues             = true
  has_wiki               = false
  delete_branch_on_merge = true
  vulnerability_alerts   = true
}

resource "github_branch_protection" "main" {
  repository_id = github_repository.repo.node_id
  pattern       = "main"

  required_status_checks {
    strict   = true
    contexts = ["ci"]
  }
  required_pull_request_reviews {
    required_approving_review_count = 1
  }
  enforce_admins = false
}

# CI/CD secrets (values supplied out-of-band; never in tfvars)
resource "github_actions_secret" "cloudflare_api_token" {
  repository      = github_repository.repo.name
  secret_name     = "CLOUDFLARE_API_TOKEN"
  plaintext_value = var.cloudflare_api_token
}

resource "github_actions_secret" "cloudflare_account_id" {
  repository      = github_repository.repo.name
  secret_name     = "CLOUDFLARE_ACCOUNT_ID"
  plaintext_value = var.cloudflare_account_id
}
