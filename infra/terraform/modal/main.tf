# Modal has no official Terraform provider; manage the secret and deployment
# through the Modal CLI so it stays in the same apply/teardown flow.
terraform {
  required_version = ">= 1.6"
  required_providers {
    null = {
      source  = "hashicorp/null"
      version = "~> 3.2"
    }
  }
  backend "s3" {}
}

resource "null_resource" "modal_secret" {
  triggers = {
    name    = var.secret_name
    version = var.secret_version
  }

  provisioner "local-exec" {
    command = "modal secret create ${var.secret_name} ASCULTO_LLM_API_KEY=${var.llm_api_key} ASCULTO_LLM_BASE_URL=${var.llm_base_url} --force"
    environment = {
      MODAL_TOKEN_ID     = var.modal_token_id
      MODAL_TOKEN_SECRET = var.modal_token_secret
    }
  }
}

resource "null_resource" "modal_deploy" {
  depends_on = [null_resource.modal_secret]
  triggers = {
    app    = var.app_file
    deploy = var.deploy_revision
  }

  provisioner "local-exec" {
    working_dir = var.repo_root
    command     = "modal deploy ${var.app_file}"
    environment = {
      MODAL_TOKEN_ID     = var.modal_token_id
      MODAL_TOKEN_SECRET = var.modal_token_secret
    }
  }
}
