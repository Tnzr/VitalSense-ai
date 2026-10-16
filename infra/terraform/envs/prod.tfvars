# Shared Stage-1 variables (no secrets). Apply with:
#   terraform apply -var-file=../envs/prod.tfvars
project           = "asculto"
environment       = "prod"
r2_location       = "EEUR"             # EU-first
region_id         = "aws-eu-central-1" # Neon EU
workers_subdomain = "asculto"
domain            = "" # set only after trademark clearance + registration
repo_name         = "cardiasense-ai"
