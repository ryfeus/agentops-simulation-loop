output "vpc_id" {
  value = aws_vpc.this.id
}

output "subnet_id" {
  value = aws_subnet.public.id
}

output "security_group_id" {
  value = aws_security_group.workers.id
}

output "key_name" {
  value = aws_key_pair.harbor.key_name
}

output "ami_id" {
  # AMI IDs are public identifiers. Do not inherit unrelated public-key taint
  # into the controller config output.
  value = nonsensitive(local.ami_id)
}
