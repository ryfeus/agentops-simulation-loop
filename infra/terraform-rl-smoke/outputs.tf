output "instance_id" {
  value = aws_instance.smoke.id
}

output "instance_type" {
  value = aws_instance.smoke.instance_type
}

output "ami_id" {
  # AMI IDs are public identifiers. The public SSM parameter is provider-tainted
  # as sensitive, but Phase 8A evidence must record the resolved AMI.
  value = nonsensitive(local.ami_id)
}

output "availability_zone" {
  value = aws_instance.smoke.availability_zone
}

output "private_ip" {
  value = aws_instance.smoke.private_ip
}
