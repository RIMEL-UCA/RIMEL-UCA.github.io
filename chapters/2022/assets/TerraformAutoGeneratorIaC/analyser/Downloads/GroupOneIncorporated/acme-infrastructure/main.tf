provider "openstack" {
  version = "~> 1.17"
}

# -- SSH -- #

resource "openstack_compute_keypair_v2" "k8s" {
  name       = "k8s-keys"
  public_key = chomp(file(var.public_key_path))
}

# -- Networking -- #

// Security groups
resource "openstack_networking_secgroup_v2" "k8s_secgroup" {
  name        = "k8s_secgroup"
  description = "k8s cluster security group"
}

// SSH
resource "openstack_networking_secgroup_rule_v2" "k8s_rule_ssh" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 22
  port_range_max    = 22
  remote_ip_prefix  = "0.0.0.0/0" # Should probably be a whitelisted network ! SECURITY !
  security_group_id = openstack_networking_secgroup_v2.k8s_secgroup.id
}

// Ping/ICMP
resource "openstack_networking_secgroup_rule_v2" "k8s_rule_ICMP" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "icmp"
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.k8s_secgroup.id
}

// TCP
resource "openstack_networking_secgroup_rule_v2" "k8s_rule_tcp" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 1
  port_range_max    = 65535
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.k8s_secgroup.id
}

// UDP 
resource "openstack_networking_secgroup_rule_v2" "k8s_rule_udp" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "udp"
  port_range_min    = 1
  port_range_max    = 65535
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.k8s_secgroup.id
}

// Internal network
resource "openstack_networking_network_v2" "k8s_network" {
  name           = "k8s-network"
  admin_state_up = "true"
}

// Internal network subnet
resource "openstack_networking_subnet_v2" "k8s_subnet" {
  name            = "k8s-subnet"
  network_id      = openstack_networking_network_v2.k8s_network.id
  cidr            = "192.168.199.0/24"
  ip_version      = 4
  dns_nameservers = ["194.47.199.41", "194.47.110.97"]
}

// Router for connecting internal network with the public
resource "openstack_networking_router_v2" "k8s_router" {
  name                = "k8s-router"
  external_network_id = "fd401e50-9484-4883-9672-a2814089528c"
}

// Interface to the internal network for the router
resource "openstack_networking_router_interface_v2" "k8s_router_interface" {
  router_id = openstack_networking_router_v2.k8s_router.id
  subnet_id = openstack_networking_subnet_v2.k8s_subnet.id
}

# -- Master node(s) -- #

// Compute instance
resource "openstack_compute_instance_v2" "k8s_master" {
  name        = "k8s-master-${count.index + 1}"
  count       = var.num_k8s_masters
  image_name  = "GroupOneInc-Debian9-Docker"
  flavor_name = "c2-r4-d20"
  key_pair    = openstack_compute_keypair_v2.k8s.name

  availability_zone_hints = "Education"

  network {
    name        = "k8s-network"
    fixed_ip_v4 = "192.168.199.${count.index + 21}"
  }

  security_groups = ["default", "k8s_secgroup"]

  depends_on = [openstack_networking_network_v2.k8s_network, openstack_networking_subnet_v2.k8s_subnet]
}

// Floating IP
resource "openstack_networking_floatingip_v2" "k8s_master" {
  count = var.num_k8s_masters
  pool  = "public"
}

// Floating IP associate
resource "openstack_compute_floatingip_associate_v2" "k8s_master" {
  count       = var.num_k8s_masters
  instance_id = element(openstack_compute_instance_v2.k8s_master.*.id, count.index)
  floating_ip = element(
    openstack_networking_floatingip_v2.k8s_master.*.address,
    count.index,
  )
}

# -- Worker node(s) -- #

// Compute instance
resource "openstack_compute_instance_v2" "k8s_node" {
  name        = "k8s-node-${count.index + 1}"
  count       = var.num_k8s_nodes
  image_name  = "GroupOneInc-Debian9-Docker"
  flavor_name = "c2-r8-d20"
  key_pair    = openstack_compute_keypair_v2.k8s.name

  availability_zone_hints = "Education"

  network {
    name        = "k8s-network"
    fixed_ip_v4 = "192.168.199.${count.index + 11}"
  }

  security_groups = ["default", "k8s_secgroup"]

  depends_on = [openstack_networking_network_v2.k8s_network, openstack_networking_subnet_v2.k8s_subnet]
}

# -- Monitoring -- #

resource "openstack_compute_instance_v2" "monitoring" {
  name        = "monitoring"
  image_name  = "GroupOneInc-Debian9-Monitoring"
  flavor_name = "c2-r4-d10"
  key_pair    = openstack_compute_keypair_v2.k8s.name

  availability_zone_hints = "Education"

  network {
    name        = "k8s-network"
    fixed_ip_v4 = "192.168.199.31"
  }

  security_groups = ["default", "k8s_secgroup"]

  depends_on = [openstack_networking_network_v2.k8s_network, openstack_networking_subnet_v2.k8s_subnet]
}

// Floating IP for Monitoring
resource "openstack_networking_floatingip_v2" "monitoring" {
  pool  = "public"
}

// Floating IP associate Monitoring
resource "openstack_compute_floatingip_associate_v2" "monitoring" {
  instance_id = openstack_compute_instance_v2.monitoring.id
  floating_ip = openstack_networking_floatingip_v2.monitoring.address
}
