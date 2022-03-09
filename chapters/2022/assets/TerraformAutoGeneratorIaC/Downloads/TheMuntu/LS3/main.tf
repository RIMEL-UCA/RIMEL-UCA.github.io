provider "google" {
  
  credentials = file("elemental-shine.json")
  project = var.project_id
  region = var.region
  zone = var.zone
  
}


# VPC Network
resource "google_compute_network" "ls3_lab_network" {  
  name = var.network
  
}

# GCP Instance
resource "google_compute_instance" "ls3_instance" {
  name         = "ls3-instance"
  machine_type = "e2-micro"
  

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-10"
    }
  }

  network_interface {
    network = google_compute_network.ls3_lab_network.name
    access_config {
    }
  }
}


# firewall rules
resource "google_compute_firewall" "ls3_lab_firewall" {
 name    = "ls3-firewall"
 network = google_compute_network.ls3_lab_network.name

 allow {
   protocol = "tcp"
   ports    = ["8000","80","22","443",]
 }
 source_tags = ["ls3-lab-firewall"]
 source_ranges = ["0.0.0.0/0"]
}
