use std::{
    fs,
    path::{Path, PathBuf},
    process::Command,
    str::from_utf8,
};

use anyhow::Context;

pub fn is_helm_command_present() -> bool {
    let test_helm_command = Command::new("helm").arg("-h").spawn();
    match test_helm_command {
        Ok(_) => return true,
        Err(_) => return false,
    }
}

pub fn add_helm_repo(name: &str, url: &str) -> anyhow::Result<()> {
    let output = Command::new("helm")
        .arg("repo")
        .arg("add")
        .arg(name)
        .arg(url)
        .output()?;
    if output.status.success() {
        println!("added helm repo: {}", name)
    } else {
        let stderr = from_utf8(&output.stderr[..]).unwrap_or_else(|_| "");
        if stderr.contains("already exists") {
            println!("helm repo already exists: {}", name);
        } else {
            println!("warning adding repo {}: {}", name, stderr);
        }
    }
    Ok(())
}

pub fn update_helm_repos() -> anyhow::Result<()> {
    let output = Command::new("helm").arg("repo").arg("update").output()?;
    if output.status.success() {
        println!("helm repos updated successfully")
    } else {
        println!(
            "warning updating repos: {}",
            from_utf8(&output.stderr[..]).unwrap_or_else(|_| "")
        )
    }
    Ok(())
}

pub fn pull_chart(chart_path: &str) -> anyhow::Result<()> {
    let output = Command::new("helm")
        .arg("pull")
        .arg(chart_path)
        .arg("--untar")
        .current_dir("./charts")
        .output()?;
    if output.status.success() {
        println!("pulled: {}", chart_path)
    } else {
        let stderr = from_utf8(&output.stderr[..]).unwrap_or_else(|_| "");
        println!("error pulling chart {}: {}", chart_path, stderr);
    }
    Ok(())
}

pub fn clone_chart(repo_url: &str) -> anyhow::Result<()> {
    let output = Command::new("git")
        .arg("clone")
        .arg("--depth")
        .arg("1")
        .arg(repo_url)
        .arg(name_from_url(repo_url))
        .current_dir("./charts")
        .output()?;
    if output.status.success() {
        println!("cloned: {}", repo_url)
    } else {
        let stderr = from_utf8(&output.stderr[..]).unwrap_or_else(|_| "");
        println!("error cloning {}: {}", repo_url, stderr);
    }
    Ok(())
}

pub fn name_from_url(repo_url: &str) -> String {
    let s: Vec<&str> = repo_url.split('/').collect();
    let name = format!("{}-{}", s[s.len() - 2], s[s.len() - 1]);
    name
}

pub fn collect_charts_from_repo(repo_url: &str) -> anyhow::Result<Vec<PathBuf>> {
    clone_chart(repo_url)?;
    let name = name_from_url(repo_url);
    let chart_path = format!("./charts/{}/charts", name);
    let mut chart_names: Vec<PathBuf> = Vec::new();
    match fs::read_dir(chart_path) {
        Ok(entries) => {
            for entry in entries {
                let entry = entry?;
                if !entry.path().is_dir() {
                    continue;
                }
                let file_name = entry.file_name();
                let destination = Path::new("./charts").join(file_name);
                match fs::rename(entry.path(), destination.clone()) {
                    Ok(_) => {
                        chart_names.push(destination.clone());
                    }
                    Err(e) => {
                        println!("renaming failed for this entry: {}", e);
                        continue;
                    }
                };
            }
        }
        Err(e) => {
            println!("error reading charts path, skipping this repository {}", e);
        }
    }
    let chart_dir_path = format!("./charts/{}", name);
    fs::remove_dir_all(Path::new(&chart_dir_path[..])).context("when removing parent dir")?;
    println!("collected charts from {}", name);
    Ok(chart_names)
}

pub struct Collector {
    artifacts: Vec<String>,
    repositories: Vec<String>,
}

impl Collector {
    pub fn new(artifacts: Vec<String>, repositories: Vec<String>) -> Self {
        Self {
            artifacts,
            repositories,
        }
    }
    pub fn collect(self) -> anyhow::Result<()> {
        add_helm_repo("bitnami", "https://charts.bitnami.com/bitnami")?;
        add_helm_repo("stable", "https://charts.helm.sh/stable")?;
        add_helm_repo(
            "prometheus-community",
            "https://prometheus-community.github.io/helm-charts",
        )?;
        add_helm_repo("grafana", "https://grafana.github.io/helm-charts")?;
        add_helm_repo("argo", "https://argoproj.github.io/argo-helm")?;
        add_helm_repo("jetstack", "https://charts.jetstack.io")?;
        add_helm_repo(
            "ingress-nginx",
            "https://kubernetes.github.io/ingress-nginx",
        )?;
        add_helm_repo("traefik", "https://traefik.github.io/charts")?;
        add_helm_repo("hashicorp", "https://helm.releases.hashicorp.com")?;
        add_helm_repo("elastic", "https://helm.elastic.co")?;
        update_helm_repos()?;

        if !self.artifacts.is_empty() {
            for artifact in self.artifacts.into_iter() {
                let _ = pull_chart(&artifact[..]);
            }
        }

        if !self.repositories.is_empty() {
            for repository in self.repositories.iter() {
                match collect_charts_from_repo(&repository[..]) {
                    Ok(_) => {}
                    Err(e) => println!("failed to collect from {}: {}", repository, e),
                }
            }
        }
        Ok(())
    }
}
