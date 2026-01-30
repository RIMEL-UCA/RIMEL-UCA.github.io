use anyhow::Ok;
use collector::collect_charts_from_repo;
use metrics::{Graph, GraphMetrics};
use std::fs;

fn main() -> anyhow::Result<()> {
    let github_repos: Vec<&str> = vec![
        // --- Catégorie 1 : Les "Gold Standards" ---
        "https://github.com/argoproj/argo-helm",
        "https://github.com/prometheus-community/helm-charts",
        "https://github.com/grafana/helm-charts",
        "https://github.com/kubernetes/ingress-nginx",
        "https://github.com/linkerd/linkerd2",
        "https://github.com/aws/karpenter",
        // --- Catégorie 2 : Les "Real-World / Medium" ---
        "https://github.com/Budibase/budibase",
        "https://github.com/nocodb/nocodb",
        "https://github.com/openebs/openebs",
        "https://github.com/VictoriaMetrics/helm-charts",
        "https://github.com/falcosecurity/charts",
        "https://github.com/dask/helm-charts",
        "https://github.com/codecentric/helm-charts",
        "https://github.com/jenkinsci/helm-charts",
        "https://github.com/mattermost/mattermost-helm",
        "https://github.com/external-secrets/external-secrets",
        "https://github.com/dexidp/helm-charts",
        "https://github.com/SonarSource/helm-chart-sonarqube",
        "https://github.com/questdb/questdb-kubernetes",
        "https://github.com/redpanda-data/helm-charts",
        "https://github.com/temporalio/helm-charts",
        "https://github.com/netdata/helmchart",
        "https://github.com/kubernetes-sigs/external-dns",
        "https://github.com/qdrant/qdrant-helm",
        "https://github.com/milvus-io/milvus-helm",
        "https://github.com/metallb/metallb",
        "https://github.com/influxdata/helm-charts",
        "https://github.com/Kong/charts",
        "https://github.com/fluxcd/flagger",
        // --- Catégorie 3 : Les "Small / Simple" ---
        "https://github.com/stefanprodan/podinfo",
        "https://github.com/kubernetes-sigs/metrics-server",
        "https://github.com/kubernetes/dashboard",
        "https://github.com/kubernetes-sigs/descheduler",
        "https://github.com/stakater/Konfigurator",
        "https://github.com/piraeusdatastore/piraeus-operator",
    ];
    let mut graphs: Vec<Graph> = Vec::new();
    let mut all_metrics: Vec<GraphMetrics> = Vec::new();

    for repo_url in github_repos.iter() {
        let chart_names = collect_charts_from_repo(repo_url)?;
        for name in chart_names.iter() {
            let name = name.to_str().unwrap();
            let mut graph = Graph::new(name.to_string());
            graph.from_chart(name)?;

            // Extract chart name from path (last directory component)
            let chart_name = std::path::Path::new(name)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(name);

            let metrics = graph.compute_metrics(chart_name);
            all_metrics.push(metrics);
            graphs.push(graph);
        }
    }

    let metric_json_data = serde_json::to_string_pretty(&all_metrics)?;
    let graph_json_data = serde_json::to_string_pretty(&graphs)?;
    fs::write("metrics_data.json", metric_json_data)?;
    println!("\nMetrics data written to metrics_data.json");
    fs::write("graphs.json", graph_json_data)?;
    println!("\nWritten all graph data to graphs.json");
    println!("Total charts analyzed: {}", all_metrics.len());

    Ok(())
}
