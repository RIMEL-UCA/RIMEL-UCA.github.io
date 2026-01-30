use collector::pull_chart;
use metrics::Graph;
fn main() -> anyhow::Result<()> {
    let mut graph = Graph::new("./charts/ingress-nginx".to_string());
    pull_chart("ingress-nginx/ingress-nginx")?;
    graph.from_chart("./charts/ingress-nginx")?;
    graph.render_large("ingress-nginx")?;
    Ok(())
}
