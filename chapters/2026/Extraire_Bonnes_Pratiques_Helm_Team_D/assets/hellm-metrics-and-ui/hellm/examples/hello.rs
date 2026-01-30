use collector::collect_charts_from_repo;
use metrics::Graph;
fn main() -> anyhow::Result<()> {
    let chart_names = collect_charts_from_repo("https://github.com/helm/examples.git")?;
    let name = chart_names[0].to_str().unwrap();
    let mut graph = Graph::new(name.to_string());
    graph.from_chart(name)?;
    graph.render_large("hello")?;
    Ok(())
}
