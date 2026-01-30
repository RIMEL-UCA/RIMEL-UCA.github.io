# Hellm - Helm Chart Cognitive Complexity Analyzer

[![Rust](https://img.shields.io/badge/rust-2024-orange.svg)](https://www.rust-lang.org)
[![Helm](https://img.shields.io/badge/helm-v3.0+-blue.svg)](https://helm.sh)

**Hellm** is a comprehensive tool for analyzing the cognitive complexity of Helm charts, implementing academic research on declarative process model complexity combined with Helm best practices metrics. This tool provides quantitative insights into how difficult Helm charts are to understand, maintain, and modify.

## Overview

Hellm analyzes Helm charts by building dependency graphs and calculating two categories of metrics:

### Cognitive Complexity Metrics
Based on the paper *"Complexity in declarative process models"*, adapted for Helm charts:

- **Size** - Number of nodes in the chart graph
- **Comprehension Scope** - Fraction of the graph needed to understand per resource  
- **Cognitive Diameter** - Maximum distance between resources
- **Hub Dominance** - Dependency concentration on critical nodes
- **Modification Isolation** - Independence of resource modifications
- **Helper Justification Ratio** - Fraction of helpers that are reused
- **Blast Radius Variance** - Predictability of change impact

### Best Practice Metrics
Derived from Helm and Kubernetes best practices:

- **Max Nesting Depth** - Deepest template nesting level
- **Unguarded Nested Access** - Unsafe `.Values` property accesses
- **Array Config Count** - Array-based configurations that break `--set`
- **Hardcoded Image Count** - Non-parameterized image references
- **Multi-Resource File Count** - Files containing multiple resources
- **Unquoted String Count** - Potentially unsafe string values
- **Floating Image Tag Count** - Non-pinned image versions (latest, main, etc.)
- **Mutable Selector Label** - Changeable pod selectors
- **Missing Pod Selector** - Deployments without proper selectors

## Architecture

Hellm is built as a Rust workspace with three main components:

```
hellm/
├── collector/     # Helm chart collection and cloning utilities
├── metrics/       # Graph analysis and metric calculations  
├── hellm/         # Main application and examples
├── ui/           # Angular-based visualization interface
└── charts/       # Analyzed chart storage
```

### Components

- **`collector`**: Fetches Helm charts from GitHub repositories and Helm repositories
- **`metrics`**: Core graph building, dependency analysis, and metric computation engine
- **`hellm`**: Main application with examples for running analyses
- **`ui`**: Angular interface for interactive exploration of chart complexity data

## Quick Start

### Prerequisites

- **Rust** (edition 2024)
- **Helm** (v3.0+) installed and available in PATH
- **Graphviz** (sfdp) for graph rendering
- **Python 3.12+** for visualization scripts
- **Node.js 18+** and Angular CLI for the UI

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/mouhamadalmounayar/hellm.git
cd hellm
```

2. **Install Rust dependencies:**
```bash
cargo build --workspace
```

3. **Set up Python environment:**
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

4. **Set up the Angular UI (optional):**
```bash
cd ui
npm install
cd ..
```

### Basic Usage

#### 1. Analyze a Simple Chart
```bash
cargo run --example hello
```
This analyzes the basic Helm examples chart and generates `hello.svg` visualization.

#### 2. Analyze a Complex Chart
```bash
cargo run --example ingress
```
This pulls and analyzes the ingress-nginx chart, generating a dependency graph.

#### 3. Full Repository Analysis
```bash
cargo run --release
```
Analyzes all configured repositories (40+ charts) and generates:
- `metrics_data.json` - All computed metrics
- `graphs.json` - Graph structure data
- Individual `.svg` files for each chart

#### 4. Generate Visualizations
```bash
python plot_histograms.py
```
Creates histogram distributions for all metrics in the `plots/` directory.

#### 5. Run the Web Interface
```bash
cd ui
npm run start
```
Navigate to `http://localhost:4200` to explore charts interactively.

## Metrics Explained

### Cognitive Complexity Metrics

| Metric | Range | Interpretation |
|--------|-------|----------------|
| **Size** | 0-∞ | Total number of nodes in the dependency graph |
| **Comprehension Scope** | 0-1 | Average fraction of graph to understand per resource |
| **Cognitive Diameter** | 0-∞ | Maximum distance between resources (∞ = disconnected) |
| **Hub Dominance** | 0-1 | Dependency concentration on top-k nodes |
| **Modification Isolation** | 0-1 | How independent resource modifications are |
| **Helper Justification** | 0-1 | Fraction of helpers with ≥2 reuses |
| **Blast Radius Variance** | 0-∞ | Predictability of change impact |

### Best Practice Metrics

| Metric | Good Range | Issue |
|--------|------------|-------|
| **Max Nesting Depth** | < 5 | Deep nesting is hard to follow |
| **Unguarded Nested Access** | 0 | Unsafe property access |
| **Array Config Count** | 0 | Breaks `--set` functionality |
| **Hardcoded Image Count** | 0 | Reduces portability |
| **Multi-Resource File** | 0 | Against Kubernetes conventions |
| **Unquoted String Count** | 0 | YAML parsing ambiguities |
| **Floating Image Tag** | 0 | Deployment unreliability |
| **Mutable Selector** | 0 | Breaking changes risk |
| **Missing Pod Selector** | 0 | Network policy violations |

## Analyzed Repositories

Hellm analyzes charts from three complexity categories:

### Gold Standards
- **argoproj/argo-helm** - Argo Project charts
- **prometheus-community/helm-charts** - Monitoring stack
- **grafana/helm-charts** - Grafana ecosystem
- **kubernetes/ingress-nginx** - Ingress controller
- **linkerd/linkerd2** - Service mesh
- **aws/karpenter** - Kubernetes autoscaler

### Real-World / Medium
- **budibase/budibase** - Low-code platform
- **nocodb/nocodb** - Airtable alternative
- **openebs/openebs** - Container storage
- **victoria-metrics/helm-charts** - Time series database
- **falcosecurity/charts** - Runtime security
- **dask/helm-charts** - Distributed computing

### Small / Simple
- **stefanprodan/podinfo** - Microservice demo
- **kubernetes-sigs/metrics-server** - Cluster metrics
- **kubernetes/dashboard** - Web UI
- **kubernetes-sigs/descheduler** - Pod rebalancing

## Output Files

### Metric Data (`metrics_data.json`)
```json
[
  {
    "chart_name": "ingress-nginx",
    "size": 45,
    "comprehension_scope": 0.23,
    "cognitive_diameter": 3.0,
    "hub_dominance": 0.67,
    "modification_isolation": 0.89,
    "helper_justification_ratio": 0.75,
    "blast_radius_variance": 2.1,
    "max_nesting_depth": 4,
    "unguarded_nested_access": 12,
    "array_config_count": 3,
    "hardcoded_image_count": 2,
    "multi_resource_file_count": 5,
    "unquoted_string_count": 8,
    "floating_image_tag_count": 1,
    "mutable_selector_label": 0,
    "missing_pod_selector": 0
  }
]
```

### Graph Data (`graphs.json`)
Contains the complete dependency graph structure for each chart, enabling reproducible analysis and custom visualizations.

### Visualizations
- **Individual SVG files** - Force-directed graph layouts
- **Histogram plots** - Metric distributions across all charts
- **Combined overview** - All metrics in a single visualization

## Development

### Project Structure
```
hellm/
├── collector/
│   └── src/
│       └── lib.rs          # Chart fetching utilities
├── metrics/
│   └── src/
│       └── lib.rs          # Core analysis engine
├── hellm/
│   ├── src/
│   │   └── main.rs         # Main application
│   └── examples/
│       ├── hello.rs        # Simple analysis example
│       └── ingress.rs      # Complex analysis example
├── ui/
│   ├── src/
│   │   ├── app/           # Angular components
│   │   └── ...            # UI source code
│   └── package.json
└── charts/                 # Downloaded chart storage
```

### Running Tests
```bash
cargo test --workspace
```

### Adding New Metrics

1. **Extend `GraphMetrics` struct** in `metrics/src/lib.rs`
2. **Implement calculation logic** in the `compute_metrics` method
3. **Add histogram plotting** in `plot_histograms.py`
4. **Update UI components** to display the new metric

### Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/new-metric`
3. Make your changes and add tests
4. Commit: `git commit -am 'Add new complexity metric'`
5. Push: `git push origin feature/new-metric`
6. Submit a pull request

## Research Background

This implementation is based on academic research into the cognitive complexity of declarative process models, adapted specifically for Helm charts. The approach treats Helm templates as a directed graph where:

- **Nodes** represent Kubernetes resources, template helpers, and value references
- **Edges** represent dependencies and data flow
- **Graph topology** reveals cognitive load characteristics

The cognitive metrics capture different aspects of mental effort required to understand and modify Helm charts, while the best practice metrics identify common anti-patterns that affect maintainability and reliability.

## Use Cases

### For Chart Authors
- Identify overly complex areas in your charts
- Validate adherence to Helm best practices
- Measure complexity changes over time
- Compare different implementation approaches

### For Platform Teams  
- Evaluate third-party charts before adoption
- Set complexity thresholds for production charts
- Track cognitive load across your chart ecosystem
- Prioritize refactoring efforts

### For Researchers
- Study complexity patterns in cloud-native configurations
- Validate complexity theories in real-world deployments
- Develop new metrics for IaC analysis
- Benchmark analysis tools

## Configuration

### Chart Repository Configuration

The main application (`hellm/src/main.rs`) includes a curated list of repositories. To add new repositories:

```rust
let github_repos: Vec<&str> = vec![
    // Add your repositories here
    "https://github.com/your-org/your-helm-charts",
];
```

### Metric Thresholds

Customize acceptable ranges in the UI or by filtering the `metrics_data.json` results:

```python
# Example: Find charts exceeding complexity thresholds
high_complexity = [m for m in metrics 
                  if m['comprehension_scope'] > 0.5 and 
                     m['hub_dominance'] > 0.8]
```

## Troubleshooting

### Common Issues

**Helm not found:**
```bash
# Ensure helm is installed and in PATH
helm version
```

**Graph rendering fails:**
```bash
# Install graphviz with sfdp support
# Ubuntu/Debian:
sudo apt-get install graphviz
# macOS:
brew install graphviz
```

**Memory issues with large charts:**
- Use the `--release` flag for optimized builds
- Limit the number of concurrent analyses
- Increase available RAM or use swap space

**Python matplotlib issues:**
```bash
# Install required packages
pip install matplotlib numpy pathlib
```

### Debug Mode

Enable detailed logging by setting the `RUST_LOG` environment variable:

```bash
RUST_LOG=debug cargo run --example hello
```

## Acknowledgments

- The original research paper on *"Complexity in declarative process models"*
- The Helm community for chart examples and best practices
- Contributors to the various analyzed chart repositories
- The Rust community for excellent tooling and performance

## Contact

- **Repository**: https://github.com/mouhamadalmounayar/hellm
- **Issues**: https://github.com/mouhamadalmounayar/hellm/issues
- **Author**: Mouhamad Al Mounayar - Jim Abi Habib - Logan Brunet - Leo Quelis

---

**Hellm** - Making Helm chart complexity measurable and manageable.
