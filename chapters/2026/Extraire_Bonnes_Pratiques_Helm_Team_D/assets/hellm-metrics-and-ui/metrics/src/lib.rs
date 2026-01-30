use std::{
    collections::{HashMap, HashSet, VecDeque},
    fmt::Write,
    fs::{self, File},
    io::{BufRead, BufReader},
    path::Path,
};

use anyhow::Context;
use regex::Regex;
use serde::Serialize;
use walkdir::{DirEntry, WalkDir};

#[derive(Hash, PartialEq, Eq, Debug, Serialize)]
pub enum NodeType {
    Ressource,
    Helper,
    ValuesSection,
}

#[derive(Debug, Serialize)]
pub struct GraphMetrics {
    pub chart_name: String,
    pub size: usize,
    pub comprehension_scope: f64,
    pub cognitive_diameter: f64,
    pub hub_dominance: f64,
    pub modification_isolation: f64,
    pub helper_justification_ratio: f64,
    pub blast_radius_variance: f64,
    pub max_nesting_depth: usize,
    pub unguarded_nested_access: usize,
    pub array_config_count: usize,
    pub hardcoded_image_count: usize,
    pub multi_resource_file_count: usize,
    pub unquoted_string_count: usize,
    pub floating_image_tag_count: usize,
    pub mutable_selector_label: usize,
    pub missing_pod_selector: usize,
}

#[derive(Debug, Serialize)]
pub struct Graph {
    pub name: String,
    pub list: HashMap<String, HashSet<String>>,
    pub types: HashMap<String, NodeType>,
}

impl Graph {
    pub fn new(name: String) -> Self {
        Graph {
            name,
            list: HashMap::new(),
            types: HashMap::new(),
        }
    }

    pub fn from_chart(&mut self, chart_path: &str) -> anyhow::Result<()> {
        let define_regex =
            Regex::new(r#"\{\{-?\s*define\s+["']([^"']+)["']"#).context("when creating regex")?;
        let values_regex =
            Regex::new(r"\.Values\.[a-zA-Z0-9_-]+").context("when creating regex")?;
        let kind_regex = Regex::new(r#"(?m)^\s*kind:\s*['\"]?(\w+)['\"]?"#)?;
        let include_regex = Regex::new(r#"\{\{-?\s*include\s+["']([^"']+)["']"#)?;
        let walker = WalkDir::new(chart_path).into_iter();
        for entry in walker {
            let entry = entry?;
            match (is_tpl_file(&entry), is_yaml_file(&entry)) {
                (true, _) => {
                    self.process_tpl_file(
                        entry.path(),
                        &define_regex,
                        &values_regex,
                        &include_regex,
                    )?;
                }
                (_, true) => {
                    self.process_yaml_file(
                        entry.path(),
                        &kind_regex,
                        &values_regex,
                        &include_regex,
                    )?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    pub fn process_tpl_file(
        &mut self,
        template: impl AsRef<Path>,
        define_regex: &Regex,
        values_regex: &Regex,
        include_regex: &Regex,
    ) -> anyhow::Result<()> {
        let template = template.as_ref();
        let f = File::open(template).context("when opening file")?;
        let mut buf_reader = BufReader::new(f);
        let mut content: String = String::new();
        let mut current_matched_block: Option<String> = None;
        while buf_reader.read_line(&mut content)? != 0 {
            if let Some(define_block) = self.capture_first_node(&define_regex, &content) {
                self.insert_node(&define_block, NodeType::Helper);
                current_matched_block = Some(define_block);
            }
            if let Some(values_section) = self.capture_first_node(&values_regex, &mut content) {
                self.insert_node(&values_section, NodeType::ValuesSection);
                // add helper -> values edge
                if let Some(block_name) = &current_matched_block {
                    self.insert_to_neighbors(block_name, &values_section);
                }
            }
            // Check for include statements (helper -> helper edges)
            if let Some(include_nodes) = self.capture_all_node(include_regex, &content) {
                for included_helper in include_nodes {
                    self.insert_node(&included_helper, NodeType::Helper);
                    // add helper -> helper edge
                    if let Some(block_name) = &current_matched_block {
                        self.insert_to_neighbors(block_name, &included_helper);
                    }
                }
            }
            content.clear();
        }
        Ok(())
    }

    pub fn process_yaml_file(
        &mut self,
        resource: impl AsRef<Path>,
        kind_regex: &Regex,
        values_regex: &Regex,
        include_regex: &Regex,
    ) -> anyhow::Result<()> {
        let resource = resource.as_ref();
        let content =
            fs::read_to_string(resource).context("when reading resource file to string")?;
        if let Some(kind) = self.capture_first_node(kind_regex, &content) {
            let kind_node = format!("{}/{}", resource.to_str().unwrap(), kind);
            self.insert_node(&kind_node, NodeType::Ressource);
            if let Some(value_nodes) = self.capture_all_node(values_regex, &content) {
                for node in value_nodes {
                    self.insert_node(&node, NodeType::ValuesSection);
                    self.insert_to_neighbors(&kind_node, &node);
                }
            }
            if let Some(include_nodes) = self.capture_all_node(include_regex, &content) {
                for node in include_nodes {
                    self.insert_node(&node, NodeType::Helper);
                    self.insert_to_neighbors(&kind_node, &node);
                }
            }
        };
        Ok(())
    }

    fn insert_to_neighbors(&mut self, node_id: &str, to_insert: &str) {
        if let Some(neighbours) = self.list.get_mut(node_id) {
            neighbours.insert(to_insert.to_string());
        }
    }

    fn capture_all_node(&mut self, re: &Regex, content: &str) -> Option<Vec<String>> {
        let matches: Vec<String> = re
            .captures_iter(content)
            .filter_map(|caps| {
                caps.get(1)
                    .or_else(|| caps.get(0))
                    .map(|m| m.as_str().to_string())
            })
            .collect();

        if matches.is_empty() {
            None
        } else {
            Some(matches)
        }
    }

    fn capture_first_node(&mut self, re: &Regex, content: &str) -> Option<String> {
        let caps = re.captures(content)?;
        let captured = caps.get(1).or_else(|| caps.get(0))?.as_str();
        Some(captured.to_string())
    }

    fn insert_node(&mut self, id: &str, node_type: NodeType) {
        self.list.insert(id.to_string(), HashSet::new());
        self.types.insert(id.to_string(), node_type);
    }

    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph HelmChart {\n");

        dot.push_str("    layout=sfdp;\n");
        dot.push_str("    K=6.0;\n");
        dot.push_str("    overlap=scale;\n");
        dot.push_str("    overlap_scaling=-20;\n");
        dot.push_str("    smoothing=triangle;\n");
        dot.push_str("    repulsiveforce=4.0;\n");
        dot.push_str("    splines=true;\n");
        dot.push_str("    concentrate=false;\n");
        dot.push_str("    sep=\"+1.5\";\n");
        dot.push_str("    margin=1.0;\n");
        dot.push_str("    pad=1.0;\n");

        dot.push_str("    node [fontname=\"Arial\", fontsize=11];\n");
        dot.push_str("    edge [color=\"#00000025\", arrowsize=0.6];\n");
        dot.push_str("    bgcolor=white;\n\n");

        for (node_id, node_type) in &self.types {
            match node_type {
                NodeType::Ressource => {
                    let label = self.clean_label(node_id);
                    writeln!(
                        &mut dot,
                        "    \"{}\" [label=\"{}\", shape=box, fillcolor=\"#4A90E2\", fontcolor=white, style=\"filled,rounded\", penwidth=0, margin=0.15];",
                        node_id, label
                    ).unwrap();
                }
                NodeType::Helper => {
                    let label = self.clean_label(node_id);
                    writeln!(
                        &mut dot,
                        "    \"{}\" [label=\"{}\", shape=circle, fillcolor=\"#7ED321\", style=filled, width=0.25, height=0.25, penwidth=0, fixedsize=true];",
                        node_id, label
                    ).unwrap();
                }
                NodeType::ValuesSection => {
                    let label = self.clean_label(node_id);
                    writeln!(
                        &mut dot,
                        "    \"{}\" [label=\"{}\", shape=circle, fillcolor=\"#F5A623\", style=filled, width=0.18, height=0.18, penwidth=0, fixedsize=true];",
                        node_id, label
                    ).unwrap();
                }
            }
        }

        dot.push_str("\n");

        for (from, neighbors) in &self.list {
            for to in neighbors {
                writeln!(&mut dot, "    \"{}\" -> \"{}\";", from, to).unwrap();
            }
        }

        dot.push_str("}\n");
        dot
    }

    fn clean_label(&self, node_id: &str) -> String {
        if let Some(pos) = node_id.rfind('/') {
            node_id[pos + 1..].to_string()
        } else {
            node_id.to_string()
        }
    }

    pub fn write_dot(&self, output_path: &str) -> anyhow::Result<()> {
        let dot_content = self.to_dot();
        fs::write(output_path, dot_content).context("Failed to write DOT file")?;
        Ok(())
    }

    pub fn render_with_sfdp(&self, output_path: &str, format: &str) -> anyhow::Result<()> {
        use std::process::Command;

        let dot_file = format!("{}.dot", output_path);
        self.write_dot(&dot_file)?;

        let output = Command::new("sfdp")
            .arg("-T")
            .arg(format)
            .arg(&dot_file)
            .arg("-o")
            .arg(format!("{}.{}", output_path, format))
            .output()
            .context("Failed to execute sfdp command")?;

        if !output.status.success() {
            anyhow::bail!("sfdp failed: {}", String::from_utf8_lossy(&output.stderr));
        }

        Ok(())
    }

    pub fn render_large(&self, output_path: &str) -> anyhow::Result<()> {
        use std::process::Command;

        let dot_file = format!("{}.dot", output_path);
        self.write_dot(&dot_file)?;

        let output = Command::new("sfdp")
            .arg("-Tsvg")
            .arg(&dot_file)
            .arg("-o")
            .arg(format!("{}.svg", output_path))
            .output()
            .context("Failed to execute sfdp command")?;

        if !output.status.success() {
            anyhow::bail!("sfdp failed: {}", String::from_utf8_lossy(&output.stderr));
        }

        Ok(())
    }

    pub fn compute_metrics(&self, chart_name: &str) -> GraphMetrics {
        let size = self.types.len();
        let comprehension_scope = self.compute_comprehension_scope();
        let cognitive_diameter = self.compute_cognitive_diameter();
        let hub_dominance = self.compute_hub_dominance();
        let modification_isolation = self.compute_modification_isolation();
        let helper_justification_ratio = self.compute_helper_justification_ratio();
        let blast_radius_variance = self.compute_blast_radius_variance();
        let mut max_nesting_depth = 0;
        let mut unguarded_nested_access = 0;
        let mut array_config_count = 0;
        let mut hardcoded_image_count = 0;
        let mut multi_resource_file_count = 0;
        let mut unquoted_string_count = 0;
        let mut floating_image_tag_count = 0;
        let mut mutable_selector_label = 0;
        let mut missing_pod_selector = 0;

        let chart_root = Path::new(&self.name);
        let chart_root = if chart_root.is_dir() {
            chart_root
        } else {
            Path::new(chart_name)
        };

        let templates_dir = chart_root.join("templates");
        if templates_dir.is_dir() {
            let walker = WalkDir::new(&templates_dir).into_iter();
            for entry in walker.filter_map(|e| e.ok()) {
                if !(is_yaml_file(&entry) || is_tpl_file(&entry)) {
                    continue;
                }
                let path = entry.path().to_string_lossy();
                let path = path.as_ref();

                max_nesting_depth =
                    max_nesting_depth.max(count_max_nesting_depth(path).unwrap_or(0));

                unguarded_nested_access += count_unguarded_nested_access(path).unwrap_or(0);
                hardcoded_image_count += count_hardcoded_images(path).unwrap_or(0);

                if is_yaml_file(&entry) {
                    multi_resource_file_count += count_resource_declaration(path).unwrap_or(0);
                    floating_image_tag_count += count_floating_image_tags(path).unwrap_or(0);
                    mutable_selector_label += count_mutable_selector_labels(path).unwrap_or(0);
                    missing_pod_selector += count_missing_pod_selectors(path).unwrap_or(0);
                }
            }
        }

        let values_yaml = chart_root.join("values.yaml");
        let values_yml = chart_root.join("values.yml");
        let values_path = if values_yaml.is_file() {
            Some(values_yaml)
        } else if values_yml.is_file() {
            Some(values_yml)
        } else {
            None
        };

        if let Some(values_path) = values_path {
            let path = values_path.to_string_lossy();
            let path = path.as_ref();
            array_config_count += count_array_config(path).unwrap_or(0);
            unquoted_string_count += count_unquoted_strings(path).unwrap_or(0);
        }

        GraphMetrics {
            chart_name: chart_name.to_string(),
            size,
            comprehension_scope,
            cognitive_diameter,
            hub_dominance,
            modification_isolation,
            helper_justification_ratio,
            blast_radius_variance,
            max_nesting_depth,
            unguarded_nested_access,
            array_config_count,
            hardcoded_image_count,
            multi_resource_file_count,
            unquoted_string_count,
            floating_image_tag_count,
            mutable_selector_label,
            missing_pod_selector,
        }
    }

    // Metric 2: Comprehension Scope
    // Average fraction of the graph reachable from each resource
    fn compute_comprehension_scope(&self) -> f64 {
        let resources: Vec<&String> = self
            .types
            .iter()
            .filter(|(_, t)| matches!(t, NodeType::Ressource))
            .map(|(node, _)| node)
            .collect();

        if resources.is_empty() {
            return 0.0;
        }

        let total_nodes = self.types.len();
        if total_nodes == 0 {
            return 0.0;
        }

        let sum_reachability: f64 = resources
            .iter()
            .map(|&resource| {
                let reachable_count = self.count_reachable_forward(resource);
                reachable_count as f64 / total_nodes as f64
            })
            .sum();

        sum_reachability / resources.len() as f64
    }

    fn compute_cognitive_diameter(&self) -> f64 {
        let resources: Vec<&String> = self
            .types
            .iter()
            .filter(|(_, t)| matches!(t, NodeType::Ressource))
            .map(|(node, _)| node)
            .collect();

        if resources.len() < 2 {
            return 0.0;
        }

        let mut max_distance: usize = 0;
        let mut found_any_path = false;

        for i in 0..resources.len() {
            for j in (i + 1)..resources.len() {
                let distance = self.shortest_path(resources[i], resources[j]);
                if distance != usize::MAX {
                    // Only count valid paths
                    max_distance = max_distance.max(distance);
                    found_any_path = true;
                }
            }
        }

        if found_any_path {
            max_distance as f64
        } else {
            f64::INFINITY
        }
    } // Metric 4: Hub Dominance
    // Fraction of edges going through top-k nodes (k = sqrt(|V|))
    fn compute_hub_dominance(&self) -> f64 {
        let total_edges: usize = self.list.values().map(|neighbors| neighbors.len()).sum();

        if total_edges == 0 {
            return 0.0;
        }

        // Calculate in-degree + out-degree for each node
        let mut node_degrees: HashMap<String, usize> = HashMap::new();

        for (from, neighbors) in &self.list {
            *node_degrees.entry(from.clone()).or_insert(0) += neighbors.len();
            for to in neighbors {
                *node_degrees.entry(to.clone()).or_insert(0) += 1;
            }
        }

        let k = (self.types.len() as f64).sqrt().ceil() as usize;
        let k = k.max(1);

        let mut degrees: Vec<usize> = node_degrees.values().copied().collect();
        degrees.sort_by(|a, b| b.cmp(a));

        let top_k_total_degree: usize = degrees.iter().take(k).sum();

        top_k_total_degree as f64 / (2.0 * total_edges as f64)
    }

    // Metric 5: Modification Isolation
    // Average fraction of resources that don't share dependencies
    fn compute_modification_isolation(&self) -> f64 {
        let resources: Vec<&String> = self
            .types
            .iter()
            .filter(|(_, t)| matches!(t, NodeType::Ressource))
            .map(|(node, _)| node)
            .collect();

        if resources.is_empty() {
            return 1.0;
        }

        let sum_isolation: f64 = resources
            .iter()
            .map(|&resource| {
                let deps = self.get_all_dependencies(resource);
                let sharing_count = resources
                    .iter()
                    .filter(|&&other_resource| {
                        other_resource != resource && {
                            let other_deps = self.get_all_dependencies(other_resource);
                            !deps.is_disjoint(&other_deps)
                        }
                    })
                    .count();

                let isolation = if resources.len() <= 1 {
                    1.0
                } else {
                    1.0 - (sharing_count as f64 / (resources.len() - 1) as f64)
                };
                isolation
            })
            .sum();

        sum_isolation / resources.len() as f64
    }

    // Metric 6: Helper Justification Ratio
    // Fraction of helpers with fan-in >= 2
    fn compute_helper_justification_ratio(&self) -> f64 {
        let mut helper_in_degrees = HashMap::new();

        for (_from, neighbors) in &self.list {
            for to in neighbors {
                *helper_in_degrees.entry(to.clone()).or_insert(0) += 1;
            }
        }

        let mut helper_count = 0;
        let mut justified_helpers = 0;

        for (node, node_type) in &self.types {
            if matches!(node_type, NodeType::Helper) {
                helper_count += 1;
                let in_degree = helper_in_degrees.get(node).copied().unwrap_or(0);
                if in_degree >= 2 {
                    justified_helpers += 1;
                }
            }
        }

        if helper_count == 0 {
            0.0
        } else {
            justified_helpers as f64 / helper_count as f64
        }
    }

    // Metric 7: Blast Radius Variance
    // Variance in reachable nodes from each value node
    fn compute_blast_radius_variance(&self) -> f64 {
        let value_nodes: Vec<&String> = self
            .types
            .iter()
            .filter(|(_, t)| matches!(t, NodeType::ValuesSection))
            .map(|(node, _)| node)
            .collect();

        if value_nodes.is_empty() {
            return 0.0;
        }

        let blast_radii: Vec<usize> = value_nodes
            .iter()
            .map(|&value| self.count_reverse_reachable(value))
            .collect();

        let mean = blast_radii.iter().sum::<usize>() as f64 / blast_radii.len() as f64;

        let variance = blast_radii
            .iter()
            .map(|&radius| {
                let diff = radius as f64 - mean;
                diff * diff
            })
            .sum::<f64>()
            / blast_radii.len() as f64;

        variance
    }

    // Helper: Count nodes reachable forward from a given start node
    fn count_reachable_forward(&self, start: &str) -> usize {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(start.to_string());
        visited.insert(start.to_string());

        while let Some(node) = queue.pop_front() {
            if let Some(neighbors) = self.list.get(&node) {
                for neighbor in neighbors {
                    if !visited.contains(neighbor) {
                        visited.insert(neighbor.clone());
                        queue.push_back(neighbor.clone());
                    }
                }
            }
        }

        visited.len().saturating_sub(1)
    }

    // Helper: Count nodes that can reach this node (reverse traversal)
    fn count_reverse_reachable(&self, start: &str) -> usize {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(start.to_string());
        visited.insert(start.to_string());

        while let Some(node) = queue.pop_front() {
            for (from, neighbors) in &self.list {
                if neighbors.contains(&node) && !visited.contains(from) {
                    visited.insert(from.clone());
                    queue.push_back(from.clone());
                }
            }
        }

        visited.len().saturating_sub(1)
    }

    // Helper: Find shortest path between two nodes
    fn shortest_path(&self, start: &str, end: &str) -> usize {
        let mut visited = HashMap::new();
        let mut queue = VecDeque::new();

        queue.push_back((start.to_string(), 0));
        visited.insert(start.to_string(), 0);

        while let Some((node, dist)) = queue.pop_front() {
            if node == end {
                return dist;
            }

            // Forward edges
            if let Some(neighbors) = self.list.get(&node) {
                for neighbor in neighbors {
                    if !visited.contains_key(neighbor) {
                        visited.insert(neighbor.clone(), dist + 1);
                        queue.push_back((neighbor.clone(), dist + 1));
                    }
                }
            }

            // Backward edges (undirected graph for resource connectivity)
            for (from, neighbors) in &self.list {
                if neighbors.contains(&node) && !visited.contains_key(from) {
                    visited.insert(from.clone(), dist + 1);
                    queue.push_back((from.clone(), dist + 1));
                }
            }
        }

        usize::MAX // No path found
    }

    // Helper: Get all dependencies (helpers and values) of a resource
    fn get_all_dependencies(&self, resource: &str) -> HashSet<String> {
        let mut deps = HashSet::new();
        let mut queue = VecDeque::new();

        queue.push_back(resource.to_string());

        while let Some(node) = queue.pop_front() {
            if let Some(neighbors) = self.list.get(&node) {
                for neighbor in neighbors {
                    if !deps.contains(neighbor) {
                        deps.insert(neighbor.clone());
                        if let Some(node_type) = self.types.get(neighbor) {
                            if matches!(node_type, NodeType::Helper) {
                                queue.push_back(neighbor.clone());
                            }
                        }
                    }
                }
            }
        }

        deps
    }
}

fn is_tpl_file(e: &DirEntry) -> bool {
    e.file_name().to_str().map_or(false, |s| s.ends_with("tpl"))
}

fn is_yaml_file(e: &DirEntry) -> bool {
    e.file_name()
        .to_str()
        .map_or(false, |s| s.ends_with("yaml") || s.ends_with("yml"))
}

// Metrics functions based on Helm best practices

pub fn count_max_nesting_depth(path: &str) -> anyhow::Result<usize> {
    let f = File::open(path).context("when opening file")?;
    let buf_reader = BufReader::new(f);

    let values_access_regex =
        Regex::new(r"\.Values(?:\.[A-Za-z0-9_-]+)+").context("when creating regex")?;

    let mut max_depth: usize = 0;
    for line in buf_reader.lines() {
        let line = line.context("when reading line")?;
        for m in values_access_regex.find_iter(&line) {
            let s = m.as_str();
            let depth = s
                .trim_start_matches(".Values.")
                .split('.')
                .filter(|seg| !seg.is_empty())
                .count();
            max_depth = max_depth.max(depth);
        }
    }

    Ok(max_depth)
}

pub fn count_unguarded_nested_access(path: &str) -> anyhow::Result<usize> {
    let f = File::open(path).context("when opening file")?;
    let buf_reader = BufReader::new(f);

    let values_nested_regex =
        Regex::new(r"\.Values(?:\.[A-Za-z0-9_-]+){2,}").context("when creating regex")?;

    let mut guard_depth: i32 = 0;
    let mut nb_unguarded_access: usize = 0;

    for line in buf_reader.lines() {
        let line = line.context("when reading line")?;
        update_guard_depth(&line, &mut guard_depth)?;

        if guard_depth == 0 {
            nb_unguarded_access += values_nested_regex.find_iter(&line).count();
        }
    }

    Ok(nb_unguarded_access)
}

fn update_guard_depth(line: &str, guard_depth: &mut i32) -> anyhow::Result<()> {
    // Consider common guarding constructs around `.Values` accesses.
    // This is intentionally conservative: if inside any `if/with/range`, treat as guarded.
    let start_regex =
        Regex::new(r#"\{\{\-?\s*(if|with|range)\b"#).context("when creating start regex")?;
    let end_regex = Regex::new(r#"\{\{\-?\s*end\b"#).context("when creating end regex")?;

    if start_regex.is_match(line) {
        *guard_depth += 1;
    }
    if end_regex.is_match(line) {
        *guard_depth = (*guard_depth - 1).max(0);
    }

    Ok(())
}

// ●     array_config_count : structures en array qui cassent --set
pub fn count_array_config(path: &str) -> anyhow::Result<usize> {
    let yaml = fs::read_to_string(path).context("when reading values yaml")?;
    let value: serde_yaml::Value = serde_yaml::from_str(&yaml).context("when parsing values yaml")?;

    fn count_sequences(v: &serde_yaml::Value) -> usize {
        match v {
            serde_yaml::Value::Sequence(items) => 1 + items.iter().map(count_sequences).sum::<usize>(),
            serde_yaml::Value::Mapping(map) => map.values().map(count_sequences).sum::<usize>(),
            _ => 0,
        }
    }

    Ok(count_sequences(&value))
}

// ●     hardcoded_image_count : images hardcodées dans les templates
pub fn count_hardcoded_images(path: &str) -> anyhow::Result<usize> {
    let f = File::open(path).context("when opening file")?;
    let buf_reader = BufReader::new(f);

    let values_nested_regex =
        Regex::new(r"\.Values(?:\.[A-Za-z0-9_-]+){2,}").context("when creating regex")?;

    let mut nb_hardcoded_images: usize = 0;

    for line in buf_reader.lines() {
        let line = line.context("when reading line")?;

        if (line.contains("image:") || line.contains("image: ")) && !values_nested_regex.is_match(&line) {
            nb_hardcoded_images += 1;
        }
    }

    Ok(nb_hardcoded_images)
}

// ●     multi_resource_file_count : fichiers avec plusieurs ressources
pub fn count_resource_declaration(path: &str) -> anyhow::Result<usize> {
    let f = File::open(path).context("when opening file")?;
    let buf_reader = BufReader::new(f);

    let kind_regex = Regex::new(r#"(?m)^\s*kind:\s*['\"]?(\w+)['\"]?"#)?;
    let mut nb_resources: usize = 0;

    for line in buf_reader.lines() {
        let line = line.context("when reading line")?;

        if kind_regex.is_match(&line) {
            nb_resources += 1;
        }
    }
    Ok(if nb_resources > 1 { 1 } else { 0 })
}

// ●     unquoted_string_count : strings non quotées
pub fn count_unquoted_strings(path: &str) -> anyhow::Result<usize> {
    let f = File::open(path).context("when opening file")?;
    let buf_reader = BufReader::new(f);

    let value_regex = Regex::new(r#"(?m)^\s*[\w.-]+:\s+([^"'\s\-\{\[\n][^\n#]*)"#)?;

    // Types à exclure
    let is_boolean = Regex::new(r"^(true|false|null)$")?;
    let is_number = Regex::new(r"^-?\d+(\.\d+)?$")?;

    let mut unquoted_count: usize = 0;

    for line in buf_reader.lines() {
        let line = line.context("when reading line")?;

        if let Some(caps) = value_regex.captures(&line) {
            let val = caps.get(1).map_or("", |m| m.as_str()).trim();

            // Si ce n'est ni un boolean, ni un nombre alors on compte comme string non quotée
            if !is_boolean.is_match(val) && !is_number.is_match(val) {
                unquoted_count += 1;
            }
        }
    }

    Ok(unquoted_count)
}

// ●     floating_image_tag_count : tags d'images flottants (latest, head, canary)
pub fn count_floating_image_tags(path: &str) -> anyhow::Result<usize> {
    let f = File::open(path).context("when opening file")?;
    let buf_reader = BufReader::new(f);

    let image_tag_regex = Regex::new(
        r#"image:\s*["']?[^"':]+:(latest|head|master|main|canary|stable)["']?"#,
    )
    .context("when creating regex")?;

    let mut floating_tag_count: usize = 0;

    for line in buf_reader.lines() {
        let line = line.context("when reading line")?;

        if image_tag_regex.is_match(&line) {
            floating_tag_count += 1;
        }
    }

    Ok(floating_tag_count)
}

// ●     mutable_selector_label : labels mutables dans les selectors
pub fn count_mutable_selector_labels(path: &str) -> anyhow::Result<usize> {
    let f = File::open(path).context("when opening file")?;
    let buf_reader = BufReader::new(f);

    let selector_block = Regex::new(r#"(?i)selector:"#)?;
    let mutable_labels = Regex::new(r#"(?i)(chart|heritage|release|version|date):"#)?;

    let mut count = 0;
    let mut in_selector_block = false;
    let mut selector_indent = 0;

    for line in buf_reader.lines() {
        let line = line.context("when reading line")?;
        let current_indent = line.len() - line.trim_start().len();

        if selector_block.is_match(&line) {
            in_selector_block = true;
            selector_indent = current_indent;
            continue;
        }

        if in_selector_block {
            if !line.trim().is_empty() && current_indent <= selector_indent && !line.contains("matchLabels") {
                in_selector_block = false;
                continue;
            }
            if mutable_labels.is_match(&line) {
                count += 1;
            }
        }
    }

    Ok(count)
}

// ●     missing_pod_selector : selectors manquants sur les PodTemplates
pub fn count_missing_pod_selectors(path: &str) -> anyhow::Result<usize> {
    let f = File::open(path).context("when opening file")?;
    let buf_reader = BufReader::new(f);

    let pod_controller_kind =
        Regex::new(r#"(?i)kind:\s*(Deployment|StatefulSet|DaemonSet|Job)"#)?;
    let selector_regex = Regex::new(r#"(?i)selector:"#)?;
    let match_labels_regex = Regex::new(r#"(?i)matchLabels:"#)?;

    let mut missing_selector_count = 0;

    let mut is_pod_controller = false;
    let mut found_selector = false;
    let mut found_match_labels = false;

    for line in buf_reader.lines() {
        let line = line.context("when reading line")?;

        if line.starts_with("---") {
            if is_pod_controller && (!found_selector || !found_match_labels) {
                missing_selector_count += 1;
            }

            is_pod_controller = false;
            found_selector = false;
            found_match_labels = false;
            continue;
        }

        if pod_controller_kind.is_match(&line) {
            is_pod_controller = true;
        }
        if selector_regex.is_match(&line) {
            found_selector = true;
        }
        if match_labels_regex.is_match(&line) {
            found_match_labels = true;
        }
    }

    if is_pod_controller && (!found_selector || !found_match_labels) {
        missing_selector_count += 1;
    }

    Ok(missing_selector_count)
}
