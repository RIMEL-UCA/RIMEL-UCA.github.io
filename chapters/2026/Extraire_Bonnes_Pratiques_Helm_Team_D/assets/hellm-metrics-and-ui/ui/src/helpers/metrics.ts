export interface MetricData {
  chart_name: string;
  size: number;
  comprehension_scope: number;
  cognitive_diameter: number | null;
  hub_dominance: number;
  modification_isolation: number;
  helper_justification_ratio: number;
  blast_radius_variance: number;
  max_nesting_depth: number,
  unguarded_nested_access: number,
  array_config_count: number,
  hardcoded_image_count: number,
  multi_resource_file_count: number,
  unquoted_string_count: number,
  floating_image_tag_count: number,
  mutable_selector_label: number,
  missing_pod_selector: number,
}

export type MetricKey = keyof Omit<MetricData, 'chart_name'>;

export const METRIC_KEYS: MetricKey[] = [
  'size',
  'comprehension_scope',
  'cognitive_diameter',
  'hub_dominance',
  'modification_isolation',
  'helper_justification_ratio',
  'blast_radius_variance',
  'max_nesting_depth',
  'unguarded_nested_access',
  'array_config_count',
  'hardcoded_image_count',
  'multi_resource_file_count',
  'unquoted_string_count',
  'floating_image_tag_count',
  'mutable_selector_label',
  'missing_pod_selector',
];

export const METRIC_LABELS: Record<MetricKey, string> = {
  size: 'Size',
  comprehension_scope: 'Comprehension Scope',
  cognitive_diameter: 'Cognitive Diameter',
  hub_dominance: 'Hub Dominance',
  modification_isolation: 'Modification Isolation',
  helper_justification_ratio: 'Helper Justification Ratio',
  blast_radius_variance: 'Blast Radius Variance',
  max_nesting_depth: 'Max Nesting Depth',
  unguarded_nested_access: 'Unguarded Nest Access',
  array_config_count: 'Array Config Count',
  hardcoded_image_count: 'Hardcoded Image Count',
  multi_resource_file_count: 'Multi Resource File Count',
  unquoted_string_count: 'Unquoted String Count',
  floating_image_tag_count: 'Floating Image Tag Count',
  mutable_selector_label: 'Mutable Selector Label',
  missing_pod_selector: 'Missing Pod Selector',
};

export const METRIC_DESCRIPTIONS: Record<MetricKey, string> = {
  size: 'Number of nodes in the graph',
  comprehension_scope: 'Measure of how easy the graph is to understand',
  cognitive_diameter: 'Maximum cognitive distance between nodes',
  hub_dominance: 'Degree to which the graph is dominated by hub nodes',
  modification_isolation: 'How isolated changes are from the rest of the graph',
  helper_justification_ratio: 'Ratio of helper nodes to total nodes',
  blast_radius_variance: 'Variance in potential impact of changes',
  max_nesting_depth: 'Max Nesting Depth',
  unguarded_nested_access: 'Unguarded Nest Access',
  array_config_count: 'Array Config Count',
  hardcoded_image_count: 'Hardcoded Image Count',
  multi_resource_file_count: 'Multi Resource File Count',
  unquoted_string_count: 'Unquoted String Count',
  floating_image_tag_count: 'Floating Image Tag Count',
  mutable_selector_label: 'Mutable Selector Label',
  missing_pod_selector: 'Missing Pod Selector',
};

export interface DistributionBin {
  min: number;
  max: number;
  count: number;
  charts: string[];
}

export function normalizeChartName(name: string): string {
  const parts = name.split('/');
  return parts[parts.length - 1];
}

export function computeDistribution(metrics: MetricData[], metricKey: MetricKey, binCount: number = 10): DistributionBin[] {
  const values = metrics
    .map(m => ({ name: m.chart_name, value: m[metricKey] }))
    .filter(v => v.value !== null && v.value !== undefined) as { name: string; value: number }[];

  if (values.length === 0) return [];

  const numericValues = values.map(v => v.value);
  const min = Math.min(...numericValues);
  const max = Math.max(...numericValues);

  if (min === max) {
    return [{
      min,
      max,
      count: values.length,
      charts: values.map(v => v.name)
    }];
  }

  const binWidth = (max - min) / binCount;
  const bins: DistributionBin[] = [];

  for (let i = 0; i < binCount; i++) {
    const binMin = min + i * binWidth;
    const binMax = i === binCount - 1 ? max + 0.001 : min + (i + 1) * binWidth;
    const chartsInBin = values
      .filter(v => v.value >= binMin && v.value < binMax)
      .map(v => v.name);

    bins.push({
      min: binMin,
      max: binMax,
      count: chartsInBin.length,
      charts: chartsInBin
    });
  }

  return bins;
}

export function findChartBin(metrics: MetricData[], chartName: string, metricKey: MetricKey, binCount: number = 10): number {
  const normalizedChartName = normalizeChartName(chartName);

  const values = metrics
    .map(m => ({ name: m.chart_name, value: m[metricKey] }))
    .filter(v => v.value !== null && v.value !== undefined) as { name: string; value: number }[];

  if (values.length === 0) return -1;

  const numericValues = values.map(v => v.value);
  const min = Math.min(...numericValues);
  const max = Math.max(...numericValues);

  if (min === max) return 0;

  const chartValue = metrics.find(m => m.chart_name === normalizedChartName)?.[metricKey];
  if (chartValue === null || chartValue === undefined) return -1;

  const binWidth = (max - min) / binCount;
  const binIndex = Math.min(Math.floor((chartValue as number - min) / binWidth), binCount - 1);

  return binIndex;
}

export function getMetricValue(metrics: MetricData[], chartName: string, metricKey: MetricKey): number | null {
  const normalizedChartName = normalizeChartName(chartName);
  const metric = metrics.find(m => m.chart_name === normalizedChartName);
  if (!metric) return null;
  return metric[metricKey] as number | null;
}
