import { createFeatureSelector, createSelector } from '@ngrx/store';
import { MetricsState } from './metrics.reducer';
import { computeDistribution, METRIC_KEYS, METRIC_LABELS, METRIC_DESCRIPTIONS, findChartBin, getMetricValue } from '../helpers/metrics';

export const selectMetricsState = createFeatureSelector<MetricsState>('metrics');

export const selectAllMetrics = createSelector(
  selectMetricsState,
  (state) => state.metrics
);

export const selectSelectedMetricKey = createSelector(
  selectMetricsState,
  (state) => state.selectedMetricKey
);

export const selectMetricsLoading = createSelector(
  selectMetricsState,
  (state) => state.loading
);

export const selectMetricsError = createSelector(
  selectMetricsState,
  (state) => state.error
);

export const selectMetricKeys = createSelector(
  () => METRIC_KEYS
);

export const selectMetricLabels = createSelector(
  () => METRIC_LABELS
);

export const selectMetricDescriptions = createSelector(
  () => METRIC_DESCRIPTIONS
);

export const selectCurrentDistribution = createSelector(
  selectAllMetrics,
  selectSelectedMetricKey,
  (metrics, metricKey) => computeDistribution(metrics, metricKey, 10)
);

export const selectChartBinIndex = (chartName: string) => createSelector(
  selectAllMetrics,
  selectSelectedMetricKey,
  (metrics, metricKey) => findChartBin(metrics, chartName, metricKey, 10)
);

export const selectChartMetricValue = (chartName: string) => createSelector(
  selectAllMetrics,
  selectSelectedMetricKey,
  (metrics, metricKey) => getMetricValue(metrics, chartName, metricKey)
);

export const selectSelectedMetricLabel = createSelector(
  selectSelectedMetricKey,
  (metricKey) => METRIC_LABELS[metricKey]
);

export const selectSelectedMetricDescription = createSelector(
  selectSelectedMetricKey,
  (metricKey) => METRIC_DESCRIPTIONS[metricKey]
);
