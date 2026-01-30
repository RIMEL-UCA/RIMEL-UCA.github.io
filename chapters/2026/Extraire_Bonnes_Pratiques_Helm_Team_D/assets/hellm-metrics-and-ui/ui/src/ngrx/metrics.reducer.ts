import { createReducer, on } from '@ngrx/store';
import { MetricsActions } from './metrics.actions';
import { MetricData, MetricKey } from '../helpers/metrics';

export interface MetricsState {
  metrics: MetricData[];
  selectedMetricKey: MetricKey;
  loading: boolean;
  error: string | null;
}

export const initialMetricsState: MetricsState = {
  metrics: [],
  selectedMetricKey: 'size',
  loading: false,
  error: null
};

export const metricsReducer = createReducer(
  initialMetricsState,

  on(MetricsActions.loadMetrics, (state) => ({
    ...state,
    loading: true,
    error: null
  })),

  on(MetricsActions.loadMetricsSuccess, (state, { metrics }) => ({
    ...state,
    metrics,
    loading: false
  })),

  on(MetricsActions.loadMetricsFailure, (state, { error }) => ({
    ...state,
    loading: false,
    error
  })),

  on(MetricsActions.selectMetric, (state, { metricKey }) => ({
    ...state,
    selectedMetricKey: metricKey
  }))
);
