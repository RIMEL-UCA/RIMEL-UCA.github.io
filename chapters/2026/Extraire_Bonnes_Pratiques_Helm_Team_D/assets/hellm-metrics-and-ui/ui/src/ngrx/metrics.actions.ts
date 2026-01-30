import { createActionGroup, emptyProps, props } from '@ngrx/store';
import { MetricData, MetricKey } from '../helpers/metrics';

export const MetricsActions = createActionGroup({
  source: 'Metrics',
  events: {
    'Load Metrics': emptyProps(),
    'Load Metrics Success': props<{ metrics: MetricData[] }>(),
    'Load Metrics Failure': props<{ error: string }>(),

    'Select Metric': props<{ metricKey: MetricKey }>(),
  }
});
