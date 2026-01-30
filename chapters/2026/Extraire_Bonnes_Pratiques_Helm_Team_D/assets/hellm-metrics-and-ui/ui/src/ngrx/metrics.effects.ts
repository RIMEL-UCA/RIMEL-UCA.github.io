import { inject, Injectable } from '@angular/core';
import { Actions, createEffect, ofType } from '@ngrx/effects';
import { of } from 'rxjs';
import { map, catchError } from 'rxjs/operators';
import { MetricsActions } from './metrics.actions';
import { MetricData } from '../helpers/metrics';
import metricsData from '../../../metrics_data.json';

@Injectable()
export class MetricsEffects {
  private actions$ = inject(Actions);

  loadMetrics$ = createEffect(() =>
    this.actions$.pipe(
      ofType(MetricsActions.loadMetrics),
      map(() => {
        const metrics = metricsData as MetricData[];
        return MetricsActions.loadMetricsSuccess({ metrics });
      }),
      catchError((error) =>
        of(MetricsActions.loadMetricsFailure({
          error: error.message || 'Failed to load metrics'
        }))
      )
    )
  );
}
