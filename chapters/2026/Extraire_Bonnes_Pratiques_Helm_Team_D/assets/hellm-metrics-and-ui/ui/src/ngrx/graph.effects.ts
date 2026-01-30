import { inject, Injectable } from '@angular/core';
import { Actions, createEffect, ofType } from '@ngrx/effects';
import { of } from 'rxjs';
import { map, catchError } from 'rxjs/operators';
import { GraphActions } from './graph.actions';
import { IGraph } from '../helpers/graph';
import graphsData from '../../../graphs.json';

@Injectable()
export class GraphEffects {
  private actions$ = inject(Actions);

  loadGraphs$ = createEffect(() =>
    this.actions$.pipe(
      ofType(GraphActions.loadGraphs),
      map(() => {
        const graphs = graphsData as IGraph[];
        return GraphActions.loadGraphsSuccess({ graphs });
      }),
      catchError((error) =>
        of(GraphActions.loadGraphsFailure({
          error: error.message || 'Failed to load graphs'
        }))
      )
    )
  );
}
