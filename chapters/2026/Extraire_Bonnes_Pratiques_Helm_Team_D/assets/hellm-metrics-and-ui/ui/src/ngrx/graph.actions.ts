import { createActionGroup, emptyProps, props } from '@ngrx/store';
import { IGraph } from '../helpers/graph';

export const GraphActions = createActionGroup({
  source: 'Graph',
  events: {
    'Load Graphs': emptyProps(),
    'Load Graphs Success': props<{ graphs: IGraph[] }>(),
    'Load Graphs Failure': props<{ error: string }>(),

    'Select Graph': props<{ graphName: string }>(),
    'Clear Selection': emptyProps(),

    'Toggle Labels': emptyProps(),
    'Set Labels': props<{ showLabels: boolean }>()
  }
});
