import { createReducer, on } from '@ngrx/store';
import { GraphActions } from './graph.actions';
import { IGraph } from '../helpers/graph';

export interface GraphState {
  graphs: IGraph[];
  selectedGraphName: string | null;
  selectedGraph: IGraph | null;
  showLabels: boolean;
  loading: boolean;
  error: string | null;
}

export const initialState: GraphState = {
  graphs: [],
  selectedGraphName: null,
  selectedGraph: null,
  showLabels: true,
  loading: false,
  error: null
};

export const graphReducer = createReducer(
  initialState,

  on(GraphActions.loadGraphs, (state) => ({
    ...state,
    loading: true,
    error: null
  })),

  on(GraphActions.loadGraphsSuccess, (state, { graphs }) => ({
    ...state,
    graphs,
    loading: false,
    selectedGraphName: graphs.length > 0 ? graphs[0].name : null,
    selectedGraph: graphs.length > 0 ? graphs[0] : null
  })),

  on(GraphActions.loadGraphsFailure, (state, { error }) => ({
    ...state,
    loading: false,
    error
  })),

  on(GraphActions.selectGraph, (state, { graphName }) => ({
    ...state,
    selectedGraphName: graphName,
    selectedGraph: state.graphs.find(g => g.name === graphName) || null
  })),

  on(GraphActions.clearSelection, (state) => ({
    ...state,
    selectedGraphName: null,
    selectedGraph: null
  })),

  on(GraphActions.toggleLabels, (state) => ({
    ...state,
    showLabels: !state.showLabels
  })),

  on(GraphActions.setLabels, (state, { showLabels }) => ({
    ...state,
    showLabels
  }))
);
