import { createFeatureSelector, createSelector } from '@ngrx/store';
import { GraphState } from './graph.reducer';

export const selectGraphState = createFeatureSelector<GraphState>('graph');

export const selectAllGraphs = createSelector(
  selectGraphState,
  (state) => state.graphs
);

export const selectSelectedGraphName = createSelector(
  selectGraphState,
  (state) => state.selectedGraphName
);

export const selectSelectedGraph = createSelector(
  selectGraphState,
  (state) => state.selectedGraph
);

export const selectShowLabels = createSelector(
  selectGraphState,
  (state) => state.showLabels
);

export const selectLoading = createSelector(
  selectGraphState,
  (state) => state.loading
);

export const selectError = createSelector(
  selectGraphState,
  (state) => state.error
);

// Derived selectors
export const selectGraphNames = createSelector(
  selectAllGraphs,
  (graphs) => graphs.map(g => g.name)
);

export const selectGraphCount = createSelector(
  selectAllGraphs,
  (graphs) => graphs.length
);

export const selectHasGraphs = createSelector(
  selectGraphCount,
  (count) => count > 0
);

// View model selector (for components)
export const selectGraphViewModel = createSelector(
  selectAllGraphs,
  selectSelectedGraph,
  selectShowLabels,
  selectLoading,
  selectError,
  (graphs, selectedGraph, showLabels, loading, error) => ({
    graphs,
    selectedGraph,
    showLabels,
    loading,
    error,
    hasGraphs: graphs.length > 0
  })
);
