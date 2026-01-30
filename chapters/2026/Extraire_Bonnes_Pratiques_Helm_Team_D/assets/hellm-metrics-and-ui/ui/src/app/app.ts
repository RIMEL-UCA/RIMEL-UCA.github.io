import { Component, OnDestroy, OnInit, inject } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { RouterOutlet } from '@angular/router';
import { Store } from '@ngrx/store';
import { Graph } from '../components/graph/graph';
import { DataView } from '../components/data-view/data-view';
import { GraphActions } from '../ngrx/graph.actions';
import { Select, SelectChangeEvent } from 'primeng/select';
import { Tabs, TabList, Tab, TabPanels, TabPanel } from 'primeng/tabs';

import {
  selectGraphNames,
  selectSelectedGraphName,
  selectSelectedGraph
} from '../ngrx/graph.selectors';
import { Subscription } from 'rxjs';

@Component({
  selector: 'app-root',
  imports: [CommonModule, FormsModule, RouterOutlet, Graph, DataView, Select, Tabs, TabList, Tab, TabPanels, TabPanel],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App implements OnInit, OnDestroy {
  private store = inject(Store);
  names?: string[];
  subscription!: Subscription;
  protected graphNames$ = this.store.select(selectGraphNames);
  protected selectedGraphName$ = this.store.select(selectSelectedGraphName);
  protected selectedGraph$ = this.store.select(selectSelectedGraph);

  selectedChartName: string | null = null;

  ngOnInit(): void {
    this.store.dispatch(GraphActions.loadGraphs());
    this.subscription = this.graphNames$.subscribe((names) => {
      this.names = names;
    });
    
    this.selectedGraphName$.subscribe((name) => {
      this.selectedChartName = name;
    });
  }

  ngOnDestroy(): void {
    this.subscription.unsubscribe();
  }

  onGraphSelect(event: SelectChangeEvent): void {
    const graphName = event.value;
    if (graphName) {
      this.store.dispatch(GraphActions.selectGraph({ graphName }));
    }
  }
}
