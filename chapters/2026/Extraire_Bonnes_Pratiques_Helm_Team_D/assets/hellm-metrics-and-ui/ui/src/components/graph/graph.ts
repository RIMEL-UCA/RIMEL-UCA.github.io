import { AfterViewInit, Component, ElementRef, Inject, Input, OnChanges, PLATFORM_ID, SimpleChanges, ViewChild, inject } from '@angular/core';
import { Store } from '@ngrx/store';
import { convertToGraphData, IGraph } from '../../helpers/graph';
import { isPlatformBrowser } from '@angular/common';
import { GraphActions } from '../../ngrx/graph.actions';
import { selectShowLabels } from '../../ngrx/graph.selectors';
import { Card } from 'primeng/card';
import { Button } from 'primeng/button';

@Component({
  selector: 'app-graph',
  imports: [Card, Button],
  templateUrl: './graph.html',
  styleUrl: './graph.css',
})
export class Graph implements AfterViewInit, OnChanges {
  @ViewChild("graph") graphElement?: ElementRef;
  @Input() graphInput!: IGraph;

  private store = inject(Store);
  private graphInstance: any;

  showLabels$ = this.store.select(selectShowLabels);
  showLabels: boolean = true;

  constructor(@Inject(PLATFORM_ID) private platformID: Object) {
    this.showLabels$.subscribe(show => {
      this.showLabels = show;
      if (this.graphInstance) {
        this.graphInstance.nodeCanvasObject(this.graphInstance.nodeCanvasObject());
      }
    });
  }

  toggleLabels(): void {
    this.store.dispatch(GraphActions.toggleLabels());
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['graphInput'] && !changes['graphInput'].firstChange) {
      this.renderGraph();
    }
  }

  async ngAfterViewInit(): Promise<void> {
    await this.renderGraph();
  }

  private async renderGraph(): Promise<void> {
    if (!isPlatformBrowser(this.platformID) || !this.graphInput) return;

    const ForceGraph = (await import('force-graph')).default;

    if (this.graphInstance) {
      this.graphInstance.graphData(convertToGraphData(this.graphInput));
    } else {
      this.graphInstance = new ForceGraph(this.graphElement?.nativeElement).graphData(convertToGraphData(this.graphInput))
        .nodeLabel(node => { return `${node.id}` }).nodeColor((node) => {
          switch (this.graphInput.types[`${node.id}`]) {
            case "Ressource":
              return "#10b981";
            case "Helper":
              return "#3b82f6";
            case "ValuesSection":
              return "#f59e0b";
            default:
              return "#6b7280";
          }
        })
        .nodeCanvasObject((node, ctx, globalScale) => {
          if (!this.showLabels) return;

          const label = `${node.id}`.split('/').pop() || `${node.id}`;
          const fontSize = 10 / globalScale;
          ctx.font = `${fontSize}px Sans-Serif`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'top';

          const textWidth = ctx.measureText(label).width;
          const padding = 2;
          ctx.fillStyle = 'rgba(255, 255, 255, 0.8)';
          ctx.fillRect(node.x! - textWidth / 2 - padding, node.y! + 8, textWidth + padding * 2, fontSize + padding * 2);

          ctx.fillStyle = '#1f2937';
          ctx.fillText(label, node.x!, node.y! + 10);
        })
        .nodeCanvasObjectMode(() => 'after')
        .linkDirectionalArrowLength(6)
        .linkDirectionalArrowRelPos(1)
        .linkColor(() => '#ffffff');
    }
  }
}
