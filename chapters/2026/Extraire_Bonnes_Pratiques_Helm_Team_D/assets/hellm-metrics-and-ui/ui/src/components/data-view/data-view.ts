import { Component, Input, OnInit, OnDestroy, inject, OnChanges, SimpleChanges, ViewChild } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Store } from '@ngrx/store';
import { Subscription, combineLatest } from 'rxjs';
import { Card } from 'primeng/card';
import { Select, SelectChangeEvent } from 'primeng/select';
import { UIChart } from 'primeng/chart';
import { MetricsActions } from '../../ngrx/metrics.actions';
import {
  selectAllMetrics,
  selectSelectedMetricKey,
  selectCurrentDistribution,
  selectSelectedMetricDescription
} from '../../ngrx/metrics.selectors';
import { MetricKey, METRIC_KEYS, METRIC_LABELS, DistributionBin, findChartBin, getMetricValue } from '../../helpers/metrics';

interface MetricOption {
  label: string;
  value: MetricKey;
}

@Component({
  selector: 'app-data-view',
  imports: [CommonModule, FormsModule, Card, Select, UIChart],
  templateUrl: './data-view.html',
})
export class DataView implements OnInit, OnDestroy, OnChanges {
  @Input() selectedChartName: string | null = null;
  @ViewChild('distributionChart') distributionChart!: UIChart;

  private store = inject(Store);
  private subscription!: Subscription;

  metrics$ = this.store.select(selectAllMetrics);
  selectedMetricKey$ = this.store.select(selectSelectedMetricKey);
  distribution$ = this.store.select(selectCurrentDistribution);
  selectedMetricDescription$ = this.store.select(selectSelectedMetricDescription);

  metricOptions: MetricOption[] = METRIC_KEYS.map(key => ({
    label: METRIC_LABELS[key],
    value: key
  }));

  selectedMetricKey: MetricKey = 'size';
  distribution: DistributionBin[] = [];
  highlightedBinIndex: number = -1;
  chartMetricValue: number | null = null;
  highlightedBinRange: string = '';

  // Chart.js data and options
  chartData: any = {};
  chartOptions: any = {};

  ngOnInit(): void {
    this.store.dispatch(MetricsActions.loadMetrics());
    this.initChartOptions();
    
    this.subscription = combineLatest([
      this.selectedMetricKey$,
      this.distribution$,
      this.metrics$
    ]).subscribe(([metricKey, distribution, metrics]) => {
      this.selectedMetricKey = metricKey;
      this.distribution = distribution;
      
      if (this.selectedChartName && metrics.length > 0) {
        this.highlightedBinIndex = findChartBin(metrics, this.selectedChartName, metricKey, 10);
        this.chartMetricValue = getMetricValue(metrics, this.selectedChartName, metricKey);
      } else {
        this.highlightedBinIndex = -1;
        this.chartMetricValue = null;
      }
      
      this.updateChartData();
    });
  }

  ngOnChanges(changes: SimpleChanges): void {
    if (changes['selectedChartName'] && !changes['selectedChartName'].firstChange) {
      this.updateHighlight();
    }
  }

  private updateHighlight(): void {
    combineLatest([this.metrics$, this.selectedMetricKey$, this.distribution$]).subscribe(([metrics, metricKey, distribution]) => {
      if (this.selectedChartName && metrics.length > 0) {
        this.highlightedBinIndex = findChartBin(metrics, this.selectedChartName, metricKey, 10);
        this.chartMetricValue = getMetricValue(metrics, this.selectedChartName, metricKey);
        if (this.highlightedBinIndex >= 0 && distribution[this.highlightedBinIndex]) {
          const bin = distribution[this.highlightedBinIndex];
          this.highlightedBinRange = `${this.formatValue(bin.min)} - ${this.formatValue(bin.max)}`;
        } else {
          this.highlightedBinRange = '';
        }
      } else {
        this.highlightedBinIndex = -1;
        this.chartMetricValue = null;
        this.highlightedBinRange = '';
      }
      this.updateChartData();
    }).unsubscribe();
  }

  ngOnDestroy(): void {
    this.subscription?.unsubscribe();
  }

  onMetricSelect(event: SelectChangeEvent): void {
    const metricKey = event.value as MetricKey;
    if (metricKey) {
      this.store.dispatch(MetricsActions.selectMetric({ metricKey }));
    }
  }

  formatValue(value: number): string {
    if (Number.isInteger(value)) {
      return value.toString();
    }
    return value.toFixed(3);
  }

  private initChartOptions(): void {
    this.chartOptions = {
      responsive: true,
      maintainAspectRatio: false,
      resizeDelay: 0,
      plugins: {
        legend: {
          display: false
        },
        tooltip: {
          callbacks: {
            title: (context: any) => {
              const index = context[0].dataIndex;
              if (this.distribution[index]) {
                const bin = this.distribution[index];
                return `Range: ${this.formatValue(bin.min)} - ${this.formatValue(bin.max)}`;
              }
              return '';
            },
            label: (context: any) => {
              const index = context.dataIndex;
              if (this.distribution[index]) {
                const bin = this.distribution[index];
                const chartsText = bin.charts.length > 3 
                  ? bin.charts.slice(0, 3).join(', ') + ` +${bin.charts.length - 3} more`
                  : bin.charts.join(', ') || 'None';
                return [`Count: ${bin.count}`, `Charts: ${chartsText}`];
              }
              return `Count: ${context.raw}`;
            }
          }
        }
      },
      scales: {
        x: {
          grid: {
            color: 'rgba(255, 255, 255, 0.1)'
          },
          ticks: {
            color: '#9ca3af'
          }
        },
        y: {
          beginAtZero: true,
          grid: {
            color: 'rgba(255, 255, 255, 0.1)'
          },
          ticks: {
            color: '#9ca3af',
            stepSize: 1
          }
        }
      }
    };
  }

  private updateChartData(): void {
    const labels = this.distribution.map((bin) => {
      return this.formatValue(bin.min);
    });

    // Create background colors - highlight the bin containing selected chart
    const backgroundColors = this.distribution.map((_, index) => {
      if (index === this.highlightedBinIndex) {
        return 'rgba(239, 68, 68, 0.9)'; // Red for highlighted
      }
      return 'rgba(99, 102, 241, 0.6)'; // Indigo for others
    });

    this.chartData = {
      labels: [...labels],
      datasets: [
        {
          label: 'Count',
          data: this.distribution.map(bin => bin.count),
          backgroundColor: backgroundColors,
          borderColor: 'transparent',
          borderWidth: 0,
          borderRadius: 4
        }
      ]
    };

    if (this.distributionChart?.chart) {
      this.distributionChart.chart.update();
    }
  }
}
