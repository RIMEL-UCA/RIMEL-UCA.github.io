const { ChartJSNodeCanvas } = require('chartjs-node-canvas');
const fs = require('fs')

module.exports.generate = async (json, output, title) => {
    const width = 2900;
    const height = 700;
    const chartJSNodeCanvas = new ChartJSNodeCanvas({
        width, height, chartCallback: (Chart) => {
            Chart.plugins.register({
                beforeDraw: function (chartInstance) {
                    var ctx = chartInstance.chart.ctx;
                    ctx.fillStyle = "white";
                    ctx.fillRect(0, 0, chartInstance.chart.width, chartInstance.chart.height);
                }
            });
        }
    });
    const configuration = {
        type: 'line',
        data: {
            labels: json.map(e => e.name),
            datasets: [{
                label: 'Test unitaire',
                data: json.map(e => e.countUnit),
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1
            }, {
                label: 'Test fonctionnel',
                data: json.map(e => e.countFunc),
                backgroundColor: 'rgba(54, 162, 235, 0.2)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            title: {
                display: true,
                text: title
            },
            scales: {
                yAxes: [{
                    ticks: {
                        beginAtZero: true
                    }
                }]
            }
        }
    };
    const image = await chartJSNodeCanvas.renderToBuffer(configuration, 'image/png');
    // const dataUrl = await chartJSNodeCanvas.renderToDataURL(configuration);
    // const stream = chartJSNodeCanvas.renderToStream(configuration);

    fs.writeFileSync(output, image)
}

module.exports.generateRatio = async (json, output) => {
    const width = 2900;
    const height = 700;
    const chartJSNodeCanvas = new ChartJSNodeCanvas({
        width, height, chartCallback: (Chart) => {
            Chart.plugins.register({
                beforeDraw: function (chartInstance) {
                    var ctx = chartInstance.chart.ctx;
                    ctx.fillStyle = "white";
                    ctx.fillRect(0, 0, chartInstance.chart.width, chartInstance.chart.height);
                }
            });
        }
    });
    const configuration = {
        type: 'line',
        data: {
            labels: json.map(e => e.name),
            datasets: [{
                data: json.map(e => e.ratio),
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1
            }]
        },
        options: {
            title: {
                display: true,
                text: 'Ratio du nombre de tests fonctionnels sur le nombre de tests unitaires'
            },
            scales: {
                yAxes: [{
                    ticks: {
                        beginAtZero: true
                    }
                }]
            },
            legend: {
                display: false
            },
            tooltips: {
                callbacks: {
                   label: function(tooltipItem) {
                          return tooltipItem.yLabel;
                   }
                }
            }
        }
    };
    const image = await chartJSNodeCanvas.renderToBuffer(configuration, 'image/png');
    // const dataUrl = await chartJSNodeCanvas.renderToDataURL(configuration);
    // const stream = chartJSNodeCanvas.renderToStream(configuration);

    fs.writeFileSync(output, image)
}