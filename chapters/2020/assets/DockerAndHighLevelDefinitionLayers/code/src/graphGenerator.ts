const plotly = require('plotly')('theosM', 'dTV9XLpCiA55T7Yz6QUp');
//https://plot.ly/nodejs/line-and-scatter/

const trace1 = {
    x: [52698, 43117],
    y: [53, 31],
    mode: 'markers',
    name: 'Average',
    text: ['United States', 'Canada'],
    marker: {
        color: 'rgb(164, 194, 244)',
        size: 12,
        line: {
            color: 'white',
            width: 0.5
        }
    },
    type: 'scatter'
};

const data = [trace1];
const layout = {
    title: 'Quarter 1 Growth',
    xaxis: {
        title: 'GDP per Capita',
        showgrid: false,
        zeroline: false
    },
    yaxis: {
        title: 'Percent',
        showline: false
    }
};

const graphOptions = {layout: layout, filename: 'line-style', fileopt: 'overwrite'};

plotly.plot(data, graphOptions, function (err, msg) {
    console.log(msg);
});
