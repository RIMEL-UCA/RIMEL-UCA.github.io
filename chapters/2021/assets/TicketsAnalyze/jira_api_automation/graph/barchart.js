const fs = require('fs');
const output = require('d3node-output');
const d3 = require('d3-node')().d3;
const d3nBar = require('../util/barchart.js');
const parse =(name)=> {
    fs.readdir('./data/' + name.toString(), function (err, files) {
        if (err) {
            throw err;
        }
        files.forEach(function (file) {
            const csvString = fs.readFileSync('./data/' + name + '/' + file).toString();
            const data = d3.csvParse(csvString);
            // create output files
            console.log('../result/' + name + '/' + file);
            let file_name = file.split('.')[0];
            output('./result/' + name + '/' + file_name, d3nBar({data: data}));

        });
    });
};

module.exports = parse;


