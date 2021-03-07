const fs = require('fs');
const output = require('d3node-output');
const d3 = require('d3-node')().d3;
const d3nPie = require('../util/index.js');

const parse =( name )=>{
    console.log(name);
    fs.readdir('./data/'+name.toString(), function (err, files) {
        if (err) {
            throw err;
        }
        files.forEach(function (file) {
            const opts = {width: 720, height: 750, fontsize : 12};
            const csvString = fs.readFileSync('./data/'+name+'/'+file).toString();
            const data = d3.csvParse(csvString);
            // create output files
            console.log('../result/'+name+'/'+file);
            let file_name = file.split('.')[0];
            output('./result/'+name+'/'+file_name, d3nPie({ data: data }),opts);
        });
    });

};

module.exports = parse;

