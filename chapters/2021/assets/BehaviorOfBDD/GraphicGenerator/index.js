const fs = require('fs');
const { generate, generateRatio } = require('./generator');
const arg = require('arg');
const dirTree = require("directory-tree");
const flattenArray = require("flatten-array");

const getFileList = (res) => {
    return res.children.map(el => {
        if (el.type == 'directory') return getFileList(el);
        return el;
    })
}

const args = arg({
    // Types
    '--help': Boolean,
    '--output': String,
    '--dir': Boolean,

    // Aliases
    '-h': '--help',
    '-o': '--output',
    '-d': '--dir'
});

if (args["--help"]) {
    console.log("Usage: test-analyser [options]")
    console.log("")
    console.log("Options:")
    console.log("   --help          Print this message")
    console.log("   --output          Outputfile")
    console.log("   --dir          Input is directory")
    exit(0);
}

const output = args['--output'];
const input = args['_'];
const dir = args['--dir'];

if(dir === false) {
    const json = JSON.parse(fs.readFileSync(input[0]))
    generate(json, output)
}

else {
    const inputsFile = flattenArray(getFileList(dirTree(input[0]))).filter(e => e.name === 'matrix-tests-sum-merged.json')
    const outputs = inputsFile.map(e => e.path.split('/').slice(0, -1).join('/') + '/' + output)
    const outputsRatio = inputsFile.map(e => e.path.split('/').slice(0, -1).join('/') + '/' + output.split('.')[0] + '_ratio.' + output.split('.')[1])
    const outputsDerivate = inputsFile.map(e => e.path.split('/').slice(0, -1).join('/') + '/' + output.split('.')[0] + '_derivate.' + output.split('.')[1])
    const inputs = inputsFile.map(e => e.path)
    for(let i=0; i<outputs.length; i++) {
        const json = JSON.parse(fs.readFileSync(inputs[i]))
        const jsonRatio = json.map(e => {
            if(e.countUnit === 0 || e.countFunc === 0) return undefined;
            return {name: e.name, ratio: e.countFunc/e.countUnit}
        }).filter(e => e)
        const jsonDerivate = json.map((e, index) => {
            if(index === 0 || index === json.length-1) return undefined;
            return {name: e.name, countFunc: (e.countFunc - json[index-1].countFunc), countUnit: (e.countUnit - json[index-1].countUnit)}
        }).filter(e => e)
        generate(json, outputs[i], 'Nombre de tests couvrant la méthode')
        generateRatio(jsonRatio, outputsRatio[i])
        generate(jsonDerivate, outputsDerivate[i], 'Dérivée du nombre de tests couvrant la méthode')
    }
}