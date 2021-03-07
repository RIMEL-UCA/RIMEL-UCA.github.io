const arg = require('arg');
const { exit } = require('process');
const app = require('coverage-parser')
const { listOfTest } = require('directory-parser')
const ProgressBar = require('progress');
const fs = require('fs');
const { listOfMethod } = require('directory-parser');
const { testedMethods, testedMethodsSum } = require('methods-parser');
const { modifyExtension, resetExtension } = require("extension-modifier");
const { merge } = require('../MethodsParser');


const args = arg({
    // Types
    '--help': Boolean,
    '--path': String,
    '--type': String,
    '--method': Boolean,
    '--all': Boolean,
    '--output': String,

    // Aliases
    '-h': '--help',
    '-p': '--path',
    '-t': '--type',
    '-m': '--method',
    '-a': '--all',
    '-o': '--output'
});

if (args["--help"]) {
    console.log("Usage: test-analyser [options]")
    console.log("")
    console.log("Options:")
    console.log("   --help          Print this message")
    console.log("   --path          Path of the project")
    console.log("   --method        If true output gives the methods related to a test else the lines related to test, default set to false  ")
    console.log("   --all           Run the entire routine")
    console.log("   --type          Test type to detect :  <unit|fonctionnal|complexity>")
    exit(0);
}

const pathProject = args['--path'];
const output = args['--output'];
const type = args['--type'];
const method = args["--method"];
const all = args["--all"];

const executeEntireRoutine = async (path) => {

    if (fs.existsSync(output)) {
        fs.rmdirSync(output, { recursive: true });
    }
    fs.mkdirSync(output, { recursive: true });

    // Part 1
    const principalBar = new ProgressBar('Generating statistics : [:bar] :percent ', {
        total: 11,
        incomplete: ' ',
        width: 50,
    });
    principalBar.tick()

    const testProjectUnit = listOfTest(path, 'unit');
    let bar = new ProgressBar('Run all units tests : [:bar] :percent :etas', {
        total: testProjectUnit.length, complete: '=',
        incomplete: ' ',
        width: testProjectUnit.length < 50 ? 50 : testProjectUnit.length,
    });
    const resultUnit = await app.runAllTests(path, testProjectUnit, 'unit', bar)
    fs.writeFileSync(output + "/matrix-tu.json", JSON.stringify(resultUnit));
    principalBar.tick()

    modifyExtension(path)
    principalBar.tick()

    const testProjectFonc = listOfTest(path, 'fonctionnal');
    bar = new ProgressBar('Run all functional tests : [:bar] :percent :etas', {
        total: testProjectFonc.length, complete: '=',
        incomplete: ' ',
        width: testProjectFonc.length < 50 ? 50 : testProjectFonc.length,
    });
    const resultFonc = await app.runAllTests(path, testProjectFonc, 'fonctionnal', bar)
    fs.writeFileSync(output + "/matrix-tf.json", JSON.stringify(resultFonc));
    principalBar.tick()

    // // Reset 
    resetExtension(path)
    principalBar.tick()

    // // Part 2

    // Methods list
    const methods = listOfMethod(path)
    fs.writeFileSync(output + "/method-list.json​", JSON.stringify(methods));
    principalBar.tick()
    // Methods tested
    const methodAssociatedUnit = testedMethods(resultUnit, methods);
    fs.writeFileSync(output + "/matrix-methods-tu.json", JSON.stringify(methodAssociatedUnit));
    principalBar.tick()

    const methodAssociatedFonc = testedMethods(resultFonc, methods);
    fs.writeFileSync(output + "/matrix-methods-tf.json", JSON.stringify(methodAssociatedFonc));
    principalBar.tick()

    // Methods tested (sum)
    const sumTestMethodsTU = testedMethodsSum(methodAssociatedUnit);
    fs.writeFileSync(output + "/matrix-tests-sum-tu.json", JSON.stringify(sumTestMethodsTU));
    principalBar.tick()

    const sumTestMethodsTF = testedMethodsSum(methodAssociatedFonc);
    fs.writeFileSync(output + "/matrix-tests-sum-tf.json", JSON.stringify(sumTestMethodsTF));
    principalBar.tick()

    const mergedMatrixMethods = merge(sumTestMethodsTU, sumTestMethodsTF)
    fs.writeFileSync(output + "/matrix-tests-sum-merged.json", JSON.stringify(mergedMatrixMethods));
    principalBar.tick()

    // const sumMatrices = sumUnitFonc(methodAssociatedUnit, methodAssociatedFonc);

}


if (pathProject === undefined) {
    console.log('Undefined path... Abort')
    exit(-1)
}

if (all) {
    executeEntireRoutine(pathProject);
    return;
}

if (output === undefined) {
    console.log('Undefined output file... Abort')
    exit(-1)
}

if (type === undefined || (type !== 'unit' && type !== 'fonctionnal' && type !== 'complexity')) {
    console.log('Undefined or bad type ... Abort')
    console.log('Hint : use --type=unit or --type=fonctionnal or type=complexity');
    exit(-1)
}

if (method) {
    console.log(JSON.stringify(listOfMethod(pathProject)));
    exit(0)
}


if (type == 'complexity') {
    const result = listOfTest(pathProject, type);
    console.log(JSON.stringify(result));
    exit(0);
}

const testProject = listOfTest(pathProject, type);
console.log(testProject);

(async () => {
    var bar = new ProgressBar('Loading : [:bar] :percent :etas', { total: testProject.length });
    const result = await app.runAllTests(pathProject, testProject, type, bar)
    fs.writeFileSync(output, JSON.stringify(result));


    //result.filter(e -> e.)
    /*map(current => {
        listOfMethod(pathProject);
    })*/

})()

