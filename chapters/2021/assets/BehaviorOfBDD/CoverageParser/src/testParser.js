const command = require('./command');
const parserXML = require('./parseXML')
// const GenerateSchema = require('generate-schema')
const parserJSON = require('./parseJSON')

const runOneTest = async (pathProject, testProject, type) => {
    if (type === 'unit') {
        const arr = testProject.split('.');

        const test = arr.pop();
        const className = arr.pop();
        const package = arr.join('.');

        //TODO add verbose
        const syncDataRUN = await command.runUnit(pathProject, package, className, test)

    }
    else {
        const syncDataRUN = await command.runFonctionnal(pathProject, testProject);

    }
    // const schema = GenerateSchema.json('Result', parsed)

    return parserJSON.parse(await parserXML.parse(pathProject, 'target/site/jacoco', 'jacoco.xml'));
}

module.exports.runAllTests = async (pathProject, testsProject, type, bar) => {
    const results = []
    for (let i = 0; i < testsProject.length; i++) {
        bar.tick();
        const name = testsProject[i]
        try {
            const data = await runOneTest(pathProject, name, type)
            results.push({
                name,
                data
            })
        } catch (e) {
            console.error(e);
        }

    }
    return results;
}