const { exit } = require('process');
const util = require('util');
const exec = util.promisify(require('child_process').exec);

module.exports.runUnit = async (path, package, className, testName) => {
    const { stdout, stderr } = await exec(`cd ${path} && mvn -Dtest=${package}.${className}#${testName} clean test`);
    return { stdout, stderr }
    // -Dmaven.clean.failOnError=false
}

module.exports.runFonctionnal = async (path, scenarios) => {
    const { stdout, stderr } = await exec(`cd ${path} && mvn -Dcucumber.options="${scenarios}" clean test`);
    return { stdout, stderr }
    //-Dmaven.clean.failOnError=false
}