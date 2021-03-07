const fs = require('fs');
const parser = require('xml2json');

module.exports.parse = (pathProject, path, filename) => {
    return new Promise((resolve, reject) => {
        const data = fs.readFileSync(`${pathProject}/${path}/${filename}`);
        const json = parser.toJson(data);
        resolve(json)
    })
}