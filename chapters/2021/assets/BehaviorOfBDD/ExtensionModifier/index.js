const dirTree = require("directory-tree");
const flattenArray = require("flatten-array");
const fs = require('fs');


const getFileList = (res) => {
    return res.children.map(el => {
        if (el.type == 'directory') return getFileList(el);
        return el;
    })
}

module.exports.modifyExtension = (path) => {
    const filteredTree = dirTree(path + '/src/test/java', { extensions: /\.java/ });
    const files = flattenArray(getFileList(filteredTree)).map(e => e.path)
    const tests = flattenArray(files.map(filename => {
        const data = fs.readFileSync(filename, 'utf8');

        let functions = data.match(/@Test.*\n.*void.*\(/g);
        if (functions) {
            return filename;
        }
        else { return null }
    })).filter(e => e != null);

    for (let i = 0; i < tests.length; i++) {
        fs.renameSync(tests[i], tests[i].replace(".java", ".ignored"))
    }

}


module.exports.resetExtension = (path) => {
    const filteredTree = dirTree(path + '/src/test/java', { extensions: /\.ignored/ });
    const files = flattenArray(getFileList(filteredTree)).map(e => e.path)
    for (let i = 0; i < files.length; i++) {
        fs.renameSync(files[i], files[i].replace(".ignored", ".java"))
    }

}