const dirTree = require("directory-tree");
const parseMethod = require("./parseMethod");

const flattenArray = require("flatten-array");
const fs = require('fs');

const getFileList = (res) => {
    return res.children.map(el => {
        if (el.type == 'directory' && el.children.length > 0) {
            return getFileList(el);
        } else if (el.type == 'directory') {
            return [];
        }
        return el;
    })
}

const lineNumberByIndex = (index, string) => {
    // RegExp
    var line = 0,
        match,
        re = /(^)[\S\s]/gm;
    while (match = re.exec(string)) {
        if (match.index > index)
            break;
        line++;
    }
    return line;
}

module.exports.listOfMethod = (path) => {
    const filteredTree = dirTree(path + '/src/main/java', { extensions: /\.java/ });
    const files = flattenArray(getFileList(filteredTree)).map(e => e.path)
    const result = flattenArray(files.map(filename => {
        const data = fs.readFileSync(filename, 'utf8');
        return { file: filename, methods: parseMethod(data) };
    }));
    return result;
}


module.exports.listOfTest = (path, type) => {
    const filteredTree = dirTree(type === 'unit' ? path + '/src/test/java' : path + '/src/test/resources', { extensions: type === 'unit' ? /\.java/ : /\.feature/ });
    const files = flattenArray(getFileList(filteredTree)).map(e => e.path)
    const tests = flattenArray(files.map(filename => {
        const data = fs.readFileSync(filename, 'utf8');
        let result;
        if (type === 'unit') {
            let packages = data.match(/package.*;/g);
            let functions = data.match(/@Test.*\n.*void.*\(/g);
            let className = data.match(/[ \n]class.*{/g)
            if (functions && className) {
                let package = packages !== null ? packages[0].split(' ')[1].replace(';', '') : '';
                functions = functions.map(e => e.split('void ')[1].replace('(', ''))
                className = className[0].split('class ')[1].replace('{', '').trim();
                result = functions.map(e => package + '.' + className + '.' + e)
            }
        }
        else if (type == 'complexity') {
            let arrayOfScenarios = data.split("\n").reduce((acc, curr, index) => {
                curr = curr.trim()
                if (curr.length === 0 || ((!curr.toLowerCase().startsWith('scenario')) && acc.length == 0)) return acc;
                else if (curr.toLowerCase().startsWith('scenario')) return [...acc, [filename + ":" + index + 1]]
                else if (acc.length == 1) return [[...acc[0], curr]]
                else return [...[...acc.slice(0, -1)], [...acc.slice(-1)[0], curr]]
            }, []);
            result = {}
            arrayOfScenarios.forEach(e => result[e[0]] = e.length - 1)

        }
        else {
            let pattern = /Scenario.*\n/g;
            result = [];
            while (match = pattern.exec(data)) {
                result.push(filename + ':' + lineNumberByIndex(match.index, data));
            }
        }
        return result
    }).filter(e => e));
    return tests;
}
