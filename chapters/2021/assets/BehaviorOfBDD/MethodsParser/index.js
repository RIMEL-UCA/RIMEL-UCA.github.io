const fs = require('fs')
const os = require('os');
const flattenArray = require("flatten-array");

//const jsonSchemaGenerator = require('json-schema-generator');
module.exports.testedMethods = (matrix, listOfMethods) => {

    listOfMethods = listOfMethods.map(currentFile => {
        let table;
        if (os.platform() == 'darwin' || os.platform() == 'linux') {
            table = currentFile.file.split("/");
        } else if (os.platform() == 'win32') {
            table = currentFile.file.split("\\");
        }
        return { file: table[table.length - 1], methods: currentFile.methods }
    })
    matrix = matrix.map(eOutput => {
        return eOutput.data.map(fileOutput => {
            let methods = null;
            methods = listOfMethods.filter(currentFile => {
                return currentFile.file == fileOutput.name
            }).map(e => {
                return e.methods.map(e => {

                    for (let i = 0; ; i++) {
                        const methodC = fileOutput.line.filter(currentLineOutput => {
                            return parseInt(currentLineOutput.nr) == parseInt(e.line) + i
                        })
                        if (methodC.length == 0) {
                            continue;
                        }
                        return { ...e, c: methodC[0].c }
                    }
                })
            })
            methods = methods[0];
            return { name: fileOutput.name, methods }
        })
    })
    return matrix;


}

module.exports.testedMethodsSum = (result) => {
    return result.reduce((acc, element, index) => {
        //Si on a pas encore ajouté le fichier
        if (index === 0) {
            return element;
        }

        else {
            // console.log("=>")
            for (let i = 0; i < element.length; i++) {
                // console.log("==>")
                const methodsInCurrentFile = element[i].methods;
                for (let j = 0; j < methodsInCurrentFile.length; j++) {
                    if (acc[i].methods.length === 0) continue;
                    acc[i].methods[j].c += methodsInCurrentFile[j].c
                }
            }
            return acc;
        }
    }, [])

}


module.exports.merge = (units, funcs) => {

    const flatMatrix = (array) => {
        return flattenArray(array.map(e => {
            const name = e.name
            return e.methods.map(e => {
                return { ...e, name: `${name.split('.')[0]}#${e.name}` }
            })
        }))
    }

    const getMethodStatsFromName = (array, name) => {
        for (const method of array) {
            if (method.name === name) return method;
        }
        return null;
    }

    const mergedUnits = flatMatrix(units)
    const mergedFuncs = flatMatrix(funcs)

    return mergedUnits.map(method => {
        return { name: method.name, countUnit: method.c, countFunc: getMethodStatsFromName(mergedFuncs, method.name).c }
    })
}