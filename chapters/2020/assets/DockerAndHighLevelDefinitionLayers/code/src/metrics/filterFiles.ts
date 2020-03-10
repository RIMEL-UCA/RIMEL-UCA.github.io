import {lstatSync, readdir} from 'fs';
import {join} from 'path';


//filter ----
const isFile = fileName => {
    return lstatSync(fileName).isFile();
};

const isFolder = fileName => {
    return lstatSync(fileName).isDirectory();
};

function readDirAsync(path: string): Promise<Array<string>> {
    return new Promise((resolve, reject) => {
        readdir(path, (err, files) => {
            if (err) reject(err);
            resolve(files);
        })
    });
}

export async function filterFile(folder: string, nameFilter: string, recurse: boolean, files?: {} | undefined): Promise<string[]> {

    files = files || {paths: []};

    let nodes = (await readDirAsync(folder)).filter(f => !f.startsWith('.'));

    //get interesting file
    let f_nodes = nodes
        .filter(f => f.toUpperCase().includes(nameFilter.toUpperCase()))
        .map(fileName => join(folder, fileName))
        .filter(isFile);

    if (f_nodes.length > 0) {
        files['paths'] = files['paths'].concat(f_nodes);
    }

    // recurce on folder
    if (recurse) {
        nodes
            .map(fileName => join(folder, fileName))
            .filter(isFolder)
            .forEach(d => filterFile(join(d), nameFilter, recurse, files));
    }

    return files['paths'];
}
