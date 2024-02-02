const fs = require('fs');
const path = require('path');

const args = process.argv.slice(2);
console.log("Arguments : ", args);

function printUsage() {
    console.log("Usage is : npm start -- {starting_folder}");
}

function isValidPath(filePath) {
    try {
        const stats = fs.statSync(filePath);
        if (!stats.isDirectory())
            console.log("Path is not a folder");
        return stats.isDirectory();
    } catch (error) {
        console.log("Folder doesn't exist");
        return false;
    }
}

function getAllFolders(directoryPath) {
    return fs.readdirSync(directoryPath)
        .filter(file => fs.statSync(path.join(directoryPath, file)).isDirectory());
}

function findPythonFiles(directory, files = []) {
    const contents = fs.readdirSync(directory);

    contents.forEach(file => {
        const filePath = path.join(directory, file);
        const isDirectory = fs.statSync(filePath).isDirectory();

        if (isDirectory) {
            findPythonFiles(filePath, files);
        } else if (file.endsWith('.py') || file.endsWith('.ipynb')) {
            files.push(filePath);
        }
    });

    return files;
}

function fileContainsString(filePath) {
    const fileContent = fs.readFileSync(filePath, 'utf-8');
    for (let string of stringsToFind) {
        if(fileContent.includes(string))
            return true;
    }
    return false;
}

if (args.length !== 1) {
    console.log("Invalid number of arguments");
    printUsage();
    return;
}

if (!isValidPath(args[0])) {
    printUsage();
    return;
}

const startDirectory = args[0];
const stringsToFind = ["log_param", "autolog", "log_metric"];

const folders = getAllFolders(startDirectory);
console.log('All folders in the directory:', folders);

let sum = 0;

outerloop: for (let folder of folders) {
    console.log("Going through", folder);
    let pythonFiles = [];
    findPythonFiles(startDirectory + "/" + folder, pythonFiles);
    for (let pythonFile of pythonFiles) {
        console.log("Searching in", pythonFile, "...");
        const containsParamLogging = fileContainsString(pythonFile);
        if (containsParamLogging) {
            console.log(folder, "contains param logging in", pythonFile, "!");
            sum++;
            continue outerloop;
        }
    }
    console.log(folder, "does not contain param logging !");
}

const usagePercent = (sum / folders.length) * 100;

console.log();
console.log(usagePercent, "% of projects include param logging out of", folders.length, "projects !");

const csvFile = "Usage percent;Number of projects\n" + usagePercent.toString().replace(".", ",") + ";" + folders.length;

try {
    fs.writeFileSync("results.csv", csvFile);
    console.log("Usage percent file written successfully");
} catch (err) {
    console.error(err);
}