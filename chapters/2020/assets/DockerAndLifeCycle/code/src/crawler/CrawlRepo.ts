import {existsSync, mkdirSync, readFileSync} from 'fs-extra';
import {EOL} from 'os';
import {clone, deleteCurRepo} from './Git';
import {filterFile} from '../metrics/filterFiles';
import {AstExplorer} from '../metrics/DockerFileAstExplorer';
import {exists, readFile, writeFileSync} from 'fs';
import {join} from 'path';
import {GlobalMetrics} from '../metrics/model_metrics';
import {MardownExplorer} from '../metrics/MardownExplorer';
import {ShellAnalyser} from '../metrics/ShellAnalyser';

const workspace = './workspace/';
const langdir = './lang/';

export function parseList(file: string): { lang: string, urls: Array<string> } {
    const parts: string[] = readFileSync(file).toString().split(EOL);
    parts.reverse();
    return {
        lang: parts.pop(),
        urls: parts.reverse()
    };
}

export async function crawlLang(lang: string, urls: Array<string>, securityfile: String) {
    // if directory lang don't exists create it
    const securityParts = readFileSync(securityfile).toString().split(EOL);

    if (!existsSync(langdir)) {
        mkdirSync(langdir);
    }

    if (!existsSync(join(langdir, lang))) {
        mkdirSync(join(langdir, lang));
    }

    if (!existsSync(workspace)) {
        mkdirSync(workspace);
    }

    // for each repo if file exists delete, else crawl
    console.log(urls);

    while (urls.length > 0) {
        console.log(urls.length);
        const batch = new Array<Promise<any>>();
        for (let cpt = 0; cpt < 10 && urls.length > 0; ++cpt) {
            const r = urls.pop();
            batch.push(crawlRepo(r, langdir + lang, securityParts));
        }
        await Promise.all(batch);
    }
    /*
    urls.forEach(async r => {

        console.log(batch.length);
        if (batch.length >= max_batch) Promise.all(batch).then((_) => {
            batch = new Array<Promise<any>>();;
            batch.push();
        }).catch(err => {
            console.error(err);
            process.exit(3);
        })
        else {
            batch.push(crawlRepo(r, langdir + lang, securityParts));
        }
    });
    */
}

function existsAsync(path) {
    return new Promise((resolve) => {
        exists(path, (result) => {
            resolve(result);
        });
    });
}

function readAsync(path) {
    return new Promise((resolve, reject) => {
        readFile(path, (err, data) => {
            if (err) reject(err);
            resolve(data);
        })
    })
}

export async function crawlRepo(url: string, baseDir: string, securityParts: string[]) {
    console.log('processing ' + url);
    const parts = url.split('/');
    const name = parts[parts.length - 1];

    if (await existsAsync(join(baseDir, name))) {
        return;
    }

    if (await existsAsync(join(workspace, name))) {
        return;
    }

    // clone repo or pull last version
    await clone(url, workspace + name);


    const globalMetrics = new GlobalMetrics();
    // get all metrics (dockerfile, docker-compose, Readme) -> agregate
    // DockerFile -- Analyse build binaire and build image
    const dockerfilePath = (await filterFile(workspace + name, "DOCKERFILE", true))[0];
    try {
        const dockerfileExplorer = new AstExplorer(dockerfilePath, securityParts, globalMetrics);
        dockerfileExplorer.explore();
    } catch (e) {
        globalMetrics.makeInvalid("invalid dockerfile");
    }

    //Analyse Exec part
    //shellScript
    const shellPaths = await filterFile(workspace + name, ".sh", false);
    let findExecCommand = false;
    if (shellPaths != undefined) {
        const shellAnalyser = new ShellAnalyser(shellPaths, globalMetrics);
        findExecCommand = shellAnalyser.analyse();
        globalMetrics.execSource = "shell";
    }

    //readme
    if (!findExecCommand) {
        const readMePath = await filterFile(workspace + name, "README", false);
        if (readMePath != undefined) {
            const mardownExplorer = new MardownExplorer(readMePath, globalMetrics);
            mardownExplorer.explorer();
            globalMetrics.execSource = "readme";
        }
    }

    // store metrics
    //console.log(globalMetrics);
    writeFileSync(join(baseDir, name), JSON.stringify(globalMetrics.toPrintableJson()));

    // remove clonned repo
    await deleteCurRepo(url);
    console.log("end " + url);
}
