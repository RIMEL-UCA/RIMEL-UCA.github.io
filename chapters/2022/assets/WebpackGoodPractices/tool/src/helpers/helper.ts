import * as path from "path";
import { PathLike } from "fs";
import fs from "fs/promises";
import { Project } from "../database/project.db";
import { Octokit } from "@octokit/rest";
import pLimit from "p-limit";
import { CategorizationEnum } from "../database/categorize";

export const ForbiddenDirectory = ['node_modules','dist'];

export const ROOT_PATH = path.resolve(__dirname,'../..');
export const REPOSITORIES_PATH = path.resolve(ROOT_PATH,'repositories');
const octokit = new Octokit({
    auth: process.env.GITHUB_TOKEN,
});

const ONE_YEAR_IN_MS: number = 31536000000;

const MINIFY_JS_PACKAGE : string[] = ["express-uglify-middleware","babel-minify", "google-closure-compiler", "terser", "uglify-js", "uglify-es", "yuicompressor", "yui"];

const MINIFY_CSS_PACKAGE: string[] = ["clean-css", "crass", "cssnano", "csso", "@node-minify/sqwish", "sqwish", "yuicompressor", "yui"];

/**
 * Get all files from directory
 * @param dir - directory path
 */
export async function getFilesFromDirectory(dir: PathLike): Promise<string[]> {
    const subDirs = await fs.readdir(dir);
    const files = await Promise.all(subDirs.map(async (subdir) => {
        const res = path.resolve(dir.toLocaleString(), subdir);
        const stat = await fs.lstat(res);
        return stat.isDirectory() && !ForbiddenDirectory.includes(path.basename(res)) && !stat.isSymbolicLink() ? getFilesFromDirectory(res) : [res];
    }));
    if( files.length === 0 ){
        return [];
    }
    return files.reduce((a, f) => a.concat(f, []));
}

/**
 * Get local repository name for a project 
 * Ex: angular-first_3195709375
 */
export function resolveLocalRepositoryName(project: Project): string {
    return `${project.name}_${project.id}`;
}


/**
 * Find files in a directory matching a regex
 * @param name - File name
 * @param dir - File directory
 */
async function findFile(name: string | RegExp, dir: PathLike): Promise<string[]> {
    const filePaths = await getFilesFromDirectory(dir);
    const found = filePaths.filter((file) => path.basename(file).match(name));
    return found;
}

/**
 * Find all files in a project matching a regex
 * Alias of findFile but with project as parameter
 */
export function findFileInProject(name: string | RegExp, project: Project): Promise<string[]> {
    const projectPath = resolveLocalPath(project); 
    return findFile(name, projectPath);
}

/**
 * Find all package.json files in a directory
 * @param repoPath
 */
export async function findPackageJSONPath(repoPath: PathLike): Promise<PathLike[]> {
    const files = await findFile("package.json", repoPath);
    return files;
}

/**
 * Parse package.json and extract dependencies, devDependencies, peerDependencies, optionalDependencies
 * @param packagePath
 */
export async function parsePackageJSON(
    packagePath: PathLike,
): Promise<{
    dependencies: Record<string, string>;
    devDependencies: Record<string, string>;
    peerDependencies: Record<string, string>;
    optionalDependencies: Record<string, string>;
}> {
    const rowData = await fs.readFile(packagePath, 'utf8');
    let packageJSON: Record<string, any> = {};
    try {
        packageJSON = JSON.parse(rowData);
    } catch (e) {
        // console.log(`❌ Error parsing package.json: ${e}`);
    }
    const dependencies = packageJSON.dependencies;
    const devDependencies = packageJSON.devDependencies;
    const peerDependencies = packageJSON.peerDependencies;
    const optionalDependencies = packageJSON.optionalDependencies;
    return {
        dependencies: dependencies,
        devDependencies: devDependencies,
        peerDependencies: peerDependencies,
        optionalDependencies: optionalDependencies,
    };
}

/**
 * Return if a repository as a dependency
 * @param packageJSONDependencies
 * @param found
 */
export function hasDependency(packageJSONDependencies: Record<string,string>, found: string): boolean {
    return packageJSONDependencies.hasOwnProperty(found)
}

/**
 * Resolve path to a project with project entity
 */
export function resolveLocalPath(project: Project): PathLike {
    return path.resolve(REPOSITORIES_PATH, resolveLocalRepositoryName(project));
}

/**
 * Get repository dependencies
 * @param projectName
 * @param repoPath
 */
export async function getDependencies(project: Project): Promise<Record<string, string>> {
    const dependencies = await getStructuredDependencies(project);
    return { ...dependencies.dependencies, ...dependencies.devDependencies };
}

/**
 * Get all dependencies from a project
 * (works with several files package.json)
 */
export async function getStructuredDependenciesFromPath(path: PathLike): Promise<{
    dependencies: Record<string, string>;
    devDependencies: Record<string, string>;
    peerDependencies: Record<string, string>;
    optionalDependencies: Record<string, string>;
}>{
    const packageJSONPaths = await findPackageJSONPath(path);
    if (!packageJSONPaths) {
        throw new Error(' package.json not found');
    }
    const packageJSONParseTasks = packageJSONPaths.map((packageJSONPath) =>
        parsePackageJSON(packageJSONPath).catch(() => {
            return {
                dependencies: {},
                devDependencies: {},
                peerDependencies: {},
                optionalDependencies: {},
            };
        }),
    );
    const packageJSONs = await Promise.all(packageJSONParseTasks);
    
    const concatPackageJSON = {
        dependencies: packageJSONs.reduce((a, p) => ({ ...a, ...p.dependencies }), {}),
        devDependencies: packageJSONs.reduce((a, p) => ({ ...a, ...p.devDependencies }), {}),
        peerDependencies: packageJSONs.reduce((a, p) => ({ ...a, ...p.peerDependencies }), {}),
        optionalDependencies: packageJSONs.reduce((a, p) => ({ ...a, ...p.optionalDependencies }), {}),
    };
    return concatPackageJSON;
}

/**
 * Get all dependencies from a project
 * (works with several files package.json)
 */
export async function getStructuredDependencies(
    project: Project,
): Promise<{
    dependencies: Record<string, string>;
    devDependencies: Record<string, string>;
    peerDependencies: Record<string, string>;
    optionalDependencies: Record<string, string>;
}> {
    const repoPath = resolveLocalPath(project);
    return await getStructuredDependenciesFromPath(repoPath);
}

/**
 * Remove duplicates from array of objects
 */
export function removeDuplicates(array: any[]) {
    return Array.from(new Set(array));
}

/**
 * Return number of contributors for a repository
 * @param repo
 */
 export async function getNbContributors(repo: any): Promise<number> {
    const res = await octokit.rest.repos.listContributors({
        owner: repo.owner.login,
        repo: repo.name,
        per_page: 100,
    }).catch(async (error: any) => {
        let delay: number;
        switch (error.status) {
            case 404:
                return -1;
            case 500:
                delay = 30;
                break;
            case 403:
                delay = error?.response?.headers['x-ratelimit-reset'] - (Date.now()/1000);
                break;
            default:
                console.log(error);
                delay = 30;
                break;
        }
        process.stdout.write(`\r⏳ Delay for ${delay.toFixed(0)} seconds`);
        while(delay > 0){
            delay--;
            // eslint-disable-next-line no-await-in-loop
            await new Promise(resolve => setTimeout(resolve, 1000));
            process.stdout.write(`\r⏳ Delay for ${delay.toFixed(0)} seconds`);
        }
        return getNbContributors(repo);
    });
    if (typeof res != "number") {
        return res.data.length;
    }
    return 1;
}

/**
 * Return number of row of code for a repository
 * Matching only files with extension .js, .ts, .jsx, .tsx
 */
export async function countRowOfCode(projectName: string, projectId: string): Promise<number>{
    const repoPath = path.resolve(REPOSITORIES_PATH, `${projectName}_${projectId}`);
    const files = await getFilesFromDirectory(repoPath);
    const filesPath = files
    .filter((file) => file.match(/\.(js|ts|tsx|jsx)$/))
    .map((file) => path.resolve(repoPath, file));
    let nbLignes = 0;
    try {
        const filesLines = await Promise.all(
            filesPath.map((file) =>
                fs
                    .readFile(file, {
                        encoding: 'utf8',
                        flag: 'r',
                    })
                    .catch(() => '')
                    .then((file) => file.split('\n').length),
            ),
        );
        nbLignes = filesLines.reduce((a, b) => a + b, 0);
    } catch (_e) {
        nbLignes = -1;
    }
    return nbLignes;

    
}

export async function isMinifyJs(projectName: string, projectId: string): Promise<boolean>{
    const repoPath = path.resolve(REPOSITORIES_PATH, `${projectName}_${projectId}`);

    const structuredDependencies = await getStructuredDependenciesFromPath(repoPath).catch();
    if(!structuredDependencies){
        return false;
    }

    const dependencies = {...structuredDependencies.dependencies, ...structuredDependencies.devDependencies};

    const projectCategory = foundCategory(dependencies);
    if(projectCategory === "angular" || projectCategory === "vue"){
        return true;
    }

    const dependenciesKeys = Object.keys(dependencies);
    MINIFY_JS_PACKAGE.forEach((minifyDependency:any) => {
        if(dependenciesKeys.includes(minifyDependency)){
            console.log("trouvé !!", minifyDependency)
            return true;
        }
    });

    return false;
}

export async function isMinifyCss(projectName: string, projectId: string): Promise<boolean>{
    const repoPath = path.resolve(REPOSITORIES_PATH, `${projectName}_${projectId}`);

    const structuredDependencies = await getStructuredDependenciesFromPath(repoPath).catch();
    if(!structuredDependencies){
        return false;
    }

    const dependencies = {...structuredDependencies.dependencies, ...structuredDependencies.devDependencies};

    const projectCategory = foundCategory(dependencies);
    if(projectCategory === "angular" || projectCategory === "vue"){
        return true;
    }

    const dependenciesKeys = Object.keys(dependencies);
    MINIFY_CSS_PACKAGE.forEach((minifyDependency:any) => {
        if(dependenciesKeys.includes(minifyDependency)){
            console.log("trouvé !!", minifyDependency)
            return true;
        }
    });

    return false;
}

/**
 * Define if a repo is still maintened
 * 
 * @param repo 
 * @returns 
 */
export async function isMaintened(repo: any): Promise<boolean>{

    const releases : any = await getRepoRelease(repo);
    if(releases.data.length < 3 || new Date(Date.now()-ONE_YEAR_IN_MS)>new Date(Date.parse(releases.data[0].created_at))) {
        const commits = await getAllCommits(repo);
        return new Date(Date.now()-ONE_YEAR_IN_MS) < new Date(Date.parse(commits.data[0].commit.author.date));
    }
    return true;
}

export async function getAllCommits(repo:any): Promise<any>{
    return await octokit.rest.repos.listCommits({
        owner: repo.owner.login,
        repo: repo.name,
        per_page: 100,
    }).catch(async (error: any) => {
        let delay: number;
        switch (error.status) {
            case 404:
                return -1;
            case 500:
                delay = 30;
                break;
            case 403:
                delay = error?.response?.headers['x-ratelimit-reset'] - (Date.now()/1000);
                break;
            default:
                console.log(error);
                delay = 30;
                break;
        }
        process.stdout.write(`\r⏳ Delay for ${delay.toFixed(0)} seconds`);
        while(delay > 0){
            delay--;
            // eslint-disable-next-line no-await-in-loop
            await new Promise(resolve => setTimeout(resolve, 1000));
            process.stdout.write(`\r⏳ Delay for ${delay.toFixed(0)} seconds`);
        }
        return getAllCommits(repo);
    });
}

export async function getTagDetail(repo:any, tag: any): Promise<any>{
    // Doesnt work !!!
   return await octokit.rest.git.getTag({
        owner: repo.owner.login,
        repo: repo.name,
        tag_sha: tag.commit.sha,
      }).catch(async (error: any) => {
        let delay: number;
        switch (error.status) {
            case 404:
                console.log("404")
                return -1;
            case 500:
                delay = 30;
                break;
            case 403:
                delay = error?.response?.headers['x-ratelimit-reset'] - (Date.now()/1000);
                break;
            default:
                console.log(error);
                delay = 30;
                break;
        }
        process.stdout.write(`\r⏳ Delay for ${delay.toFixed(0)} seconds`);
        while(delay > 0){
            delay--;
            // eslint-disable-next-line no-await-in-loop
            await new Promise(resolve => setTimeout(resolve, 1000));
            process.stdout.write(`\r⏳ Delay for ${delay.toFixed(0)} seconds`);
        }
        return getTagDetail(repo, tag);
    });;
}

export function filterTags(tags:any): any[]
{
    const regex = new RegExp('^v', 'i')
    let newTags = JSON.parse(JSON.stringify(tags));
    newTags.data = [];

    tags.data.forEach((tag:any) => {
        if(regex.test(tag.name)){
            newTags.data.push(tag);
        }
    });
    return newTags;
}

export async function getRepoRelease(repo:any): Promise<any> {
    return await octokit.rest.repos.listReleases({
        owner: repo.owner.login,
        repo: repo.name,
        per_page: 100,
    }).catch(async (error: any) => {
        let delay: number;
        switch (error.status) {
            case 404:
                return -1;
            case 500:
                delay = 30;
                break;
            case 403:
                delay = error?.response?.headers['x-ratelimit-reset'] - (Date.now()/1000);
                break;
            default:
                console.log(error);
                delay = 30;
                break;
        }
        process.stdout.write(`\r⏳ Delay for ${delay.toFixed(0)} seconds`);
        while(delay > 0){
            delay--;
            // eslint-disable-next-line no-await-in-loop
            await new Promise(resolve => setTimeout(resolve, 1000));
            process.stdout.write(`\r⏳ Delay for ${delay.toFixed(0)} seconds`);
        }
        return getRepoRelease(repo);
    });


}

export async function getRepoTags(repo:any): Promise<any> {
    return await octokit.rest.repos.listTags({
        owner: repo.owner.login,
        repo: repo.name,
        per_page: 100,
    }).catch(async (error: any) => {
        let delay: number;
        switch (error.status) {
            case 404:
                return -1;
            case 500:
                delay = 30;
                break;
            case 403:
                delay = error?.response?.headers['x-ratelimit-reset'] - (Date.now()/1000);
                break;
            default:
                console.log(error);
                delay = 30;
                break;
        }
        process.stdout.write(`\r⏳ Delay for ${delay.toFixed(0)} seconds`);
        while(delay > 0){
            delay--;
            // eslint-disable-next-line no-await-in-loop
            await new Promise(resolve => setTimeout(resolve, 1000));
            process.stdout.write(`\r⏳ Delay for ${delay.toFixed(0)} seconds`);
        }
        return getRepoTags(repo);
    });
}

/**
 * Found repository category
 * @param dependencies
 */
 export function foundCategory(dependencies: Record<string, string>): CategorizationEnum {
    switch (true) {
        case hasDependency(dependencies, '@angular/core'): {
            return "angular";
        }
        case hasDependency(dependencies, 'vue'): {
            return "vue";
        }
        case hasDependency(dependencies, '@nestjs/core'): {
            return "nestjs";
        }
        case hasDependency(dependencies, 'next'): {
            return "next";
        }
        case hasDependency(dependencies, 'react'): {
            return "react";
        }
        case hasDependency(dependencies, 'express'): {
            return "express";
        }
        case hasDependency(dependencies, 'webpack'): {
            return "native";
        }
    }
    return "other";
}
