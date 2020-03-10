const fs = require('fs');
const request = require('request');
const path = require('path');
const chalk = require('chalk');

const sleep = require('./utils').sleep;
const language = require('./utils').language;
const clientSecret = require('./clientSecret');
const save = require('./utils').save;

// --------------------------------------------------------------------------- //

let projects;

const selected_repos = path.join(__dirname, '..', 'generated', language, `${language}-selected-tmp.txt`);
const with_docker_compose = path.join(__dirname, '..', 'generated', language, `${language}-compose.txt`);
const no_docker_file = path.join(__dirname, '..', 'generated', language, `${language}-no-docker.txt`);
const request_failed = path.join(__dirname, '..', 'generated', language, `${language}-request-failed.txt`);

const notFound = '404: Not Found';

main();

async function main() {

    fs.writeFileSync(selected_repos, `${language}\n`);
    fs.writeFileSync(with_docker_compose, '');
    fs.writeFileSync(no_docker_file, '');
    fs.writeFileSync(request_failed, '');

    let url = `https://api.github.com/search/repositories?q=docker+language:${language}&sort=stars&order=desc&per_page=100`;

    for (let i = 1; i < 11; i++) {
        await processPage(i, `${url}&page=${i}&client_id=AlexBolot&client_secret=${clientSecret.secret}`);
        await sleep(5000);
        console.log(chalk.green("-- waited 5 sec"));
    }
}

async function processPage(index, url) {
    console.log(chalk.green(`-- started for page ${index}`));
    await request(url, { json: true, headers: { 'User-Agent': 'AlexBolot' } }, async (err, res, body) => {

        if (body === undefined || body.items === undefined) {
            console.log(`body or items undefined for page ${index} - ${url}`);
            return;
        }

        projects = body.items;

        for (let projectId = 0; projectId < projects.length - 1; projectId++) {

            if (await getDockerCompose(projectId)) {
                continue;
            }

            if (await getRootDockerfile(projectId)) {
                continue;
            }

            await getDeepDockerfile(projectId);
        }
    });
}

function getFullName(index) {
    return projects[index]['full_name'];
}

// --------------------------------------- //
// --------------------------------------- //
// --------------------------------------- //

async function getDockerCompose(projectId) {
    const projectName = getFullName(projectId);
    const url = `https://raw.githubusercontent.com/${projectName}/master/docker-compose.yml`;

    await request(url, { headers: { 'User-Agent': 'AlexBolot' } }, (err, res, body) => {
        if (body === undefined) {
            saveError(projectName, 'docker-compose.yml');
            return true;
        }

        // If has a docker-compose -> exclude project
        if (body.trim() !== notFound) {
            saveDockerCompose(projectName);
            return true;
        }
    });
}

async function getRootDockerfile(projectId) {
    const projectName = getFullName(projectId);
    const url = `https://raw.githubusercontent.com/${projectName}/master/Dockerfile`;

    await request(url, { headers: { 'User-Agent': 'AlexBolot' } }, (err, res, body) => {
        if (body === undefined) {
            saveError(projectName, './Dockerfile');
            return true;
        }

        // If has a ./Dockerfile -> save and move on
        if (body.trim() !== notFound) {
            saveSuccess(projectName);
            return true;
        }

        console.log(chalk.yellow(`${projectName} has no ./Dockerfile`));
    });
}

async function getDeepDockerfile(projectId) {
    const projectName = getFullName(projectId);
    const url = `https://raw.githubusercontent.com/${projectName}/master/docker/Dockerfile`;

    await request.get(url, { headers: { 'User-Agent': 'AlexBolot' } }, (err, res, body) => {
        if (body === undefined) {
            saveError(projectName, './docker/Dockerfile');
            return;
        }

        // If has a ./docker/Dockerfile -> save and quit
        if (body.trim() !== notFound) {
            saveSuccess(projectName);
            return;
        }

        saveNoDockerFile(projectName);
        console.log(chalk.red(`${projectName} has no ./docker/Dockerfile`));

    });
}

function saveError(content, lookingFor) {
    save(request_failed, content + '\n');
    console.log(`--- Error while looking for ${lookingFor} ---`);
}

function saveSuccess(content) {
    save(selected_repos, `https://github.com/${content}\n`);
    console.log(chalk.blue(content));
}

function saveNoDockerFile(content) {
    save(no_docker_file, `https://github.com/${content}\n`);
}

function saveDockerCompose(content) {
    save(with_docker_compose, `https://github.com/${content}\n`);
    console.log(chalk.gray(`${content} -- has a docker-compose`));
}
