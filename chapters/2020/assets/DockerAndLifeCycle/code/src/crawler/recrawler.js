const fs = require('fs');
const request = require('request');
const path = require('path');
const chalk = require('chalk');

const language = require('./utils').language;
const save = require('./utils').save;

// --------------------------------------------------------------------------- //

let projects;

const selected_repos = path.join(__dirname, '..', 'generated', language, `${language}-selected-tmp.txt`);
const with_docker_compose = path.join(__dirname, '..', 'generated', language, `${language}-compose.txt`);
const no_docker_file = path.join(__dirname, '..', 'generated', language, `${language}-no-docker.txt`);
const request_failed = path.join(__dirname, '..', 'generated', language, `${language}-request-failed.txt`);
const request_failed_again = path.join(__dirname, '..', 'generated', language, `${language}-request-failed-again.txt`);

const notFound = '404: Not Found';

main();

async function main() {

    fs.writeFileSync(request_failed_again, '');

    projects = (fs.readFileSync(request_failed) + '').trim().split('\n');

    for (const id of projects) {
        await processRepos(id);
    }
}

async function processRepos(projectName) {

    console.log(projectName);

    if (await getDockerCompose(projectName)) {
        return;
    }

    if (await getRootDockerfile(projectName)) {
        return;
    }

    await getDeepDockerfile(projectName);
}

// --------------------------------------- //
// --------------------------------------- //
// --------------------------------------- //

async function getDockerCompose(projectName) {
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

async function getRootDockerfile(projectName) {
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

async function getDeepDockerfile(projectName) {
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
    save(request_failed_again, content + '\n');
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
