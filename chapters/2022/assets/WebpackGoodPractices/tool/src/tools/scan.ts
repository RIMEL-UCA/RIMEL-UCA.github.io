/* eslint-disable no-await-in-loop */
import { Command } from 'commander';
import { db } from '../database/database';
import { getAllProject, getProjectsByCategorie, Project, saveProject } from '../database/project.db';
import * as fs from 'fs/promises';
import { REPOSITORIES_PATH, resolveLocalPath, resolveLocalRepositoryName } from '../helpers/helper';
import pLimit from 'p-limit';
import { exists } from 'fs';

const limit = pLimit(50);

interface ScanArguments {
    local: boolean;
    db: boolean;
    other: boolean;
}
// eslint-disable-next-line @typescript-eslint/no-floating-promises
(async () => {
    await db.sync();
    const args = extractScanArguments();
    if (args.local) {
        console.log('🧹 Cleaning local db project');
        const projects = await getAllProject();
        await Project.destroy({ where: {} });
        try {
            await scanLocalRepositories(projects);
        } catch (err) {
            console.log(`Error during scanning local repositories: ${err}`);
            console.log(`Rollback of database...`);
            await Project.destroy({ where: {} });
            await Project.bulkCreate(projects);
        }
    }
    if (args.db) {
        console.log('🧹 Cleaning unsaved local project');
        await cleanUnconsistancyDBLocal();
    }
    if (args.other){
        console.log('🧹 RM sources of other projects');
        await cleanOtherProject();
    }
    console.log('🏁 End of scan');
    process.exit(0);
})();

function extractScanArguments(): ScanArguments {
    const program = new Command();
    program
        .description('Retrieves and clones projects from GitHub according to the query options')
        .version('0.0.1')
        .option('--local', 'Clean db and use LOCAL as source of truth', false)
        .option('--db', 'Use DB as source of truth and delete local unused package', false)
        .option('--other', 'Rm sources of "other" projects', false)
        .parse(process.argv);

    const options = program.opts();
    if (options.local && options.db) {
        throw new Error('You cannot use --local and --db at the same time');
    }
    return {
        local: options.local,
        db: options.db,
        other: options.other,
    };
}

async function scanLocalRepositories(dbProjects: Project[]): Promise<void> {
    const localRepositories: string[] = await fs.readdir(REPOSITORIES_PATH);
    let ended = 0;
    for (const localRepo of localRepositories) {
        const repositoryPath = `${REPOSITORIES_PATH}/${localRepo}`;
        const detailsPath = `${repositoryPath}/details.json`;
        const file = await fs.readFile(detailsPath, 'utf8').catch(() => null);
        if (!file) {
            continue;
        }
        const details = JSON.parse(file);
        if (Object.keys(details).length <= 1) {
            continue;
        }
        const dbProject = dbProjects.find((project: Project) => project.id === details.id);
        if (dbProject) {
            await Project.create(dbProject);
        } else {
            await saveProject(details);
        }
        ended++;
        process.stdout.write(`\r Pendings scanning local repositories: ${ended}/${localRepositories.length}`);
    }
}

async function cleanOtherProject(): Promise<void>{
    const projects = await getProjectsByCategorie('other');
    console.log(`🧹 Clean Other projects: ${projects.length}`);
    let ended = 0;
    const tasks = projects.map(async (project: Project) => {
        const projectPath = resolveLocalPath(project);
        const sourcePath = `${projectPath}/source/`;
        const isExist = await fs.stat(sourcePath)
        .then(fsStat => {
            return fsStat.isDirectory();
        })
        .catch(err => {
            return false;
        });
        if (isExist) {
            await fs.rm(sourcePath, { recursive: true });
        }
        ended++;
        process.stdout.write(`\r🧹 Cleaning other projects: ${ended}/${projects.length}`);
    });
    await Promise.all(tasks);
}

async function cleanUnconsistancyDBLocal(): Promise<void> {
    const localRepositories: string[] = await fs.readdir(REPOSITORIES_PATH);
    const dbProjects = await getAllProject();
    const dbRepositorires: string[] = dbProjects.map((project: Project) => resolveLocalRepositoryName(project));

    const unsavedLocalRepositories = localRepositories.filter(
        (localRepository: string) => !dbRepositorires.includes(localRepository),
    );
    console.log(`Unsaved local repositories: ${unsavedLocalRepositories.length}`);
    const tasks = unsavedLocalRepositories.map((repository: string) => {
        return fs.rm(`${REPOSITORIES_PATH}/${repository}`, { recursive: true });
    });
    await Promise.all(tasks);

    const unsavedDBRepositories = dbProjects.filter(
        (project: Project) => !localRepositories.includes(resolveLocalRepositoryName(project)),
    );
    console.log(`Unsaved db repositories: ${unsavedDBRepositories.length}`);
    const tasks2 = unsavedDBRepositories.map((repository: Project) => {
        return Project.destroy({ where: { id: repository.id } });
    });
    await Promise.all(tasks2);
}
