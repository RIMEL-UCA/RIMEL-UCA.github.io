import { Octokit } from '@octokit/rest';
import * as dotenv from "dotenv";
import { Command } from 'commander';
import { db } from '../database/database';
import { getAllProject, saveProject } from '../database/project.db';
import path from 'path';
import fs from 'fs';
import { REPOSITORIES_PATH } from '../helpers/helper';
import AdmZip from "adm-zip";

dotenv.config();

const octokit = new Octokit({
    auth: process.env.GITHUB_TOKEN,
});

interface DownloadArguments {
    query: string;
    limit: number;
    sort: string;
}
function extractDownloadArguments(): DownloadArguments {
    const program = new Command();
    program
        .description('Retrieves and clones projects from GitHub according to the query options')
        .version('0.0.1')
        .option('--query <string>', 'Query term in package.json', 'webpack')
        .option('--limit <number>', 'Limit the number of repositories to download', '1000')
        .option('--sort <string>', 'Sort by (updated, stars, forks)', 'updated')
        .parse(process.argv);

    const options = program.opts();
    return {
        query: options.query,
        limit: options.limit,
        sort: options.sort,
    };
}

// eslint-disable-next-line @typescript-eslint/no-floating-promises
(async () => {
    await db.sync();
    const args = extractDownloadArguments();
    const repositories = await retrieveRepositoriesFromGithub(args.query, args.limit, {
        sort: args.sort,
    });
    await multiDownloadRepo(repositories);

    process.exit(0);
})();

/**
 * Make parallel download of repositories
 * @params concurrentDownloads Max concurrent downloads is concurrentDownloads
 */
async function multiDownloadRepo(repos: any[], concurrentDownloads = 50): Promise<any> {
    function recursiveLoopingDownload(repositoriesStackTask: any[], repo: any): Promise<any> {
        return downloadRepository(repo, true)
        .catch((_err) => {
            console.log('📥 Downloading project failed');
        }) 
        .then(async () => {
            await saveProject(repo).catch((_err) => {
                console.log('📥 Saving project failed');
            });
        })
        .finally(async () => {
            ended++;
            process.stdout.write(`\r📥 Processing...   ${ended}/${repos.length}`);
            const nextRepo = repositoriesStackTask.pop();
            if (nextRepo) {
                await recursiveLoopingDownload(repositoriesStackTask, nextRepo);
            }
        });
    }
    console.log(`\n📥 Downloading ${repos.length} repositories`);
    let ended = 0;
    process.stdout.write(`\r📥 Processing...   ${ended}/${repos.length}`);

    const repositoriesStackTask = [...repos];
    const downloadTask = [];
    while (downloadTask.length < concurrentDownloads && repositoriesStackTask.length > 0) {
        downloadTask.push(recursiveLoopingDownload(repositoriesStackTask, repositoriesStackTask.pop()));
    }
    await Promise.all(downloadTask);

    process.stdout.write(`\n📥 Processing...   ${ended}/${repos.length} ended✅`);
    return;
}

/**
 * Retrieve repositories from GitHub
 * @link Example: https://github.com/search?q=webpack+in%3Apackage.json+language%3Ajavascript+archived%3Afalse+is%3Apublic&type=Repositories
 * @param termInPackageJson - Search for projects containing this term in package.json
 * @param limit - Limit of projects wanted
 */
async function retrieveRepositoriesFromGithub(termInPackageJson: string, limit: number, options: any): Promise<any> {
    //Optimization if limit > PER_PAGE_MAX
    const alreadyLoadedRepositories = await getAllProject();

    const repositories: any[] = [];
    console.log(`🔎 Search project from GitHub`);
    let page = 1;
    // Max result is 1000
    while (repositories.length < limit && page <= 10) {
        // eslint-disable-next-line no-await-in-loop
        const githubResponse = await githubCall({
            termInPackageJson: termInPackageJson,
            per_page: 100,
            page: page,
            sort: options.sort,
        },
        );
        //Cleaning data
        const queryRepositories = githubResponse.data.items
            .flat()
            .filter((repo: any) => !(alreadyLoadedRepositories.find((r: any) => r.id === repo.id)))
            .slice(0, limit - repositories.length);

        repositories.push(...queryRepositories);
        process.stdout.write(`\r🔎 Retrieve ${termInPackageJson}... ${repositories.length}/${limit}`);
        page++;
    }
    // 
    process.stdout.write(`\n🔎 Search ${termInPackageJson} on GitHub ended ✅\n`);
    return repositories;
}

/**
 * Execute github HTTP GET request
 * @param params
 */
function githubCall(params: any): Promise<any> {
    return octokit.rest.search.repos({
        q: params.termInPackageJson + '+in:package.json+language:javascript+language:typescript+archived:false+is:public',
        sort: params.sort,
        order: 'desc',
        per_page: params.per_page,
        page: params.page,
    }).catch(async (error: any) => {
        process.stdout.write('\n');
        let delay = 70;
        console.log();
        while(delay > 0) {
            process.stdout.write(`\rAPI rate limit exceeded waiting ${delay} seconds`);
            // eslint-disable-next-line no-await-in-loop
            await new Promise((resolve) => setTimeout(resolve, 1000));
            delay--;
        }
        return githubCall(params);
    });
}

/**
 * Download a repository in the right path
 * @param repo - Repository object return by Github's API
 * @param saveDetails - Save details to a json
 */
async function downloadRepository(repo: any, saveDetails: boolean): Promise<string> {
    if (!repo){
        throw new Error('Repository is undefined');
    }
    const repoPath = path.resolve(REPOSITORIES_PATH, `${repo.name}_${repo.id}`);

    await fs.promises.mkdir(repoPath, { recursive: true });
    
    try {
        const repoData = await octokit.rest.repos.downloadZipballArchive({
            owner: repo.owner.login,
            repo: repo.name,
            ref: repo.default_branch,
        });
        // eslint-disable-next-line @typescript-eslint/ban-ts-comment
        // @ts-ignore
        const buffer = Buffer.from(repoData.data);
        const zip = new AdmZip(buffer);
        zip.extractAllTo(repoPath,true);
        const sourcePath = await fs.promises.readdir(repoPath);
        await fs.promises.rename(path.resolve(repoPath, sourcePath[0]), path.resolve(repoPath, 'source'));
        if (saveDetails) {
            await fs.promises.writeFile(path.resolve(repoPath, 'details.json'), JSON.stringify(repo, null, 2));
        }
    } catch (e) {
        throw new Error(`Error while downloading repository ${repo.name}`);
    }
    return repoPath;
}