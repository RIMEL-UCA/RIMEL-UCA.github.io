
import {$} from 'zx';
import { Command } from 'commander';


interface MainArguments {
    'query': string;
    'limit': number;
    'sort': string;
}
// eslint-disable-next-line @typescript-eslint/no-floating-promises
(async () => {
    const args = extractMainArguments();
    const prefix = `${__dirname}/..`;
    await $`npm run download --prefix ${prefix} -- --query ${args.query} --limit ${args.limit} --sort ${args.sort}`;
    await $`npm run categorize --prefix ${prefix}`;
    await $`npm run analyze:1 --prefix ${prefix}`;
    await $`npm run analyze:2 --prefix ${prefix}`;

    
    process.exit(0);
})();

/**
 * Copy from tools/download.ts
 */
function extractMainArguments(): MainArguments {
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