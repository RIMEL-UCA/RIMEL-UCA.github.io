import { db } from '../database/database';
import { getAllProject, Project } from '../database/project.db';
import { findFileInProject } from '../helpers/helper';
import path from 'path';
import YAML from 'yaml'
import fs from 'fs/promises';
import { parse } from 'comment-json';
import { clearAllEsLintAnalyze, saveEsLintAnalyzeAttributes } from '../database/eslintAnalyze';

// eslint-disable-next-line @typescript-eslint/no-floating-promises
(async () => {
    await db.sync();
    console.log('🚀 Start of ESlint analyze \n');

    await clearAllEsLintAnalyze();
    const projects = await getAllProject(['other']);
    let ended = 0;
    const tasks = projects
    .map(async (project: Project) => {
        const esLintData = await extractEsLintConfig(project);
        await saveEsLintAnalyzeAttributes({
            id: project.id,
            eslintFiles: esLintData.esLintFiles.length > 0,
            quantityOfRules: esLintData.rules.length,
            quantityOfPlugins: esLintData?.plugins.length,
            error: esLintData.error,
        });
        ended++;
        process.stdout.write(`\r⌛ Pending ESlint anaylze... ${ended}/${projects.length}`);
    });
    await Promise.all(tasks);
    console.log('\n🏁 End of ESlint categorization');
    process.exit(0);
})();

/**
 * Regex : https://regex101.com/r/1btp8R/1
 */
async function getESLintFiles(project: Project) {
    const files = await findFileInProject(new RegExp(/^(\.)*eslintrc(.ts|.js|.json||.yml|.yaml)$/mgi), project);
    return files;
}

type EsLintData = {
    esLintFiles: string[];
    rules: string[];
    plugins: string[];
    error: number;
}
/**
 * Extract all eslint rules and plugins from project
 * Handles eslintrc.js, eslintrc.ts, eslintrc.json, eslintrc.yml, eslintrc.yaml
 */
async function extractEsLintConfig(project: Project): Promise<EsLintData>{
    const esLintFiles = await getESLintFiles(project);
    if (esLintFiles.length === 0) {
        return {
            esLintFiles,
            rules: [],
            plugins: [],
            error: 0,
        };
    }
    const tasks = esLintFiles.map(async (file) => {
        const fileContent = await fs.readFile(file, 'utf8');
        let eslintConfig;
        switch (path.extname(file)) {
            case '.js':
            case '.ts':
                eslintConfig = fileContent;
                break;
            case '.json':
            case '.eslintrc':
            case '':
                try {
                    eslintConfig = parse(fileContent);
                } catch (error) { }
                break;
            case '.yaml':
            case '.yml':
                eslintConfig = YAML.parse(fileContent);
                break;
            default:
                console.log(`\n${file}  ${path.extname(file)} is not a valid eslint config file`);
                break;
        }
        const esLintData: EsLintData = {
            esLintFiles: esLintFiles,
            rules: Object.keys(eslintConfig?.rules ?? []),
            plugins: Object.values(eslintConfig?.plugins ?? []),
            error: 0,
        };
        return esLintData;
    });
    const esLintDataArray = await Promise.all(tasks);
    if (esLintDataArray.length === 0) {
        return {
            esLintFiles: esLintFiles,
            rules: [],
            plugins: [],
            error: 1,
        };
    }
    const esLintDataConcat = esLintDataArray.reduce((acc, value) => {
        return {
            esLintFiles: [...acc.esLintFiles,...value.esLintFiles],
            rules: [...acc.rules, ...value.rules],
            plugins: [...acc.plugins, ...value.plugins],
            error: acc.error + value.error,
        };
    });
    return esLintDataConcat;
}
