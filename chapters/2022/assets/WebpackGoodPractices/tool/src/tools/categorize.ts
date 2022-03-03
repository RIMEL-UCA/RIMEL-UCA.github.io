import { CategorizationEnum, categorizeProject, clearAllCategorization } from '../database/categorize';
import { db } from '../database/database';
import { getAllProject, Project } from '../database/project.db';
import { foundCategory, getDependencies, hasDependency } from '../helpers/helper';

// eslint-disable-next-line @typescript-eslint/no-floating-promises
(async () => {
    await db.sync();
    console.log('🚀 Start categorization \n');

    await clearAllCategorization();

    const projects = await getAllProject();
    let ended = 0;
    process.stdout.write(`\r⌛ Categorization... ${ended}/${projects.length}`);
    const tasks = projects.map(async (project: Project) => {
        const dependencies = await getDependencies(project).catch(() => { return {} });
        const category = foundCategory(dependencies);
        await categorizeProject(project, category);
        ended++;
        process.stdout.write(`\r⌛ Categorization... ${ended}/${projects.length}`);
    });
    await Promise.all(tasks);
    console.log('\n🏁 End of categorization');
    process.exit(0);
})();


/**
 * Return if a repository use webpack
 * @param dependencies
 * @param category
 */
function isWebpackRepository(dependencies: Record<string, string> | {}, category: string): boolean {
    if (['angular', 'next', 'vue'].includes(category)) {
        return true;
    }
    return hasDependency(dependencies, 'webpack');
}