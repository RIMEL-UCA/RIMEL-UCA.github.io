import { db } from '../database/database';
import { getAllProject, Project } from '../database/project.db';
import { getStructuredDependencies } from '../helpers/helper';
import { clearAllDevDependenciesAnalyze, DevDependenciesAnalyzeAttributes, saveDevDependenciesAnalyze } from '../database/devDependenciesAnalyze';

// eslint-disable-next-line @typescript-eslint/no-floating-promises
(async () => {
    await db.sync();
    console.log('🚀 Start of DevDependencies categorization \n');

    const mostAverageDevDependencies = await findMostAverageDevDependenciesAccuracy();

    await clearAllDevDependenciesAnalyze();

    const projects = await getAllProject(['other']);
    let ended = 0;
    const tasks = projects.map(async (project: Project) => {
        await checkWrongPlaceForDependencies(project, mostAverageDevDependencies);
        ended++;
        process.stdout.write(`\r⌛ Pending DevDependencies anaylze... ${ended}/${projects.length}`);
    });
    await Promise.all(tasks);
    console.log('\n🏁 End of DevDependencies categorization');
    process.exit(0);
})();

/**
 * Find the most common dev dependencies
 */
async function findMostAverageDevDependenciesAccuracy(): Promise<string[]>{
    const dependenciesOccurences: Record<string, {
        dependencies: number;
        devDependencies: number;
    }> = {};
    const projects = await getAllProject(['other']);
    const tasks = projects.map(async (project: Project) => {
        const dependencies = await getStructuredDependencies(project);
        Object.keys(dependencies.dependencies).forEach(key => {
            dependenciesOccurences[key] = dependenciesOccurences[key] ?? {dependencies: 0, devDependencies: 0};
            dependenciesOccurences[key].dependencies++;
        });
        Object.keys(dependencies.devDependencies).forEach(key => {
            dependenciesOccurences[key] = dependenciesOccurences[key] ?? {dependencies: 0, devDependencies: 0};
            dependenciesOccurences[key].devDependencies++;
        });
    });
    await Promise.all(tasks);
    return filtreDependancies(dependenciesOccurences);
}

/**
 * Filter dependancies to keep only the most common ones
 */
function filtreDependancies(dependenciesOccurences: Record<string, {
    dependencies: number;
    devDependencies: number;
}>): string[] {
     const object = Object.entries(dependenciesOccurences)
     .sort((a, b) => {
        return b[1].devDependencies - a[1].devDependencies;
    })
    .filter((value) => {
        return value[1].devDependencies / (value[1].devDependencies + value[1].dependencies) ?? 1 > 0.8;
    })
    .filter((value) => {
        return value[1].devDependencies > 25;
    });
    return object.map(value => value[0]);
}

/**
 * Checks the number of misplaced dev dependencies
 * @param project
 */
 async function checkWrongPlaceForDependencies(project: Project, mostAverageDevDependencies: string[]): Promise<void> {
    const mostCommonDevDependencies: string[] = mostAverageDevDependencies;
    
    const dependencies = await getStructuredDependencies(project);
    
    const quantityOfDependencies = Object.keys(dependencies.dependencies ?? {}).length;
    const quantityOfDevDependencies = Object.keys(dependencies.devDependencies ?? {}).length;

    const quantityOfTargetDependencies = Object.keys({...dependencies.dependencies, ...dependencies.devDependencies}).filter(dependency => mostCommonDevDependencies.includes(dependency)).length;
    const quantityOfWrongDevDependencies = Object.keys(dependencies.dependencies ?? {}).filter(dependency => mostCommonDevDependencies.includes(dependency)).length;
    
    const devDependenciesAnalyze: DevDependenciesAnalyzeAttributes = {
        id: project.id,
        quantityOfDependencies: quantityOfDependencies,
        quantityOfDevDependencies: quantityOfDevDependencies,
        quantityOfTargetDependencies: quantityOfTargetDependencies,
        quantityOfWrongDevDependencies: quantityOfWrongDevDependencies,
    };
    await saveDevDependenciesAnalyze(devDependenciesAnalyze);


}