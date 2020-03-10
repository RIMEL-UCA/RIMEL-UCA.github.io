import * as fs from 'fs';
import {analyseLine} from './ShellAnalyser';
import {GlobalMetrics} from './model_metrics';

export class MardownExplorer {
    pathToMd: string[];
    globalMetrics: GlobalMetrics;

    constructor(path: string[], globalMetrics: GlobalMetrics) {
        this.pathToMd = path;
        this.globalMetrics = globalMetrics;
    }

    explorer() {
        this.pathToMd.forEach(element => {
            const contentFile = fs.readFileSync(element, 'utf8').trim();
            const array = contentFile.split("```");
            this.explorerCommandLine(array);
        });
    }

    explorerCommandLine(array) {
        array.forEach((element, index, arr) => {
            if (element.includes("docker")) {
                analyseLine(arr, index, this.globalMetrics);
                return;
            }
        });
    }
}
