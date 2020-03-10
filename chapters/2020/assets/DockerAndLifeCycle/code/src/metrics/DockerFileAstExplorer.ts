import * as fs from 'fs';
import {Dockerfile, DockerfileParser, Instruction} from 'dockerfile-ast';
import {GlobalMetrics, metrics} from './model_metrics';

export class AstExplorer {

    stages: Array<Array<Instruction>>;
    dockerfile: Dockerfile;
    currentStage: number;
    stageCount: number;
    metrics: GlobalMetrics;

    securityparts: string[];

    constructor(file: string, securityparts: string[], globalMetrics: GlobalMetrics) {
        this.stages = [];
        this.currentStage = 0;
        this.metrics = globalMetrics;

        this.dockerfile = DockerfileParser.parse(fs.readFileSync(file).toString());
        this.stageCount = this.check();
        this.securityparts = securityparts;
    }

    check(): number {
        let stageCounts: number = 0;
        const instructions = this.dockerfile.getInstructions();
        let curInstructions: Instruction[] = [];
        instructions.forEach(element => {
            if (element.getKeyword() === 'FROM') {
                ++stageCounts;
                if (curInstructions.length != 0) {
                    this.stages.push(curInstructions);
                    curInstructions = [];
                }
            } else {
                curInstructions.push(element);
            }
        });
        this.stages.push(curInstructions);
        if (stageCounts > 2) throw 'too many stage';
        if (stageCounts === 2) this.metrics.makeBuildPresent();
        return stageCounts;
    }

    explore(): GlobalMetrics {
        let res = this.metrics;
        for (let curstage = 0; curstage < this.stageCount; ++curstage) {
            let stage = this.exploreStage(this.stages[curstage]);
            if (this.stageCount == 2) {
                switch (curstage) {
                    case 0:
                        res.buildMetrics = stage != null ? stage : res.buildMetrics;
                        break;

                    case 1:
                        res.runMetrics = stage != null ? stage : res.runMetrics;
                        break;
                }
            } else {
                res.runMetrics = stage;
                res.buildMetrics = null;
            }
        }
        return res;
    }


    exploreStage(stage: Instruction[]): metrics {
        const res = new metrics();
        if (stage == null) return res;
        stage.forEach(i => {
            switch (i.getKeyword().toUpperCase()) {
                case 'RUN':
                    this.exploreRUN(i.getArgumentsContent()).forEach(e => res.envVariables.add(e));
                    break;
                case 'ENV':
                    this.exploreENV(i.getArgumentsContent()).forEach(e => res.envVariables.add(e));
                    break;
                case 'ARG':
                    res.args++;
                    break;

                case 'EXPOSE':
                    res.expose++;
                    break;
                case 'VOLUME':
                    res.volumes++;
                    break;
                default:
                    res.unknown.add(i.getKeyword().toUpperCase());
                    break;
            }
        });

        //security variables
        Array.from(res.envVariables).filter(v => {
            let predicate = false;
            this.securityparts.forEach(p => {
                if (v.toUpperCase().includes(p.toUpperCase())) {
                    predicate = true;
                }
            });
            return predicate;
        }).forEach(v => res.securityVariable.add(v));

        return res;
    }

    exploreENV(cmd: string): Array<string> {
        let res = new Array<string>();
        if (cmd.includes('=')) {
            cmd.split(' ').forEach(e => res.push(e.split('=')[0]));
        } else {
            res.push(cmd.split(' ')[0]);
        }
        return res.filter(e => e.length > 0);
    }

    exploreRUN(cmd: string): Array<string> {
        let res = cmd.match(/\$([A-Z_]+[A-Z0-9_]*)|\${([A-Z0-9_]*)}/ig);
        return res !== null ? res : [];
    }
}

