export class GlobalMetrics {
    buildMetrics: metrics;
    runMetrics: metrics;
    execMetrics: metrics;

    isValid: boolean;
    whyInvalid: string;

    buildPresent: boolean;
    execPresent: boolean;
    execSource: string;

    constructor() {
        this.buildMetrics = new metrics();
        this.runMetrics = new metrics();
        this.execMetrics = new metrics();
        this.isValid = true;
        this.buildPresent = false;
        this.execPresent = false;
        this.execSource = "";
    }

    toSting(): string {
        return JSON.stringify(this);
    }

    toPrintableJson(): any {
        const build = this.buildMetrics == null ? null : this.buildMetrics.toPrintableJson();
        const run = this.runMetrics == null ? null : this.runMetrics.toPrintableJson();
        const exe = this.execMetrics == null ? null : this.execMetrics.toPrintableJson();
        return {
            buildMetrics: build,
            runMetrics: run,
            execMetrics: exe,
            isValid: this.isValid,
            whyInvalid: this.whyInvalid,
            buildPresent: this.buildPresent,
            execPresent: this.execPresent,
            execSource: this.execSource
        }
    }

    makeInvalid(why: string) {
        this.isValid = false;
        this.whyInvalid = why;
    }

    makeBuildPresent() {
        this.buildPresent = true;
    }

    makeExecPresent() {
        this.execPresent = true;
    }

}

export class metrics {
    expose: number;
    args: number;
    volumes: number;
    envVariables: Set<string>;
    securityVariable: Set<String>;
    unknown: Set<string>;

    constructor() {
        this.expose = 0;
        this.args = 0;
        this.volumes = 0;
        this.envVariables = new Set();
        this.unknown = new Set();
        this.securityVariable = new Set();
    }

    toPrintableJson() {
        return {
            expose : this.expose,
            args : this.args,
            volumes: this.volumes,
            EnvVariables : Array.from(this.envVariables),
            unknown : this.unknown,
            SecurityVariable : Array.from(this.securityVariable)
        };
    }
}
