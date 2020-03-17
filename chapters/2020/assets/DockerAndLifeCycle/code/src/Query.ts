import { readdir, readFile } from "fs";
import * as _ from "lodash";

class repo {
    metrics: repoMetrics;
    name: string;
}

class repoMetrics {
    buildMetrics: metrics;
    runMetrics: metrics;
    execMetrics: metrics;

    isValid: boolean;
    whyInvalid: string;

    buildPresent: boolean;
    execPresent: boolean;
    execSource: string;
}

class metrics {
    expose: number;
    Args: number;
    volumes: number;
    EnvVariables: Array<string>;
    SecurityVariable: Array<String>;
    unknown: Array<string>;
}

function read(file: string) : Promise<repo> {
    const parts = file.split("/");
    return new Promise((resolve, reject) => {
        readFile(file,(err, data) => {
            if (err) {
                reject(err);
            }else { 
                const res = new repo();
                res.name = parts[parts.length -1];
                res.metrics = (JSON.parse(data.toString()) as repoMetrics)
                resolve(res);
            }
        })
    })
}

async function load(files: string[]): Promise<Array<repo>> {
    const pending = new Array<Promise<Object>>();
    files.forEach(f => {
        pending.push(read(f));
    })
    return (await Promise.all(pending) as repo[]);
}

// main

const lang = process.argv[2];
const select = process.argv[3];
const where = process.argv[4];

readdir("./lang/"+lang, async (err,files) => {
    if (err) {
        console.error(err);
        return;
    }

    const repos = await load(files.map(f => "./lang/"+lang+"/"+f));
    const res = new Array<Object>();

    let filtered = repos;
    where.split(" ").forEach(filter => {
        const key = filter.split("=")[0];
        const value = filter.split("=")[1];
        filtered = filtered.filter(v => {
            //console.log(JSON.stringify(_.get(v, key))+" === "+JSON.stringify(value).replace("\"", "").replace("\"",""));
            return JSON.stringify(_.get(v, key)) === JSON.stringify(value).replace("\"", "").replace("\"", "");
        });
    });


    filtered.map(v => {
        const res = {};
        select.split(" ").forEach(k => {
            res[k] = _.get(v, k);
        });
        return res;
    }).forEach(v => {
        console.log(v);
    });
});




