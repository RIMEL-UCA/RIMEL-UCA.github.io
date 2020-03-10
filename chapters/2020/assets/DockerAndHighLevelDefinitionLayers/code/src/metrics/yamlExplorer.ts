import * as yaml from 'js-yaml';
import * as fs from 'fs';
import {GlobalMetrics, metrics} from './model_metrics';

export function parseYaml(path: string, globalMetrics: GlobalMetrics) {
    let doc = null;
    try {
        doc = yaml.safeLoad(fs.readFileSync(path, 'utf8'));
        exploreServices(doc['services'], globalMetrics.execMetrics);

    } catch (e) {
        console.log(e);
    }
}

function exploreServices(services, metrics: metrics) {
    for (let ser of services) {
        let port = services[ser].ports;
        console.log(port);
        if (port != undefined) {
            metrics.expose++;
        }

    }
}
