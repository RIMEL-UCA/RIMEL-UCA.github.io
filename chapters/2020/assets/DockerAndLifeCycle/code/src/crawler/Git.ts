import {exec} from 'child_process';
import {remove} from 'fs-extra';

export function clone(url: string, output: string) {
    const command = `git clone ${url}.git ${output}`;
    return new Promise((resolve, reject) => {
        exec(command, err => {
            if (err) reject(err);
            resolve();
        });
    })
}

export async function deleteCurRepo(url: string) {
    const parts = url.split('/');
    await remove('workspace/' + parts[parts.length - 1]);
}
