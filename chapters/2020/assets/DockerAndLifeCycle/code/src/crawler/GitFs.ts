import {request, RequestOptions} from 'https';
import {join} from 'path';

/**
 * unused
 */

export class Node {
    private repoOwner: string;
    private readonly repoName: string;
    private path: string;

    private readonly _isDir: boolean = false;
    private readonly _isFile: boolean = false;

    private readonly childs: Array<Node>;

    private contentUrl: string | null;

    /**
     *
     * @param apiObjetct see https://developer.github.com/v3/repos/contents/
     * @param repoOwner
     * @param repoName
     */
    constructor(apiObjetct: any | null, repoOwner: string, repoName: string) {
        this.repoName = repoName;
        this.repoOwner = repoOwner;

        if (apiObjetct == null) {
            this.contentUrl = null;
            this._isFile = false;
            this._isDir = true;
            this.path = '/';
        } else {
            const apiNode = apiObjetct as apiNode;
            this.path = apiNode.path;
            this.contentUrl = apiNode.download_url;

            switch (apiNode.type) {
                case 'file':
                    this._isFile = true;
                    break;
                case 'dir':
                    this._isDir = true;
                    break;
            }
        }

        this.childs = new Array<Node>();
    }

    public isvalid(): boolean {
        return (this._isDir && !this._isFile) || (!this._isDir && this._isFile);
    }

    public getContent(): Buffer {
        if (!this._isFile) throw 'invalid method, the current node is not a file';
        return
    }

    public async walk() {
        if (!this._isDir) throw 'invalid method, the current node is not a directory';
        for (const e of (await execGetNodeContent(this.repoOwner, this.repoName, this.path))) {
            this.childs.push(e);
            console.log(this.repoName + ': ' + e.path);
            if (e.isDir()) {
                await e.walk();
            }
        }
    }

    public getChilds(): Array<Node> {
        return this.childs;
    }

    public isFile(): boolean {
        return this._isFile;
    }

    public isDir(): boolean {
        return this._isDir;
    }
}

export async function getRooOfRepo(owner: string, repo: string): Promise<Node> {
    const res = new Node(null, owner, repo);
    await res.walk();
    return res;
}

async function execGetNodeContent(owner: string, repo: string, path: string): Promise<Array<Node>> {
    const content = JSON.parse((await executeGetContent(owner, repo, path)).toString());
    const res = new Array<Node>();
    content.forEach(element => {
        res.push(new Node(element, owner, repo));
    });
    return res;
}

function executeGetContent(owner: string, repo: string, path: string): Promise<Buffer> {
    console.log('execute for ' + owner + '/' + repo + path);
    const resource = join('repos', owner, repo, 'contents', path);
    const host = 'api.github.com';

    const options: RequestOptions = {
        hostname: host,
        port: 443,
        path: resource,
        method: 'GET',
        headers: {
            'User-Agent': 'rimel',
            'Authorization': ''
        }
    };

    return new Promise((resolve, reject) => {
        request(options, res => {

            const body = new Array<Buffer>();
            res.on('data', chunk => {
                const part = chunk as Buffer;
                body.push(part);
            });
            if (res.statusCode != 200) {
                reject('status code for ' + resource + ' : ' + res.statusCode + '\nBody:' + body.toString());
            }
            res.on('end', () => resolve(Buffer.concat(body)));
            res.on('close', () => resolve(Buffer.concat(body)));
            res.on('error', (err) => reject(err));
        }).end();
    });
}

class apiNode {
    type: string;
    name: string;
    path: string;
    url: string;
    download_url: string;
}
