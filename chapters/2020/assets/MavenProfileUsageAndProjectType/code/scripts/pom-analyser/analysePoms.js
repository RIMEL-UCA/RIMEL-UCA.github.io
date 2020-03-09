'use strict';

const fs = require('fs');
const pomParser = require("pom-parser");

// An assertion that can be a closed question or a calculated metric
class Assertion {
    // name: String
    // assertion: Function (pomObject) => boolean | number

    constructor(name, assertion) {
        this.name = name;
        this.assertion = assertion;
    }
}

/**
 * Analyse a pom.xml file to extract some interesting metrics
 * 
 * @param {String} pomPath Path to the pom.xml to parse
 * @param {Assertion[]} assertions The assertions to test on this pom
 * @returns {Promise<Result[]>} An array of assertion results for this pom: {name: String, result: boolean | number}
 */
async function analyse(pomPath, assertions, repo) {
    var opts = {
        filePath: pomPath, // The path to a pom file
    };

    return new Promise((res, rej) => {
        // Parse the pom based on a path
        pomParser.parse(opts, (err, pomResponse) => {
            if (err) rej(err);
            
            // The original pom xml that was loaded is provided.
            // console.log("XML: " + pomResponse.pomXml);
            // The parsed pom pbject.
            // console.log("OBJECT: " + JSON.stringify(pomResponse.pomObject));
    
            const pomObj = pomResponse.pomObject;
            res(
                {
                    repo: repo,
                    result: assertions.map(assertion => ({
                        name: assertion.name,
                        result: assertion.assertion(pomObj)
                    }))
                }
            );
        });
    });
}

const assertionsToTest = [
    {
        name: "ProfilesCount",
        assertion: (pom) => {
            if (!pom.project) return 0;
            if (!pom.project.profiles) return 0;
            if (!pom.project.profiles.profile) return 0;
            return pom.project.profiles.profile.length || 1;
        }
    },
    {
        name: "ContainsReleaseCount",
        assertion: (pom) => {
            if (!pom.project) return 0;
            if (!pom.project.profiles) return 0;
            if (!pom.project.profiles.profile) return 0;
            return pom.project.profiles.profile.length > 0 ? 
                pom.project.profiles.profile.filter(p => p.id.includes('release')).length : 
                (pom.project.profiles.profile.id.includes('release') ? 1 : 0);
        }
    },
    {
        name: "ContainsPropertiesCount",
        assertion: (pom) => {
            if (!pom.project) return 0;
            if (!pom.project.profiles) return 0;
            if (!pom.project.profiles.profile) return 0;
            return pom.project.profiles.profile.length > 0 ? 
                pom.project.profiles.profile.filter(p => p.properties).length : 
                (pom.project.profiles.profile.properties ? 1 : 0);
        }
    }
];

// analyse(__dirname + '/results/apache/hbase/pom.xml', assertionsToTest)
//     .then(res => console.log(res));

const promises = [];
const roots = fs.readdirSync(__dirname + '/results');
roots.forEach(root => {
    const repos = fs.readdirSync(__dirname + '/results/' + root);
    repos.forEach(repo => {
        promises.push(analyse(__dirname + '/results/' + root + '/' + repo + '/pom.xml', assertionsToTest, `${root}/${repo}`));
    });
});

Promise.all(promises).then(res => {
    console.log(JSON.stringify(res));
});