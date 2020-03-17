'use strict';

require('dotenv').config({ path: '../../.env' });

const csv = require('csv-parser');
const fs = require('fs');
const { createReadStream } = fs;
const rp = require('request-promise');
const https = require('https');
const results = [];
 
function sleep(ms) {
  return new Promise((resolve) => {
    setTimeout(resolve, ms);
  });
}

createReadStream('sources/dataset.csv')
  .pipe(csv())
  .on('data', (data) => {
    if (data.stars > 1000) {
      results.push(data);
    }
  })
  .on('end', async () => {
    console.log("CSV read");
    results.forEach(async repo => {
      if (!fs.existsSync(__dirname + `/results/${repo.name}/`)) {
        await getPomInformation(repo.name);
      }
      await sleep(7000);
    });
  });

async function getPomInformation(gitName) {
    var options = {
        uri: `https://api.github.com/search/code?q=repo:${gitName}+filename:pom.xml+profile+in:file`,
        qs: {
            access_token: process.env.GITHUB_KEY
        },
        headers: {
            'User-Agent': 'Request-Promise'
        },
        json: true // Automatically parses the JSON string in the response
    };
     
    const res = await rp(options);
    if (!res.items) return;

    let filesToDownload = res.items.map(meta => (
      {
        path: meta.path,
        url: meta.html_url.replace('/blob/', '/').replace('github.com', 'raw.githubusercontent.com')
      }
    ));

    filesToDownload = filesToDownload.filter(f => f.path.split('/').length === 1);
    downloadFiles(filesToDownload, gitName);
}

function downloadFiles(toDownload, repoName) {
  toDownload.forEach(fileObj => {
    const tab = fileObj.path.split('/');
    tab.pop();

    const pathWithoutFile = tab.length > 0 ? tab.join('/') : '';
    fs.mkdirSync(__dirname + `/results/${repoName}/${pathWithoutFile}`, { recursive: true });

    const file = fs.createWriteStream(`results/${repoName}/${fileObj.path}`);
    https.get(fileObj.url, function(response) {
      response.pipe(file);
    });
  });
}