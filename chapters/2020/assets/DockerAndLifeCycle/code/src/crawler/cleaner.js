const path = require('path');
const neek = require('neek');
const language = require('./utils').language;
// --------------------------------------------------------------------------- //

const readable = path.join(__dirname, '..', 'generated', language, `${language}-selected-tmp.txt`);
const writable = path.join(__dirname, '..', 'generated', language, `${language}-selected.txt`);

neek.unique(readable, writable, function (result) {
    console.log(result);
});

