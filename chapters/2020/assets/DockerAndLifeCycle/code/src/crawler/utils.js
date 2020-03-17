const fs = require('fs');
const promisify = require("util").promisify;

const sleep = promisify(function (i, cb) {
    setTimeout(cb, i);
});


module.exports = {
    sleep,
    language: 'C%23',
    save
};

function save(fileName, data) {
    fs.appendFileSync(fileName, data);
}
