const https = require('https');
const fs = require('fs');
const path = require('path');

// URL of the JSON file
const url = 'https://raw.githubusercontent.com/gururise/AlpacaDataCleaned/refs/heads/main/alpaca_data.json';

// Function to download the JSON file
const downloadJson = (url, callback) => {
    https.get(url, (response) => {
        let data = '';

        // A chunk of data has been received.
        response.on('data', (chunk) => {
            data += chunk;
        });

        // The whole response has been received.
        response.on('end', () => {
            callback(JSON.parse(data));
        });
    }).on('error', (err) => {
        console.error('Error downloading the JSON file:', err.message);
    });
};

// Function to process the JSON data
const processJson = (data) => {
    // Retrieve the first 4096 objects from the root array
    const first4096Objects = data.slice(0, 4096);

    // Extract the instruction attribute from each object
    const instructions = first4096Objects.map(obj => obj.instruction);

    // Save the instructions to a new file
    const outputFilePath = path.join(__dirname, 'instructions.json');
    fs.writeFileSync(outputFilePath, JSON.stringify(instructions, null, 2));

    console.log('Instructions saved to', outputFilePath);
};

// Download and process the JSON file
downloadJson(url, processJson);