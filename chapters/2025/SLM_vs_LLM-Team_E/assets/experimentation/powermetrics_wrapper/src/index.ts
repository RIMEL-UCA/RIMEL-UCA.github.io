import express from 'express';
import { spawn } from 'child_process';
import { XMLParser } from 'fast-xml-parser';

const UPDATE_INTERVAL = 100; // 100 ms

const app = express();
const port = 3033;

type Metric = {
    batteryPercentage: number;
    cpuPowerMw: number;
    gpuPowerMw: number;
    combinedPowerMw: number;
};

// Buffer for the metrics
let lastMetric: Metric = {
    batteryPercentage: 0,
    cpuPowerMw: 0,
    gpuPowerMw: 0,
    combinedPowerMw: 0
};

const xmlParser = new XMLParser({
    ignoreAttributes: false,
    attributeNamePrefix: "@_",
    allowBooleanAttributes: true,
    parseAttributeValue: true,
    trimValues: true
});

let buffer = '';

const handlePowerMetricsData = (data: string) => {
    try {
        // Accumulate the data in the buffer
        buffer += data.toString();

        // Check if the buffer contains a complete XML
        const startIndex = buffer.indexOf('<plist');
        const endIndex = buffer.indexOf('</plist>');

        if (startIndex !== -1 && endIndex !== -1 && endIndex > startIndex) {
            // Extract the complete XML
            const completeXml = buffer.substring(startIndex, endIndex + 8); // 8 is the length of '</plist>'

            // Parse the data
            const parsedData = xmlParser.parse(completeXml);

            const batteryPercentage = parsedData.plist.dict.dict[0].integer;

            const cpuPowerMw = parsedData.plist.dict.dict[1].real[0];
            const gpuPowerMw = parsedData.plist.dict.dict[1].real[1];
            const combinedPowerMw = parsedData.plist.dict.dict[1].real[3];

            lastMetric = { batteryPercentage, cpuPowerMw, gpuPowerMw, combinedPowerMw };

            // Reset the buffer
            buffer = buffer.substring(endIndex + 8);
        }
    } catch (error) {
        // Reset the buffer in case of error
        buffer = '';
    }
};

const listenPowerMetrics = () => {
    console.log("Lauching powermetrics command...");

    // Commande to launch : sudo powermetrics --format plist -i 500
    const command = spawn('sudo', ['powermetrics', '--format', 'plist', '-i', UPDATE_INTERVAL.toString(), '-s', 'battery,cpu_power'], { stdio: 'pipe' });

    // Listen to the output (stdout)
    command.stdout.on('data', handlePowerMetricsData);

    // Handle error
    command.on('error', (error) => {
        console.error(`Error: ${error.message}`);
    });

    // Handle close
    command.on('close', (code) => {
        console.log(`Process closed with code ${code}`);
    });

    // Handle exit
    command.on('exit', (code) => {
        console.log(`Command exited with code ${code}`);
        if (code === 0) {
            console.log("Relaunching command in 5 seconds...");
            setTimeout(listenPowerMetrics, 5000);
        }
    });
};

listenPowerMetrics();

// Export the metrics
app.get('/metrics', async (req, res) => {
    // console.log("GET /metrics", lastMetric);
    console.log("GET /metrics");
    res.set('Content-Type', 'application/json');
    res.json(lastMetric);
});

// Start the server
app.listen(port, () => {
    console.log(`Metrics server running on http://localhost:${port}/metrics`);
});
