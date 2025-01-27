const fs = require('fs');
const path = require('path');
const axios = require('axios');

// Load instructions from the JSON file
const instructionsPath = path.join(__dirname, '../data/instructions.json');
const instructions = JSON.parse(fs.readFileSync(instructionsPath, 'utf8'));

// GPT4ALL URL
const gpt4allUrl = 'http://localhost:4891/v1/completions';

const models = ['Llama 3.2 1B Instruct', 'Llama 3.2 3B Instruct', 'Llama 3 8B Instruct'];

// Infer function to run one instruction on the model
const infer = async (instruction, model) => {
    const startTime = Date.now();
    const response = await axios.post(gpt4allUrl, {
        model,
        prompt: instruction,
        max_tokens: 1024,
        temperature: 0.7
    });
    const endTime = Date.now();
    const inferenceTime = endTime - startTime;

    return {
        inferenceTime,
        input: instruction,
        inputLength: instruction.length,
        inputToken: response.data.usage.prompt_tokens,
        output: response.data.choices[0].text,
        outputLength: response.data.choices[0].text.length,
        outputToken: response.data.usage.completion_tokens,
    };
};

let lastEnergyMetrics = {
    batteryPercentage: 0,
    cpuPowerMw: 0,
    gpuPowerMw: 0,
    combinedPowerMw: 0
};
// Get energy metrics from the energy server
const getEnergyMetrics = async () => {
    // Double try request to avoid errors
    // If the first request fails, try again
    // If the second request fails, return the last known metrics
    try {
        const response = await axios.get('http://localhost:3033/metrics', { timeout: 25 });
        lastEnergyMetrics = response.data;
    } catch (error) {
        try {
            const response = await axios.get('http://localhost:3033/metrics', { timeout: 25 });
            lastEnergyMetrics = response.data;
        } catch (error) {
            console.error('Failed to get energy metrics, returning last known metrics:', error.message);
        }
    }
    return lastEnergyMetrics;
};

// Run instructions on the model
const runInstructions = async (model, index) => {
    const limitedInstructionsSize = 256;
    let limitedInstructions = instructions.slice(0, limitedInstructionsSize);

    const outputFilePath = path.join(__dirname, './results/results_' + index + '.json');

    // Open JSON array
    fs.appendFileSync(outputFilePath, '[');

    const instructionsSize = limitedInstructions.length;
    for (let i = 0; i < instructionsSize; i++) {
        console.log('Processing instruction', i + 1, 'of', instructionsSize);
        const instruction = limitedInstructions[i];
        // Retrieve energy metrics every 100ms during the inference
        const energyMetrics = [];
        let timeElapsed = 0;
        const interval = setInterval(async () => {
            const metrics = await getEnergyMetrics();
            energyMetrics.push({
                ...metrics,
                timeElapsed,
            });
            timeElapsed += 100;
        }, 100);

        const response = await infer(instruction, model);

        clearInterval(interval);

        const result = {
            ...response,
            energyMetrics
        };

        // Save to the JSON file after each instruction
        if (i === instructionsSize - 1) {
            fs.appendFileSync(outputFilePath, JSON.stringify(result, null, 2));
        } else {
            fs.appendFileSync(outputFilePath, JSON.stringify(result, null, 2) + ',\n');
        }

        console.log(`Instruction processed: ${instruction}`);
    }
    // Close the JSON array
    fs.appendFileSync(outputFilePath, ']');
    console.log('Results saved to', outputFilePath);
};

const runModels = async () => {
    // Erase folder results if it exists
    const resultsPath = path.join(__dirname, './results');
    if (fs.existsSync(resultsPath)) {
        fs.rmSync(resultsPath, { recursive: true });
    }

    // Create folder results
    fs.mkdirSync(resultsPath);

    for (const model of models) {
        // Run a first instruction to warm up the model
        console.log(`Warming up model: ${model}`);
        await infer('How to create a website', model);

        console.log(`Running instructions for model: ${model}`);
        await runInstructions(model, models.indexOf(model));
    }
};


runModels();