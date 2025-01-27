import json
import matplotlib.pyplot as plt
import numpy as np
import os

models = ['Llama 3.2 1B Instruct', 'Llama 3.2 3B Instruct', 'Llama 3 8B Instruct']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

# Init list for average metrics
avg_words_per_sec = []
avg_tokens_per_sec = []
avg_responses_per_sec = []
avg_power_per_model = []
avg_energy_per_model = []
avg_inference_time = []

# Browse through the models
for i, model in enumerate(models):
    # Load the results from the file
    results_path = f'../infer/results/results_{i}.json'
    if not os.path.exists(results_path):
        print(f'File {results_path} does not exist. Skipping model {model}.')
        continue

    with open(results_path, 'r') as f:
        results = json.load(f)

    words_per_sec = []
    tokens_per_sec = []
    responses_per_sec = []
    power_per_model = []
    energy_per_model = []
    inference_times = []

    # Browse through the results
    for result in results:
        inference_time = result['inferenceTime'] / 1000  # Convert to seconds
        output_length = result['outputLength']
        output_tokens = result['outputToken']
        energy_metrics = result['energyMetrics']

        # Calculating the metrics
        words_per_sec.append(output_length / inference_time)
        tokens_per_sec.append(output_tokens / inference_time)
        responses_per_sec.append(1 / inference_time)
        inference_times.append(inference_time)

        # Calculating average power in watts
        avg_power = np.mean([metric['gpuPowerMw'] for metric in energy_metrics]) / 1000  # Convert in watts
        power_per_model.append(avg_power)

        # Calculate total energy in joules
        total_energy = np.sum([metric['gpuPowerMw'] * 0.1 for metric in energy_metrics]) / 1000  # Convert in joules
        energy_per_model.append(total_energy)

    # Calculate average metrics for the model
    avg_words_per_sec.append(np.mean(words_per_sec))
    avg_tokens_per_sec.append(np.mean(tokens_per_sec))
    avg_responses_per_sec.append(np.mean(responses_per_sec))
    avg_power_per_model.append(np.mean(power_per_model))
    avg_energy_per_model.append(np.mean(energy_per_model))
    avg_inference_time.append(np.mean(inference_times))

# Create results directory if it doesn't exist
if not os.path.exists('results'):
    os.makedirs('results')
else:
    # Delete existing files
    for file in os.listdir('results'):
        os.remove(os.path.join('results', file))

# Generate the bar graphs
# Graph words/sec
plt.figure(figsize=(10, 5))
plt.bar(models, avg_words_per_sec, color=colors)
plt.title('Average Words per Second')
plt.xlabel('Model')
plt.ylabel('Words/sec')
plt.legend()
plt.savefig('results/words_per_sec.png')

# Graph tokens/sec
plt.figure(figsize=(10, 5))
plt.bar(models, avg_tokens_per_sec, color=colors)
plt.title('Average Tokens per Second')
plt.xlabel('Model')
plt.ylabel('Tokens/sec')
plt.legend()
plt.savefig('results/tokens_per_sec.png')

# Graph responses/sec
plt.figure(figsize=(10, 5))
plt.bar(models, avg_responses_per_sec, color=colors)
plt.title('Average Responses per Second')
plt.xlabel('Model')
plt.ylabel('Responses/sec')
plt.legend()
plt.savefig('results/responses_per_sec.png')

# Graph average power
plt.figure(figsize=(10, 5))
plt.bar(models, avg_power_per_model, color=colors)
plt.title('Average GPU Power')
plt.xlabel('Model')
plt.ylabel('GPU Power (W)')
plt.legend()
plt.savefig('results/average_power.png')

# Graph average energy per inference
plt.figure(figsize=(10, 5))
plt.bar(models, avg_energy_per_model, color=colors)
plt.title('Average GPU Energy per Inference')
plt.xlabel('Model')
plt.ylabel('Energy (J)')
plt.legend()
plt.savefig('results/average_energy.png')

# Graph average inference time
plt.figure(figsize=(10, 5))
plt.bar(models, avg_inference_time, color=colors)
plt.title('Average Inference Time')
plt.xlabel('Model')
plt.ylabel('Inference Time (s)')
plt.legend()
plt.savefig('results/inference_time.png')

print('Graphs saved in results directory.')