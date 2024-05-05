r"""
    Compare
    1. reduction_scatter_reduce_cap for different input sizes (guess=candidate), output size (colors), capacity (1-threads)
    2. reduction_scatter_reduce_row (guess parallel)
    3. reduction_scatter_reduce (candidate parallel naive)
"""
import subprocess
import csv
import math
from tqdm import tqdm
# Configuration settings
# Common variables
# input_pow = range(12, 16, 4)
input_pow = [10, 11, 12, 13, 14, 15, 16]
input_range = [int(math.pow(2, i)) for i in input_pow]
# output_pow = [5]
output_pow = [5, 6, 7, 8]
output_range = [int(math.pow(3, j)) for j in output_pow]

thread_range = [1, 2, 4, 8]
seed = [0, 1, 2]

specific_params = []


specific_params.append(
    {'-m': 'G'} # guess parallel, reduction_scatter_reduce_row
)
specific_params.append(
    {'-m': 'R'}
)
for i in [1, 3, 5, 7, 9]:
    specific_params.append(
        {'-m': 'C', '-c': i} # candidate parallel, reduction_scatter_reduce
    )

# Cross product specific with common parameters
exp_params = []

for param in specific_params:
    for i in input_range:
        for o in output_range:
            for n in thread_range:
                for s in seed:
                    combo = param.copy()  # Start with the specific settings
                    combo.update({'-i': i, '-o': o, '-n': n, '-s': s})  # Add common settings
                    exp_params.append(combo)
# Add serial
for i in input_range:
    for o in output_range:
        for s in seed:
            exp_params.append({'-m': 'S', '-i': i, '-o': o, '-n': 1, '-s': s})

executable_path = './sred-2d'
output_csv = 'benchmark_results_2d.csv'

# Function to run the benchmark and capture output
def run_benchmark(params):
    command = [executable_path]
    for key, value in params.items():
        command.append(key)
        command.append(str(value))
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout

# Parse the output to extract metrics
def parse_output(output):
    lines = output.split('\n')
    metrics = {}
    for line in lines:
        if 'computation time' in line:
            metrics['time'] = float(line.split(': ')[1])
    return metrics

def format_metrics(metrics):
    formatted = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted[key] = f"{value:.15f}"  # Format as string with 15 decimal places
        else:
            formatted[key] = value
    return formatted
# Write to CSV
with open(output_csv, 'w', newline='') as file:
    fieldnames = ['time', 'mode', 'input', 'output', 'threads', 'seed', 'capacity']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    
    for params in (pbar := tqdm(exp_params)):
        pbar.set_postfix_str(params)
        output = run_benchmark(params)
        metrics = parse_output(output)
        metrics.update({
            'mode': params['-m'],
            'input': params['-i'],
            'output': params['-o'],
            'threads': params['-n'],
            'seed': params['-s'],
            'capacity': params.get('-c', 0)
        })
        formatted_metric = format_metrics(metrics)
        writer.writerow(formatted_metric)

print("Benchmarking completed. Results saved to", output_csv)
