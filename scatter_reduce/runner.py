import subprocess
import csv
import math
import pickle # needed for full precision storage, rather than writing to csv again
# Configuration settings
# Common variables
# input_pow = range(12, 16, 4)
input_pow = range(12, 32, 4)
input_range = [int(math.pow(2, i)) for i in input_pow]
# output_pow = [5]
output_pow = [5, 6, 7, 8]
output_range = [int(math.pow(3, j)) for j in output_pow]
# thread_range = [4]
thread_range = [1, 2, 4, 8, 16]
seed = [0, 1, 2]

specific_params = []


# mode = 'F' - this is silly
# specific_params.append(
#     {'-m': 'F', '-l': 0}
# )

# mode = 'R'
specific_params.append(
    {'-m': 'R', '-l': 0}
)
specific_params.append(
    {'-m': 'M', '-l': 0} # OMP native
)


# Cross product specific with common parameters
exp_params = []

# mode = 'L' specific
# use a reduced set of inputs - hangs on full
for i in input_range[:int(len(input_range) / 2)]:
    for o in output_range:
        for n in thread_range:
            for s in seed:
                for l in [16, 64]:
                    exp_params.append({'-m': 'L', '-l': l, '-i': i, '-o': o, '-n': n, '-s': s})

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
            exp_params.append({'-m': 'S', '-l': 0, '-i': i, '-o': o, '-n': 1, '-s': s})


executable_path = './sred'
output_csv = 'benchmark_results.csv'

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
    fieldnames = ['time', 'mode', 'l_value', 'input', 'output', 'threads', 'seed']
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    
    for params in exp_params:
        output = run_benchmark(params)
        metrics = parse_output(output)
        metrics.update({
            'mode': params['-m'],
            'l_value': params['-l'],
            'input': params['-i'],
            'output': params['-o'],
            'threads': params['-n'],
            'seed': params['-s']
        })
        formatted_metric = format_metrics(metrics)
        writer.writerow(formatted_metric)

print("Benchmarking completed. Results saved to", output_csv)
