import subprocess
import csv
import math
# Configuration settings
# Common variables
# input_pow = range(12, 16, 4)
input_pow = range(12, 32, 4)
input_range = [int(math.pow(2, i)) for i in input_pow]
output_pow = [5]
# output_pow = [5, 6, 7, 8]
output_range = [int(math.pow(3, j)) for j in output_pow]
thread_range = [4]
# thread_range = [1, 2, 4, 8, 16]

specific_params = []
# mode = 'L' specific
for l in [16, 64]:
    specific_params.append(
        {'-l': l, '-m': 'L'}
    )

# mode = 'F'
specific_params.append(
    {'-m': 'F', '-l': 0}
)

# mode = 'R'
specific_params.append(
    {'-m': 'R', '-l': 0}
)
specific_params.append(
    {'-m': 'M', '-l': 0} # OMP native
)

# Cross product specific with common parameters
# TODO
exp_params = []
for param in specific_params:
    for i in input_range:
        for o in output_range:
            for n in thread_range:
                combo = param.copy()  # Start with the specific settings
                combo.update({'-i': i, '-o': o, '-n': n})  # Add common settings
                exp_params.append(combo)
                
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
        if 'serial computation time' in line:
            metrics['serial_time'] = float(line.split(': ')[1])
        elif 'parallel computation time' in line:
            metrics['parallel_time'] = float(line.split(': ')[1])
        elif 'Parallel Speedup' in line:
            metrics['speedup'] = float(line.split(':')[1])
    return metrics

# Write to CSV
with open(output_csv, 'w', newline='') as file:
    fieldnames = ['serial_time', 'parallel_time', 'speedup', 'mode', 'l_value', 'input', 'output', 'threads']
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
            'threads': params['-n']
        })
        writer.writerow(metrics)

print("Benchmarking completed. Results saved to", output_csv)
