r"""
    Compare OMP settings on subset_100.txt
    # 5 letter ex
    ./out -f allowed_words.txt -t subset_100.txt -n 12 -x 'c' -c 6
"""
import subprocess
import csv
from tqdm import tqdm
import pandas as pd

# Configuration settings
thread_range = [1, 2, 4, 8]
# seed = [0, 1, 2]

specific_params = []

specific_params.append(
    {'-x': 'g', '-c': 1} # guess parallel, reduction_scatter_reduce_row
)
specific_params.append(
    {'-x': 'g', '-c': 1, '-b': ''}
)
specific_params.append(
    {'-x': 'h', '-c': 6}
)
specific_params.append(
    {'-x': 'h', '-c': 6, '-b': ''}
)
specific_params.append(
    {'-x': 'c', '-c': 1},
)
# Cross product specific with common parameters
exp_params = []

for param in specific_params:
    for n in thread_range:
        for legal_file in ['allowed_words.txt']:
            for test_file in ['subset_1.txt']:
            # for test_file in ['subset_100.txt']:
                combo = param.copy()  # Start with the specific settings
                combo.update({'-n': n, '-t': test_file, '-f': legal_file})  # Add common settings
                exp_params.append(combo)
executable_path = './out'
output_csv = 'benchmark_results_main.csv'

# Function to run the benchmark and capture output
def run_benchmark(params):
    command = [executable_path]
    for key, value in params.items():
        command.append(key)
        if bool(value):
            command.append(str(value))
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout

# Parse`` the output to extract metrics
# Unlike in sim, each run will generate many words this time
def parse_output(output):
    lines = output.split('\n')
    metrics = []
    word_ctr = 0
    for line in lines:
        if 'Inner Time' in line:
            metrics.append({
                'round': -1, # total
                'time': float(line.split(': ')[1]),
                'word': word_ctr
            })
        if 'Round Time' in line:
            metrics.append({
                'round': int(line.split()[1]),
                'time': float(line.split(': ')[1]),
                'word': word_ctr
            })
    return metrics

# Write to CSV
all_metrics = []
for params in (pbar := tqdm(exp_params)):
    print(params)
    pbar.set_postfix_str(params)
    output = run_benchmark(params)
    metrics = parse_output(output)
    for k in metrics:
        k.update({
            'mode': params['-x'],
            'rebuild': '-b' in params,
            'threads': params['-n'],
            'capacity': params.get('-c', 0)    
        })
    all_metrics.extend(metrics)
pd.DataFrame(all_metrics).to_csv(output_csv, index=False)

print("Benchmarking completed. Results saved to", output_csv)
