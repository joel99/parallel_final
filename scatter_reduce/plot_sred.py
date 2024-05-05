#%%
from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns

ghc_result = pd.read_csv('scatter_reduce/ghc_benchmark_results.csv')

serial_df = ghc_result[ghc_result['mode'] == 'S']
# print(serial_df)
# Compute normative times for input/output pairs
# print(serial_df.groupby('input')['time'].transform(lambda x: x.mean()))
mean_times = serial_df.groupby(['input', 'output'])['time'].mean().reset_index()
mean_times.rename(columns={'time': 'avg_time'}, inplace=True)
print(mean_times)
ghc_result = ghc_result.merge(mean_times, on=['input', 'output'], how='left')
ghc_result['speedup'] = ghc_result['avg_time'] / ghc_result['time']
print(ghc_result.head())

#%%
# plot serial times
serial_df = ghc_result[ghc_result['mode'] == 'S']
sns.set_style("whitegrid")
sns.despine()
palette = sns.color_palette("flare", ghc_result['threads'].nunique())
f = plt.figure(figsize=(4, 4))
sns.lineplot(data=serial_df, x='input', y='time', hue='output', ax=f.gca())
# change legend title to "Scatter Output"
plt.legend(title='Scatter Output')

# set labels to "Input Size" and "Time (s)"
plt.xlabel('Input Size')
plt.ylabel('Time (s)')
plt.title('Serial Execution Time (1D)')
#%%
lock_df = ghc_result[ghc_result['mode'] == 'L']
print(lock_df)
sns.set_style("whitegrid")
sns.despine()
# facet by l_value
palette = sns.color_palette("flare", lock_df['threads'].nunique())
g = sns.FacetGrid(lock_df, row='l_value', col='output', hue='threads', palette=palette)
g.map(sns.lineplot, 'input', 'speedup')
g.set_axis_labels('Input Size', 'Speedup')
g.set_titles('Locks: {row_name} Output Size: {col_name}')
# show legend
g.add_legend()

#%%
reduct_df = ghc_result[ghc_result['mode'] == 'R']
f = plt.figure(figsize=(6, 6))
# ax = sns.scatterplot(data=reduct_df, x='input', y='output', hue='speedup', palette=palette, ax=f.gca())
g = sns.FacetGrid(reduct_df, col='output', hue='threads', palette=palette, aspect=0.6)
g.map(sns.lineplot, 'input', 'speedup')
# labels
sns.set_style("whitegrid")
sns.despine()
g.set_axis_labels('In', 'Speedup')
g.set_titles('Out: {col_name}')
g.add_legend(title='Threads')
#%%
reduct_df = ghc_result[ghc_result['mode'] == 'M']
f = plt.figure(figsize=(6, 6))
# ax = sns.scatterplot(data=reduct_df, x='input', y='output', hue='speedup', palette=palette, ax=f.gca())
g = sns.FacetGrid(reduct_df, col='output', hue='threads', palette=palette, aspect=0.6)
g.map(sns.lineplot, 'input', 'speedup')
# labels
sns.set_style("whitegrid")
sns.despine()
g.set_axis_labels('Input Size', 'Speedup')
g.set_titles('Output Size: {col_name}')
g.add_legend()

#%%
# mode = M and mode = R have identical parameter ranges, compare them

comp_df = ghc_result[ghc_result['mode'].isin(['M', 'R'])]
pivot_df = comp_df.pivot_table(index=['input', 'output', 'threads'], columns='mode', values='speedup').reset_index()

# Drop any rows that do not have both 'M' and 'R' values to ensure direct comparability
pivot_df = pivot_df.dropna(subset=['M', 'R'])
plt.figure(figsize=(6, 6))
sns.scatterplot(data=pivot_df, x='R', y='M', hue='threads', palette='viridis', s=100)

# Enhance the plot
plt.title('(1D) Speedup: Manual vs OMP Reduce')
plt.xlabel('Speedup - Manual')
plt.ylabel('Speedup - OMP Reduce')
plt.axline((1, 1), slope=1, color='red', linestyle='--')  # Adding a diagonal line for reference
# match axes
plt.rcParams.update({'font.size': 16})
plt.gca().set_aspect('equal', adjustable='box')
# match axis ticks
plt.locator_params(axis='x', nbins=5)
plt.locator_params(axis='y', nbins=5)
plt.xlim(0, 4)
plt.ylim(0, 4)
plt.legend(title='Threads')
# Show the plot
plt.grid(True)
plt.show()