#%%
from matplotlib import pyplot as plt
from pathlib import Path
import pandas as pd
import seaborn as sns

ghc_result = pd.read_csv('openMP/ghc_benchmark_results_main.csv')

serial_df = ghc_result[ghc_result['mode'] == 's']
mean_times = serial_df.groupby(['round', 'rebuild'])['time'].mean().reset_index()
mean_times.rename(columns={'time': 'avg_time'}, inplace=True)
ghc_result = ghc_result.merge(mean_times, on=['round', 'rebuild'], how='left')
ghc_result['speedup'] = ghc_result['avg_time'] / ghc_result['time']
print(ghc_result.head())

#%%
# plot serial times
serial_df = ghc_result[ghc_result['mode'] == 'g']
# serial_df = ghc_result[ghc_result['mode'] == 's']
serial_df = serial_df[serial_df['round'] != -1]
print(serial_df)
sns.set_style("whitegrid")
sns.despine()
f = plt.figure(figsize=(4, 4))
plt.rcParams.update({'font.size': 16})
sns.lineplot(data=serial_df, x='round', y='time', hue='rebuild', ax=f.gca())
# change legend title to "Scatter Output"
plt.legend(title='Rebuild')
# set labels to "Input Size" and "Time (s)"
plt.xlabel('Round')
# only whole number xticks
plt.locator_params(axis='x', nbins=3)
plt.ylabel('Time (s)')
plt.title('Serial Time / Word')

#%%
net_df = ghc_result[ghc_result['round'] == -1]
print(net_df)
sns.set_style("whitegrid")
sns.despine()
# capacity plots
# palette = sns.color_palette("flare", net_df['capacity'].nunique())
g = sns.FacetGrid(net_df, col='rebuild', hue='mode')
# ax = sns.lineplot(data=net_df, x='threads', y='speedup', hue='mode')
g.map(sns.lineplot, 'threads', 'speedup')
# g.map(sns.lineplot, 'threads', 'time')
g.set_axis_labels('Threads', 'Speedup')
g.set_titles('Rebuild: {col_name}')
# plot unity line
plt.rcParams.update({'font.size': 16})
plt.locator_params(axis='x', nbins=5)
plt.locator_params(axis='y', nbins=5)
# ax.legend(title='Mode')
# ax.set_xlabel('Threads')
# ax.set_ylabel('Speedup')
# show legend
# g.add_legend(title='Mode')# , labels=['Across', None, 'Hybrid', None, 'Within', None, 'Serial', None])
# After all your plotting code
axes = g.axes.flatten()  # Get the array of axes in the FacetGrid
axes[0].legend(title='Mode')
handles, labels = g.axes.flat[0].get_legend_handles_labels()
axes[0].legend().remove()
g.add_legend(handles=handles[:4], labels=['Across', 'Hybrid', 'Within', 'Serial'], title='Mode')
g.map(plt.axline, xy1=(1,1), xy2=(8,8), color='black', linestyle='--', alpha=0.2)
#%%
palette = sns.color_palette("flare", ghc_result['threads'].nunique())
# no_rebuild_df = ghc_result[(ghc_result['rebuild'] == False) & (ghc_result['round'] != -1)]
no_rebuild_df = ghc_result[(ghc_result['rebuild'] == True) & (ghc_result['round'] != -1)]
g = sns.FacetGrid(no_rebuild_df, col='mode', hue='threads', palette=palette)
g.map(sns.lineplot, 'round', 'time')
g.set_axis_labels('Round', 'Time')
# log scale
g.set(yscale='log')
g.set_titles('Mode: {col_name}')
axes = g.axes.flatten()  # Get the array of axes in the FacetGrid
axes[0].set_title('Mode: Across')
axes[1].set_title('Mode: Hybrid')
axes[2].set_title('Mode: Serial')
# Hybrid parallelism conveys no gains in sufficiently small rounds over serial