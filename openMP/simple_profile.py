#%%
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

log_file = 'openMP/basic_profile.txt'
log_file_rebuild = 'openMP/basic_profile_rebuild.txt'
# Define a function to parse the log trace
def parse_log(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]
    print(lines)
    data = []
    inner_times = []
    scatter_times = []
    entropy_times = []
    rebuild_times = []
    round_times = []
    words = []
    for line in lines:
        if "Benchmarking word:" in line:
            words.append(line.split()[-1])
        elif "Inner Time" in line:
            inner_times.append(float(line.split()[-1]))
        elif "Scatter Time" in line:
            scatter_times.append(float(line.split()[-1]))
        elif "Entropy Time" in line:
            entropy_times.append(float(line.split()[-1]))
        elif "Rebuild Time" in line:
            rebuild_times.append(float(line.split()[-1]))
        elif "Round Time" in line:
            round_num = int(line.split()[1])
            round_time = float(line.split()[-1])
            round_times.append({'round': round_num, 'time': round_time, 'word': len(words) - 1})
    # Calculate inner times as the sum of scatter, entropy, and rebuild times
    return words, inner_times, round_times, scatter_times, entropy_times, rebuild_times

# Parse the log trace
words, inner_times, round_times, scatter_times, entropy_times, rebuild_times = parse_log(log_file)
words_rebuild, inner_times_rebuild, round_times_rebuild, scatter_times_rebuild, entropy_times_rebuild, rebuild_times_rebuild = parse_log(log_file_rebuild)
# Create a DataFrame
df = pd.DataFrame({
    'Word': words,
    'Inner Time': inner_times,
    # 'Round Time': round_times,
    'Scatter Time': scatter_times,
    'Entropy Time': entropy_times,
    'Update Time': rebuild_times,
    'Rebuild': False
})
# delete inner time
df_rebuild = pd.DataFrame({
    'Word': words_rebuild,
    'Inner Time': inner_times_rebuild,
    # 'Round Time': round_times,
    'Scatter Time': scatter_times_rebuild,
    'Entropy Time': entropy_times_rebuild,
    'Update Time': rebuild_times_rebuild,
    'Rebuild': True
})
df = pd.concat([df, df_rebuild])
df = df.drop(columns=['Inner Time'])
df_melted = pd.melt(df, id_vars=['Rebuild'], value_vars=['Scatter Time', 'Entropy Time', 'Update Time'], var_name='Component', value_name='Time')
# Display the melted DataFrame
print(df_melted)
#%%
# # Create the plot using Seaborn
ax = sns.barplot(data=df_melted, x='Component', y='Time', hue='Rebuild')
ax.set_yscale('log')
ax.set_ylabel('Time (s)')
# large font, labels
fontsize = 20
plt.rcParams.update({'font.size': fontsize, 'axes.labelsize': fontsize, 'xtick.labelsize': fontsize, 'ytick.labelsize': fontsize, 'legend.fontsize': fontsize, 'figure.titlesize': fontsize, 'legend.title_fontsize': fontsize})
plt.xticks(rotation=15)
ax.set_xlabel('')
# ax.set_title('Serial Workload Profile', fontsize=fontsize)

#%%
# Make new df based on round times
df_round_times = pd.DataFrame(round_times)
df_round_times['Rebuild'] = False
df_round_times_rebuild = pd.DataFrame(round_times_rebuild)
df_round_times_rebuild['Rebuild'] = True
df_round = pd.concat([df_round_times, df_round_times_rebuild])

ax = sns.lineplot(data=df_round, x='round', y='time', hue='Rebuild')
ax.set_xlabel('Round')
ax.set_ylabel('Time (s)')
ax.set_yscale('log')