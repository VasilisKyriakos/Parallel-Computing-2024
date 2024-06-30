import json
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Number of runs for averaging
num_runs = 5

# Executable commands
executables = {
    'seq': './multistart_mds_seq',
    'openmp': './multistart_mds_omp 16',
    'openmp_tasks': './multistart_mds_omp_tasks 8',
    'mpi': 'mpirun -np 8 ./multistart_mds_mpi'
}

# File paths for results
result_files = {
    'seq': 'results_seq.json',
    'openmp': 'results_openmp.json',
    'openmp_tasks': 'results_openmp_tasks.json',
    'mpi': 'results_mpi.json'
}

# Function to clear previous results
def clear_results_files():
    for file in result_files.values():
        open(file, 'w').close()

# Function to run executable and collect results
def run_executable(executable, result_file, num_runs):
    results = []
    for _ in range(num_runs):
        subprocess.run(executable.split())
        with open(result_file, 'r') as f:
            data = json.load(f)
            results.extend(data)
    return results

# Clear previous results
clear_results_files()

# Run each executable and collect results
all_results = {version: run_executable(exec_cmd, result_files[version], num_runs) for version, exec_cmd in executables.items()}

# Save results to JSON files
for version, results in all_results.items():
    with open(result_files[version], 'w') as f:
        json.dump(results, f, indent=2)

# Read and process results
all_results = {}
for version, result_file in result_files.items():
    with open(result_file, 'r') as f:
        results = json.load(f)
        all_results[version] = results

# Calculate averages
averaged_results = []
for version, results in all_results.items():
    df = pd.DataFrame(results)
    numeric_cols = df.select_dtypes(include=np.number).columns
    averaged_result = df[numeric_cols].mean().to_dict()
    averaged_result['version'] = version
    averaged_result['best_pt'] = df['best_pt'].tolist()  # Keep the list of best points as is
    averaged_results.append(averaged_result)

# Create DataFrame from averaged results
df = pd.DataFrame(averaged_results)
df['version'] = pd.Categorical(df['version'], categories=['seq', 'openmp', 'openmp_tasks', 'mpi'], ordered=True)
df = df.sort_values('version')

# Calculate speedup and efficiency
df['speedup'] = df.loc[df['version'] == 'seq', 'elapsed_time'].values[0] / df['elapsed_time']
df['efficiency'] = df['speedup'] / 4  # Assuming 4 threads or processors for simplicity

# Set up the Seaborn style
sns.set(style="whitegrid")

# Helper function to add annotations
def add_annotations(ax, data, y_field):
    for line in range(0, data.shape[0]):
        ax.annotate(f'{data[y_field].iloc[line]:.3f}' if y_field != 'best_fx' else f'{data[y_field].iloc[line]:.7f}', 
                    (line, data[y_field].iloc[line]), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center',
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

# Create subplots
fig, axs = plt.subplots(1, 3, figsize=(21, 7))

# Plot execution times
ax1 = sns.lineplot(data=df, x='version', y='elapsed_time', marker='o', color='blue', ax=axs[0])
add_annotations(ax1, df, 'elapsed_time')
axs[0].set_xlabel('Version', fontsize=12, fontweight='bold')
axs[0].set_ylabel('Time (s)', fontsize=12, fontweight='bold')
axs[0].set_title('Execution Time', fontsize=14, fontweight='bold')

# Plot efficiency
ax2 = sns.lineplot(data=df, x='version', y='efficiency', marker='o', color='orange', ax=axs[1])
add_annotations(ax2, df, 'efficiency')
axs[1].set_xlabel('Version', fontsize=12, fontweight='bold')
axs[1].set_ylabel('Efficiency', fontsize=12, fontweight='bold')
axs[1].set_title('Efficiency', fontsize=14, fontweight='bold')

# Plot best_fx
ax3 = sns.lineplot(data=df, x='version', y='best_fx', marker='o', color='green', ax=axs[2])
add_annotations(ax3, df, 'best_fx')
axs[2].set_xlabel('Version', fontsize=12, fontweight='bold')
axs[2].set_ylabel('Best f(x)', fontsize=12, fontweight='bold')
axs[2].set_title('Best f(x)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()


