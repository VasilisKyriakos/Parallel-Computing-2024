import json
import subprocess
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Number of runs for averaging
num_runs = 1

# Number of threads/processes to test
thread_counts = [4, 8, 12, 16]

# Executable commands templates
executables = {
    'openmp': './multistart_mds_omp {threads}',
    'openmp_tasks': './multistart_mds_omp_tasks {threads}',
    'mpi': 'mpirun -n {threads} ./multistart_mds_mpi'
}

# File paths for results
result_files = {
    'openmp': 'results_openmp.json',
    'openmp_tasks': 'results_openmp_tasks.json',
    'mpi': 'results_mpi.json'
}

# Function to run executable and collect results
def run_executable(executable, result_file, num_runs):
    results = []
    for _ in range(num_runs):
        open(result_file, 'w').close()
        subprocess.run(executable.split())
        with open(result_file, 'r') as f:
            data = json.load(f)
            results.extend(data)
    return results

# Run each executable with different thread counts and collect results
all_results = {}
for version, exec_template in executables.items():
    for threads in thread_counts:
        # Skip MPI if threads > 8
        if version == 'mpi' and threads > 8:
            continue
        exec_cmd = exec_template.format(threads=threads)
        result_file = result_files[version]
        results = run_executable(exec_cmd, result_file, num_runs)
        
        # Save results to JSON files
        file_key = f"{version}_{threads}"
        with open(f'results_{file_key}.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        all_results[file_key] = results

# Read and process results
processed_results = []
for version, exec_template in executables.items():
    for threads in thread_counts:
        # Skip MPI if threads > 8
        if version == 'mpi' and threads > 8:
            continue
        file_key = f"{version}_{threads}"
        result_file = f'results_{file_key}.json'
        with open(result_file, 'r') as f:
            results = json.load(f)
            df = pd.DataFrame(results)
            numeric_cols = df.select_dtypes(include=np.number).columns
            averaged_result = df[numeric_cols].mean().to_dict()
            averaged_result['version'] = version
            averaged_result['threads'] = threads
            averaged_result['best_pt'] = df['best_pt'].tolist() 
            processed_results.append(averaged_result)

# Create DataFrame from processed results
df = pd.DataFrame(processed_results)
df['version'] = pd.Categorical(df['version'], categories=['openmp', 'openmp_tasks', 'mpi'], ordered=True)
df = df.sort_values(['version', 'threads'])

# Calculate speedup and efficiency
for version in executables.keys():
    baseline_time = df[(df['version'] == version) & (df['threads'] == 4)]['elapsed_time'].values[0]
    df.loc[df['version'] == version, 'speedup'] = baseline_time / df[df['version'] == version]['elapsed_time']
    df.loc[df['version'] == version, 'efficiency'] = df.loc[df['version'] == version, 'speedup'] / df.loc[df['version'] == version, 'threads']

# Set up the Seaborn style
sns.set(style="whitegrid")

# Helper function to add annotations
def add_annotations(ax, data, y_field):
    for line in range(0, data.shape[0]):
        ax.annotate(f'{data[y_field].iloc[line]:.3f}' if y_field != 'best_fx' else f'{data[y_field].iloc[line]:.7f}', 
                    (data['threads'].iloc[line], data[y_field].iloc[line]), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center',
                    fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="white"))

# Plot for OpenMP
df_openmp = df[df['version'] == 'openmp']
fig, axs = plt.subplots(1, 4, figsize=(28, 7))

# Plot execution times
ax1 = sns.lineplot(data=df_openmp, x='threads', y='elapsed_time', marker='o', ax=axs[0])
add_annotations(ax1, df_openmp, 'elapsed_time')
axs[0].set_xlabel('Threads', fontsize=12, fontweight='bold')
axs[0].set_ylabel('Time (s)', fontsize=12, fontweight='bold')
axs[0].set_title('OpenMP Execution Time', fontsize=14, fontweight='bold')

# Plot speedup
ax2 = sns.lineplot(data=df_openmp, x='threads', y='speedup', marker='o', ax=axs[1])
add_annotations(ax2, df_openmp, 'speedup')
axs[1].set_xlabel('Threads', fontsize=12, fontweight='bold')
axs[1].set_ylabel('Speedup', fontsize=12, fontweight='bold')
axs[1].set_title('OpenMP Speedup', fontsize=14, fontweight='bold')

# Plot efficiency
ax3 = sns.lineplot(data=df_openmp, x='threads', y='efficiency', marker='o', ax=axs[2])
add_annotations(ax3, df_openmp, 'efficiency')
axs[2].set_xlabel('Threads', fontsize=12, fontweight='bold')
axs[2].set_ylabel('Efficiency', fontsize=12, fontweight='bold')
axs[2].set_title('OpenMP Efficiency', fontsize=14, fontweight='bold')

# Plot best_fx
ax4 = sns.lineplot(data=df_openmp, x='threads', y='best_fx', marker='o', ax=axs[3])
add_annotations(ax4, df_openmp, 'best_fx')
axs[3].set_xlabel('Threads', fontsize=12, fontweight='bold')
axs[3].set_ylabel('Best f(x)', fontsize=12, fontweight='bold')
axs[3].set_title('OpenMP Best f(x)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# Plot for OpenMP tasks
df_openmp_tasks = df[df['version'] == 'openmp_tasks']
fig, axs = plt.subplots(1, 4, figsize=(28, 7))

# Plot execution times
ax1 = sns.lineplot(data=df_openmp_tasks, x='threads', y='elapsed_time', marker='o', ax=axs[0])
add_annotations(ax1, df_openmp_tasks, 'elapsed_time')
axs[0].set_xlabel('Threads', fontsize=12, fontweight='bold')
axs[0].set_ylabel('Time (s)', fontsize=12, fontweight='bold')
axs[0].set_title('OpenMP Tasks Execution Time', fontsize=14, fontweight='bold')

# Plot speedup
ax2 = sns.lineplot(data=df_openmp_tasks, x='threads', y='speedup', marker='o', ax=axs[1])
add_annotations(ax2, df_openmp_tasks, 'speedup')
axs[1].set_xlabel('Threads', fontsize=12, fontweight='bold')
axs[1].set_ylabel('Speedup', fontsize=12, fontweight='bold')
axs[1].set_title('OpenMP Tasks Speedup', fontsize=14, fontweight='bold')

# Plot efficiency
ax3 = sns.lineplot(data=df_openmp_tasks, x='threads', y='efficiency', marker='o', ax=axs[2])
add_annotations(ax3, df_openmp_tasks, 'efficiency')
axs[2].set_xlabel('Threads', fontsize=12, fontweight='bold')
axs[2].set_ylabel('Efficiency', fontsize=12, fontweight='bold')
axs[2].set_title('OpenMP Tasks Efficiency', fontsize=14, fontweight='bold')

# Plot best_fx
ax4 = sns.lineplot(data=df_openmp_tasks, x='threads', y='best_fx', marker='o', ax=axs[3])
add_annotations(ax4, df_openmp_tasks, 'best_fx')
axs[3].set_xlabel('Threads', fontsize=12, fontweight='bold')
axs[3].set_ylabel('Best f(x)', fontsize=12, fontweight='bold')
axs[3].set_title('OpenMP Tasks Best f(x)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()

# Plot for MPI
df_mpi = df[df['version'] == 'mpi']
fig, axs = plt.subplots(1, 4, figsize=(28, 7))

# Plot execution times
ax1 = sns.lineplot(data=df_mpi, x='threads', y='elapsed_time', marker='o', ax=axs[0])
add_annotations(ax1, df_mpi, 'elapsed_time')
axs[0].set_xlabel('Threads', fontsize=12, fontweight='bold')
axs[0].set_ylabel('Time (s)', fontsize=12, fontweight='bold')
axs[0].set_title('MPI Execution Time', fontsize=14, fontweight='bold')

# Plot speedup
ax2 = sns.lineplot(data=df_mpi, x='threads', y='speedup', marker='o', ax=axs[1])
add_annotations(ax2, df_mpi, 'speedup')
axs[1].set_xlabel('Threads', fontsize=12, fontweight='bold')
axs[1].set_ylabel('Speedup', fontsize=12, fontweight='bold')
axs[1].set_title('MPI Speedup', fontsize=14, fontweight='bold')

# Plot efficiency
ax3 = sns.lineplot(data=df_mpi, x='threads', y='efficiency', marker='o', ax=axs[2])
add_annotations(ax3, df_mpi, 'efficiency')
axs[2].set_xlabel('Threads', fontsize=12, fontweight='bold')
axs[2].set_ylabel('Efficiency', fontsize=12, fontweight='bold')
axs[2].set_title('MPI Efficiency', fontsize=14, fontweight='bold')

# Plot best_fx
ax4 = sns.lineplot(data=df_mpi, x='threads', y='best_fx', marker='o', ax=axs[3])
add_annotations(ax4, df_mpi, 'best_fx')
axs[3].set_xlabel('Threads', fontsize=12, fontweight='bold')
axs[3].set_ylabel('Best f(x)', fontsize=12, fontweight='bold')
axs[3].set_title('MPI Best f(x)', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.show()
