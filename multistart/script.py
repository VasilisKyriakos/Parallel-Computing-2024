import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV file
data = pd.read_csv('simplex_vertices.csv', header=None, comment='=')
data.columns = ['vertex', 'function_value'] + [f'coord_{i}' for i in range(1, data.shape[1] - 1)]

# Extract unique iterations based on '=='
iterations = data[data['vertex'] == '='].index

# Define pairs of dimensions to plot (e.g., 4D means you have (1,2), (1,3), (1,4), (2,3), (2,4), (3,4))
dimension_pairs = [(1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)]

for i in range(len(iterations) - 1):
    iteration_data = data.iloc[iterations[i] + 1:iterations[i + 1]]
    for (dim1, dim2) in dimension_pairs:
        plt.figure(figsize=(8, 6))
        plt.scatter(iteration_data[f'coord_{dim1}'], iteration_data[f'coord_{dim2}'], c=iteration_data['function_value'], cmap='viridis')
        plt.colorbar(label='Function Value')
        plt.title(f'Iteration {i} - Dimension {dim1} vs Dimension {dim2}')
        plt.xlabel(f'Coordinate {dim1}')
        plt.ylabel(f'Coordinate {dim2}')
        plt.show()
