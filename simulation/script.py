import pandas as pd
import numpy as np
import plotly.graph_objects as go

data = pd.read_csv('../simulation/simplex_vertices.csv', header=None, comment='=')
data.columns = ['vertex', 'function_value', 'coord_1', 'coord_2', 'coord_3']

iteration_interval = 10
vertices_per_iteration = 4

initial_state = data.iloc[:vertices_per_iteration].copy()
initial_state['iteration'] = 0  # Add iteration index
initial_state['vertex_index'] = np.tile(np.arange(vertices_per_iteration), 1)

# Extract sampled data: every 10 iterations (4 rows per iteration)
sampled_indices = []
for i in range(0, len(data), vertices_per_iteration * iteration_interval):
    sampled_indices.extend(range(i, i + vertices_per_iteration))

sampled_data = data.iloc[sampled_indices].copy()
sampled_data['iteration'] = np.repeat(np.arange(0, len(sampled_data) // vertices_per_iteration), vertices_per_iteration) * iteration_interval
sampled_data['vertex_index'] = np.tile(np.arange(vertices_per_iteration), len(sampled_data) // vertices_per_iteration)

# Extract terminal state
terminal_state = data.iloc[-vertices_per_iteration:].copy()
terminal_state['iteration'] = (len(data) // vertices_per_iteration - 1)  # Add iteration index
terminal_state['vertex_index'] = np.tile(np.arange(vertices_per_iteration), 1)

# Print the number of samples taken
total_iterations = len(data) // vertices_per_iteration
num_samples = len(sampled_indices) // vertices_per_iteration

print(f'Total number of iterations: {total_iterations}')
print(f'Number of samples taken: {num_samples}')

# Combine initial, sampled, and terminal states for plotting
combined_data = pd.concat([initial_state, sampled_data, terminal_state])

# Plot 1: Movement of vertices over iterations with lines
fig_movement = go.Figure()

for vertex_index in combined_data['vertex_index'].unique():
    subset = combined_data[combined_data['vertex_index'] == vertex_index]
    fig_movement.add_trace(go.Scatter3d(
        x=subset['coord_1'],
        y=subset['coord_2'],
        z=subset['coord_3'],
        mode='markers+lines',
        marker=dict(
            size=5,
            opacity=0.8
        ),
        name=f'Vertex {vertex_index}'
    ))

fig_movement.update_layout(
    title='Movement of Simplex Vertices Over Iterations',
    scene=dict(
        xaxis_title='Coordinate 1',
        yaxis_title='Coordinate 2',
        zaxis_title='Coordinate 3'
    ),
    legend_title='Vertex Index',
    legend=dict(x=0.1, y=0.9)
)

# Plot 2: Only the dots representing the vertices
fig_dots = go.Figure()

fig_dots.add_trace(go.Scatter3d(
    x=combined_data['coord_1'],
    y=combined_data['coord_2'],
    z=combined_data['coord_3'],
    mode='markers',
    marker=dict(
        size=5,
        opacity=0.8,
        color=combined_data['vertex_index'],
        colorscale='Viridis'
    ),
    name='Vertices'
))

fig_dots.update_layout(
    title='Vertices of Simplex in 3D',
    scene=dict(
        xaxis_title='Coordinate 1',
        yaxis_title='Coordinate 2',
        zaxis_title='Coordinate 3'
    )
)

# Plot 3: Initial and terminal states showing the movement with lines
initial_and_terminal = pd.concat([initial_state, terminal_state])
fig_initial_terminal = go.Figure()

for vertex_index in initial_and_terminal['vertex_index'].unique():
    subset = initial_and_terminal[initial_and_terminal['vertex_index'] == vertex_index]
    fig_initial_terminal.add_trace(go.Scatter3d(
        x=subset['coord_1'],
        y=subset['coord_2'],
        z=subset['coord_3'],
        mode='markers+lines',
        marker=dict(
            size=5,
            opacity=0.8
        ),
        name=f'Vertex {vertex_index}'
    ))

fig_initial_terminal.update_layout(
    title='Initial and Terminal States of Simplex Vertices',
    scene=dict(
        xaxis_title='Coordinate 1',
        yaxis_title='Coordinate 2',
        zaxis_title='Coordinate 3'
    ),
    legend_title='Vertex Index',
    legend=dict(x=0.1, y=0.9)
)

# Save the plots as separate HTML files
fig_movement.write_html('../simulation/simplex_vertices_movement.html')
fig_dots.write_html('../simulation/simplex_vertices_dots.html')
fig_initial_terminal.write_html('../simulation/simplex_vertices_initial_terminal.html')

# Show the plots
fig_movement.show()
fig_dots.show()
fig_initial_terminal.show()

print("\nInitial State:\n")
print(initial_state)

print("\nTerminal State:\n")
print(terminal_state)
