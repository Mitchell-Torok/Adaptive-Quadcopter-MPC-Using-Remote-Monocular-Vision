import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Plot settings
FIGURE_SIZE = (1, 3)
MAIN_TITLE_SIZE = 20
SUBPLOT_TITLE_SIZE = 22
AXIS_LABEL_SIZE = 20
TICK_LABEL_SIZE = 20
LEGEND_SIZE = 22
LINE_WIDTH = 2
GRID_ALPHA = 0.3

# Define the color palette
colors = ['#e97d00', '#008e00', '#0049bd', '#911515', '#000000', '#97c6d1', '#b697ff']

# Define the log file path (relative to this script's directory)
#log_file = '../../logs/ExperimentDataSets/Cinewhoop/z_sin_xy_adapt_4/log.csv'
#log_file = '../../logs/ExperimentDataSets/Cinewhoop/z_sin_no_xy_adapt_5/log.csv'

#log_file = '../../logs/ExperimentDataSets/Cinewhoop/circle_xy_adapt_3/log.csv'
#log_file = '../../logs/ExperimentDataSets/Cinewhoop/xyz_xy_adapt_1/log.csv'
#log_file = '../../logs/ExperimentDataSets/CinewhoopXYZ/log.csv'

log_file = '../../logs/ExperimentDataSets/Cinewhoop/xyz_212_test_2/log.csv'


# Create figure with 2 subplots
fig, axes = plt.subplots(2, 1, figsize=(FIGURE_SIZE[0] * 10, FIGURE_SIZE[1] * 2))
fig.subplots_adjust(hspace=0.3)

# Create separate figure for legend
fig_legend = plt.figure(figsize=(10, 0.5))

# Store handles and labels for legend
legend_handles = []
legend_labels = []

try:
    # Read the CSV file
    df = pd.read_csv(log_file)
    
    # Calculate time from timestamp
    time = df['timestamp'] - df['timestamp'].iloc[0]
    
    # Extract data for X and Y position
    x_position = df['ukf_pose_x']
    y_position = df['ukf_pose_y']
    x_desired = df['traj_x_ref']
    y_desired = df['traj_y_ref']
    
    # Calculate errors
    x_error = x_position - x_desired
    y_error = y_position - y_desired
    
    # Extract roll and pitch offset estimates
    roll_offset = df['est_param_fc_roll_offset_deg']
    pitch_offset = df['est_param_fc_pitch_offset_deg']
    
    # Plot 0: X and Y positions (actual and desired)
    line_x_actual, = axes[0].plot(time, x_position, color=colors[0], linewidth=LINE_WIDTH, label='X Position')
    line_x_desired, = axes[0].plot(time, x_desired, color=colors[0], linewidth=LINE_WIDTH, linestyle='--', alpha=0.7, label='X Desired')
    line_y_actual, = axes[0].plot(time, y_position, color=colors[1], linewidth=LINE_WIDTH, label='Y Position')
    line_y_desired, = axes[0].plot(time, y_desired, color=colors[1], linewidth=LINE_WIDTH, linestyle='--', alpha=0.7, label='Y Desired')
    
    # Plot 1: Roll and Pitch offset estimates
    line_roll, = axes[1].plot(time, roll_offset, color=colors[2], linewidth=LINE_WIDTH, label=r'Roll offset $\hat{\delta}_\phi$')
    line_pitch, = axes[1].plot(time, pitch_offset, color=colors[3], linewidth=LINE_WIDTH, label=r'Pitch offset $\hat{\delta}_\theta$')
    
    # Store for legend
    legend_handles = [line_x_actual, line_x_desired, line_y_actual, line_y_desired, line_roll, line_pitch]
    legend_labels = ['X position', 'X desired', 'Y position', 'Y desired', r'Roll offset $\hat{\delta}_\phi$', r'Pitch offset $\hat{\delta}_\theta$']
    
    # Reorder for 3 columns layout (matplotlib fills DOWN columns first)
    # Want: [X_pos, X_des, Pitch] [Y_pos, Y_des, Roll]
    # Need: [X_pos, Y_pos, X_des, Y_des, Pitch, Roll]
    legend_handles_ordered = [line_x_actual, line_y_actual, line_x_desired, line_y_desired, line_pitch, line_roll]
    legend_labels_ordered = ['X position', 'Y position', 'X desired', 'Y desired', r'Pitch offset $\hat{\delta}_\theta$', r'Roll offset $\hat{\delta}_\phi$']
    
except FileNotFoundError:
    print(f"Warning: File not found: {log_file}")
except Exception as e:
    print(f"Error processing {log_file}: {e}")

# Configure axis 0: X-Y Positions
axes[0].set_ylabel('Position (m)', fontsize=AXIS_LABEL_SIZE)
axes[0].set_xlabel('Time (s)', fontsize=AXIS_LABEL_SIZE)
axes[0].grid(True, alpha=GRID_ALPHA)
axes[0].set_title('(a) X-Y Position Tracking Performance', fontsize=SUBPLOT_TITLE_SIZE, fontweight='bold')
axes[0].tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)
axes[0].set_xlim(0, time.max())

# Configure axis 1: Offset Estimates
axes[1].set_ylabel(r'Offset ($^\circ$)', fontsize=AXIS_LABEL_SIZE)
axes[1].set_xlabel('Time (s)', fontsize=AXIS_LABEL_SIZE)
axes[1].grid(True, alpha=GRID_ALPHA)
axes[1].set_title('(b) Estimated Roll and Pitch Offsets', fontsize=SUBPLOT_TITLE_SIZE, fontweight='bold')
axes[1].tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)
axes[1].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
axes[1].set_xlim(0, time.max())

# Create legend in separate figure
# Layout with ncol=3, 2 rows:
# Row 1: X Position, X Desired, Pitch Offset
# Row 2: Y Position, Y Desired, Roll Offset
fig_legend.legend(legend_handles_ordered, legend_labels_ordered, 
                  loc='center', ncol=3, frameon=True, fontsize=LEGEND_SIZE, borderaxespad=0)
fig_legend.tight_layout()

# Adjust main figure layout
fig.tight_layout()

# Save figures as PDFs
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(os.path.dirname(script_dir), 'outputs')
os.makedirs(output_dir, exist_ok=True)
fig.savefig(os.path.join(output_dir, 'angle_offset_circle.pdf'), format='pdf', bbox_inches='tight')
fig_legend.savefig(os.path.join(output_dir, 'angle_offset_circle_legend.pdf'), format='pdf', bbox_inches='tight')
print(f"Saved figures to {script_dir}")

# Show both figures
plt.show()
