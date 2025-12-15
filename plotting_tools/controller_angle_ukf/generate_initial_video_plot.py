import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter


# Plot settings
FIGURE_SIZE = (1, 3)
MAIN_TITLE_SIZE = 20
SUBPLOT_TITLE_SIZE = 22
AXIS_LABEL_SIZE = 20
TICK_LABEL_SIZE = 20
LEGEND_SIZE = 18
LINE_WIDTH = 2
GRID_ALPHA = 0.3

# Define the color palette
colors = ['#e97d00', '#008e00', '#0049bd', '#911515', '#000000', '#97c6d1', '#b697ff']



OUTPUT_PREFIX = 'xyz_212_video' 
DURATION = 70  # Plot duration in seconds
log_files = [
    '../../logs/ExperimentDataSets/Cinewhoop/xyz_212_test_2/log.csv',
]






# Labels for the legend
labels = [ 'Cinewhoop']

# Animation settings
VIDEO_WIDTH = 3840/3  # Video width in pixels
PLOT_HEIGHT = 2160/2  # Height for plots
LEGEND_HEIGHT = 100  # Height for legend
DPI = 100  # Dots per inch for the figure

import os
script_dir = os.path.dirname(os.path.abspath(__file__))

# ============================================================================
# 3D Position Error Analysis Section
# ============================================================================

print("\n" + "="*80)
print("Creating 3D Position Error Analysis...")
print("="*80)

# Create figure for 3D error plot
fig_error, ax_error = plt.subplots(1, 1, figsize=(FIGURE_SIZE[0] * 10, FIGURE_SIZE[1] * 1))

# Store error data and RMSE values
error_data = []
rmse_values = []

# Calculate and plot 3D position errors for each dataset
for idx, (log_file, label) in enumerate(zip(log_files, labels)):
    try:
        df = pd.read_csv(log_file)
        time = df['timestamp'] - df['timestamp'].iloc[0]
        
        # Filter data based on DURATION (-1 means plot entire file)
        if DURATION == -1:
            mask = time >= 0  # Include all data
        else:
            mask = time <= DURATION
        
        time_filtered = time[mask]
        
        # Extract actual and desired positions
        x_actual = df['ukf_pose_x'][mask].values
        y_actual = df['ukf_pose_y'][mask].values
        z_actual = df['ukf_pose_z'][mask].values
        
        x_desired = df['traj_x_ref'][mask].values
        y_desired = df['traj_y_ref'][mask].values
        z_desired = df['traj_z_ref'][mask].values
        
        # Calculate 3D Euclidean error
        error_3d = np.sqrt(
            (x_actual - x_desired)**2 + 
            (y_actual - y_desired)**2 + 
            (z_actual - z_desired)**2
        )
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean(error_3d**2))
        rmse_values.append((label, rmse))
        
        # Store for animation
        error_data.append({
            'time': time_filtered,
            'error_3d': error_3d,
            'color': colors[idx],
            'label': label,
            'rmse': rmse
        })
        
        # Plot error
        ax_error.plot(time_filtered, error_3d, color=colors[idx], 
                     linewidth=LINE_WIDTH, label=f'{label} (RMSE: {rmse:.3f} m)')
        
        print(f"{label:15s} - RMSE: {rmse:.4f} m")
        
    except Exception as e:
        print(f"Error processing {log_file}: {e}")

# Configure error plot
ax_error.set_ylabel('3D Position Error (m)', fontsize=AXIS_LABEL_SIZE)
ax_error.set_xlabel('Time (s)', fontsize=AXIS_LABEL_SIZE)
ax_error.grid(True, alpha=GRID_ALPHA)
ax_error.set_title('3D Position Tracking Error', fontsize=SUBPLOT_TITLE_SIZE, fontweight='bold')
# Set x-axis limits based on actual data duration
max_error_time = max([data['time'].max() for data in error_data])
ax_error.set_xlim(0, max_error_time)
ax_error.set_xticks(np.arange(0, max_error_time + 10, 10))
ax_error.tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)

# Save error plot
fig_error.tight_layout()
output_dir = os.path.join(os.path.dirname(script_dir), 'outputs')
os.makedirs(output_dir, exist_ok=True)
error_plot_path = os.path.join(output_dir, f'{OUTPUT_PREFIX}_3d_position_error.pdf')
fig_error.savefig(error_plot_path, format='pdf', bbox_inches='tight')
print(f"\nSaved 3D error plot to {error_plot_path}")

plt.show()

print("Creating animation...")

# Calculate figure sizes in inches
plot_width_inches = VIDEO_WIDTH / DPI
plot_height_inches = PLOT_HEIGHT / DPI
legend_width_inches = VIDEO_WIDTH / DPI
legend_height_inches = LEGEND_HEIGHT / DPI

# Scale text sizes for the wide aspect ratio
ANIM_AXIS_LABEL_SIZE = 24
ANIM_TICK_LABEL_SIZE = 22
ANIM_TITLE_SIZE = 24
ANIM_LEGEND_SIZE = 32
ANIM_LINE_WIDTH = 2.5

print(f"Plot video dimensions: {VIDEO_WIDTH} x {PLOT_HEIGHT} pixels at {DPI} DPI")
print(f"Legend video dimensions: {VIDEO_WIDTH} x {LEGEND_HEIGHT} pixels at {DPI} DPI")

# Create figure for plots (no legend) - 3 rows, 1 column for vertical stacking
fig_anim, axes_anim = plt.subplots(3, 1, figsize=(plot_width_inches, plot_height_inches), dpi=DPI)
fig_anim.subplots_adjust(left=0.12, right=0.92, top=0.95, bottom=0.08, hspace=0.65)

# Store all data for animation
all_data = []
max_time = 0
for idx, (log_file, label) in enumerate(zip(log_files, labels)):
    try:
        df = pd.read_csv(log_file)
        time = df['timestamp'] - df['timestamp'].iloc[0]
        
        # Filter data based on DURATION (-1 means plot entire file)
        if DURATION == -1:
            mask = time >= 0  # Include all data
        else:
            mask = time <= DURATION
        
        time_filtered = time[mask].values
        
        # Skip if no data after filtering
        if len(time_filtered) == 0:
            print(f"Warning: No data for {label} after time filtering")
            continue
        
        # Extract actual and desired positions for 3D error
        x_actual = df['ukf_pose_x'][mask].values
        y_actual = df['ukf_pose_y'][mask].values
        z_actual = df['ukf_pose_z'][mask].values
        
        x_desired = df['traj_x_ref'][mask].values
        y_desired = df['traj_y_ref'][mask].values
        z_desired = df['traj_z_ref'][mask].values
        
        # Calculate 3D Euclidean error
        error_3d = np.sqrt(
            (x_actual - x_desired)**2 + 
            (y_actual - y_desired)**2 + 
            (z_actual - z_desired)**2
        )
        
        # Calculate RMSE
        rmse = np.sqrt(np.mean(error_3d**2))
        
        all_data.append({
            'time': time_filtered,
            'thrust_estimate': df['est_param_thrust_ratio'][mask].values,
            'roll_offset': df['est_param_fc_roll_offset_deg'][mask].values,
            'pitch_offset': df['est_param_fc_pitch_offset_deg'][mask].values,
            'error_3d': error_3d,
            'rmse': rmse,
            'color': colors[idx],
            'label': label
        })
        max_time = max(max_time, time_filtered[-1])
    except Exception as e:
        print(f"Error loading data for animation from {log_file}: {e}")

# Check if we have any data
if not all_data:
    print("Error: No valid data loaded for animation!")
    import sys
    sys.exit(1)

# Animation parameters - sync to real time
FPS = 30  # Frames per second for video
total_frames = int(max_time * FPS)  # Total frames = actual duration * FPS

print(f"Animation duration: {max_time:.2f} seconds")
print(f"Total frames: {total_frames}")
print(f"Loaded data for {len(all_data)} datasets")

# Initialize line objects for each dataset
lines_error = []
lines_thrust = []
lines_roll = []
lines_pitch = []

for idx, data in enumerate(all_data):
    line_error, = axes_anim[0].plot([], [], color=data['color'], linewidth=ANIM_LINE_WIDTH,
                                     label=data['label'])
    line_thrust, = axes_anim[1].plot([], [], color=colors[1], linewidth=ANIM_LINE_WIDTH, 
                                     label=data['label'])
    line_roll, = axes_anim[2].plot([], [], color=colors[2], linewidth=ANIM_LINE_WIDTH,
                                    label='Roll offset')
    line_pitch, = axes_anim[2].plot([], [], color=colors[3], linewidth=ANIM_LINE_WIDTH,
                                     label='Pitch offset')
    
    lines_error.append(line_error)
    lines_thrust.append(line_thrust)
    lines_roll.append(line_roll)
    lines_pitch.append(line_pitch)

# Configure axes for animation - TOP: Error, BOTTOM: Thrust
axes_anim[0].set_ylabel('Error (m)', fontsize=ANIM_AXIS_LABEL_SIZE)
axes_anim[0].set_xlabel('Time (s)', fontsize=ANIM_AXIS_LABEL_SIZE)
axes_anim[0].grid(True, alpha=GRID_ALPHA)
axes_anim[0].set_title('3D Position Tracking Error', fontsize=ANIM_TITLE_SIZE, fontweight='bold', pad=15)
axes_anim[0].set_xlim(0, max_time)
axes_anim[0].set_xticks(np.arange(0, max_time + 10, 10))
axes_anim[0].tick_params(axis='both', which='major', labelsize=ANIM_TICK_LABEL_SIZE)
# Determine y-limits from error data
all_errors = np.concatenate([data['error_3d'] for data in all_data])
error_min, error_max = 0, all_errors.max()
error_margin = error_max * 0.1
axes_anim[0].set_ylim(error_min, error_max + error_margin)

axes_anim[1].set_ylabel(r'Throttle gain $\hat{k}_t$', fontsize=ANIM_AXIS_LABEL_SIZE)
axes_anim[1].set_xlabel('Time (s)', fontsize=ANIM_AXIS_LABEL_SIZE)
axes_anim[1].grid(True, alpha=GRID_ALPHA)
axes_anim[1].set_title(r'Estimated Throttle Gain $\hat{k}_t$', fontsize=ANIM_TITLE_SIZE, fontweight='bold', pad=15)
axes_anim[1].set_xlim(0, max_time)
axes_anim[1].set_xticks(np.arange(0, max_time + 10, 10))
axes_anim[1].tick_params(axis='both', which='major', labelsize=ANIM_TICK_LABEL_SIZE)
# Determine y-limits from data
all_thrust = np.concatenate([data['thrust_estimate'] for data in all_data])
thrust_min, thrust_max = all_thrust.min(), all_thrust.max()
thrust_margin = (thrust_max - thrust_min) * 0.1
axes_anim[1].set_ylim(thrust_min - thrust_margin, thrust_max + thrust_margin)

axes_anim[2].set_ylabel(r'Offset ($^\circ$)', fontsize=ANIM_AXIS_LABEL_SIZE)
axes_anim[2].set_xlabel('Time (s)', fontsize=ANIM_AXIS_LABEL_SIZE)
axes_anim[2].grid(True, alpha=GRID_ALPHA)
axes_anim[2].set_title('Estimated Flight Controller Roll and Pitch Offsets', fontsize=ANIM_TITLE_SIZE, fontweight='bold', pad=15)
axes_anim[2].set_xlim(0, max_time)
axes_anim[2].set_xticks(np.arange(0, max_time + 10, 10))
axes_anim[2].tick_params(axis='both', which='major', labelsize=ANIM_TICK_LABEL_SIZE)
axes_anim[2].legend(loc='upper right', fontsize=ANIM_LEGEND_SIZE-8)
axes_anim[2].axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.5)
# Determine y-limits from data
all_roll = np.concatenate([data['roll_offset'] for data in all_data])
all_pitch = np.concatenate([data['pitch_offset'] for data in all_data])
offset_min = min(all_roll.min(), all_pitch.min())
offset_max = max(all_roll.max(), all_pitch.max())
offset_margin = (offset_max - offset_min) * 0.1
axes_anim[2].set_ylim(offset_min - offset_margin, offset_max + offset_margin)

# Animation update function
def update(frame):
    # Calculate current time: frame number / FPS = actual time in seconds
    current_time = frame / FPS
    
    for idx, data in enumerate(all_data):
        # Find indices up to current time
        time_mask = data['time'] <= current_time
        
        # Update line data (error on top [0], thrust middle [1], offsets bottom [2])
        lines_error[idx].set_data(data['time'][time_mask], data['error_3d'][time_mask])
        lines_thrust[idx].set_data(data['time'][time_mask], data['thrust_estimate'][time_mask])
        lines_roll[idx].set_data(data['time'][time_mask], data['roll_offset'][time_mask])
        lines_pitch[idx].set_data(data['time'][time_mask], data['pitch_offset'][time_mask])
    
    return lines_error + lines_thrust + lines_roll + lines_pitch

# Initialize first frame
update(0)

# Save first frame as PNG
first_frame_path = os.path.join(output_dir, f'{OUTPUT_PREFIX}_first_frame.png')
fig_anim.savefig(first_frame_path, format='png', dpi=DPI, bbox_inches='tight')
print(f"Saved first frame to {first_frame_path}")

# Display the first frame
plt.show()

# Create animation
anim = FuncAnimation(fig_anim, update, frames=total_frames, interval=1000/FPS, blit=True)

# Save plot animation as MP4 with final name
final_output_path = os.path.join(output_dir, f'{OUTPUT_PREFIX}.mp4')
writer = FFMpegWriter(fps=FPS, metadata=dict(artist='Matplotlib'), bitrate=5000)

print(f"Saving animation to {final_output_path}...")
anim.save(final_output_path, writer=writer, dpi=DPI)
print(f"Animation saved successfully to {final_output_path}!")

plt.close(fig_anim)

print("\n" + "="*80)
print("All plots and animations completed!")
print("="*80)