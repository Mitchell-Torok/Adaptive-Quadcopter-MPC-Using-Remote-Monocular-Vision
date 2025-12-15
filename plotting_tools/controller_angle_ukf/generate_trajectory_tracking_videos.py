import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, FFMpegWriter


# Plot settings
FIGURE_SIZE = (1, 3)
MAIN_TITLE_SIZE = 24
SUBPLOT_TITLE_SIZE = 26
AXIS_LABEL_SIZE = 24
TICK_LABEL_SIZE = 24
LEGEND_SIZE = 22
LINE_WIDTH = 2
GRID_ALPHA = 0.3

# Define the color palette
colors = ['#e97d00', '#008e00', '#0049bd', '#911515', '#000000', '#97c6d1', '#b697ff']


'''


DURATION = 30 
OUTPUT_PREFIX = 'ukf_z_sin' 
log_files = [
    '../../logs/ExperimentDataSets/Cinewhoop/z_sin_video/log.csv',
    '../../logs/ExperimentDataSets/Freestyle/z_sin_2/log.csv',
    '../../logs/ExperimentDataSets/TinyWhoop/z_sin_2/log.csv',
    '../../logs/ExperimentDataSets/TinyTrainer/z_sin_2/log.csv',
]

'''


DURATION = 70
OUTPUT_PREFIX = 'ukf_circle' 
log_files = [
    '../../logs/ExperimentDataSets/Cinewhoop/circle_video/log.csv',
    '../../logs/ExperimentDataSets/Freestyle/circle_3/log.csv',
    '../../logs/ExperimentDataSets/TinyWhoop/circle_3/log.csv',
    '../../logs/ExperimentDataSets/TinyTrainer/circle_2_video/log.csv',
]



'''
DURATION = -1
OUTPUT_PREFIX = 'ukf_xyz' 
log_files = [
    '../../logs/ExperimentDataSets/Cinewhoop/xyz_sine_video/log.csv',
    '../../logs/ExperimentDataSets/Freestyle/xyz_sine_2/log.csv',
    '../../logs/ExperimentDataSets/TinyWhoop/xyz_sine_3/log.csv',
    '../../logs/ExperimentDataSets/TinyTrainer/xyz_sine_2/log.csv',
]

'''



# Labels for the legend
labels = [ 'Cinewhoop', 'Freestyle', 'Tinywhoop', 'Tinytrainer']

# Animation settings
VIDEO_WIDTH = 3840  # Video width in pixels
PLOT_HEIGHT = 620  # Height for plots
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

# Create figure for plots (no legend)
fig_anim, axes_anim = plt.subplots(1, 2, figsize=(plot_width_inches, plot_height_inches), dpi=DPI)
fig_anim.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.12, wspace=0.15)

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
lines_thrust = []
lines_error = []

for idx, data in enumerate(all_data):
    line_error, = axes_anim[0].plot([], [], color=data['color'], linewidth=ANIM_LINE_WIDTH)
    line_thrust, = axes_anim[1].plot([], [], color=data['color'], linewidth=ANIM_LINE_WIDTH, 
                                     label=data['label'])
    
    lines_thrust.append(line_thrust)
    lines_error.append(line_error)

# Configure axes for animation - LEFT: Error, RIGHT: Thrust
axes_anim[0].set_ylabel('3D Position Error (m)', fontsize=ANIM_AXIS_LABEL_SIZE)
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

axes_anim[1].set_ylabel(r'Throttle Gain $\hat{k}_t$', fontsize=ANIM_AXIS_LABEL_SIZE)
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

# Animation update function
def update(frame):
    # Calculate current time: frame number / FPS = actual time in seconds
    current_time = frame / FPS
    
    for idx, data in enumerate(all_data):
        # Find indices up to current time
        time_mask = data['time'] <= current_time
        
        # Update line data (error on left [0], thrust on right [1])
        lines_error[idx].set_data(data['time'][time_mask], data['error_3d'][time_mask])
        lines_thrust[idx].set_data(data['time'][time_mask], data['thrust_estimate'][time_mask])
    
    return lines_error + lines_thrust

# Create animation
anim = FuncAnimation(fig_anim, update, frames=total_frames, interval=1000/FPS, blit=True)

# Save plot animation as MP4
plot_output_path = os.path.join(output_dir, f'{OUTPUT_PREFIX}_kt_and_error_plots.mp4')
writer = FFMpegWriter(fps=FPS, metadata=dict(artist='Matplotlib'), bitrate=5000)

print(f"Saving plot animation to {plot_output_path}...")
anim.save(plot_output_path, writer=writer, dpi=DPI)
print(f"Plot animation saved successfully!")

plt.close(fig_anim)

# ============================================================================
# Create Legend Video (static)
# ============================================================================

print("Creating legend video...")

# Create figure for legend only
fig_legend_anim = plt.figure(figsize=(legend_width_inches, legend_height_inches), dpi=DPI)
fig_legend_anim.patch.set_facecolor('white')

# Create legend handles (without desired height)
from matplotlib.lines import Line2D
legend_handles = []
legend_labels_list = []
for idx, label in enumerate(labels):
    line = Line2D([0], [0], color=colors[idx], linewidth=ANIM_LINE_WIDTH, label=label)
    legend_handles.append(line)
    legend_labels_list.append(label)

# Create legend centered in the figure
legend = fig_legend_anim.legend(legend_handles, legend_labels_list, 
                                loc='center', ncol=4, frameon=True, 
                                fontsize=ANIM_LEGEND_SIZE)

# Save legend as static video (all frames identical)
legend_output_path = os.path.join(script_dir, f'{OUTPUT_PREFIX}_thrust_legend.mp4')
print(f"Saving legend video to {legend_output_path}...")

# Create a simple animation that keeps the legend static
def update_legend(frame):
    return []

anim_legend = FuncAnimation(fig_legend_anim, update_legend, frames=total_frames, interval=1000/FPS, blit=True)
writer_legend = FFMpegWriter(fps=FPS, metadata=dict(artist='Matplotlib'), bitrate=1000)
anim_legend.save(legend_output_path, writer=writer_legend, dpi=DPI)
print(f"Legend video saved successfully!")

plt.close(fig_legend_anim)

# Save individual legend labels as separate PNGs
print("Saving individual legend labels as PNGs...")

for idx, (label_text, color_val) in enumerate(zip(legend_labels_list, [colors[i] for i in range(len(labels))])):
    # Create small figure for individual legend item
    fig_item = plt.figure(figsize=(3, 0.5), dpi=150)
    fig_item.patch.set_facecolor('white')
    
    # Regular colored line
    line_item = Line2D([0], [0], color=color_val, linewidth=ANIM_LINE_WIDTH)
    
    legend_item = fig_item.legend([line_item], [label_text], 
                                   loc='center', frameon=False, 
                                   fontsize=ANIM_LEGEND_SIZE)
    
    # Save individual legend item
    safe_filename = label_text.replace('$', '').replace('{', '').replace('}', '').replace(',', '_').replace(' ', '_').replace('=', '')
    item_path = os.path.join(output_dir, f'legend_{safe_filename}.png')
    fig_item.savefig(item_path, format='png', bbox_inches='tight', dpi=150, transparent=False, facecolor='white')
    plt.close(fig_item)
    print(f"  Saved: legend_{safe_filename}.png")

print("Individual legend labels saved!")

# ============================================================================
# Stitch videos together using ffmpeg
# ============================================================================

print("Stitching videos together...")

final_output_path = os.path.join(output_dir, f'{OUTPUT_PREFIX}_kt_and_error_animation.mp4')

# Use ffmpeg to stack videos vertically (plots on top, legend on bottom)
import subprocess
ffmpeg_cmd = [
    'ffmpeg', '-y',  # Overwrite output file
    '-i', plot_output_path,
    '-i', legend_output_path,
    '-filter_complex', '[0:v][1:v]vstack=inputs=2',
    '-c:v', 'libx264',
    '-preset', 'medium',
    '-crf', '18',
    final_output_path
]

try:
    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
    print(f"Final animation saved to {final_output_path}")
    
    # Clean up intermediate files
    import os
    os.remove(plot_output_path)
    os.remove(legend_output_path)
    print("Intermediate files cleaned up")
except subprocess.CalledProcessError as e:
    print(f"Error stitching videos: {e}")
    print(f"Intermediate files saved: {plot_output_path}, {legend_output_path}")
except FileNotFoundError:
    print("ffmpeg not found. Please install ffmpeg to stitch videos.")
    print(f"Intermediate files saved: {plot_output_path}, {legend_output_path}")

print("\n" + "="*80)
print("All plots and animations completed!")
print("="*80)