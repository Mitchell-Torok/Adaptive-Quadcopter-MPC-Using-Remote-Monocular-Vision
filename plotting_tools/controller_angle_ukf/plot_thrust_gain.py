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
LEGEND_SIZE = 22
LINE_WIDTH = 2
GRID_ALPHA = 0.3

# Define the color palette
colors = ['#e97d00', '#008e00', '#0049bd', '#911515', '#000000', '#97c6d1', '#b697ff']

# Define the log file paths (relative to this script's directory)
log_files = [
    '../../logs/ExperimentDataSets/Cinewhoop/20/log.csv',
    '../../logs/ExperimentDataSets/Cinewhoop/24/log.csv',
    '../../logs/ExperimentDataSets/Cinewhoop/28/log.csv',
    '../../logs/ExperimentDataSets/Cinewhoop/32/log.csv',
]

# Labels for the legend
labels = [ r'$k_{t,0} = 20$', r'$k_{t,0} = 24$', r'$k_{t,0} = 28$', r'$k_{t,0} = 32$']

# Create figure with 2 subplots
fig, axes = plt.subplots(2, 1, figsize=(FIGURE_SIZE[0] * 10, FIGURE_SIZE[1] * 2))
fig.subplots_adjust(hspace=0.3)

# Create separate figure for legend
fig_legend = plt.figure(figsize=(10, 0.5))

# Store handles and labels for legend
legend_handles = []
legend_labels = []

# Read and plot data from each log file
for idx, (log_file, label) in enumerate(zip(log_files, labels)):
    try:
        # Read the CSV file
        df = pd.read_csv(log_file)
        
        # Calculate time from timestamp
        time = df['timestamp'] - df['timestamp'].iloc[0]
        
        # Filter data to first 48 seconds
        mask = time <= 60
        time = time[mask]
        
        # Extract data
        z_position = df['ukf_pose_z'][mask]
        z_desired = df['traj_z_ref'][mask]
        thrust_estimate = df['est_param_thrust_ratio'][mask]
        
        color = colors[idx]
        
        # Plot 1: Z position and desired position
        line1, = axes[0].plot(time, z_position, color=color, linewidth=LINE_WIDTH, label=label)
        axes[0].plot(time, z_desired, color='#000000', linewidth=LINE_WIDTH, linestyle='--', alpha=0.7)
        
        # Plot 2: Estimated thrust value
        axes[1].plot(time, thrust_estimate, color=color, linewidth=LINE_WIDTH, label=label)
        
        # Store for legend
        legend_handles.append(line1)
        legend_labels.append(label)
        
    except FileNotFoundError:
        print(f"Warning: File not found: {log_file}")
    except Exception as e:
        print(f"Error processing {log_file}: {e}")

# Add desired trajectory to legend last
from matplotlib.lines import Line2D
desired_line = Line2D([0], [0], color='#000000', linewidth=LINE_WIDTH, linestyle='--', alpha=0.7)

# Configure axis 0: Z Position
axes[0].set_ylabel('Altitude (m)', fontsize=AXIS_LABEL_SIZE)
axes[0].grid(True, alpha=GRID_ALPHA)
axes[0].set_title('(a) Altitude Tracking Performance', fontsize=SUBPLOT_TITLE_SIZE, fontweight='bold')
axes[0].set_xlim(0, 45)
axes[0].set_xticks(np.arange(0, 50, 10))
axes[0].tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)

# Configure axis 1: Thrust Estimate
axes[1].set_ylabel(r'Throttle gain $\hat{k}_t$', fontsize=AXIS_LABEL_SIZE)
axes[1].set_xlabel('Time (s)', fontsize=AXIS_LABEL_SIZE)
axes[1].grid(True, alpha=GRID_ALPHA)
axes[1].set_title(r'(b) Estimated Throttle Gain $\hat{k}_t$', fontsize=SUBPLOT_TITLE_SIZE, fontweight='bold')
axes[1].set_xlim(0, 45)
axes[1].set_xticks(np.arange(0, 50, 10))
axes[1].tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)

# Create legend in separate figure
all_handles = legend_handles + [desired_line]
all_labels = legend_labels + ['Desired height']

fig_legend.legend(all_handles, all_labels, 
                  loc='center', ncol=3, frameon=True, fontsize=LEGEND_SIZE, borderaxespad=0)
fig_legend.tight_layout()

# Adjust main figure layout
fig.tight_layout()

# Save figures as PDFs
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(os.path.dirname(script_dir), 'outputs')
os.makedirs(output_dir, exist_ok=True)
fig.savefig(os.path.join(output_dir, 'throttle_gain_comparison.pdf'), format='pdf', bbox_inches='tight')
fig_legend.savefig(os.path.join(output_dir, 'throttle_gain_comparison_legend.pdf'), format='pdf', bbox_inches='tight')
print(f"Saved figures to {script_dir}")

# Show both figures
plt.show()

# ============================================================================
# Animation Section - Generate MP4 synchronized to actual timestamps
# ============================================================================

print("Creating animation...")

# Animation settings
VIDEO_WIDTH = 3840  # Video width in pixels
PLOT_HEIGHT = 620  # Height for plots
LEGEND_HEIGHT = 100  # Height for legend
DPI = 100  # Dots per inch for the figure

# Calculate figure sizes in inches
plot_width_inches = VIDEO_WIDTH / DPI
plot_height_inches = PLOT_HEIGHT / DPI
legend_width_inches = VIDEO_WIDTH / DPI
legend_height_inches = LEGEND_HEIGHT / DPI

# Scale text sizes for the wide aspect ratio
ANIM_AXIS_LABEL_SIZE = 20
ANIM_TICK_LABEL_SIZE = 20
ANIM_TITLE_SIZE = 20
ANIM_LEGEND_SIZE = 32  # Reduced from 40 to create buffer space
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
        mask = time <= 60
        all_data.append({
            'time': time[mask].values,
            'z_position': df['ukf_pose_z'][mask].values,
            'z_desired': df['traj_z_ref'][mask].values,
            'thrust_estimate': df['est_param_thrust_ratio'][mask].values,
            'color': colors[idx],
            'label': label
        })
        max_time = max(max_time, time[mask].values[-1])
    except Exception as e:
        print(f"Error loading data for animation: {e}")

# Animation parameters - sync to real time
FPS = 30  # Frames per second for video
total_frames = int(max_time * FPS)  # Total frames = actual duration * FPS

print(f"Animation duration: {max_time:.2f} seconds")
print(f"Total frames: {total_frames}")

# Initialize line objects for each dataset
lines_z = []
lines_z_des = []
lines_thrust = []
legend_lines = []  # For legend handles

for idx, data in enumerate(all_data):
    line_z, = axes_anim[0].plot([], [], color=data['color'], linewidth=ANIM_LINE_WIDTH, label=data['label'])
    line_z_des, = axes_anim[0].plot([], [], color='#000000', linewidth=ANIM_LINE_WIDTH, linestyle='--', alpha=0.7)
    line_thrust, = axes_anim[1].plot([], [], color=data['color'], linewidth=ANIM_LINE_WIDTH, label=data['label'])
    
    lines_z.append(line_z)
    lines_z_des.append(line_z_des)
    lines_thrust.append(line_thrust)
    legend_lines.append(line_z)  # Add to legend handles

# Configure axes for animation
axes_anim[0].set_ylabel('Altitude (m)', fontsize=ANIM_AXIS_LABEL_SIZE)
axes_anim[0].set_xlabel('Time (s)', fontsize=ANIM_AXIS_LABEL_SIZE)
axes_anim[0].grid(True, alpha=GRID_ALPHA)
axes_anim[0].set_title('Altitude Tracking Performance', fontsize=ANIM_TITLE_SIZE, fontweight='bold', pad=15)
axes_anim[0].set_xlim(0, 45)
# Determine y-limits from data
all_z = np.concatenate([data['z_position'] for data in all_data] + [data['z_desired'] for data in all_data])
z_min, z_max = all_z.min(), all_z.max()
z_margin = (z_max - z_min) * 0.1
axes_anim[0].set_ylim(z_min - z_margin, z_max + z_margin)
axes_anim[0].set_xticks(np.arange(0, 50, 10))
axes_anim[0].tick_params(axis='both', which='major', labelsize=ANIM_TICK_LABEL_SIZE)

axes_anim[1].set_ylabel(r'Throttle Gain $\hat{k}_t$', fontsize=ANIM_AXIS_LABEL_SIZE)
axes_anim[1].set_xlabel('Time (s)', fontsize=ANIM_AXIS_LABEL_SIZE)
axes_anim[1].grid(True, alpha=GRID_ALPHA)
axes_anim[1].set_title(r'Throttle gain $\hat{k}_t$', fontsize=ANIM_TITLE_SIZE, fontweight='bold', pad=15)
axes_anim[1].set_xlim(0, 45)
# Determine y-limits from data
all_thrust = np.concatenate([data['thrust_estimate'] for data in all_data])
thrust_min, thrust_max = all_thrust.min(), all_thrust.max()
thrust_margin = (thrust_max - thrust_min) * 0.1
axes_anim[1].set_ylim(thrust_min - thrust_margin, thrust_max + thrust_margin)
axes_anim[1].set_xticks(np.arange(0, 50, 10))
axes_anim[1].tick_params(axis='both', which='major', labelsize=ANIM_TICK_LABEL_SIZE)

# Animation update function
def update(frame):
    # Calculate current time: frame number / FPS = actual time in seconds
    current_time = frame / FPS
    
    for idx, data in enumerate(all_data):
        # Find indices up to current time
        time_mask = data['time'] <= current_time
        
        # Update line data
        lines_z[idx].set_data(data['time'][time_mask], data['z_position'][time_mask])
        # Always show full desired trajectory (not animated)
        lines_z_des[idx].set_data(data['time'], data['z_desired'])
        lines_thrust[idx].set_data(data['time'][time_mask], data['thrust_estimate'][time_mask])
    
    return lines_z + lines_z_des + lines_thrust

# Create animation
anim = FuncAnimation(fig_anim, update, frames=total_frames, interval=1000/FPS, blit=True)

# Save plot animation as MP4
plot_output_path = os.path.join(script_dir, 'throttle_gain_plots.mp4')
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

# Create legend handles
from matplotlib.lines import Line2D
legend_handles = []
legend_labels_list = []
for idx, label in enumerate(labels):
    line = Line2D([0], [0], color=colors[idx], linewidth=ANIM_LINE_WIDTH, label=label)
    legend_handles.append(line)
    legend_labels_list.append(label)

# Add desired trajectory
desired_line = Line2D([0], [0], color='#000000', linewidth=ANIM_LINE_WIDTH, linestyle='--', alpha=0.7)
legend_handles.append(desired_line)
legend_labels_list.append('Desired Height')

# Create legend centered in the figure
legend = fig_legend_anim.legend(legend_handles, legend_labels_list, 
                                loc='center', ncol=5, frameon=True, 
                                fontsize=ANIM_LEGEND_SIZE)

# Save legend as static video (all frames identical)
legend_output_path = os.path.join(script_dir, 'throttle_gain_legend.mp4')
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

for idx, (label_text, color_val) in enumerate(zip(legend_labels_list, [colors[i] for i in range(len(labels))] + ['#000000'])):
    # Create small figure for individual legend item
    fig_item = plt.figure(figsize=(3, 0.5), dpi=150)
    fig_item.patch.set_facecolor('white')
    
    if idx < len(labels):
        # Regular colored line
        line_item = Line2D([0], [0], color=color_val, linewidth=ANIM_LINE_WIDTH)
    else:
        # Dashed line for desired trajectory
        line_item = Line2D([0], [0], color=color_val, linewidth=ANIM_LINE_WIDTH, linestyle='--', alpha=0.7)
    
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

final_output_path = os.path.join(output_dir, 'throttle_gain_animation.mp4')

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