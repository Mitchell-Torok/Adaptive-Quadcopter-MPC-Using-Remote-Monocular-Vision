


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D

FIGURE_SIZE = (1, 3)
MAIN_TITLE_SIZE = 15
SUBPLOT_TITLE_SIZE = 16
AXIS_LABEL_SIZE = 15
TICK_LABEL_SIZE = 15
LEGEND_SIZE = 16
LINE_WIDTH = 2
GRID_ALPHA = 0.3

colors = ['#e97d00', '#008e00', '#0049bd', '#911515', '#000000', '#97c6d1', '#b697ff']

CAMERA_ELEVATION = 25
CAMERA_AZIMUTH = 45
CAMERA_ROLL = 0
START_TIME_3D = 35.0
END_TIME_3D = 70.0

DURATION = 70
OUTPUT_PREFIX = 'ukf_circle' 
log_files = [
    '../../logs/ExperimentDataSets/Cinewhoop/circle_video/log.csv',
    '../../logs/ExperimentDataSets/Freestyle/circle_3/log.csv',
    '../../logs/ExperimentDataSets/TinyWhoop/circle_3/log.csv',
    '../../logs/ExperimentDataSets/TinyTrainer/circle_2_video/log.csv',
]

labels = ['Cinewhoop', 'Freestyle', 'Tinywhoop', 'Tinytrainer']

fig = plt.figure(figsize=(FIGURE_SIZE[0] * 10, FIGURE_SIZE[1] * 4))
gs_main = fig.add_gridspec(2, 1, hspace=0.2, height_ratios=[1.6, 1])

gs_3d = gs_main[0].subgridspec(1, 3, wspace=0.1, width_ratios=[0.1, 0.8, 0.1])
ax_3d = fig.add_subplot(gs_3d[1], projection='3d')

gs_2d_container = gs_main[1].subgridspec(1, 3, wspace=0.1, width_ratios=[0.1, 0.8, 0.1])
gs_2d = gs_2d_container[1].subgridspec(2, 1, hspace=1.0)
axes = [fig.add_subplot(gs_2d[0]), fig.add_subplot(gs_2d[1])]

all_x, all_y, all_z = [], [], []
desired_plotted = False

print("Plotting 3D trajectories...")
for idx, (log_file, label) in enumerate(zip(log_files, labels)):
    try:
        df = pd.read_csv(log_file)
        time = df['timestamp'] - df['timestamp'].iloc[0]
        mask = (time >= START_TIME_3D) & (time <= END_TIME_3D)
        time_filtered = time[mask]
        
        if len(time_filtered) == 0:
            print(f"Warning: No 3D data in time range for {label}")
            continue
        
        x_actual = df['ukf_pose_x'][mask].values
        y_actual = df['ukf_pose_y'][mask].values
        z_actual = df['ukf_pose_z'][mask].values
        
        x_desired = df['traj_x_ref'][mask].values
        y_desired = df['traj_y_ref'][mask].values
        z_desired = df['traj_z_ref'][mask].values
        
        color = colors[idx]
        ax_3d.plot(x_actual, y_actual, z_actual, color=color, linewidth=LINE_WIDTH, label=f'{label}', linestyle='-')
        
        if not desired_plotted:
            ax_3d.plot(x_desired, y_desired, z_desired, color='black', linewidth=LINE_WIDTH*0.7, 
                       label='Reference Trajectory', linestyle='--', alpha=0.8)
            desired_plotted = True
        
        all_x.extend(x_actual)
        all_x.extend(x_desired)
        all_y.extend(y_actual)
        all_y.extend(y_desired)
        all_z.extend(z_actual)
        all_z.extend(z_desired)
        
        print(f"  {label}: {len(time_filtered)} points")
    except FileNotFoundError:
        print(f"Warning: File not found: {log_file}")
    except Exception as e:
        print(f"Error: {e}")

if all_x and all_y and all_z:
    max_range = np.array([max(all_x) - min(all_x), max(all_y) - min(all_y), 
                          max(all_z) - min(all_z)]).max() / 2.0
    mid_x = (max(all_x) + min(all_x)) * 0.5
    mid_y = (max(all_y) + min(all_y)) * 0.5
    mid_z = (max(all_z) + min(all_z)) * 0.5
    ax_3d.set_xlim(mid_x - max_range, mid_x + max_range)
    ax_3d.set_ylim(mid_y - max_range, mid_y + max_range)
    ax_3d.set_zlim(mid_z - max_range, mid_z + max_range)

ax_3d.view_init(elev=CAMERA_ELEVATION, azim=CAMERA_AZIMUTH, roll=CAMERA_ROLL)
ax_3d.set_xlabel('X Position (m)', fontsize=AXIS_LABEL_SIZE, labelpad=15)
ax_3d.set_ylabel('Y Position (m)', fontsize=AXIS_LABEL_SIZE, labelpad=15)
ax_3d.set_zlabel('Z Position (m)', fontsize=AXIS_LABEL_SIZE, labelpad=15)
ax_3d.set_title(f'(a) 3D Pose Trajectories', fontsize=SUBPLOT_TITLE_SIZE, fontweight='bold', y=0.98)
ax_3d.tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)
ax_3d.grid(True, alpha=GRID_ALPHA)

fig_legend = plt.figure(figsize=(10, 0.5))
legend_handles = []
legend_labels = []

print("Plotting tracking error and thrust estimates...")
for idx, (log_file, label) in enumerate(zip(log_files, labels)):
    try:
        df = pd.read_csv(log_file)
        time = df['timestamp'] - df['timestamp'].iloc[0]
        mask = time <= DURATION
        time = time[mask]
        
        x_actual = df['ukf_pose_x'][mask].values
        y_actual = df['ukf_pose_y'][mask].values
        z_actual = df['ukf_pose_z'][mask].values
        x_desired = df['traj_x_ref'][mask].values
        y_desired = df['traj_y_ref'][mask].values
        z_desired = df['traj_z_ref'][mask].values
        
        tracking_error_3d = np.sqrt((x_actual - x_desired)**2 + (y_actual - y_desired)**2 + (z_actual - z_desired)**2)
        thrust_estimate = df['est_param_thrust_ratio'][mask]
        color = colors[idx]
        
        line1, = axes[0].plot(time, tracking_error_3d, color=color, linewidth=LINE_WIDTH, label=label)
        axes[1].plot(time, thrust_estimate, color=color, linewidth=LINE_WIDTH, label=label)
        legend_handles.append(line1)
        legend_labels.append(label)
    except FileNotFoundError:
        print(f"Warning: File not found: {log_file}")
    except Exception as e:
        print(f"Error: {e}")

desired_line = Line2D([0], [0], color='black', linewidth=LINE_WIDTH*0.7, linestyle='--', alpha=0.8)
legend_handles.append(desired_line)
legend_labels.append('Reference trajectory')

axes[0].set_ylabel('Error (m)', fontsize=AXIS_LABEL_SIZE)
axes[0].grid(True, alpha=GRID_ALPHA)
axes[0].set_title('(b) 3D Position Tracking Error', fontsize=SUBPLOT_TITLE_SIZE, fontweight='bold')
axes[0].set_xlim(0, DURATION)
axes[0].set_xticks(np.arange(0, DURATION + 10, 20))
axes[0].tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)

axes[1].set_ylabel(r'Throttle gain $\hat{k}_t$', fontsize=AXIS_LABEL_SIZE)
axes[1].set_xlabel('Time (s)', fontsize=AXIS_LABEL_SIZE)
axes[1].grid(True, alpha=GRID_ALPHA)
axes[1].set_title(r'(c) Estimated Throttle Gain $\hat{k}_t$', fontsize=SUBPLOT_TITLE_SIZE, fontweight='bold')
axes[1].set_xlim(0, DURATION)
axes[1].set_xticks(np.arange(0, DURATION + 10, 20))
axes[1].tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE)

fig_legend.legend(legend_handles, legend_labels, loc='center', ncol=3, frameon=True, 
                  fontsize=LEGEND_SIZE, borderaxespad=0)
fig_legend.tight_layout()
fig.tight_layout()

script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(os.path.dirname(script_dir), 'outputs')
os.makedirs(output_dir, exist_ok=True)
fig.savefig(os.path.join(output_dir, 'trajectory_tracking_comparison.pdf'), format='pdf', bbox_inches='tight')
fig_legend.savefig(os.path.join(output_dir, 'trajectory_tracking_comparison_legend.pdf'), format='pdf', bbox_inches='tight')
print(f"Saved figures to {output_dir}")

plt.show()