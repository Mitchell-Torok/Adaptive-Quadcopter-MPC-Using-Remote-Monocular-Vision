import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
df = pd.read_csv('../../logs/ExperimentDataSets/Cinewhoop/circle_video/log.csv')

# Replace these with your actual column names if different
axis_columns = ['MPC_setup_time', 'MPC_solve_time', 'Visualisation_time', 'UKF_update_time']

# Calculate statistics for MPC (setup + solve)
mpc_total_time = df['MPC_setup_time'] + df['MPC_solve_time']
mpc_mean = mpc_total_time.mean()
mpc_std = mpc_total_time.std()

# Calculate statistics for UKF
ukf_mean = df['UKF_update_time'].mean()
ukf_std = df['UKF_update_time'].std()

# Print statistics
print("=" * 50)
print("Timing Statistics")
print("=" * 50)
print(f"MPC Total (Setup + Solve):")
print(f"  Mean: {mpc_mean:.6f} s ({mpc_mean*1000:.4f} ms)")
print(f"  Std:  {mpc_std:.6f} s ({mpc_std*1000:.4f} ms)")
print()
print(f"UKF Update:")
print(f"  Mean: {ukf_mean:.6f} s ({ukf_mean*1000:.4f} ms)")
print(f"  Std:  {ukf_std:.6f} s ({ukf_std*1000:.4f} ms)")
print("=" * 50)

# Use a sample index or timestamp for x-axis
if 'timestamp' in df.columns:
    x = df['timestamp']
    xlabel = 'Timestamp'
elif 'sample' in df.columns:
    x = df['sample']
    xlabel = 'Sample'
else:
    x = df.index
    xlabel = 'Sample Index'

plt.figure(figsize=(10, 6))
for axis in axis_columns:
    plt.plot(x, df[axis], label=axis)

plt.xlabel(xlabel)
plt.ylabel('Time Taken (s)')
plt.title('Controller Run Time on 4 Axes')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()