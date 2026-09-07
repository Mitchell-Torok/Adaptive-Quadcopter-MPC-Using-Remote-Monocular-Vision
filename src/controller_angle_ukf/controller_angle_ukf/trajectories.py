import numpy as np
from scipy.spatial.transform import Rotation as R

def takeoff_trajectory(dt):

    steps_takeoff = int(4 / dt)  # 4 seconds of takeoff
    steps_hover = int(4 / dt)    # 4 seconds of hover

    # Takeoff phase
    time_space_takeoff = np.linspace(0, steps_takeoff * dt, steps_takeoff)
    x_traj_takeoff = np.zeros_like(time_space_takeoff)
    y_traj_takeoff = np.zeros_like(time_space_takeoff)
    z_traj_takeoff = np.linspace(0.0, 1.0, steps_takeoff)

    # Hover phase
    time_space_hover = np.linspace(0, steps_hover * dt, steps_hover)
    x_traj_hover = np.zeros_like(time_space_hover)
    y_traj_hover = np.zeros_like(time_space_hover)
    z_traj_hover = np.ones_like(time_space_hover) * 1.0

    # Concatenate phases
    x_traj = np.concatenate((x_traj_takeoff, x_traj_hover))
    y_traj = np.concatenate((y_traj_takeoff, y_traj_hover))
    z_traj = np.concatenate((z_traj_takeoff, z_traj_hover))
    time_space_total = np.concatenate((time_space_takeoff, time_space_hover))

    roll_traj = np.zeros_like(time_space_total)
    pitch_traj = np.zeros_like(time_space_total)
    yaw_traj = np.zeros_like(time_space_total)
    rpy_traj = np.vstack((roll_traj, pitch_traj, yaw_traj)).T
    quaternions = R.from_euler('xyz', rpy_traj).as_quat()  # Convert to quaternions
    qx_traj = quaternions[:, 0]
    qy_traj = quaternions[:, 1]
    qz_traj = quaternions[:, 2]
    qw_traj = quaternions[:, 3]
    vx_traj = np.gradient(x_traj, dt)
    vy_traj = np.gradient(y_traj, dt)
    vz_traj = np.gradient(z_traj, dt)
    ax_traj = np.zeros_like(time_space_total)
    ay_traj = np.zeros_like(time_space_total)
    az_traj = np.zeros_like(time_space_total)
    u1 = np.zeros_like(time_space_total)
    u2 = np.zeros_like(time_space_total)
    u3 = np.zeros_like(time_space_total)
    u4 = np.zeros_like(time_space_total)
    traj = np.array([x_traj, y_traj, z_traj, qw_traj, qx_traj, qy_traj, qz_traj,
                     vx_traj, vy_traj, vz_traj, ax_traj, ay_traj, az_traj, u1, u2, u3, u4])
    return traj, "takeoff"
    

def land_trajectory(dt, init_pose):
    steps_move_back = int(5 / dt)  # 5 seconds of move back
    steps_descend = int(5 / dt)  # 5 seconds of descend
    steps_ground = int(3 / dt)  # 3 seconds on ground
    time_space_move_back = np.linspace(0, steps_move_back * dt, steps_move_back)
    time_space_descend = np.linspace(0, steps_descend * dt, steps_descend)
    time_space_ground = np.linspace(0, steps_ground * dt, steps_ground)

    time_space_total = np.concatenate((time_space_move_back, time_space_descend, time_space_ground))

    x_traj_move_back = np.linspace(init_pose[0], 0, steps_move_back)
    x_traj_descend = np.linspace(0, 0, steps_descend)
    x_traj_ground = np.zeros_like(time_space_ground)
    x_traj = np.concatenate((x_traj_move_back, x_traj_descend, x_traj_ground))

    y_traj_move_back = np.linspace(init_pose[1], 0, steps_move_back)
    y_traj_descend = np.linspace(0, 0, steps_descend)
    y_traj_ground = np.zeros_like(time_space_ground)
    y_traj = np.concatenate((y_traj_move_back, y_traj_descend, y_traj_ground))

    z_traj_move_back = np.linspace(init_pose[2], 1.0, steps_move_back)
    z_traj_descend  = np.linspace(1.0, 0.0, steps_descend)
    z_traj_ground = np.zeros_like(time_space_ground)
    z_traj = np.concatenate((z_traj_move_back, z_traj_descend, z_traj_ground))

    roll_traj = np.zeros_like(time_space_total)
    pitch_traj = np.zeros_like(time_space_total)
    yaw_traj = np.zeros_like(time_space_total)
    rpy_traj = np.vstack((roll_traj, pitch_traj, yaw_traj)).T
    quaternions = R.from_euler('xyz', rpy_traj).as_quat()  # Convert to quaternions
    qx_traj = quaternions[:, 0]
    qy_traj = quaternions[:, 1]
    qz_traj = quaternions[:, 2]
    qw_traj = quaternions[:, 3]
    vx_traj = np.gradient(x_traj, dt)
    vy_traj = np.gradient(y_traj, dt)
    vz_traj = np.gradient(z_traj, dt)
    ax_traj = np.zeros_like(time_space_total)
    ay_traj = np.zeros_like(time_space_total)
    az_traj = np.zeros_like(time_space_total)
    u1 = np.zeros_like(time_space_total)
    u2 = np.zeros_like(time_space_total)
    u3 = np.zeros_like(time_space_total)
    u4 = np.zeros_like(time_space_total)
    traj = np.array([x_traj, y_traj, z_traj, qw_traj, qx_traj, qy_traj, qz_traj,
                     vx_traj, vy_traj, vz_traj, ax_traj, ay_traj, az_traj, u1, u2, u3, u4])
    return traj, "land"



def hover_trajectory(dt):
    take_off_traj, _ = takeoff_trajectory(dt)

    steps = int(15 / dt)  # 5 seconds of hover
    time_space = np.linspace(0, steps * dt, steps)
    x_traj = np.zeros_like(time_space) #np.zeros_like(time_space)
    y_traj = np.zeros_like(time_space) #np.zeros_like(time_space) # 0.0 + 0.5 * np.sin(1.0 * np.pi * time_space / 5.0)
    z_traj = 1.0 * np.ones_like(time_space)#1.2 * np.ones_like(time_space) #1.5 + 0.5 * np.sin(1.5 * np.pi * time_space / 5.0)    #

    roll_traj = np.zeros_like(time_space)
    pitch_traj = np.zeros_like(time_space)
    yaw_traj = np.zeros_like(time_space)
    rpy_traj = np.vstack((roll_traj, pitch_traj, yaw_traj)).T
    quaternions = R.from_euler('xyz', rpy_traj).as_quat()  # Convert to quaternions
    qx_traj = quaternions[:, 0]
    qy_traj = quaternions[:, 1]
    qz_traj = quaternions[:, 2]
    qw_traj = quaternions[:, 3]
    vx_traj = np.gradient(x_traj, dt)
    vy_traj = np.gradient(y_traj, dt)
    vz_traj = np.gradient(z_traj, dt)
    ax_traj = np.zeros_like(time_space)
    ay_traj = np.zeros_like(time_space)
    az_traj = np.zeros_like(time_space)
    u1 = np.zeros_like(time_space)
    u2 = np.zeros_like(time_space)
    u3 = np.zeros_like(time_space)
    u4 = np.zeros_like(time_space)
    hover_traj =  np.array([x_traj, y_traj, z_traj, qw_traj, qx_traj, qy_traj, qz_traj,
                 vx_traj, vy_traj, vz_traj, ax_traj, ay_traj, az_traj, u1, u2, u3, u4])

    land_traj, _ = land_trajectory(dt, take_off_traj[:, -1])
    zeros = np.zeros((take_off_traj.shape[0], int(1 / dt)))

    traj = np.concatenate((take_off_traj,hover_traj, land_traj,zeros), axis=1)
    return traj, "hover"




def z_sin_trajectory(dt):
    trajectory_name = "ZSINE"
    take_off_traj, _ = takeoff_trajectory(dt)

    # Phase 1: Hover for 1 second
    steps_hover = int(1 / dt)  # 1 second of hover
    time_space_hover = np.linspace(0, steps_hover * dt, steps_hover)
    x_traj_hover = np.zeros_like(time_space_hover)
    y_traj_hover = np.zeros_like(time_space_hover)
    z_traj_hover = 1.0 * np.ones_like(time_space_hover)  # Constant hover at 1.0m
    
    # Phase 2: Variable frequency sinusoidal motion for 10 seconds
    steps_sin = int(30 / dt)  # 10 seconds of sinusoidal motion
    time_space_sin = np.linspace(0, steps_sin * dt, steps_sin)
    
    # Frequency increases linearly from 1.0 to 3.0 Hz over 40 seconds
    freq_start = 0.1  # Hz
    freq_end = 0.1    # Hz
    frequencies = np.linspace(freq_start, freq_end, steps_sin)
    
    # Calculate the instantaneous phase by integrating frequency
    # phase(t) = 2π * ∫frequency(τ)dτ from 0 to t
    phase = 2 * np.pi * np.cumsum(frequencies) * dt
    
    x_traj_sin = np.zeros_like(time_space_sin)
    y_traj_sin = np.zeros_like(time_space_sin)
    z_traj_sin = 1.0 + 0.5 * np.sin(phase)  # 1m amplitude around 1.5m center
    
    # Combine both phases
    x_traj = np.concatenate((x_traj_hover, x_traj_sin))
    y_traj = np.concatenate((y_traj_hover, y_traj_sin))
    z_traj = np.concatenate((z_traj_hover, z_traj_sin))
    time_space = np.concatenate((time_space_hover, time_space_sin))

    roll_traj = np.zeros_like(time_space)
    pitch_traj = np.zeros_like(time_space)
    yaw_traj = np.zeros_like(time_space)
    rpy_traj = np.vstack((roll_traj, pitch_traj, yaw_traj)).T
    quaternions = R.from_euler('xyz', rpy_traj).as_quat()  # Convert to quaternions
    qx_traj = quaternions[:, 0]
    qy_traj = quaternions[:, 1]
    qz_traj = quaternions[:, 2]
    qw_traj = quaternions[:, 3]
    vx_traj = np.gradient(x_traj, dt)
    vy_traj = np.gradient(y_traj, dt)
    vz_traj = np.gradient(z_traj, dt)
    ax_traj = np.zeros_like(time_space)
    ay_traj = np.zeros_like(time_space)
    az_traj = np.zeros_like(time_space)
    u1 = np.zeros_like(time_space)
    u2 = np.zeros_like(time_space)
    u3 = np.zeros_like(time_space)
    u4 = np.zeros_like(time_space)
    hover_traj =  np.array([x_traj, y_traj, z_traj, qw_traj, qx_traj, qy_traj, qz_traj,
                 vx_traj, vy_traj, vz_traj, ax_traj, ay_traj, az_traj, u1, u2, u3, u4])

    land_traj, _ = land_trajectory(dt, hover_traj[:, -1])
    zeros = np.zeros((take_off_traj.shape[0], int(1 / dt)))

    traj = np.concatenate((take_off_traj,hover_traj, land_traj,zeros), axis=1)
    return traj, "z_sin"

def foward_z_sin_trajectory(dt):
    """
    Forward Z-sine trajectory that:
    1. Takes off from (-1, 0, 0) to position (-1, 0, 1)
    2. Hovers for 5 seconds
    3. Performs 3.5 Z-sine oscillations over 60 seconds while moving forward from x=-1 to x=1 (y stays at 0)
    4. Hovers for 5 seconds at the end
    5. Lands at (1, 0, 0.1)
    """
    # Custom takeoff from (-1, 0, 0) to (-1, 0, 1)
    steps_takeoff = int(5 / dt)  # 4 seconds of takeoff
    steps_hover_start = int(0 / dt)  # 5 seconds of hover at start
    
    # Takeoff phase from (-1, 0, 0) to (-1, 0, 1)
    time_space_takeoff = np.linspace(0, steps_takeoff * dt, steps_takeoff)
    x_traj_takeoff = np.full_like(time_space_takeoff, -1.0)
    y_traj_takeoff = np.zeros_like(time_space_takeoff)
    z_traj_takeoff = np.linspace(0.0, 1.0, steps_takeoff)
    
    # Initial hover phase at (-1, 0, 1)
    time_space_hover_start = np.linspace(0, steps_hover_start * dt, steps_hover_start)
    x_traj_hover_start = np.full_like(time_space_hover_start, -1.0)
    y_traj_hover_start = np.zeros_like(time_space_hover_start)
    z_traj_hover_start = np.ones_like(time_space_hover_start) * 1.0


    trajectory_duration = 40 #seconds
    
    # Phase 2: Forward motion with Z-sine oscillations (60 seconds, 3.5 oscillations)
    steps_motion = int(trajectory_duration / dt)  # 60 seconds of forward motion with z-sine
    time_space_motion = np.linspace(0, steps_motion * dt, steps_motion)
    
    # 3.5 oscillations over 60 seconds (ends on downward phase)
    num_oscillations = 1.0
    frequency = num_oscillations / trajectory_duration  # Hz
    phase = 2 * np.pi * frequency * time_space_motion
    
    # Move from x=-1 to x=1 linearly, y stays at 0
    x_traj_motion = np.linspace(-1.0, 1.0, steps_motion)
    y_traj_motion = np.zeros_like(time_space_motion)
    z_traj_motion = 1.0 + 0.5 * np.sin(phase)  # 0.5m amplitude around 1.0m center
    
    # Combine all phases
    x_traj = np.concatenate((x_traj_takeoff, x_traj_hover_start, x_traj_motion))
    y_traj = np.concatenate((y_traj_takeoff, y_traj_hover_start, y_traj_motion))
    z_traj = np.concatenate((z_traj_takeoff, z_traj_hover_start, z_traj_motion))
    time_space = np.concatenate((time_space_takeoff, time_space_hover_start, time_space_motion))

    # 5 second hover at the end
    steps_hover_end = int(2 / dt)
    time_space_hover_end = np.linspace(0, steps_hover_end * dt, steps_hover_end)
    x_traj_hover_end = np.full_like(time_space_hover_end, x_traj[-1])
    y_traj_hover_end = np.full_like(time_space_hover_end, y_traj[-1])
    z_traj_hover_end = np.full_like(time_space_hover_end, z_traj[-1])
    
    # Append end hover
    x_traj = np.concatenate((x_traj, x_traj_hover_end))
    y_traj = np.concatenate((y_traj, y_traj_hover_end))
    z_traj = np.concatenate((z_traj, z_traj_hover_end))
    time_space = np.concatenate((time_space, time_space_hover_end))

    roll_traj = np.zeros_like(time_space)
    pitch_traj = np.zeros_like(time_space)
    yaw_traj = np.zeros_like(time_space)
    rpy_traj = np.vstack((roll_traj, pitch_traj, yaw_traj)).T
    quaternions = R.from_euler('xyz', rpy_traj).as_quat()  # Convert to quaternions
    qx_traj = quaternions[:, 0]
    qy_traj = quaternions[:, 1]
    qz_traj = quaternions[:, 2]
    qw_traj = quaternions[:, 3]
    vx_traj = np.gradient(x_traj, dt)
    vy_traj = np.gradient(y_traj, dt)
    vz_traj = np.gradient(z_traj, dt)
    ax_traj = np.zeros_like(time_space)
    ay_traj = np.zeros_like(time_space)
    az_traj = np.zeros_like(time_space)
    u1 = np.zeros_like(time_space)
    u2 = np.zeros_like(time_space)
    u3 = np.zeros_like(time_space)
    u4 = np.zeros_like(time_space)
    motion_traj = np.array([x_traj, y_traj, z_traj, qw_traj, qx_traj, qy_traj, qz_traj,
                            vx_traj, vy_traj, vz_traj, ax_traj, ay_traj, az_traj, u1, u2, u3, u4])

    # Custom landing to (1, 0, 0.1)
    steps_land = int(10 / dt)  # 10 seconds to land
    time_space_land = np.linspace(0, steps_land * dt, steps_land)
    x_traj_land = np.linspace(motion_traj[0, -1], 1.0, steps_land)
    y_traj_land = np.linspace(motion_traj[1, -1], 0.0, steps_land)
    z_traj_land = np.linspace(motion_traj[2, -1], 0.0, steps_land)
    
    roll_traj_land = np.zeros_like(time_space_land) 
    pitch_traj_land = np.zeros_like(time_space_land)
    yaw_traj_land = np.zeros_like(time_space_land)
    rpy_traj_land = np.vstack((roll_traj_land, pitch_traj_land, yaw_traj_land)).T
    quaternions_land = R.from_euler('xyz', rpy_traj_land).as_quat()
    qx_traj_land = quaternions_land[:, 0]
    qy_traj_land = quaternions_land[:, 1]
    qz_traj_land = quaternions_land[:, 2]
    qw_traj_land = quaternions_land[:, 3]
    vx_traj_land = np.gradient(x_traj_land, dt)
    vy_traj_land = np.gradient(y_traj_land, dt)
    vz_traj_land = np.gradient(z_traj_land, dt)
    ax_traj_land = np.zeros_like(time_space_land)
    ay_traj_land = np.zeros_like(time_space_land)
    az_traj_land = np.zeros_like(time_space_land)
    u1_land = np.zeros_like(time_space_land)
    u2_land = np.zeros_like(time_space_land)
    u3_land = np.zeros_like(time_space_land)
    u4_land = np.zeros_like(time_space_land)
    land_traj = np.array([x_traj_land, y_traj_land, z_traj_land, qw_traj_land, qx_traj_land, qy_traj_land, qz_traj_land,
                          vx_traj_land, vy_traj_land, vz_traj_land, ax_traj_land, ay_traj_land, az_traj_land, 
                          u1_land, u2_land, u3_land, u4_land])
    
    zeros = np.zeros((motion_traj.shape[0], int(1 / dt)))

    traj = np.concatenate((motion_traj, land_traj, zeros), axis=1)
    return traj, "forward_z_sin"


def xyz_sine_trajectory(dt):
    take_off_traj, _ = takeoff_trajectory(dt)

    # 5 second hover before sine motion
    steps_hover_start = int(5 / dt)
    time_space_hover_start = np.linspace(0, steps_hover_start * dt, steps_hover_start)
    x_traj_hover_start = np.zeros_like(time_space_hover_start)
    y_traj_hover_start = np.zeros_like(time_space_hover_start)
    z_traj_hover_start = np.ones_like(time_space_hover_start) * 0.75

    # 40 seconds of sine motion
    steps = int(40 / dt)
    time_space = np.linspace(0, steps * dt, steps)
    x_traj_sine = 0.0 + 1.0 * np.sin(0.5 * np.pi * time_space / 5.0)
    y_traj_sine = 0.0 + 1.0 * np.sin(0.25 * np.pi * time_space / 5.0)
    z_traj_sine = 0.75 + 0.25 * np.sin(0.25 * np.pi * time_space / 5.0)

    # 5 second hover after sine motion
    steps_hover_end = int(5 / dt)
    time_space_hover_end = np.linspace(0, steps_hover_end * dt, steps_hover_end)
    x_traj_hover_end = np.full_like(time_space_hover_end, x_traj_sine[-1])
    y_traj_hover_end = np.full_like(time_space_hover_end, y_traj_sine[-1])
    z_traj_hover_end = np.full_like(time_space_hover_end, z_traj_sine[-1])

    # Combine all phases
    x_traj = np.concatenate((x_traj_hover_start, x_traj_sine, x_traj_hover_end))
    y_traj = np.concatenate((y_traj_hover_start, y_traj_sine, y_traj_hover_end))
    z_traj = np.concatenate((z_traj_hover_start, z_traj_sine, z_traj_hover_end))
    time_space_combined = np.concatenate((time_space_hover_start, time_space, time_space_hover_end))

    roll_traj = np.zeros_like(time_space_combined)
    pitch_traj = np.zeros_like(time_space_combined)
    yaw_traj = np.zeros_like(time_space_combined)
    rpy_traj = np.vstack((roll_traj, pitch_traj, yaw_traj)).T
    quaternions = R.from_euler('xyz', rpy_traj).as_quat()  # Convert to quaternions
    qx_traj = quaternions[:, 0]
    qy_traj = quaternions[:, 1]
    qz_traj = quaternions[:, 2]
    qw_traj = quaternions[:, 3]
    vx_traj = np.gradient(x_traj, dt)
    vy_traj = np.gradient(y_traj, dt)
    vz_traj = np.gradient(z_traj, dt)
    ax_traj = np.zeros_like(time_space_combined)
    ay_traj = np.zeros_like(time_space_combined)
    az_traj = np.zeros_like(time_space_combined)
    u1 = np.zeros_like(time_space_combined)
    u2 = np.zeros_like(time_space_combined)
    u3 = np.zeros_like(time_space_combined)
    u4 = np.zeros_like(time_space_combined)
    hover_traj =  np.array([x_traj, y_traj, z_traj, qw_traj, qx_traj, qy_traj, qz_traj,
                 vx_traj, vy_traj, vz_traj, ax_traj, ay_traj, az_traj, u1, u2, u3, u4])

    land_traj, _ = land_trajectory(dt, hover_traj[:, -1])
    zeros = np.zeros((take_off_traj.shape[0], int(1 / dt)))

    traj = np.concatenate((take_off_traj,hover_traj, land_traj,zeros), axis=1)
    return traj, "xyz_sine"


def fast_xyz_sine_trajectory(dt):
    take_off_traj, _ = takeoff_trajectory(dt)

    steps = int(20 / dt)  # 20 seconds of sine motion
    time_space = np.linspace(0, steps * dt, steps)
    y_traj = 0.0 + 0.75 * np.sin(1.0 * np.pi * time_space / 5.0) #np.zeros_like(time_space)
    x_traj = 0.0 + 0.75 * np.sin(1.0 * np.pi * time_space / 5.0) #np.zeros_like(time_space)
    z_traj = 1.0 + 0.5 * np.sin(1.0 * np.pi * time_space / 5.0)#1.2 * np.ones_like(time_space) #1.5 + 0.5 * np.sin(1.5 * np.pi * time_space / 5.0)    #

    roll_traj = np.zeros_like(time_space)
    pitch_traj = np.zeros_like(time_space)
    yaw_traj = np.zeros_like(time_space)
    rpy_traj = np.vstack((roll_traj, pitch_traj, yaw_traj)).T
    quaternions = R.from_euler('xyz', rpy_traj).as_quat()  # Convert to quaternions
    qx_traj = quaternions[:, 0]
    qy_traj = quaternions[:, 1]
    qz_traj = quaternions[:, 2]
    qw_traj = quaternions[:, 3]
    vx_traj = np.gradient(x_traj, dt)
    vy_traj = np.gradient(y_traj, dt)
    vz_traj = np.gradient(z_traj, dt)
    ax_traj = np.zeros_like(time_space)
    ay_traj = np.zeros_like(time_space)
    az_traj = np.zeros_like(time_space)
    u1 = np.zeros_like(time_space)
    u2 = np.zeros_like(time_space)
    u3 = np.zeros_like(time_space)
    u4 = np.zeros_like(time_space)
    hover_traj =  np.array([x_traj, y_traj, z_traj, qw_traj, qx_traj, qy_traj, qz_traj,
                 vx_traj, vy_traj, vz_traj, ax_traj, ay_traj, az_traj, u1, u2, u3, u4])

    land_traj, _ = land_trajectory(dt, take_off_traj[:, -1])
    zeros = np.zeros((take_off_traj.shape[0], int(1 / dt)))

    traj = np.concatenate((take_off_traj,hover_traj, land_traj,zeros), axis=1)
    return traj, "fast_xyz_sine"







def circle_trajectory(dt):
    take_off_traj, _ = takeoff_trajectory(dt)

    # Transition from hover to circle start (5 seconds)
    steps_transition = int(5 / dt)  # 5 seconds transition
    time_space_transition = np.linspace(0, steps_transition * dt, steps_transition)
    
    # Circle parameters
    radius = 1.0  # 1 meter
    period = 30  # seconds per rotation
    num_revolutions = 2  # number of full circles
    omega = 2 * np.pi / period  # angular velocity (rad/s)
    
    # Get the final position from takeoff (should be 0, 0, 1.0)
    takeoff_end_x = take_off_traj[0, -1]  # x position
    takeoff_end_y = take_off_traj[1, -1]  # y position
    takeoff_end_z = take_off_traj[2, -1]  # z position
    
    # Transition trajectory from takeoff end to circle start
    x_traj_transition = np.linspace(takeoff_end_x, radius, steps_transition)  # Move to circle start
    y_traj_transition = np.linspace(takeoff_end_y, 0.0, steps_transition)     # Move to y=0
    z_traj_transition = np.linspace(takeoff_end_z, 0.75, steps_transition)     # Move to circle altitude

    # Hover at circle start for 3 seconds
    steps_hover = int(3 / dt)
    time_space_hover = np.linspace(0, steps_hover * dt, steps_hover)
    x_traj_hover = np.full_like(time_space_hover, radius)
    y_traj_hover = np.zeros_like(time_space_hover)
    z_traj_hover = np.full_like(time_space_hover, 0.75)

    # Three full revolutions, 10 seconds each = 30 seconds total
    steps = int(period * num_revolutions / dt)  # 30 seconds of circle motion
    time_space = np.linspace(0, steps * dt, steps)

    x_traj = radius * np.cos(omega * time_space)
    y_traj = radius * np.sin(omega * time_space)
    z_traj = 0.75 * np.ones_like(time_space)  # constant altitude

    # Combine transition, hover, and circle time spaces for trajectory calculations
    time_space_combined = np.concatenate((time_space_transition, time_space_hover, time_space))
    x_traj_combined = np.concatenate((x_traj_transition, x_traj_hover, x_traj))
    y_traj_combined = np.concatenate((y_traj_transition, y_traj_hover, y_traj))
    z_traj_combined = np.concatenate((z_traj_transition, z_traj_hover, z_traj))

    roll_traj = np.zeros_like(time_space_combined)
    pitch_traj = np.zeros_like(time_space_combined)
    yaw_traj = np.zeros_like(time_space_combined)
    rpy_traj = np.vstack((roll_traj, pitch_traj, yaw_traj)).T
    quaternions = R.from_euler('xyz', rpy_traj).as_quat()  # Convert to quaternions
    qx_traj = quaternions[:, 0]
    qy_traj = quaternions[:, 1]
    qz_traj = quaternions[:, 2]
    qw_traj = quaternions[:, 3]
    vx_traj = np.gradient(x_traj_combined, dt)
    vy_traj = np.gradient(y_traj_combined, dt)
    vz_traj = np.gradient(z_traj_combined, dt)
    ax_traj = np.zeros_like(time_space_combined)
    ay_traj = np.zeros_like(time_space_combined)
    az_traj = np.zeros_like(time_space_combined)
    u1 = np.zeros_like(time_space_combined)
    u2 = np.zeros_like(time_space_combined)
    u3 = np.zeros_like(time_space_combined)
    u4 = np.zeros_like(time_space_combined)
    hover_traj = np.array([x_traj_combined, y_traj_combined, z_traj_combined, qw_traj, qx_traj, qy_traj, qz_traj,
                           vx_traj, vy_traj, vz_traj, ax_traj, ay_traj, az_traj, u1, u2, u3, u4])

    land_traj, _ = land_trajectory(dt, hover_traj[:, -1])
    zeros = np.zeros((take_off_traj.shape[0], int(1 / dt)))

    traj = np.concatenate((take_off_traj, hover_traj, land_traj, zeros), axis=1)
    return traj, "circle"






def figure8_trajectory(dt, duration=10.0, repeats=3, z_amplitude=0.5, z_offset=1.5):
    """
    Generates a repeated figure-8 trajectory (each loop duration, repeated N times) on a zsine slope.
    Args:
        dt: timestep (s)
        duration: duration of one figure-8 (s)
        repeats: number of times to repeat the figure-8
        z_amplitude: amplitude of zsine (m)
        z_offset: base z height (m)
    Returns:
        traj: np.ndarray shape (16, N)
    """
    take_off_traj, _ = takeoff_trajectory(dt)

    # Build one figure-8 segment
    steps = int(duration / dt)
    t = np.linspace(0, duration, steps)
    R_ = 1.0  # 1m radius
    y_seg = R_ * np.sin(2 * np.pi * t / duration)
    x_seg = R_ * np.sin(2 * np.pi * t / duration) * np.cos(2 * np.pi * t / duration)
    z_seg = z_offset + z_amplitude * np.sin(2 * np.pi * t / duration)

    vx = np.gradient(x_seg, dt)
    vy = np.gradient(y_seg, dt)
    vz = np.gradient(z_seg, dt)
    
    # Calculate yaw to point in direction of motion
    yaw_seg = np.arctan2(vy, vx)
    roll_seg = np.zeros_like(yaw_seg)
    pitch_seg = np.zeros_like(yaw_seg)
    rpy_seg = np.vstack((roll_seg, pitch_seg, yaw_seg)).T
    quats = R.from_euler('xyz', rpy_seg).as_quat()
    qx_seg = quats[:, 0]
    qy_seg = quats[:, 1]
    qz_seg = quats[:, 2]
    qw_seg = quats[:, 3]
    ax_seg = np.zeros_like(x_seg)
    ay_seg = np.zeros_like(y_seg)
    az_seg = np.zeros_like(z_seg)
    u1 = np.zeros_like(x_seg)
    u2 = np.zeros_like(x_seg)
    u3 = np.zeros_like(x_seg)
    u4 = np.zeros_like(x_seg)

    # Repeat the segment
    x_traj = np.tile(x_seg, repeats)
    y_traj = np.tile(y_seg, repeats)
    z_traj = np.tile(z_seg, repeats)
    qw_traj = np.tile(qw_seg, repeats)
    qx_traj = np.tile(qx_seg, repeats)
    qy_traj = np.tile(qy_seg, repeats)
    qz_traj = np.tile(qz_seg, repeats)
    vx_traj = np.tile(vx, repeats)
    vy_traj = np.tile(vy, repeats)
    vz_traj = np.tile(vz, repeats)
    ax_traj = np.tile(ax_seg, repeats)
    ay_traj = np.tile(ay_seg, repeats)
    az_traj = np.tile(az_seg, repeats)
    u1 = np.tile(u1, repeats)
    u2 = np.tile(u2, repeats)
    u3 = np.tile(u3, repeats)
    u4 = np.tile(u4, repeats)

    hover_traj = np.array([
        x_traj, y_traj, z_traj, qw_traj, qx_traj, qy_traj, qz_traj,
        vx_traj, vy_traj, vz_traj, ax_traj, ay_traj, az_traj, u1, u2, u3, u4
    ])

    land_traj, _ = land_trajectory(dt, hover_traj[:, -1])
    zeros = np.zeros((take_off_traj.shape[0], int(1 / dt)))

    traj = np.concatenate((take_off_traj, hover_traj, land_traj, zeros), axis=1)
    return traj, "figure8_zsine"



