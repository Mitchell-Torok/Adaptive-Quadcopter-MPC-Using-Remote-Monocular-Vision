import socket
import struct
import rclpy
import numpy as np
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import Imu
from actuator_msgs.msg import Actuators
from interfaces.msg import MotionCaptureState, ELRSCommand, Telemetry
from geometry_msgs.msg import Twist, PoseArray, Pose, PoseStamped
from tf_transformations import euler_from_quaternion, quaternion_multiply, quaternion_inverse, quaternion_matrix

class BetaflightInterfaceNode(Node):
    def __init__(self):
        super().__init__('betaflight_interface')

        # --- subs/pubs ---
        self.subscription_motion_capture = self.create_subscription(PoseArray, '/model/x3/pose', self.pose_callback, 10)
        self.subscription_control = self.create_subscription(ELRSCommand, 'ELRSCommand', self.controller_commands_callback, 10)
        self.publisher = self.create_publisher(Actuators, '/X3/gazebo/command/motor_speed', 10)

        # --- state ---
        self.set_point = None               # [roll_rate_sp, pitch_rate_sp, throttle, yaw_rate_sp]
        self.current_pose = None
        self.last_pose = None
        self.last_orientation = None
        self.last_time = None

        # --- inner RATE PID (deg/s tracking of p,q,r) ---
        self.kp = 1.0
        self.ki = 0.0
        self.kd = 0.0
        self.integral_error = np.zeros(3, dtype=float)
        self.previous_error = np.zeros(3, dtype=float)

        # --- outer ANGLE PI (deg tracking of roll, pitch) → desired rate (deg/s) ---
        self.declare_parameter('angle_max_deg', 55.0)   # Betaflight default is ~55°
        self.declare_parameter('angle_kp', 4.0)         # deg/s per deg of error
        self.declare_parameter('angle_ki', 0.5)         # deg/s per deg·s (small bias cancel)

        self.angle_max_deg = float(self.get_parameter('angle_max_deg').value)
        self.angle_kp = float(self.get_parameter('angle_kp').value)
        self.angle_ki = float(self.get_parameter('angle_ki').value)

        self.angle_int = np.zeros(2, dtype=float)       # roll,pitch integrator
        self.angle_int_limit = 200.0                    # anti-windup clamp on produced rate (deg/s equivalent)

        # --- FC mounting angle disturbance (simulates FC not perfectly level) ---
        self.declare_parameter('fc_roll_offset_deg', 0.2)   # FC roll mounting error (deg)
        self.declare_parameter('fc_pitch_offset_deg', 0.2)  # FC pitch mounting error (deg)
        self.fc_roll_offset_deg = float(self.get_parameter('fc_roll_offset_deg').value)
        self.fc_pitch_offset_deg = float(self.get_parameter('fc_pitch_offset_deg').value)

        # --- Betaflight rates params (we keep for YAW; roll/pitch now angle-mode) ---
        self.declare_parameter('rates_d_val', 70.0)
        self.declare_parameter('rates_f_val', 670.0)
        self.declare_parameter('rates_g_val', 0.5)
        self.rates_d_val = float(self.get_parameter('rates_d_val').value)
        self.rates_f_val = float(self.get_parameter('rates_f_val').value)
        self.rates_g_val = float(self.get_parameter('rates_g_val').value)

        # cached last angle setpoints (for debugging/telemetry if you want)
        self.last_angle_sp_deg = np.zeros(2, dtype=float)

    # ------------------------ Utilities ------------------------

    def betaflight_rates(self, x):
        """
        Betaflight unified rates-like curve.
        We will only use this for YAW (rate), not for roll/pitch in Angle mode.
        """
        import math
        x = max(-1.0, min(1.0, x))
        ax = math.sqrt(x*x + 1e-6)
        sgn = x / ax if ax > 0 else 0.0
        h_abs = ax * (pow(ax, 5) * self.rates_g_val + ax * (1.0 - self.rates_g_val))
        j_abs = self.rates_d_val * ax + (self.rates_f_val - self.rates_d_val) * h_abs
        return sgn * j_abs  # deg/s

    def normalize_quaternion_positive_w(self, x, y, z, w):
        if w < 0.0:
            return -x, -y, -z, -w
        return x, y, z, w

    # ------------------------ Callbacks ------------------------

    def pose_callback(self, msg: PoseArray):
        # Pick your body link index; you used 5 previously
        current_position = msg.poses[5].position
        q = msg.poses[5].orientation
        q.x, q.y, q.z, q.w = self.normalize_quaternion_positive_w(q.x, q.y, q.z, q.w)

        now = self.get_clock().now().to_msg()
        if self.last_pose is None:
            self.last_pose = current_position
            self.last_orientation = q
            self.last_time = now
            return

        # dt
        dt = (now.sec - self.last_time.sec) + (now.nanosec - self.last_time.nanosec) * 1e-9
        if dt <= 0.0 or dt > 0.2:  # guard against bad timestamps / sim hiccups
            self.last_pose = current_position
            self.last_orientation = q
            self.last_time = now
            return

        # measured angular rates from quaternion delta → body rates
        q1 = [self.last_orientation.x, self.last_orientation.y, self.last_orientation.z, self.last_orientation.w]
        q2 = [q.x, q.y, q.z, q.w]
        q_rel = quaternion_multiply(q2, quaternion_inverse(q1))
        ang_vel_enu = 2.0 * np.array([q_rel[0], q_rel[1], q_rel[2]]) / dt  # approx in world frame of q1
        R_enu_to_body = quaternion_matrix(q1)[:3, :3].T
        ang_vel_body = R_enu_to_body @ ang_vel_enu
        ang_vel_body_deg = np.degrees(ang_vel_body)  # [p,q,r] deg/s

        # current attitude (roll, pitch, yaw) in deg
        roll, pitch, yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        roll_deg, pitch_deg = np.degrees([roll, pitch])

        # If we have a setpoint, run control
        if self.set_point is not None:
            # ---- OUTER ANGLE PI (roll/pitch) → desired rate (deg/s) ----
            # set_point_angle_deg contains desired [roll, pitch] in deg set in controller_commands_callback
            # Apply FC mounting angle disturbance (simulates FC not being perfectly level)
            angle_sp_deg = self.last_angle_sp_deg + np.array([self.fc_roll_offset_deg, self.fc_pitch_offset_deg], dtype=float)
            angle_err = angle_sp_deg - np.array([roll_deg, pitch_deg], dtype=float)

            # integrate with clamp
            self.angle_int += angle_err * dt
            # prevent runaway integrator (clamp the rate contribution)
            self.angle_int = np.clip(self.angle_int, -self.angle_int_limit / max(self.angle_ki, 1e-6),
                                     self.angle_int_limit / max(self.angle_ki, 1e-6))

            # desired body rates from angle loop (roll,pitch); yaw stays from sticks curve
            desired_rates_rp = self.angle_kp * angle_err + self.angle_ki * self.angle_int  # deg/s
            roll_rate_sp = float(desired_rates_rp[0])
            pitch_rate_sp = float(desired_rates_rp[1])

            # build full desired rates vector [p,q,r] (deg/s)
            desired_rates_body_deg = np.array([roll_rate_sp, pitch_rate_sp, self.set_point[3]], dtype=float)

            # ---- INNER RATE PID (track desired p,q,r) ----
            motor_speeds = self.calculate_motor_speeds(ang_vel_body_deg, desired_rates_body_deg)

            # publish
            actuator_msg = Actuators()
            actuator_msg.header.stamp = self.get_clock().now().to_msg()
            actuator_msg.velocity = motor_speeds.tolist()
            self.publisher.publish(actuator_msg)

        # update last
        self.last_pose = current_position
        self.last_orientation = q
        self.last_time = now

    def calculate_motor_speeds(self, measured_rates_deg, desired_rates_deg):
        """
        Inner rate PID on body rates: error = desired - measured  (deg/s)
        Mixer: +roll/-roll, +pitch/-pitch, +yaw/-yaw, plus common throttle.
        """
        error = np.array(desired_rates_deg, dtype=float) - np.array(measured_rates_deg, dtype=float)

        # PID terms
        self.integral_error += error
        proportional = self.kp * error
        integral = self.ki * self.integral_error
        derivative = self.kd * (error - self.previous_error)
        self.previous_error = error
        offset = proportional + integral + derivative  # [roll,pitch,yaw] contributions

        throttle = float(self.set_point[2])  # already scaled to 0..4631

        # X quad mix (assuming motors: 0 front-right, 1 rear-right, 2 rear-left, 3 front-left)
        # roll: + on left motors, - on right motors
        # pitch: + on rear motors, - on front motors
        # yaw: + on CCW motors, - on CW motors (tune signs for your sim)
        motor_speeds = np.zeros(4, dtype=float)
        motor_speeds[0] = throttle - offset[0] + offset[1] + offset[2]  # FR
        motor_speeds[1] = throttle - offset[0] - offset[1] - offset[2]  # RR
        motor_speeds[2] = throttle + offset[0] + offset[1] - offset[2]  # RL
        motor_speeds[3] = throttle + offset[0] - offset[1] + offset[2]  # FL

        # physical limits
        motor_speeds = np.clip(motor_speeds, 0.0, 4631.0)
        return motor_speeds

    def controller_commands_callback(self, msg: ELRSCommand):
        """
        Sticks: channel_0 roll, channel_1 pitch, channel_2 throttle, channel_3 yaw
        For Angle mode:
          - roll/pitch sticks map to angle setpoints in deg using angle_max_deg
          - yaw stick maps to desired yaw rate (deg/s) via Betaflight rates curve
        """
        # --- throttle (scale  -1..1  →  0..4631)
        throttle = (msg.channel_2 + 1.0) * 0.5 * 4631.0
        if msg.armed and throttle < (0.05 * 4631.0):
            throttle = 0.05 * 4631.0

        # --- roll/pitch target ANGLES (deg) ---
        roll_sp_deg  = float(np.clip(msg.channel_0, -1.0, 1.0) * self.angle_max_deg)
        pitch_sp_deg = float(np.clip(msg.channel_1, -1.0, 1.0) * self.angle_max_deg)

        # cache for pose_callback (outer loop uses it)
        self.last_angle_sp_deg = np.array([roll_sp_deg, pitch_sp_deg], dtype=float)

        # --- yaw desired RATE (deg/s) via BF rates curve ---
        yaw_rate_sp = self.betaflight_rates(-msg.channel_3)  # your original sign

        # set_point shape (we keep same layout): [roll_rate_sp, pitch_rate_sp, throttle, yaw_rate_sp]
        # roll_rate_sp / pitch_rate_sp are *produced in pose_callback* by angle loop; set dummy here.
        self.set_point = [0.0, 0.0, throttle, yaw_rate_sp]

def main(args=None):
    rclpy.init(args=args)
    node = BetaflightInterfaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
