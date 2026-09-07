#!/usr/bin/env python3
import time
import collections
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import Int32
from interfaces.msg import ControlApplied

from .acados import generate_ocp_controller

# Must match the MPC settings used in main.py
FREQUENCY_HZ = 15
DT = 1.0 / FREQUENCY_HZ
N_HORIZON = 15
SKIP_STEPS = 3


class DelayEstimatorNode(Node):
    def __init__(self):
        super().__init__('delay_estimator')

        # ---------- Parameters ----------
        self.declare_parameter('window_len', 30)            # comparison window length
        self.declare_parameter('min_delay', 1)              # inclusive
        self.declare_parameter('max_delay', 11)             # inclusive
        self.declare_parameter('rate_hz', 4.0)              # estimator loop rate
        self.declare_parameter('alpha', 0.05)               # low-pass smoothing factor

        self.window_len = int(self.get_parameter('window_len').value)
        self.min_delay = int(self.get_parameter('min_delay').value)
        self.max_delay = int(self.get_parameter('max_delay').value)
        self.rate_hz = float(self.get_parameter('rate_hz').value)
        self.alpha = float(self.get_parameter('alpha').value)

        # ---------- Dedicated simulator instance ----------
        _ocp, self.sim = generate_ocp_controller(dt=DT, N_horizon=N_HORIZON, skip_steps=SKIP_STEPS)

        # ---------- Histories ----------
        self.obs_hist = collections.deque(maxlen=512)    # pose/state, shape (13,)
        self.ctrl_hist = collections.deque(maxlen=512)   # [u(4), u_rate(4)], shape (8,)
        self.param_hist = collections.deque(maxlen=512)  # est_params, shape (6,)

        # Raw + smoothed delay tracking
        self._last_best_delay = max(self.min_delay, 1)
        self._delay_states_float = float(self._last_best_delay)

        # ---------- ROS I/O ----------
        qos_rel1 = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            durability=DurabilityPolicy.VOLATILE,
        )

        self.create_subscription(ControlApplied, '/control_applied',
                                 self._on_control_applied, qos_rel1)
        self.pub_delay = self.create_publisher(Int32, '/estimated_delay', qos_rel1)

        self.timer = self.create_timer(1.0 / self.rate_hz, self._tick)

        self.get_logger().info(
            f"DelayEstimatorNode: W={self.window_len}, delay∈[{self.min_delay},{self.max_delay}], "
            f"{self.rate_hz}Hz, alpha={self.alpha}"
        )

    # ---------- Callbacks ----------
    def _on_control_applied(self, msg: ControlApplied):
        # Pose vector (13): [x,y,z, qw,qx,qy,qz, vx,vy,vz, wx,wy,wz]
        self.obs_hist.append(np.array(msg.pose, dtype=float))

        # Controls (u:4) + rates (u_rate:4) → 8-vector
        u = np.array(msg.u, dtype=float)
        u_rate = np.array(msg.u_rate, dtype=float)
        self.ctrl_hist.append(np.concatenate((u, u_rate)))

        # Estimated parameters (6)
        self.param_hist.append(np.array(msg.est_params, dtype=float))

    # ---------- Estimation loop ----------
    def _tick(self):
        W = self.window_len
        if len(self.obs_hist) < W or len(self.ctrl_hist) < (W + self.min_delay) or len(self.param_hist) < (W + self.min_delay):
            return

        # Search a local band around the last best (±2), clamped to [min_delay, max_delay]
        c = int(self._last_best_delay)
        lo = max(self.min_delay, c - 2)
        hi = min(self.max_delay, c + 2)
        sweep = range(lo, hi + 1)

        # Snapshot arrays
        obs = np.array(self.obs_hist, dtype=float)       # (N, 13)
        ctrl = np.array(self.ctrl_hist, dtype=float)     # (N, 8)
        params = np.array(self.param_hist, dtype=float)  # (N, 6)

        obs_window = obs[-W:]  # oldest→newest, shape (W, 13)

        min_err = float('inf')
        best_delay = self._last_best_delay

        for delay in sweep:
            start = -(W + delay)
            end = -delay if delay != 0 else None

            delayed_controls = ctrl[start:end]
            delayed_params = params[start:end]

            if delayed_controls.shape[0] < W or delayed_params.shape[0] < W:
                continue

            est_state = obs_window[0].copy()
            total_err = 0.0

            # Predict W-1 steps; compare with obs_window[1..W-1]
            for i in range(W - 1):
                u4 = delayed_controls[i, 0:4]
                ur4 = delayed_controls[i, 4:8]
                est_params = delayed_params[i]  # (6,)

                x_in = np.concatenate([est_state, u4])  # 13 + 4
                self.sim.set("x", x_in)
                self.sim.set("u", ur4)
                sim_p = np.concatenate([est_params, np.array([1.0, 0.0, 0.0, 0.0])])
                self.sim.set("p", sim_p)

                if self.sim.solve() != 0:
                    total_err = float('inf')
                    break

                x_next = self.sim.get("x")
                est_state = x_next[:13]

                # Position error (extend to quat/vel if desired)
                total_err += np.linalg.norm(est_state[:3] - obs_window[i + 1, :3])

            if total_err < min_err:
                min_err = total_err
                best_delay = int(delay)

        # ---------- Low-pass filter the integer delay ----------
        self._delay_states_float = (1.0 - self.alpha) * self._delay_states_float + self.alpha * float(best_delay)
        smoothed_delay = int(round(self._delay_states_float))
        self._last_best_delay = smoothed_delay

        # Publish the smoothed delay
        self.pub_delay.publish(Int32(data=smoothed_delay))


def main(args=None):
    rclpy.init(args=args)
    node = DelayEstimatorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
