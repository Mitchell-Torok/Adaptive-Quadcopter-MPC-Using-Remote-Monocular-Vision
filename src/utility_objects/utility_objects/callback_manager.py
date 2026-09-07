# utility_objects/callbacks.py
import rclpy
import time
import threading
import numpy as np
import os
import signal


from std_msgs.msg import String, Bool
from interfaces.msg import MotionCaptureState, Telemetry, ELRSCommand
from interfaces.srv import SetArming

class CallbackManager:
    def __init__( self,node,USE_MOTION_CAPTURE: bool = True):      
        self.node = node

        self.cmd_publisher_ = self.node.create_publisher(ELRSCommand, '/ELRSCommand', 1)

        self.pose_subscription_ = self.node.create_subscription(MotionCaptureState, '/motion_capture_state', self.pose_callback, 5)
        self.telemetry_subscription_ = self.node.create_subscription(Telemetry, '/telemetry', self.telemetry_callback, 5)

        self.arming_service_ = self.node.create_service(SetArming, 'drone_arming_service', self.handle_arming_service)
        self.command_subscription_ = self.node.create_subscription(String, 'drone_command', self.command_callback, 5)
        self.arming_state_publisher_ = self.node.create_publisher(Bool, 'drone_arming_state_feedback', 5)

        self.motion_capture_pose = [0,0,0,1,0,0,0,0,0,0,0,0,0]  # x,y,z,qw,qx,qy,qz,vx,vy,vz,avx,avy,avz
        self.use_motion_capture = USE_MOTION_CAPTURE


    def pose_callback(self, msg: MotionCaptureState):
        p, o, lv, av = msg.pose.position, msg.pose.orientation, msg.twist.linear, msg.twist.angular
        arr = np.round(np.array([
            p.x, p.y, p.z,
            o.w, o.x, o.y, o.z,
            lv.x, lv.y, lv.z,
            av.x, av.y, av.z
        ]), 3)

        self.motion_capture_pose = arr

        if self.use_motion_capture:
            self.node.current_pose = self.motion_capture_pose
            self.node.last_pose_update_time = time.time()

            if hasattr(self.node, "ukf_update_from_current_pose"):
                self.node.ukf_update_from_current_pose()

    def handle_arming_service(self, request, response):
        if request.arm:
            if self.node.current_pose is not None:
                self.node.armed = True
                response.success = True
                response.message = "Drone armed successfully"
                self.node.get_logger().info("Drone armed via service")
            else:
                response.success = False
                response.message = "Cannot arm: No pose data available"
                self.node.get_logger().warn("Arming failed: No pose data")
        else:
            self.node.armed = False
            self.node.takeoff_requested = False
            self.node.shutdown_requested = True
            response.success = True
            response.message = "Drone disarmed successfully - shutting down controller"
            self.node.get_logger().info("Drone disarmed via service - initiating shutdown")
        
        self.publish_arming_state()
        return response
    

    def command_callback(self, msg: String):
        command = msg.data.upper()
        
        if command == "ARM":
            if self.node.current_pose is not None:
                self.node.armed = True
                self.node.get_logger().info("Drone armed via command")
                self.publish_arming_state()
            else:
                self.node.get_logger().warn("Cannot arm: No pose data available")
        
        elif command == "DISARM":
            self.node.armed = False
            self.node.takeoff_requested = False
            self.node.shutdown_requested = True
            self.node.get_logger().info("Drone disarmed via command - initiating shutdown")
            self.publish_arming_state()
        
        elif command == "TAKEOFF":
            if self.node.armed:
                self.node.takeoff_requested = True
                self.node.get_logger().info("Takeoff requested")
            else:
                self.node.get_logger().warn("Cannot takeoff: Drone not armed")

    def publish_arming_state(self):
        msg = Bool()
        msg.data = self.node.armed
        self.arming_state_publisher_.publish(msg)
    
    def telemetry_callback(self, msg: Telemetry):
        self.node.battery_voltage = msg.battery_voltage

    def request_shutdown(self):
        self.node.get_logger().info("Shutdown requested - closing controller")
        def shutdown_thread():
            time.sleep(0.1)
            self.node.on_close()
            try:
                self.node.destroy_node()
            except:
                pass
            try:
                rclpy.shutdown()
            except:
                pass
            os.kill(os.getpid(), signal.SIGTERM)
        thread = threading.Thread(target=shutdown_thread, daemon=True)
        thread.start()

    def safety_disarm(self, msg: ELRSCommand):
        self.node.armed = False
        self.node.takeoff_requested = False
        self.node.get_logger().warn("Safety disarm triggered - disarming drone")
        self.cmd_publisher_.publish(msg)
        self.publish_arming_state()

    def disarm(self, msg: ELRSCommand):
        self.node.armed = False
        self.node.takeoff_requested = False
        self.node.get_logger().warn("disarm triggered - disarming drone")
        self.cmd_publisher_.publish(msg)
        self.publish_arming_state()