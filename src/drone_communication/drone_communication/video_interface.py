import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import os
import time
import datetime
import numpy as np

class VideoInterface(Node):
    def __init__(self):
        super().__init__('video_interface')
        self.subscription = self.create_subscription(String, 'controller_topic', self.controller_callback, 10)
        
        self.publisher_ = self.create_publisher(Image, 'video_frames', 10)
        self.start_recording = False
        self.frames = []
        self.frame_timestamps = []
        self.video_capture = cv2.VideoCapture(4)  # Open the default camera
        self.cap = cv2.VideoCapture(0)
        self.bridge = CvBridge()

        self.timer = self.create_timer(0.02, self.timer_callback)


    def timer_callback(self):
        
        ret, frame = self.video_capture.read()
        if ret:
            
            msg = self.bridge.cv2_to_imgmsg(frame, encoding='bgr8')
            self.publisher_.publish(msg)

            if self.start_recording:
                self.frames.append(frame)
                self.frame_timestamps.append(time.time() - self.start_time)
                
            

    def controller_callback(self, msg):
        self.get_logger().info(msg.data)
        if msg.data == 'pre_arm':
            self.start_recording = True
            self.start_time = time.time()  # Record the start time
        elif msg.data == 'armed':
            pass
        else:
            self.start_recording = False
            if self.frames:
                self.save_video()
                self.frames = []


    def save_video(self):
        # Generate a timestamp for the folder name
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join('video_output', f'recording_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)

        # Save frames with timestamps
        frame_dir = os.path.join(output_dir, 'frames')
        os.makedirs(frame_dir, exist_ok=True)
        for i, (frame, timestamp) in enumerate(zip(self.frames, self.frame_timestamps)):
            frame_filename = os.path.join(frame_dir, f'frame_{timestamp}.png')
            cv2.imwrite(frame_filename, frame)

        # Compile frames into a video
        video_filename = os.path.join(output_dir, f'video.avi')
        frame_height, frame_width, _ = self.frames[0].shape
        out = cv2.VideoWriter(video_filename, cv2.VideoWriter_fourcc(*'XVID'), 24.0, (frame_width, frame_height))
        for frame in self.frames:
            out.write(frame)
        out.release()

def main(args=None):
    rclpy.init(args=args)
    video_interface = VideoInterface()
    rclpy.spin(video_interface)
    video_interface.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
