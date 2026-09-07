#ifndef COMMON_HPP
#define COMMON_HPP

// C++ std
#include <iostream>
#include <algorithm>
#include <fstream>
#include <chrono>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <map>
#include <deque>
#include <atomic>

// ROS 2
#include "rclcpp/rclcpp.hpp"
#include <std_msgs/msg/header.hpp>
#include "std_msgs/msg/float64.hpp"
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/bool.hpp>
#include "sensor_msgs/msg/image.hpp"
#include "interfaces/msg/motion_capture_state.hpp"

// Eigen
#include <Eigen/Dense>

// OpenCV / cv_bridge
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/core/eigen.hpp>

// ORB-SLAM3
#include "System.h"

// FS
#include <filesystem>
namespace fs = std::filesystem;

class MonocularMode : public rclcpp::Node
{
public:
  MonocularMode();
  ~MonocularMode();

private:
  // -------- Paths / params --------
  std::string homeDir = "";
  std::string packagePath = "ros2_test/src/ros2_orb_slam3/";
  std::string vocFilePath = "";
  std::string settingsFilePath = "";

  // Camera / SLAM params
  double  scaleFilterAlpha = 0.5;
  float   markerLength = 0.15f;
  double  cameraOffsetX = 0.0;
  double  cameraOffsetZ = 0.0;
  double  uptiltDeg = 0.0;

  // Input selection
  std::string input_mode_ = "ros";     // "ros" or "v4l2"
  std::string cap_device_ = "/dev/video0";
  int         cap_width_  = 1280;
  int         cap_height_ = 720;
  double      cap_fps_    = 60.0;

  // Recording
  std::mutex rec_mutex_;
  bool recording_ = false;
  bool writers_ready_ = false;
  int  rec_fps_ = 30;
  std::string out_dir_;
  std::string session_prefix_;
  cv::VideoWriter raw_writer_;
  cv::VideoWriter slam_writer_;
  cv::Size writer_size_{0,0};

  // Delay queue (already in your code)
  int delay_states_{0};
  std::deque<interfaces::msg::MotionCaptureState> out_queue_;

  // ORB-SLAM3
  ORB_SLAM3::System* pAgent = nullptr;
  ORB_SLAM3::System::eSensor sensorType;
  bool enablePangolinWindow = true;
  bool enableOpenCVWindow   = true;

  // State
  Sophus::SE3f lastPose = Sophus::SE3f(Eigen::Matrix3f::Identity(), Eigen::Vector3f::Zero());
  double lastTimestamp = 0.0;
  int trackingState = -1;
  int previousTrackingState = -1;
  bool  scaleInitialized = false;
  float currentScale = 1.0f;

  Eigen::Quaternionf previousQuaternion;
  Eigen::Vector3f    previousTranslation;

  // Velocity filtering
  Eigen::Vector3f filteredLinearVelocity_{0,0,0};
  Eigen::Vector3f filteredAngularVelocity_{0,0,0};
  bool velocityInitialized_{false};

  struct Observation {
    Eigen::Vector3f orbslam_position;
    Eigen::Vector3f aruco_position;
  };
  std::vector<Observation> observations;

  // ROS pubs/subs
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr expConfig_subscription_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr     configAck_publisher_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr subImgMsg_subscription_;
  rclcpp::Subscription<std_msgs::msg::Float64>::SharedPtr  subTimestepMsg_subscription_;
  rclcpp::Publisher<interfaces::msg::MotionCaptureState>::SharedPtr orbSlamState_publisher_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr     arming_state_sub_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr    image_debug_pub_;

  // --- V4L2 capture thread ---
  cv::VideoCapture cap_;
  bool cap_run_{false};
  std::thread cap_thread_;
  void startCaptureLoop();
  void stopCaptureLoop();

  // --- Latest frame buffer (ensures newest frame is always used) ---
  std::mutex latest_frame_mutex_;
  struct FrameBuffer {
    cv::Mat gray;
    rclcpp::Time stamp;
    bool has_frame = false;
  };
  FrameBuffer latest_frame_ros_;
  FrameBuffer latest_frame_v4l2_;
  std::atomic<bool> frame_ready_{false};
  std::thread frame_processor_thread_;
  void frameProcessorLoop();

  // --- Unified frame handler ---
  void handleFrame(const cv::Mat& gray, const rclcpp::Time& stamp);

  // --- Recording helpers ---
  void onArmingState(const std_msgs::msg::Bool& msg);
  void startRecordingIfNeeded(int width, int height, const rclcpp::Time& stamp);
  void stopRecording();
  std::string timeStampString(const rclcpp::Time& t);

  // --- ROS callbacks (placeholders kept) ---
  void experimentSetting_callback(const std_msgs::msg::String& msg);
  void Timestep_callback(const std_msgs::msg::Float64& time_msg);
  void Img_callback(const sensor_msgs::msg::Image &msg);
  void initializeVSLAM(std::string& cfg);

  // -------- Frequency tracking --------
  int hz_log_period_ms_{1000};
  rclcpp::TimerBase::SharedPtr hz_timer_;
  std::atomic<uint64_t> cap_frames_{0};      // V4L2 capture thread frames
  std::atomic<uint64_t> ros_frames_{0};      // /image_raw frames
  std::atomic<uint64_t> orb_frames_{0};      // TrackMonocular calls
  std::atomic<uint64_t> orb_proc_ns_sum_{0}; // total ORB processing time

  // last printed counters
  uint64_t prev_cap_frames_{0};
  uint64_t prev_ros_frames_{0};
  uint64_t prev_orb_frames_{0};
  uint64_t prev_orb_proc_ns_sum_{0};

  std::ofstream timing_log_;

  // Frame recording members
  std::thread frame_saver_thread_;
  std::queue<std::tuple<cv::Mat, std::string, double>> frame_queue_;  // frame, path, timestamp
  std::mutex frame_queue_mutex_;
  std::condition_variable frame_queue_cv_;
  std::atomic<bool> saver_running_{false};
  fs::path current_session_dir_;
  std::ofstream raw_timestamps_file_;
  std::ofstream slam_timestamps_file_;

  // Add this method declaration to the class (in the private section):
  void createVideoScripts();
};

#endif // COMMON_HPP
