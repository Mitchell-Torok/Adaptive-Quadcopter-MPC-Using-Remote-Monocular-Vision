#include "ros2_orb_slam3/common.hpp"
#include <ament_index_cpp/get_package_share_directory.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/core.hpp>   // for FileStorage (YAML intrinsics)
#include <cfloat>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <sys/stat.h>  // Add this for chmod

using std::placeholders::_1;

// ---------------- Constructor ----------------
MonocularMode::MonocularMode() : Node("mono_node_cpp")
{
  homeDir = getenv("HOME");
  RCLCPP_INFO(this->get_logger(), "\nORB-SLAM3-V1 NODE STARTED");

  // ---------- Parameters ----------
  this->declare_parameter("voc_file_arg", "file_not_set");
  this->declare_parameter("settings_file_path_arg", "file_path_not_set");
  this->declare_parameter("scale_filter_alpha", 1.0);
  this->declare_parameter("marker_size", 0.15);
  this->declare_parameter("camera_offset_x", 0.0);
  this->declare_parameter("camera_offset_z", 0.0);
  this->declare_parameter("uptilt_deg", 0.0);

  // Capture parameters
  this->declare_parameter<std::string>("input_mode", "ros");
  this->declare_parameter<std::string>("device", "/dev/video6");
  this->declare_parameter<int>("width", 1280);
  this->declare_parameter<int>("height", 720);
  this->declare_parameter<double>("fps", 60.0);

  // Recording parameters
  this->declare_parameter("recording_fps", 30);
  this->declare_parameter("output_dir", std::string(getenv("HOME")) + "/Videos/ros_orbslam");

  // Frequency logging parameters
  this->declare_parameter<int>("hz_log_period_ms", 1000);

  // ---------- Load parameters ----------
  vocFilePath       = this->get_parameter("voc_file_arg").as_string();
  settingsFilePath  = this->get_parameter("settings_file_path_arg").as_string();
  scaleFilterAlpha  = this->get_parameter("scale_filter_alpha").as_double();
  markerLength      = this->get_parameter("marker_size").as_double();
  cameraOffsetX     = this->get_parameter("camera_offset_x").as_double();
  cameraOffsetZ     = this->get_parameter("camera_offset_z").as_double();
  uptiltDeg         = this->get_parameter("uptilt_deg").as_double();

  input_mode_       = this->get_parameter("input_mode").as_string();
  cap_device_       = this->get_parameter("device").as_string();
  cap_width_        = this->get_parameter("width").as_int();
  cap_height_       = this->get_parameter("height").as_int();
  cap_fps_          = this->get_parameter("fps").as_double();

  rec_fps_          = this->get_parameter("recording_fps").as_int();
  out_dir_          = this->get_parameter("output_dir").as_string();
  hz_log_period_ms_ = this->get_parameter("hz_log_period_ms").as_int();

  try { fs::create_directories(out_dir_); } catch (...) {}

  // Default to the vocabulary and simulation settings shipped with this package.
  // Override with: --ros-args -p voc_file_arg:=<path> -p settings_file_path_arg:=<path>
  const std::string shareDir = ament_index_cpp::get_package_share_directory("ros2_orb_slam3");
  if (vocFilePath == "file_not_set") {
    vocFilePath = shareDir + "/orb_slam3/Vocabulary/ORBvoc.txt.bin";
  }
  if (settingsFilePath == "file_path_not_set" || settingsFilePath == "file_not_set") {
    settingsFilePath = shareDir + "/orb_slam3/config/Monocular/Simulation.yaml";
  }

  RCLCPP_INFO(this->get_logger(), "voc_file: %s", vocFilePath.c_str());
  RCLCPP_INFO(this->get_logger(), "settings_file_path: %s", settingsFilePath.c_str());
  RCLCPP_INFO(this->get_logger(), "Input mode: %s (%s, %dx%d@%.1f)",
              input_mode_.c_str(), cap_device_.c_str(), cap_width_, cap_height_, cap_fps_);

  // ---------- ORB-SLAM3 ----------
  sensorType = ORB_SLAM3::System::MONOCULAR;
  enablePangolinWindow = true;
  enableOpenCVWindow   = true;
  pAgent = new ORB_SLAM3::System(vocFilePath, settingsFilePath, sensorType, enablePangolinWindow);

  // ---------- ROS setup ----------
  orbSlamState_publisher_ = this->create_publisher<interfaces::msg::MotionCaptureState>("orb_slam_state", 10);
  image_debug_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/image_debug", 10);

  // Arm-state subscriber
  arming_state_sub_ = this->create_subscription<std_msgs::msg::Bool>(
      "drone_arming_state_feedback", 10, std::bind(&MonocularMode::onArmingState, this, _1));

  // Keep normal subscriber for simulation with QoS=keep_last(1) to ensure only newest frame
  auto qos = rclcpp::QoS(rclcpp::KeepLast(1));
  subImgMsg_subscription_ = this->create_subscription<sensor_msgs::msg::Image>(
      "/image_raw", qos, std::bind(&MonocularMode::Img_callback, this, _1));

  // --------- Frequency tracking timer ---------
  cap_frames_.store(0);
  ros_frames_.store(0);
  orb_frames_.store(0);
  orb_proc_ns_sum_.store(0);
  prev_cap_frames_ = prev_ros_frames_ = prev_orb_frames_ = prev_orb_proc_ns_sum_ = 0;

  hz_timer_ = this->create_wall_timer(
      std::chrono::milliseconds(std::max(100, hz_log_period_ms_)),
      [this]() {
        const uint64_t c_cap = cap_frames_.load(std::memory_order_relaxed);
        const uint64_t c_ros = ros_frames_.load(std::memory_order_relaxed);
        const uint64_t c_orb = orb_frames_.load(std::memory_order_relaxed);
        const uint64_t c_ns  = orb_proc_ns_sum_.load(std::memory_order_relaxed);

        const uint64_t d_cap = c_cap - prev_cap_frames_;
        const uint64_t d_ros = c_ros - prev_ros_frames_;
        const uint64_t d_orb = c_orb - prev_orb_frames_;
        const uint64_t d_ns  = c_ns  - prev_orb_proc_ns_sum_;

        const double period_s = std::max(0.001, hz_log_period_ms_ / 1000.0);
        const double hz_cap = d_cap / period_s;
        const double hz_ros = d_ros / period_s;
        const double hz_orb = d_orb / period_s;
        const double ms_orb = (d_orb > 0) ? (d_ns * 1e-6 / d_orb) : 0.0;

        RCLCPP_INFO(this->get_logger(),
          "[Hz] cap: %.1f  ros: %.1f  ORB: %.1f  (ORB avg %.2f ms)  [tot cap:%llu ros:%llu orb:%llu]",
          hz_cap, hz_ros, hz_orb, ms_orb,
          (unsigned long long)c_cap, (unsigned long long)c_ros, (unsigned long long)c_orb);

        prev_cap_frames_ = c_cap;
        prev_ros_frames_ = c_ros;
        prev_orb_frames_ = c_orb;
        prev_orb_proc_ns_sum_ = c_ns;
      });

  if (input_mode_ == "v4l2") startCaptureLoop();

  // Always start recording when node starts
  {
    std::lock_guard<std::mutex> lk(rec_mutex_);
    recording_ = true;
    writers_ready_ = false;
    session_prefix_.clear();
    RCLCPP_INFO(this->get_logger(), "[REC] Auto-recording enabled - will start on first frame.");
  }

  // Start frame processor thread
  frame_ready_.store(true, std::memory_order_release);
  frame_processor_thread_ = std::thread([this] { frameProcessorLoop(); });

  RCLCPP_INFO(this->get_logger(), "ORB-SLAM3 node initialised.");
}

// ---------------- Destructor ----------------
MonocularMode::~MonocularMode()
{
  { std::lock_guard<std::mutex> lk(rec_mutex_); stopRecording(); }
  stopCaptureLoop();
  frame_ready_.store(false, std::memory_order_release);
  if (frame_processor_thread_.joinable()) frame_processor_thread_.join();
  if (pAgent) { pAgent->Shutdown(); delete pAgent; pAgent = nullptr; }

  if (timing_log_.is_open()) timing_log_.close();
}

// ---------------- Frame Processor Loop (runs at a fixed rate) ----------------
void MonocularMode::frameProcessorLoop()
{
  rclcpp::WallRate rate(30);  // Run at 15 Hz
  while (rclcpp::ok()) {
    FrameBuffer frame_to_process;
    {
      std::lock_guard<std::mutex> lk(latest_frame_mutex_);
      // Prioritize V4L2 if available, else ROS. Only proceed if a frame is available.
      if (input_mode_ == "v4l2" && latest_frame_v4l2_.has_frame) {
        frame_to_process = latest_frame_v4l2_;
        latest_frame_v4l2_.has_frame = false;  // Mark as processed
      } else if (latest_frame_ros_.has_frame) {
        frame_to_process = latest_frame_ros_;
        latest_frame_ros_.has_frame = false;   // Mark as processed
      }
    }
    
    if (!frame_to_process.gray.empty()) {
      handleFrame(frame_to_process.gray, frame_to_process.stamp);
    }

    rate.sleep();
  }
}

// ---------------- Timestamp utility ----------------
std::string MonocularMode::timeStampString(const rclcpp::Time& t) {
  auto ns = t.nanoseconds();
  std::time_t secs = ns / 1000000000LL;
  std::tm tm_utc; gmtime_r(&secs, &tm_utc);
  char buf[64];
  std::snprintf(buf, sizeof(buf), "%04d%02d%02d_%02d%02d%02d_%03d",
                tm_utc.tm_year + 1900, tm_utc.tm_mon + 1, tm_utc.tm_mday,
                tm_utc.tm_hour, tm_utc.tm_min, tm_utc.tm_sec,
                int((ns/1000000LL)%1000));
  return std::string(buf);
}

// ---------------- Recording ----------------
void MonocularMode::onArmingState(const std_msgs::msg::Bool& msg)
{
  std::lock_guard<std::mutex> lk(rec_mutex_);
  if (msg.data) {
    // Close any session that is already running (e.g. the auto-started one) so the
    // saver thread is joined before a new session re-creates it.
    if (writers_ready_) stopRecording();
    recording_ = true; writers_ready_ = false; session_prefix_.clear();
    RCLCPP_INFO(this->get_logger(), "[REC] Armed -> waiting for first frame.");
  } else {
    if (recording_) RCLCPP_INFO(this->get_logger(), "[REC] Disarmed -> stop.");
    recording_ = false; stopRecording();
  }
}

void MonocularMode::startRecordingIfNeeded(int width, int height, const rclcpp::Time& stamp)
{
  if (!recording_ || writers_ready_) return;
  if (width <= 0 || height <= 0) return;

  session_prefix_ = timeStampString(stamp);
  current_session_dir_ = fs::path(out_dir_) / session_prefix_;
  
  try { 
    fs::create_directories(current_session_dir_ / "raw");
    fs::create_directories(current_session_dir_ / "slam");
  } catch (...) {}

  // Open timestamp files
  raw_timestamps_file_.open((current_session_dir_ / "raw_timestamps.txt").string());
  slam_timestamps_file_.open((current_session_dir_ / "slam_timestamps.txt").string());

  // Close any existing timing log before opening new one
  if (timing_log_.is_open()) {
    timing_log_.close();
  }
  
  // Open timing log file in session directory
  std::string timing_file = (current_session_dir_ / "timing.csv").string();
  timing_log_.open(timing_file, std::ios::out | std::ios::trunc);
  if (timing_log_.is_open()) {
    timing_log_ << "timestamp,handleFrame_ms\n";
    timing_log_.flush();
    RCLCPP_INFO(this->get_logger(), "Logging timing to: %s", timing_file.c_str());
  } else {
    RCLCPP_ERROR(this->get_logger(), "Failed to open timing log: %s", timing_file.c_str());
  }

  // Start background saver thread (join a previous one first; assigning over a
  // joinable std::thread would call std::terminate)
  if (frame_saver_thread_.joinable()) {
    saver_running_ = false;
    frame_queue_cv_.notify_all();
    frame_saver_thread_.join();
  }
  saver_running_ = true;
  frame_saver_thread_ = std::thread([this]() {
    // Open local file handles in the thread for thread-safety
    std::ofstream raw_ts((current_session_dir_ / "raw_timestamps.txt").string(), std::ios::app);
    std::ofstream slam_ts((current_session_dir_ / "slam_timestamps.txt").string(), std::ios::app);
    
    while (saver_running_ || !frame_queue_.empty()) {
      std::tuple<cv::Mat, std::string, double> frame_data;
      {
        std::unique_lock<std::mutex> lk(frame_queue_mutex_);
        frame_queue_cv_.wait(lk, [this] { return !frame_queue_.empty() || !saver_running_; });
        if (frame_queue_.empty()) continue;
        frame_data = std::move(frame_queue_.front());
        frame_queue_.pop();
      }
      
      cv::Mat& frame = std::get<0>(frame_data);
      std::string& path = std::get<1>(frame_data);
      double timestamp = std::get<2>(frame_data);
      
      // Write frame (JPEG quality 95 for speed/quality balance)
      std::vector<int> compression = {cv::IMWRITE_JPEG_QUALITY, 95};
      cv::imwrite(path, frame, compression);
      
      // Write timestamp entry - determine which file based on path
      if (path.find("/raw/") != std::string::npos && raw_ts.is_open()) {
        raw_ts << path << "," << std::fixed << std::setprecision(6) << timestamp << "\n";
        raw_ts.flush();
      } else if (path.find("/slam/") != std::string::npos && slam_ts.is_open()) {
        slam_ts << path << "," << std::fixed << std::setprecision(6) << timestamp << "\n";
        slam_ts.flush();
      }
    }
    
    raw_ts.close();
    slam_ts.close();
  });

  writers_ready_ = true;
  RCLCPP_INFO(this->get_logger(), "[REC] Started frame recording to: %s", current_session_dir_.c_str());
}

void MonocularMode::stopRecording()
{
  if (!writers_ready_) return;
  
  saver_running_ = false;
  frame_queue_cv_.notify_all();
  if (frame_saver_thread_.joinable()) frame_saver_thread_.join();
  
  // Close timestamp files
  if (raw_timestamps_file_.is_open()) raw_timestamps_file_.close();
  if (slam_timestamps_file_.is_open()) slam_timestamps_file_.close();
  
  // Close timing log
  if (timing_log_.is_open()) {
    timing_log_.flush();
    timing_log_.close();
  }
  
  // Clear any remaining frames
  std::lock_guard<std::mutex> lk(frame_queue_mutex_);
  while (!frame_queue_.empty()) frame_queue_.pop();
  
  writers_ready_ = false;
  
  // Create video compile scripts
  createVideoScripts();
  
  RCLCPP_INFO(this->get_logger(), "[REC] Stopped. Run: %s/compile_videos.sh", current_session_dir_.c_str());
}

void MonocularMode::createVideoScripts()
{
  std::string script_path = (current_session_dir_ / "compile_videos.sh").string();
  std::ofstream script(script_path);
  
  script << "#!/bin/bash\n";
  script << "# Auto-generated video compilation script\n\n";
  script << "DIR=\"$(cd \"$(dirname \"${BASH_SOURCE[0]}\")\" && pwd)\"\n\n";
  
  // Function to create concat file with timestamps
  script << "create_concat_file() {\n";
  script << "  local frames_dir=\"$1\"\n";
  script << "  local timestamps_file=\"$2\"\n";
  script << "  local concat_file=\"$3\"\n";
  script << "  \n";
  script << "  if [ ! -f \"$timestamps_file\" ]; then\n";
  script << "    echo \"Error: $timestamps_file not found\"\n";
  script << "    return 1\n";
  script << "  fi\n";
  script << "  \n";
  script << "  > \"$concat_file\"\n";
  script << "  \n";
  script << "  local prev_ts=\"\"\n";
  script << "  local frame_num=0\n";
  script << "  local prev_path=\"\"\n";
  script << "  \n";
  script << "  while IFS=',' read -r frame_path timestamp; do\n";
  script << "    if [ ! -f \"$frame_path\" ]; then continue; fi\n";
  script << "    \n";
  script << "    if [ $frame_num -gt 0 ] && [ -n \"$prev_ts\" ]; then\n";
  script << "      local duration=$(awk \"BEGIN {printf \\\"%.6f\\\", $timestamp - $prev_ts}\")\n";
  script << "      if (( $(awk \"BEGIN {print ($duration > 0)}\") )); then\n";
  script << "        echo \"file '$prev_path'\" >> \"$concat_file\"\n";
  script << "        echo \"duration $duration\" >> \"$concat_file\"\n";
  script << "      fi\n";
  script << "    fi\n";
  script << "    \n";
  script << "    prev_ts=$timestamp\n";
  script << "    prev_path=$frame_path\n";
  script << "    frame_num=$((frame_num + 1))\n";
  script << "  done < \"$timestamps_file\"\n";
  script << "  \n";
  script << "  # Add last frame (repeat for final duration)\n";
  script << "  if [ $frame_num -gt 0 ] && [ -n \"$prev_path\" ]; then\n";
  script << "    echo \"file '$prev_path'\" >> \"$concat_file\"\n";
  script << "  fi\n";
  script << "}\n\n";
  
  // Compile raw video
  script << "echo \"Compiling raw video...\"\n";
  script << "if create_concat_file \"$DIR/raw\" \"$DIR/raw_timestamps.txt\" \"$DIR/raw_concat.txt\"; then\n";
  script << "  ffmpeg -y -f concat -safe 0 -i \"$DIR/raw_concat.txt\" -c:v libx264 -crf 18 -pix_fmt yuv420p \"$DIR/raw.mp4\"\n";
  script << "  echo \"Raw video saved: $DIR/raw.mp4\"\n";
  script << "else\n";
  script << "  echo \"Failed to create raw concat file\"\n";
  script << "fi\n\n";
  
  // Compile SLAM video
  script << "echo \"Compiling SLAM video...\"\n";
  script << "if create_concat_file \"$DIR/slam\" \"$DIR/slam_timestamps.txt\" \"$DIR/slam_concat.txt\"; then\n";
  script << "  ffmpeg -y -f concat -safe 0 -i \"$DIR/slam_concat.txt\" -c:v libx264 -crf 18 -pix_fmt yuv420p \"$DIR/slam.mp4\"\n";
  script << "  echo \"SLAM video saved: $DIR/slam.mp4\"\n";
  script << "else\n";
  script << "  echo \"Failed to create SLAM concat file\"\n";
  script << "fi\n\n";
  
  script << "echo \"Done!\"\n";
  
  script.close();
  
  // Make executable
  chmod(script_path.c_str(), 0755);
}

// ---------------- ROS Image (Sim) ----------------
void MonocularMode::Img_callback(const sensor_msgs::msg::Image &msg)
{
  try {
    auto cv_ptr = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::MONO8);
    if (cv_ptr->image.empty()) return;
    ros_frames_.fetch_add(1, std::memory_order_relaxed);
    
    // Store the latest frame (overwrites any older frame in buffer)
    {
      std::lock_guard<std::mutex> lk(latest_frame_mutex_);
      latest_frame_ros_.gray = cv_ptr->image.clone();
      latest_frame_ros_.stamp = msg.header.stamp;
      latest_frame_ros_.has_frame = true;
    }
    frame_ready_.store(true, std::memory_order_release);
  } catch (const cv_bridge::Exception &e) {
    RCLCPP_ERROR(this->get_logger(), "cv_bridge exception: %s", e.what());
  }
}

// ---------------- V4L2 Capture ----------------
void MonocularMode::startCaptureLoop()
{
  if (cap_run_) return;
  if (!cap_.open(cap_device_, cv::CAP_V4L2)) {
    RCLCPP_ERROR(this->get_logger(), "Failed to open capture device: %s", cap_device_.c_str());
    return;
  }
  cap_.set(cv::CAP_PROP_FRAME_WIDTH,  cap_width_);
  cap_.set(cv::CAP_PROP_FRAME_HEIGHT, cap_height_);
  cap_.set(cv::CAP_PROP_FPS,          cap_fps_);

  // Log negotiated
  double got_w = cap_.get(cv::CAP_PROP_FRAME_WIDTH);
  double got_h = cap_.get(cv::CAP_PROP_FRAME_HEIGHT);
  double got_f = cap_.get(cv::CAP_PROP_FPS);
  RCLCPP_INFO(this->get_logger(), "V4L2 negotiated %dx%d @ %.2f FPS",
              (int)got_w, (int)got_h, got_f);

  cap_run_ = true;
  cap_thread_ = std::thread([this] {
    cv::Mat frame, gray;
    rclcpp::WallRate rate(1000);
    while (rclcpp::ok() && cap_run_) {
      if (!cap_.read(frame) || frame.empty()) { rate.sleep(); continue; }

      cap_frames_.fetch_add(1, std::memory_order_relaxed);

      // Convert to gray
      if (frame.channels() == 1) {
        gray = frame;
      } else if (frame.channels() == 3) {
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
      } else if (frame.channels() == 2) {
        if ((frame.cols & 1) != 0) frame = frame.colRange(0, frame.cols - 1);
        cv::cvtColor(frame, gray, cv::COLOR_YUV2GRAY_YUY2);
      } else {
        RCLCPP_WARN_THROTTLE(this->get_logger(), *this->get_clock(), 2000,
                             "Unexpected channels=%d", frame.channels());
        continue;
      }

      if (gray.empty() || gray.cols <= 0 || gray.rows <= 0) continue;
      auto stamp = this->get_clock()->now();
      
      // Store the latest frame (overwrites any older frame in buffer)
      {
        std::lock_guard<std::mutex> lk(latest_frame_mutex_);
        latest_frame_v4l2_.gray = gray.clone();
        latest_frame_v4l2_.stamp = stamp;
        latest_frame_v4l2_.has_frame = true;
      }
      frame_ready_.store(true, std::memory_order_release);
    }
  });
}

void MonocularMode::stopCaptureLoop()
{
  if (!cap_run_) return;
  cap_run_ = false;
  if (cap_thread_.joinable()) cap_thread_.join();
  if (cap_.isOpened()) cap_.release();
}

// ---------------- Unified Frame Handler ----------------
void MonocularMode::handleFrame(const cv::Mat& gray, const rclcpp::Time& stamp)
{
  auto handle_t0 = std::chrono::steady_clock::now();
  
  if (gray.empty() || gray.cols <= 0 || gray.rows <= 0) return;


  // --- ORB-SLAM3 processing with timing ---
  auto t0 = std::chrono::steady_clock::now();
  double ts = stamp.seconds();
  Sophus::SE3f Tcw = pAgent->TrackMonocular(gray, ts);
  auto t1 = std::chrono::steady_clock::now();

  orb_frames_.fetch_add(1, std::memory_order_relaxed);
  const uint64_t dt_ns =
      (uint64_t) std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
  orb_proc_ns_sum_.fetch_add(dt_ns, std::memory_order_relaxed);

  previousTrackingState = trackingState;
  trackingState = pAgent->GetTrackingState();

  // Pose transform
  Eigen::Matrix3f R_transform;
  R_transform << 0,0,-1, 1,0,0, 0,1,0;
  Eigen::Matrix3f R_new = (R_transform * Tcw.so3().matrix() * R_transform.transpose());
  Eigen::Vector3f t_new = R_transform * Tcw.translation();

  Eigen::Vector3f t_scale = (scaleInitialized) ? t_new / currentScale : t_new;
  Eigen::Quaternionf q(R_new);
  q.x() = -q.x(); q.y() = -q.y(); q.z() = -q.z();

  const float deg = static_cast<float>(uptiltDeg);
  const float rad = -deg * float(M_PI) / 180.0f;  // negative to remove uptilt
  Eigen::AngleAxisf Ry(rad, Eigen::Vector3f::UnitY());
  Eigen::Vector3f t_corr = Ry * t_scale;

  // Velocities & publish
  if (lastTimestamp > 0.0) {
    double dt = ts - lastTimestamp;
    if (dt > 0.0001) {
      Eigen::Vector3f v_lin = (t_corr - previousTranslation) / (float)dt;
      Eigen::Quaternionf dq = q * previousQuaternion.conjugate();
      Eigen::Vector3f w_world = 2.0f * Eigen::Vector3f(dq.x(), dq.y(), dq.z()) / (float)dt;
      Eigen::Vector3f w_body  = q.toRotationMatrix().transpose() * w_world;

      // Apply low-pass filter to velocities
      if (!velocityInitialized_) {
        filteredLinearVelocity_ = v_lin;
        filteredAngularVelocity_ = w_body;
        velocityInitialized_ = true;
      } else {
        const float alpha = static_cast<float>(scaleFilterAlpha);
        filteredLinearVelocity_ = alpha * v_lin + (1.0f - alpha) * filteredLinearVelocity_;
        filteredAngularVelocity_ = alpha * w_body + (1.0f - alpha) * filteredAngularVelocity_;
      }

      interfaces::msg::MotionCaptureState msg;
      msg.header.stamp = stamp;
      msg.header.frame_id = "map";
      msg.child_frame_id = "base_link";
      msg.pose.position.x = t_corr.x();
      msg.pose.position.y = t_corr.y();
      msg.pose.position.z = t_corr.z();
      msg.pose.orientation.x = q.x();
      msg.pose.orientation.y = q.y();
      msg.pose.orientation.z = q.z();
      msg.pose.orientation.w = q.w();
      msg.twist.linear.x = filteredLinearVelocity_.x();
      msg.twist.linear.y = filteredLinearVelocity_.y();
      msg.twist.linear.z = filteredLinearVelocity_.z();
      msg.twist.angular.x = filteredAngularVelocity_.x();
      msg.twist.angular.y = filteredAngularVelocity_.y();
      msg.twist.angular.z = filteredAngularVelocity_.z();

      orbSlamState_publisher_->publish(msg);

      /*
        out_queue_.push_back(msg);
        if ((int)out_queue_.size() > delay_states_) {
          orbSlamState_publisher_->publish(out_queue_.front());
          out_queue_.pop_front();
        }*/
      
    }
  }


  if (!scaleInitialized) {
    // Detect ArUco markers in the current grayscale frame
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_5X5_50);
    std::vector<int> markerIds;
    std::vector<std::vector<cv::Point2f>> markerCorners;
    cv::aruco::detectMarkers(gray, dictionary, markerCorners, markerIds);

    if (!markerIds.empty()) {
      // Load camera intrinsics from settings YAML (Camera1.* keys)
      cv::Mat cameraMatrix, distCoeffs;
      try {
        cv::FileStorage fs(settingsFilePath, cv::FileStorage::READ);
        if (!fs.isOpened()) {
          RCLCPP_ERROR(this->get_logger(), "Failed to open settings file: %s", settingsFilePath.c_str());
        } else {
          double fx, fy, cx, cy;
          fs["Camera1.fx"] >> fx;  fs["Camera1.fy"] >> fy;
          fs["Camera1.cx"] >> cx;  fs["Camera1.cy"] >> cy;
          cameraMatrix = (cv::Mat_<double>(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);

          double k1, k2, p1, p2, k3;
          fs["Camera1.k1"] >> k1; fs["Camera1.k2"] >> k2;
          fs["Camera1.p1"] >> p1; fs["Camera1.p2"] >> p2;
          fs["Camera1.k3"] >> k3;
          distCoeffs = (cv::Mat_<double>(5,1) << k1, k2, p1, p2, k3);
          fs.release();

          // Pose of each marker (in camera frame)
          std::vector<cv::Vec3d> rvecs, tvecs;
          cv::aruco::estimatePoseSingleMarkers(markerCorners, markerLength, cameraMatrix, distCoeffs, rvecs, tvecs);

          // Log detected markers
          for (size_t i = 0; i < markerIds.size(); i++) {
              double distance = cv::norm(tvecs[i]);
          }

          // Define the transformation matrix to align with ORB-SLAM
          Eigen::Matrix3f R_transform;
          R_transform << 
                  0,  0, -1,
                  1,  0,  0,
                  0,  1,  0;

          // Declare tvec_transformed outside the loop to use it later
          Eigen::Vector3f tvec_transformed;

          // Adjust the translation vectors to align with ORB-SLAM
          for (size_t i = 0; i < markerIds.size(); i++) {
              Eigen::Vector3f tvec_original(tvecs[i][0], tvecs[i][1], tvecs[i][2]);
              tvec_transformed = R_transform * tvec_original;

              // Convert Eigen::Vector3f to cv::Mat for norm calculation
              cv::Mat tvec_transformed_mat = (cv::Mat_<double>(3, 1) << tvec_transformed.x(), tvec_transformed.y(), tvec_transformed.z());
              double distance = cv::norm(tvec_transformed_mat);
          }

          RCLCPP_INFO(this->get_logger(), "Pos mark: [x: %.4f, y: %.4f, z: %.4f]", tvec_transformed.x(), tvec_transformed.y(), tvec_transformed.z());
          RCLCPP_INFO(this->get_logger(), "Have %zu/16 reference points", observations.size());

          // Collect observation when SLAM is tracking (state==2) and we saw at least 1 marker
          if (pAgent->GetTrackingState() == 2 && !markerIds.empty()) {
            Eigen::Vector3f orbslam_position = t_new;
            Eigen::Vector3f aruco_position   = tvec_transformed;

            if (observations.empty() ||
                (aruco_position - observations.back().aruco_position).norm() > 0.25f) {
              observations.push_back({orbslam_position, aruco_position});
            }

            // Compute scale once we have ≥10 spaced observations
            if (observations.size() >= 16) {
              double scale_sum = 0.0;
              int    count     = 0;
              for (size_t i = 1; i < observations.size(); ++i) {
                Eigen::Vector3f orb_diff  = observations[i].orbslam_position - observations[i-1].orbslam_position;
                Eigen::Vector3f aru_diff  = observations[i].aruco_position - observations[i-1].aruco_position;
                const double d_orb = orb_diff.norm();
                const double d_aru = aru_diff.norm();
                if (d_aru > 0.0) {
                  scale_sum += d_orb / d_aru;
                  ++count;
                }
              }
              if (count > 0) {
                currentScale = scale_sum / static_cast<double>(count);
                RCLCPP_INFO(this->get_logger(), "Updated scale: %.3f", currentScale);
                scaleInitialized = true;
              }
            }
          }
        }
      } catch (const std::exception &e) {
        RCLCPP_ERROR(this->get_logger(),  "Failed to load camera parameters: %s", e.what());
      }
    }
  }

  lastPose = Tcw;
  lastTimestamp = ts;
  previousQuaternion = q;
  previousTranslation = t_corr;

  // Log timing only after scale initialized
  if (scaleInitialized && recording_ && timing_log_.is_open()) {
    auto handle_t1 = std::chrono::steady_clock::now();
    double handle_ms = std::chrono::duration<double, std::milli>(handle_t1 - handle_t0).count();
    timing_log_ << std::fixed << std::setprecision(6) << ts << "," << handle_ms << "\n";
    timing_log_.flush();
  }

  // Recording
  if (recording_) {
    std::lock_guard<std::mutex> lk(rec_mutex_);
    startRecordingIfNeeded(gray.cols, gray.rows, stamp);
    
    if (writers_ready_) {
      static int raw_frame_num = 0;
      static int slam_frame_num = 0;
      double timestamp = stamp.seconds();
      
      // Queue raw frame with timestamp
      {
        std::lock_guard<std::mutex> qlk(frame_queue_mutex_);
        std::string raw_path = (current_session_dir_ / "raw" / 
                               (std::to_string(++raw_frame_num) + ".jpg")).string();
        
        frame_queue_.push({gray.clone(), raw_path, timestamp});
        frame_queue_cv_.notify_one();
      }
      
      // Get and save SLAM visualization frame
      cv::Mat slam_frame = pAgent->GetCurrentFrame();
      if (!slam_frame.empty()) {
        std::lock_guard<std::mutex> qlk(frame_queue_mutex_);
        std::string slam_path = (current_session_dir_ / "slam" / 
                                (std::to_string(++slam_frame_num) + ".jpg")).string();
        
        // Convert to BGR if it's grayscale (for better visualization)
        cv::Mat slam_bgr;
        if (slam_frame.channels() == 1) {
          cv::cvtColor(slam_frame, slam_bgr, cv::COLOR_GRAY2BGR);
        } else {
          slam_bgr = slam_frame;
        }
        
        frame_queue_.push({slam_bgr.clone(), slam_path, timestamp});
        frame_queue_cv_.notify_one();
      }
    }
  }
}

// ---------------- Placeholders ----------------
void MonocularMode::experimentSetting_callback(const std_msgs::msg::String& msg) { (void)msg; }
void MonocularMode::Timestep_callback(const std_msgs::msg::Float64& msg) { (void)msg; }
void MonocularMode::initializeVSLAM(std::string& cfg) { (void)cfg; }
