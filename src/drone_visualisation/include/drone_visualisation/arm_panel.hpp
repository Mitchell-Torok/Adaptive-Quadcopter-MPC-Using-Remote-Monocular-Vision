#ifndef DRONE_VISUALISATION__ARM_PANEL_HPP_
#define DRONE_VISUALISATION__ARM_PANEL_HPP_

#include <QVBoxLayout>
#include <QPushButton>
#include <QShortcut>
#include <QLabel>
#include <rviz_common/panel.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_msgs/msg/bool.hpp>
#include <interfaces/srv/set_arming.hpp>
#include <interfaces/msg/telemetry.hpp>
#include <interfaces/srv/set_arming.hpp>
#include <chrono>

namespace drone_visualisation
{

class ArmPanel : public rviz_common::Panel
{
  Q_OBJECT

public:
  explicit ArmPanel(QWidget* parent = nullptr);
  
  void onInitialize() override;

private Q_SLOTS:
  void onButtonPressed();
  void onSpacePressed();
  void onTakeoffPressed();

private:
  void updateButtonState();
  void updateStatusLabel();
  void armingStateCallback(const std_msgs::msg::Bool::SharedPtr msg);
  void telemetryCallback(const interfaces::msg::Telemetry::SharedPtr msg);
  void callArmingService(bool arm);
  
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr command_pub_;
  rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr arming_state_sub_;
  rclcpp::Subscription<interfaces::msg::Telemetry>::SharedPtr telemetry_sub_;
  rclcpp::Client<interfaces::srv::SetArming>::SharedPtr arming_client_;
  
  QPushButton* arm_button_;
  QPushButton* takeoff_button_;
  QShortcut* space_shortcut_;
  QLabel* status_label_;
  QLabel* battery_label_;
  
  bool is_armed_;
  float battery_voltage_;
  std::string status_message_;
};

}  // namespace drone_visualisation

#endif  // DRONE_VISUALISATION__ARM_PANEL_HPP_
