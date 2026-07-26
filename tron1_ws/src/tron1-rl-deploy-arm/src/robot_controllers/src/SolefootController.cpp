// Copyright information
//
// © [2024] LimX Dynamics Technology Co., Ltd. All rights reserved.

#include "robot_controllers/SolefootController.h"

#include <angles/angles.h>
#include <pluginlib/class_list_macros.hpp>
#include <std_msgs/Float32MultiArray.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>

namespace robot_controller {

// Initialize the controller
bool SolefootController::init(hardware_interface::RobotHW *robot_hw, ros::NodeHandle &nh) {
  // initialize initial positions
  standCenterPos_.setZero(8 + 6);
  stopSitPos_.setZero(8 + 6);
  standJointPos_.setZero(8);

  ee_init_pos_ << 0.145308, 0.000000, 0.140205;
  ee_init_rpy_ << 0.000000, 0.500000, 0.000000;


  // register publishers and subscribers
  gripper_cmd_pub_ = nh.advertise<std_msgs::Bool>("/gripper_cmd", 10, false);
  obs_debug_pub_ = nh.advertise<std_msgs::Float32MultiArray>("/obs_debug_info", 10, false);

  ee_pos_cmd_rc_delta_ =
      nh.subscribe<std_msgs::Float32MultiArray>("/EEPose_cmd_rc", 10, &SolefootController::EEPoseCmdRCCallback, this);
  ee_pos_cmd_rc_delta_msg_.data.resize(6+1);
  rc_ee_cmd.zero();
  lastEePos_ = ee_init_pos_;
  lastEeRpy_ = ee_init_rpy_;

  // Simulation-only: subscribe to gazebo ground truth odometry for base linear velocity.
  // This makes the policy obs closer to IsaacLab (which uses true base_lin_vel).
  bool isSim = false;
  nh.param<bool>("/is_sim", isSim, false);
  useGroundTruthBaseLinVel_ = isSim;
  nh.param<bool>("/PointfootCfg/use_ground_truth_base_lin_vel", useGroundTruthBaseLinVel_, useGroundTruthBaseLinVel_);
  nh.param<bool>("/PointfootCfg/ground_truth_twist_in_world_frame", groundTruthTwistInWorldFrame_, false);
  nh.param<bool>("/PointfootCfg/ee_target/use_world_frame", useWorldFrameEeTarget_, false);
  nh.param<bool>("/PointfootCfg/ee_target/anchor_base_pose", anchorEeTargetToFirstBasePose_, true);
  nh.param<bool>("/PointfootCfg/ee_target/manual_enable", manualEeTargetWorldEnabled_, false);
  {
    std::vector<double> pos;
    if (nh.getParam("/PointfootCfg/ee_target/manual_set_pos", pos) && pos.size() == 3) {
      manualEeTargetWorldPos_ << pos[0], pos[1], pos[2];
    }
    std::vector<double> rpy;
    if (nh.getParam("/PointfootCfg/ee_target/manual_set_rpy", rpy) && rpy.size() == 3) {
      manualEeTargetWorldRpy_ << rpy[0], rpy[1], rpy[2];
    }
  }

  // Subscribe ground truth odom if needed for either base_lin_vel obs or world-frame EE target tracking.
  if (useGroundTruthBaseLinVel_ || (isSim && useWorldFrameEeTarget_)) {
    ground_truth_sub_ =
        nh.subscribe<nav_msgs::Odometry>("/ground_truth/state", 10, &SolefootController::groundTruthCallback, this);
  }

  // Optional: world-frame EE target pose (odom/world). 
  // If not received, we use manual target if enabled, otherwise we can anchor base-frame targets into world.
  if (isSim && useWorldFrameEeTarget_) {
    std::string topic = "/EEPose_target_world";
    nh.param<std::string>("/PointfootCfg/ee_target/world_topic", topic, topic);
    ee_target_world_sub_ =
        nh.subscribe<geometry_msgs::PoseStamped>(topic, 10, &SolefootController::EEPoseTargetWorldCallback, this);
    ROS_INFO_STREAM("EE target in world frame enabled. topic=" << topic
                                                               << " anchor_base_pose="
                                                               << (anchorEeTargetToFirstBasePose_ ? "true" : "false"));
  }

  return ControllerBase::init(robot_hw, nh);
}

// Perform initialization when the controller starts
void SolefootController::starting(const ros::Time &time) {
  const int numJoints = static_cast<int>(hybridJointHandles_.size());
  if (defaultJointAngles_.size() != numJoints) {
    defaultJointAngles_.resize(numJoints);
  }

  for (size_t i = 0; i < hybridJointHandles_.size(); i++) {
    ROS_INFO_STREAM("starting hybridJointHandle: " << hybridJointHandles_[i].getPosition());
    defaultJointAngles_[i] = hybridJointHandles_[i].getPosition();
  }

  standCenterPos_.setZero(static_cast<int>(hybridJointHandles_.size()));
  stopSitPos_.setZero(static_cast<int>(hybridJointHandles_.size()));
  standJointPos_.setZero(8);

  for (size_t i = 0; i < jointNames_.size() && i < static_cast<size_t>(standCenterPos_.size()); ++i) {
    const std::string& name = jointNames_[i];
    if (name == "J2") {
      standCenterPos_(static_cast<int>(i)) = 0.5;
    }
    // stand posture
    if (name.find("hip_L") != std::string::npos) {
      standCenterPos_(static_cast<int>(i)) = 0.6;
      stopSitPos_(static_cast<int>(i)) = 1.0;
    } else if (name.find("knee_L") != std::string::npos) {
      standCenterPos_(static_cast<int>(i)) = 1.2;
      stopSitPos_(static_cast<int>(i)) = 1.35;
    } else if (name.find("ankle_L") != std::string::npos) {
      standCenterPos_(static_cast<int>(i)) = -0.7;
      stopSitPos_(static_cast<int>(i)) = -1.1;
    } else if (name.find("hip_R") != std::string::npos) {
      standCenterPos_(static_cast<int>(i)) = -0.6;
      stopSitPos_(static_cast<int>(i)) = -1.0;
    } else if (name.find("knee_R") != std::string::npos) {
      standCenterPos_(static_cast<int>(i)) = -1.2;
      stopSitPos_(static_cast<int>(i)) = -1.35;
    } else if (name.find("ankle_R") != std::string::npos) {
      standCenterPos_(static_cast<int>(i)) = -0.7;
      stopSitPos_(static_cast<int>(i)) = -1.1;
    }
  }

  // Ankle joint correction in stand stage (legacy)
  if (jointNames_.size() >= 8 && standJointPos_.size() >= 8) {
    for (int i = 0; i < 8 && i < static_cast<int>(jointNames_.size()); ++i) {
      const std::string& name = jointNames_[static_cast<size_t>(i)];
      if (name.find("ankle") != std::string::npos) {
      }
    }
  }

  scalar_t durationSecs = 1.0;
  standDuration_ = durationSecs * 1800.0;
  standCenterDuration_ = durationSecs * 1000.0;
  standCenterPercent_ = 0.0;
  stopCenterPercent_ = 0.0;
  stopCenterDuration_ = durationSecs * 750.0;

  initStandPercent_ = 0.0;
  initStandDuration_ = durationSecs * 800.0;
  // commands are not used in this task
  commandX_.clear();
  commandY_.clear();
  commandYaw_.clear();

  commandX_.push_back(0.0);
  commandY_.push_back(0.0);
  commandYaw_.push_back(0.0);

  loopCount_ = 0;
  loopCountKeep_ = 0;
  work_mode_flag_ = 10;
  mode_ = Mode::SOLE_WALK;

  gaitIndex_ = 0.0;
}

// Update function called periodically
void SolefootController::update(const ros::Time &time, const ros::Duration &period) {
  switch (mode_)
  {
  case Mode::SOLE_STAND:
      handleSoleStandMode();
      break;
  case Mode::SOLE_WALK:
      handleRLSoleWalkMode();
      break;
  case Mode::SOLE_STOP:
      handleSoleStopMode();
      break;
  }
  loopCount_++;
  loopCountKeep_++;
  last_mode_ = mode_;
}

void SolefootController::handleSoleStandMode() {
  const int numJoints = static_cast<int>(hybridJointHandles_.size());
  auto ensureSizeAndZeroNew = [](vector_t& vec, int size) {
    if (vec.size() >= size) {
      return;
    }
    const int oldSize = static_cast<int>(vec.size());
    vec.conservativeResize(size);
    vec.segment(oldSize, size - oldSize).setZero();
  };
  ensureSizeAndZeroNew(defaultJointAngles_, numJoints);
  ensureSizeAndZeroNew(standCenterPos_, numJoints);

  if (standCenterPercent_ < 1)
  {
    for (int j = 0; j < hybridJointHandles_.size(); j++)
    {
      const std::string* namePtr =
          (static_cast<size_t>(j) < jointNames_.size()) ? &jointNames_[static_cast<size_t>(j)] : nullptr;
      const bool isArmJ123 = (namePtr != nullptr) && (*namePtr == "J1" || *namePtr == "J2" || *namePtr == "J3");
      const bool isArmJ456 = (namePtr != nullptr) && (*namePtr == "J4" || *namePtr == "J5" || *namePtr == "J6");

      if (isArmJ123) { // arm j123
        scalar_t pos_des = defaultJointAngles_[j] * (1 - standCenterPercent_) + standCenterPos_[j] * standCenterPercent_;
        hybridJointHandles_[j].setCommand(pos_des, 0, 20, 1, 0, 0);
      }
      else if (isArmJ456) { // arm j456
        scalar_t pos_des = defaultJointAngles_[j] * (1 - standCenterPercent_) + standCenterPos_[j] * standCenterPercent_;
        hybridJointHandles_[j].setCommand(pos_des, 0, 10, 1, 0, 0);
      }
      else{
        scalar_t pos_des = defaultJointAngles_[j] * (1 - standCenterPercent_) + standCenterPos_[j] * standCenterPercent_;
        hybridJointHandles_[j].setCommand(pos_des, 0, 100, 6, 0, 0);
      }
    }
    standCenterPercent_ += 1 / standCenterDuration_;
  }
  else
  {
    mode_ = Mode::SOLE_WALK;
  }
}

void SolefootController::handleSoleStopMode() {
  const int numJoints = static_cast<int>(hybridJointHandles_.size());
  auto ensureSizeAndZeroNew = [](vector_t& vec, int size) {
    if (vec.size() >= size) {
      return;
    }
    const int oldSize = static_cast<int>(vec.size());
    vec.conservativeResize(size);
    vec.segment(oldSize, size - oldSize).setZero();
  };
  ensureSizeAndZeroNew(defaultJointAngles_, numJoints);
  ensureSizeAndZeroNew(stopSitPos_, numJoints);

  if (!stopJointAnglesUpdated_) {
    for (size_t i = 0; i < hybridJointHandles_.size(); i++) {
      ROS_INFO_STREAM("updating stop hybridJointHandle: " << hybridJointHandles_[i].getPosition());
      defaultJointAngles_[i] = hybridJointHandles_[i].getPosition();
    }
    stopJointAnglesUpdated_ = true;
  }
  if (stopCenterPercent_ < 1)
  {
    for (int j = 0; j < hybridJointHandles_.size(); j++)
    {
      const std::string* namePtr =
          (static_cast<size_t>(j) < jointNames_.size()) ? &jointNames_[static_cast<size_t>(j)] : nullptr;
      const bool isArmJ123 = (namePtr != nullptr) && (*namePtr == "J1" || *namePtr == "J2" || *namePtr == "J3");
      const bool isArmJ456 = (namePtr != nullptr) && (*namePtr == "J4" || *namePtr == "J5" || *namePtr == "J6");

      if (isArmJ123) { // arm j123
        scalar_t pos_des = defaultJointAngles_[j] * (1 - stopCenterPercent_) + stopSitPos_[j] * stopCenterPercent_;
        hybridJointHandles_[j].setCommand(pos_des, 0, 20, 1, 0, 0);
      }
      else if (isArmJ456) { // arm j456
        scalar_t pos_des = defaultJointAngles_[j] * (1 - stopCenterPercent_) + stopSitPos_[j] * stopCenterPercent_;
        hybridJointHandles_[j].setCommand(pos_des, 0, 10, 1, 0, 0);
      }
      else{
        scalar_t pos_des = defaultJointAngles_[j] * (1 - stopCenterPercent_) + stopSitPos_[j] * stopCenterPercent_;
        hybridJointHandles_[j].setCommand(pos_des, 0, 100, 6, 0, 0);
      }
    }
    stopCenterPercent_ += 1 / stopCenterDuration_;
  }
}

void SolefootController::handleRLSoleWalkMode()
{
  locomotionFlag_ = 0;
  sitDownFlag_ = 0;
  if (work_mode_flag_ != 1)
  {
    ROS_INFO_STREAM("------------------RL Sole Walk------------------");

    if (!loadRLCfg()) {
      ROS_ERROR("Failed to load RL config. Stay in SOLE_STAND.");
      work_mode_flag_ = 10;
      mode_ = Mode::SOLE_STAND;
      return;
    }
    if (!loadModel()) {
      ROS_ERROR("Failed to load ONNX models. Stay in SOLE_STAND.");
      work_mode_flag_ = 10;
      mode_ = Mode::SOLE_STAND;
      return;
    }

    if (static_cast<int>(hybridJointHandles_.size()) != actionsSize_) {
      ROS_ERROR_STREAM("actions_size(" << actionsSize_ << ") != num_joints(" << hybridJointHandles_.size() << ")");
      work_mode_flag_ = 10;
      mode_ = Mode::SOLE_STAND;
      return;
    }
    if (static_cast<int>(initJointAngles_.size()) != static_cast<int>(hybridJointHandles_.size())) {
      ROS_ERROR_STREAM("initJointAngles_ size(" << initJointAngles_.size() << ") != num_joints("
                                               << hybridJointHandles_.size() << ")");
      work_mode_flag_ = 10;
      mode_ = Mode::SOLE_STAND;
      return;
    }
    if (lastActions_.size() != actionsSize_) {
      lastActions_.resize(actionsSize_);
      lastActions_.setZero();
    }

    isfirstRecObs_ = true;
    work_mode_flag_ = 1;
  }
  
  // compute observation & actions
  if (soleRobotCfg_.controlCfg.decimation == 0)
  {
    std::cerr << "----error----  soleRobotCfg_.controlCfg.decimation" << std::endl;
    return;
  }
  if (loopCount_ % soleRobotCfg_.controlCfg.decimation == 0)
  {
    computeObservation();
    computeEncoder();
    computeActions();
    // limit action range
    scalar_t actionMin = -soleRobotCfg_.soleRlCfg.clipActions;
    scalar_t actionMax = soleRobotCfg_.soleRlCfg.clipActions;
    std::transform(actions_.begin(), actions_.end(), actions_.begin(),
                    [actionMin, actionMax](scalar_t x)
                    { return std::max(actionMin, std::min(actionMax, x)); });
  }
  
  // set action
  vector_t jointPos(hybridJointHandles_.size()), jointVel(hybridJointHandles_.size());
  for (size_t i = 0; i < hybridJointHandles_.size(); i++)
  {
    jointPos(i) = hybridJointHandles_[i].getPosition();
    jointVel(i) = hybridJointHandles_[i].getVelocity();
  }
  for (int i = 0; i < hybridJointHandles_.size(); i++)
  {
    const std::string* namePtr =
        (static_cast<size_t>(i) < jointNames_.size()) ? &jointNames_[static_cast<size_t>(i)] : nullptr;
    const bool isArmJ123 = (namePtr != nullptr) && (*namePtr == "J1" || *namePtr == "J2" || *namePtr == "J3");
    const bool isArmJ456 = (namePtr != nullptr) && (*namePtr == "J4" || *namePtr == "J5" || *namePtr == "J6");
    const bool isAnkle = (namePtr != nullptr) && (namePtr->find("ankle") != std::string::npos);

    scalar_t stiffness = soleRobotCfg_.controlCfg.leg_joint_stiffness;
    scalar_t damping = soleRobotCfg_.controlCfg.leg_joint_damping;
    scalar_t torqueLimit = soleRobotCfg_.controlCfg.leg_joint_torque_limit;

    if (isAnkle) {
      stiffness = soleRobotCfg_.controlCfg.ankle_joint_stiffness;
      damping = soleRobotCfg_.controlCfg.ankle_joint_damping;
      torqueLimit = soleRobotCfg_.controlCfg.ankle_joint_torque_limit;
    } else if (isArmJ123) {
      stiffness = soleRobotCfg_.controlCfg.arm_j123_stiffness;
      damping = soleRobotCfg_.controlCfg.arm_j123_damping;
      torqueLimit = soleRobotCfg_.controlCfg.arm_j123_torque_limit;
    } else if (isArmJ456) {
      stiffness = soleRobotCfg_.controlCfg.arm_j456_stiffness;
      damping = soleRobotCfg_.controlCfg.arm_j456_damping;
      torqueLimit = soleRobotCfg_.controlCfg.arm_j456_torque_limit;
    }

    const scalar_t scale = soleRobotCfg_.controlCfg.action_scale_pos;
    const scalar_t actionMin =
        jointPos(i) - initJointAngles_(i) + (damping * jointVel(i) - torqueLimit) / stiffness;
    const scalar_t actionMax =
        jointPos(i) - initJointAngles_(i) + (damping * jointVel(i) + torqueLimit) / stiffness;
    actions_[i] = std::max(actionMin / scale, std::min(actionMax / scale, static_cast<scalar_t>(actions_[i])));

    lastActions_(i) = actions_[i];
    const scalar_t pos_des = initJointAngles_(i) + actions_[i] * scale;

    // Optional: lock J6 (wrist roll) to a fixed angle to prevent spinning if policy doesn't use it.
    // NOTE: J6 is not useful in this task. 
    // TODO: add J6 to track the target pose.
    if (lockArmJ6_ && namePtr != nullptr && *namePtr == "J6") {
      const scalar_t locked_pos_des = static_cast<scalar_t>(lockArmJ6Angle_);
      actions_[i] = (locked_pos_des - initJointAngles_(i)) / scale;
      lastActions_(i) = actions_[i];
      hybridJointHandles_[i].setCommand(locked_pos_des, 0, stiffness, damping, 0, 0);
      continue;
    }

    if (isArmJ123) {
      const float factor_kpkd = armHoldStill_ ? 2.0f : 1.0f;
      hybridJointHandles_[i].setCommand(pos_des, 0, stiffness / factor_kpkd, damping * factor_kpkd, 0, 0);
    } else {
      hybridJointHandles_[i].setCommand(pos_des, 0, stiffness, damping, 0, 0);
    }
  }
}

bool SolefootController::loadModel() {
  // Load ONNX models for actor(policy), contactNet(encoder) and GRU.
  std::string policyModelPath;
  if (!nh_.getParam("/policyFile", policyModelPath)) {
    ROS_ERROR("Failed to retrieve policy path from the parameter server!");
    return false;
  }

  std::string encoderModelPath;
  if (!nh_.getParam("/encoderFile", encoderModelPath)) {
    ROS_ERROR("Failed to retrieve encoder path from the parameter server!");
    return false;
  }

  std::string gruModelPath;
  if (!nh_.getParam("/gruFile", gruModelPath)) {
    ROS_ERROR("Failed to retrieve gru path from the parameter server!");
    return false;
  }

  // create env
  onnxEnvPtr_.reset(new Ort::Env(ORT_LOGGING_LEVEL_WARNING, "LeggedOnnxController"));
  // create session
  Ort::SessionOptions sessionOptions;
  sessionOptions.SetIntraOpNumThreads(1);
  sessionOptions.SetInterOpNumThreads(1);

  std::cerr << "onnxEnvPtr_.use_count = " << onnxEnvPtr_.use_count() << std::endl; 
  Ort::AllocatorWithDefaultOptions allocator;
  // policy session
  std::cout << "load policy from" << policyModelPath.c_str() << std::endl;
  policySessionPtr_ = std::make_unique<Ort::Session>(*onnxEnvPtr_, policyModelPath.c_str(), sessionOptions);
  policyInputNames_.clear();
  policyOutputNames_.clear();
  policyInputShapes_.clear();
  policyOutputShapes_.clear();
  for (int i = 0; i < policySessionPtr_->GetInputCount(); i++)
  {
    policyInputNames_.push_back(policySessionPtr_->GetInputName(i, allocator));
    policyInputShapes_.push_back(policySessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    std::cerr << policySessionPtr_->GetInputName(i, allocator) << std::endl;
    std::vector<int64_t> shape = policySessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    std::cerr << "Shape: [";
    for (size_t j = 0; j < shape.size(); ++j)
    {
      std::cout << shape[j];
      if (j != shape.size() - 1)
      {
          std::cerr << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }
  for (int i = 0; i < policySessionPtr_->GetOutputCount(); i++)
  {
    policyOutputNames_.push_back(policySessionPtr_->GetOutputName(i, allocator));
    std::cerr << policySessionPtr_->GetOutputName(i, allocator) << std::endl;
    policyOutputShapes_.push_back(
        policySessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    std::vector<int64_t> shape = policySessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    std::cerr << "Shape: [";
    for (size_t j = 0; j < shape.size(); ++j)
    {
      std::cout << shape[j];
      if (j != shape.size() - 1)
      {
          std::cerr << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }

  // encoder session
  std::cout << "load encoder from" << encoderModelPath.c_str() << std::endl;
  encoderSessionPtr_ = std::make_unique<Ort::Session>(*onnxEnvPtr_, encoderModelPath.c_str(), sessionOptions);
  encoderInputNames_.clear();
  encoderOutputNames_.clear();
  encoderInputShapes_.clear();
  encoderOutputShapes_.clear();
  for (int i = 0; i < encoderSessionPtr_->GetInputCount(); i++)
  {
    encoderInputNames_.push_back(encoderSessionPtr_->GetInputName(i, allocator));
    encoderInputShapes_.push_back(
        encoderSessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    std::cerr << encoderSessionPtr_->GetInputName(i, allocator) << std::endl;
    std::vector<int64_t> shape = encoderSessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    std::cerr << "Shape: [";
    for (size_t j = 0; j < shape.size(); ++j)
    {
      std::cout << shape[j];
      if (j != shape.size() - 1)
      {
        std::cerr << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }
  for (int i = 0; i < encoderSessionPtr_->GetOutputCount(); i++)
  {
    encoderOutputNames_.push_back(encoderSessionPtr_->GetOutputName(i, allocator));
    std::cerr << encoderSessionPtr_->GetOutputName(i, allocator) << std::endl;
    encoderOutputShapes_.push_back(
        encoderSessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    std::vector<int64_t> shape = encoderSessionPtr_->GetOutputTypeInfo(
                                                        i)
                                      .GetTensorTypeAndShapeInfo()
                                      .GetShape();
    std::cerr << "Shape: [";
    for (size_t j = 0; j < shape.size(); ++j)
    {
      std::cout << shape[j];
      if (j != shape.size() - 1)
      {
        std::cerr << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }

  // GRU session
  std::cout << "load gru from " << gruModelPath.c_str() << std::endl;
  gruSessionPtr_ = std::make_unique<Ort::Session>(*onnxEnvPtr_, gruModelPath.c_str(), sessionOptions);
  gruInputNames_.clear();
  gruOutputNames_.clear();
  gruInputShapes_.clear();
  gruOutputShapes_.clear();
  for (int i = 0; i < gruSessionPtr_->GetInputCount(); i++)
  {
    gruInputNames_.push_back(gruSessionPtr_->GetInputName(i, allocator));
    gruInputShapes_.push_back(gruSessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    std::cerr << gruSessionPtr_->GetInputName(i, allocator) << std::endl;
    std::vector<int64_t> shape = gruSessionPtr_->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    std::cerr << "Shape: [";
    for (size_t j = 0; j < shape.size(); ++j)
    {
      std::cout << shape[j];
      if (j != shape.size() - 1)
      {
        std::cerr << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }
  for (int i = 0; i < gruSessionPtr_->GetOutputCount(); i++)
  {
    gruOutputNames_.push_back(gruSessionPtr_->GetOutputName(i, allocator));
    std::cerr << gruSessionPtr_->GetOutputName(i, allocator) << std::endl;
    gruOutputShapes_.push_back(gruSessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape());
    std::vector<int64_t> shape = gruSessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
    std::cerr << "Shape: [";
    for (size_t j = 0; j < shape.size(); ++j)
    {
      std::cout << shape[j];
      if (j != shape.size() - 1)
      {
        std::cerr << ", ";
      }
    }
    std::cout << "]" << std::endl;
  }

  // Resolve RFM model sizes from ONNX meta.
  auto resolveLastPositiveDim = [](const std::vector<int64_t>& shape) -> int {
    for (auto it = shape.rbegin(); it != shape.rend(); ++it) {
      const int64_t d = *it;
      if (d > 0 && d <= std::numeric_limits<int>::max()) {
        return static_cast<int>(d);
      }
    }
    return 0;
  };

  if (!policyInputShapes_.empty() && policyInputShapes_[0].size() >= 2) {
    int actorObsSize = static_cast<int>(policyInputShapes_[0].back());
    if (actorObsSize != observationSize_) {
      ROS_WARN_STREAM("Actor obs dim(" << actorObsSize << ") != observationSize_(" << observationSize_ << ")");
    }
  }
  if (policyInputShapes_.size() >= 2 && policyInputShapes_[1].size() >= 2) {
    int actorLatentSize = static_cast<int>(policyInputShapes_[1].back());
    if (actorLatentSize != encoderOutputSize_) {
      ROS_WARN_STREAM("Actor latent dim(" << actorLatentSize << ") != encoderOutputSize_(" << encoderOutputSize_ << ")");
    }
  }
  if (!encoderInputShapes_.empty()) {
    contactNetObsSize_ = resolveLastPositiveDim(encoderInputShapes_[0]);
  }
  if (!encoderOutputShapes_.empty()) {
    contactNetOutputSize_ = resolveLastPositiveDim(encoderOutputShapes_[0]);
  }
  if (!gruOutputShapes_.empty()) {
    gruLatentSize_ = resolveLastPositiveDim(gruOutputShapes_[0]);
  }
  if (gruLatentSize_ >= 3 && (gruLatentSize_ - 3) % 2 == 0) {
    nextObsLatentSize_ = (gruLatentSize_ - 3) / 2;
  } else {
    ROS_ERROR_STREAM("Unexpected GRU latent dim: " << gruLatentSize_);
    return false;
  }

  if (contactNetOutputSize_ <= 0) {
    ROS_WARN("contactNet output dim is dynamic/unknown from meta; will resolve from runtime output.");
    contactNetOutputSize_ = 0;
  }

  cnOutput_.assign(std::max(0, contactNetOutputSize_), 0.0f);
  gruHiddenState_.assign(std::max(0, gruLatentSize_), 0.0f);

  ROS_INFO_STREAM("RFM dims resolved: contactNetObsSize=" << contactNetObsSize_
                                                         << " contactNetOutputSize=" << contactNetOutputSize_
                                                         << " gruLatentSize=" << gruLatentSize_
                                                         << " nextObsLatentSize=" << nextObsLatentSize_
                                                         << " actorObsSize=" << observationSize_
                                                         << " actorLatentSize=" << encoderOutputSize_);

  ROS_INFO_STREAM("Load Onnx model from successfully !!!");
  return true;
}

// Loads the reinforcement learning configuration.
bool SolefootController::loadRLCfg() {
  auto &initState = soleRobotCfg_.initState;
  auto &initStandState = soleRobotCfg_.initStandState;
  auto &controlCfg = soleRobotCfg_.controlCfg;
  auto &obsScales = soleRobotCfg_.soleRlCfg.obsScales;
  auto &gaitCfg = soleRobotCfg_.gaitCfg;
  auto &estimationCfg = soleRobotCfg_.estimationCfg;

  try {
    // Load parameters from ROS parameter server.
    int error = 0;
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/init_state/joint_names", jointNames_));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/init_state/default_joint_angle/abad_L_Joint", initState["abad_L_Joint"]));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/init_state/default_joint_angle/hip_L_Joint", initState["hip_L_Joint"]));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/init_state/default_joint_angle/knee_L_Joint", initState["knee_L_Joint"]));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/init_state/default_joint_angle/abad_R_Joint", initState["abad_R_Joint"]));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/init_state/default_joint_angle/hip_R_Joint", initState["hip_R_Joint"]));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/init_state/default_joint_angle/knee_R_Joint", initState["knee_R_Joint"]));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/init_state/default_joint_angle/J1", initState["J1"]));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/init_state/default_joint_angle/J2", initState["J2"]));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/init_state/default_joint_angle/J3", initState["J3"]));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/init_state/default_joint_angle/J4", initState["J4"]));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/init_state/default_joint_angle/J5", initState["J5"]));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/init_state/default_joint_angle/J6", initState["J6"]));
    standDuration_ = 0.5;
    // kp, kd
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/control/leg_joint_stiffness", controlCfg.leg_joint_stiffness));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/control/leg_joint_damping", controlCfg.leg_joint_damping));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/control/ankle_joint_stiffness", controlCfg.ankle_joint_stiffness));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/control/ankle_joint_damping", controlCfg.ankle_joint_damping));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/control/arm_j123_stiffness", controlCfg.arm_j123_stiffness));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/control/arm_j123_damping", controlCfg.arm_j123_damping));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/control/arm_j456_stiffness", controlCfg.arm_j456_stiffness));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/control/arm_j456_damping", controlCfg.arm_j456_damping));
    // torque limits
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/control/leg_joint_torque_limit", controlCfg.leg_joint_torque_limit));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/control/ankle_joint_torque_limit", controlCfg.ankle_joint_torque_limit));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/control/arm_j123_torque_limit", controlCfg.arm_j123_torque_limit));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/control/arm_j456_torque_limit", controlCfg.arm_j456_torque_limit));
    // others
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/control/action_scale_pos", controlCfg.action_scale_pos));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/control/decimation", controlCfg.decimation));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/normalization/clip_scales/clip_observations", soleRobotCfg_.soleRlCfg.clipObs));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/normalization/clip_scales/clip_actions", soleRobotCfg_.soleRlCfg.clipActions));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/normalization/obs_scales/lin_vel", obsScales.linVel));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/normalization/obs_scales/ang_vel", obsScales.angVel));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/normalization/obs_scales/dof_pos", obsScales.dofPos));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/normalization/obs_scales/dof_vel", obsScales.dofVel));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/normalization/obs_scales/height_measurements", obsScales.heightMeasurements));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/size/actions_size", actionsSize_));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/size/observations_size", observationSize_));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/size/obs_history_length", obsHistoryLength_));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/size/encoder_output_size", encoderOutputSize_));

    error += static_cast<int>(!nh_.getParam("/PointfootCfg/imu_orientation_offset/yaw", imuOrientationOffset_[0]));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/imu_orientation_offset/pitch", imuOrientationOffset_[1]));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/imu_orientation_offset/roll", imuOrientationOffset_[2]));

    error += static_cast<int>(!nh_.getParam("/PointfootCfg/user_cmd_scales/lin_vel_x", soleRobotCfg_.userCmdCfg.linVel_x));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/user_cmd_scales/lin_vel_y", soleRobotCfg_.userCmdCfg.linVel_y));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/user_cmd_scales/ang_vel_yaw", soleRobotCfg_.userCmdCfg.angVel_yaw));

    error += static_cast<int>(!nh_.getParam("/PointfootCfg/init_state/default_joint_angle/ankle_L_Joint", initState["ankle_L_Joint"]));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/init_state/default_joint_angle/ankle_R_Joint", initState["ankle_R_Joint"]));

    error += static_cast<int>(!nh_.getParam("/PointfootCfg/gait/frequencies", soleRobotCfg_.gaitCfg.frequencies));
    error += static_cast<int>(!nh_.getParam("/PointfootCfg/gait/swing_height", soleRobotCfg_.gaitCfg.swing_height));
    
    if (error) {
      ROS_ERROR("Load parameters from ROS parameter server error!!!");
    }

    nh_.param<bool>("/PointfootCfg/rfm/sample_next_obs_latent", sampleNextObsLatent_, true);
    nh_.param<bool>("/PointfootCfg/arm_lock_j6", lockArmJ6_, true);
    double lock_j6_angle_param = std::numeric_limits<double>::quiet_NaN();
    nh_.param<double>("/PointfootCfg/arm_lock_j6_angle", lock_j6_angle_param, lock_j6_angle_param);

    // Init joint offsets (action=0.0) in the same order as jointNames_/handles.
    initJointAngles_.resize(static_cast<int>(jointNames_.size()));
    for (size_t i = 0; i < jointNames_.size(); ++i) {
      const auto it = initState.find(jointNames_[i]);
      if (it == initState.end()) {
        ROS_ERROR_STREAM("Missing default_joint_angle for joint: " << jointNames_[i]);
        initJointAngles_(static_cast<int>(i)) = 0.0;
      } else {
        initJointAngles_(static_cast<int>(i)) = it->second;
      }
    }

    // Resolve J6 lock angle (if enabled). Default: use J6 init angle from config.
    if (lockArmJ6_) {
      if (std::isfinite(lock_j6_angle_param)) {
        lockArmJ6Angle_ = static_cast<tensor_element_t>(lock_j6_angle_param);
      } else {
        for (size_t i = 0; i < jointNames_.size(); ++i) {
          if (jointNames_[i] == "J6") {
            lockArmJ6Angle_ = static_cast<tensor_element_t>(initJointAngles_(static_cast<int>(i)));
            break;
          }
        }
      }
      ROS_WARN_STREAM("arm_lock_j6 enabled. J6 will be held at angle: " << static_cast<double>(lockArmJ6Angle_));
    }

    jointNameToIndex_.clear();
    for (size_t i = 0; i < jointNames_.size(); ++i) {
      jointNameToIndex_[jointNames_[i]] = i;
    }

    // Init the configured EE FK chain once.
    if (!eeFkReady_) {
      std::string eeBaseLink;
      std::string eeLink;
      nh_.param<std::string>("/PointfootCfg/ee/base_link", eeBaseLink, "base_Link");
      nh_.param<std::string>("/PointfootCfg/ee/ee_link", eeLink, "eef_link");
      std::string urdfString;
      if (!nh_.getParam("/robot_description", urdfString) || urdfString.empty()) {
        ROS_WARN("Failed to get robot_description for EE FK.");
      } else {
        KDL::Tree tree;
        if (!kdl_parser::treeFromString(urdfString, tree)) {
          ROS_WARN("Failed to parse URDF into KDL tree for EE FK.");
        } else if (!tree.getChain(eeBaseLink, eeLink, eeChain_)) {
          ROS_WARN_STREAM("Failed to build KDL chain " << eeBaseLink << " -> " << eeLink << ".");
        } else {
          eeFkSolver_ = std::make_unique<KDL::ChainFkSolverPos_recursive>(eeChain_);
          eeChainJointNames_.clear();
          for (unsigned int s = 0; s < eeChain_.getNrOfSegments(); ++s) {
            const auto& joint = eeChain_.getSegment(s).getJoint();
            if (joint.getType() != KDL::Joint::None) {
              eeChainJointNames_.push_back(joint.getName());
            }
          }
          eeFkReady_ = (eeChainJointNames_.size() == eeChain_.getNrOfJoints());
          if (!eeFkReady_) {
            ROS_WARN("EE FK chain joint parsing mismatch.");
          }
        }
      }
    }

    encoderInputSize_ = obsHistoryLength_ * observationSize_;
    soleRobotCfg_.print();
    clearData();

    // Resize vectors.
    actions_.resize(actionsSize_);
    observations_.resize(observationSize_);
    proprioHistoryVector_.resize(observationSize_ * obsHistoryLength_);
    encoderOut_.resize(encoderOutputSize_);
    lastActions_.resize(actionsSize_);
    obs_debug_msg_.data.resize(observationSize_);

    // Initialize vectors.
    lastActions_.setZero();
    commands_.setZero();
    scaledCommandsSole_.setZero();
    baseLinVel_.setZero();
    basePosition_.setZero();
  } catch (const std::exception &e) {
    // Error handling.
    ROS_ERROR("Error in the PointfootCfg: %s", e.what());
    return false;
  }
  ROS_INFO_STREAM("Load Sole Robot Cfg from successfully !!!");
  return true;
}

void SolefootController::computeObservation() {
  // ---- IMU -> projected gravity / angular velocity (base frame) ----
  Eigen::Quaterniond q_wi;
  for (size_t i = 0; i < 4; ++i) {
    q_wi.coeffs()(i) = imuSensorHandles_.getOrientation()[i];
  }

  vector3_t zyx = quatToZyx(q_wi);
  matrix_t inverseRot = getRotationMatrixFromZyxEulerAngles(zyx).inverse();

  vector3_t gravityVector(0, 0, -1);
  vector3_t projectedGravity(inverseRot * gravityVector);

  vector3_t _zyx(0.0, imuOrientationOffset_[1], imuOrientationOffset_[0]);
  matrix_t rot = getRotationMatrixFromZyxEulerAngles(_zyx);
  projectedGravity = rot * projectedGravity;

  vector3_t baseAngVel(imuSensorHandles_.getAngularVelocity()[0], imuSensorHandles_.getAngularVelocity()[1],
                       imuSensorHandles_.getAngularVelocity()[2]);
  baseAngVel = rot * baseAngVel;

  // ---- Joint states ----
  const size_t numJoints = hybridJointHandles_.size();
  vector_t jointPos(numJoints);
  vector_t jointVel(numJoints);
  vector_t jointTor(numJoints);
  for (size_t i = 0; i < numJoints; ++i) {
    jointPos(i) = hybridJointHandles_[i].getPosition();
    jointVel(i) = hybridJointHandles_[i].getVelocity();
    jointTor(i) = hybridJointHandles_[i].getEffort();
  }

  // ---- Build joint_pos_rel excluding ankles (policy/contactNet expects 12 dims) ----
  std::vector<scalar_t> jointPosRelNoAnkle;
  jointPosRelNoAnkle.reserve(numJoints);
  for (size_t i = 0; i < numJoints; ++i) {
    const std::string name = (i < jointNames_.size()) ? jointNames_[i] : std::string();
    if (!name.empty() && name.find("ankle") != std::string::npos) {
      continue;
    }
    const scalar_t initAngle = (i < static_cast<size_t>(initJointAngles_.size())) ? initJointAngles_(static_cast<int>(i)) : 0.0;
    jointPosRelNoAnkle.push_back(jointPos(i) - initAngle);
  }
  vector_t jointPosRelNoAnkleVec(static_cast<int>(jointPosRelNoAnkle.size()));
  for (size_t i = 0; i < jointPosRelNoAnkle.size(); ++i) {
    jointPosRelNoAnkleVec(static_cast<int>(i)) = jointPosRelNoAnkle[i];
  }

  // ---- EE forward kinematics (configured base link -> EEF) ----
  Eigen::Vector3d eePosCurrent = Eigen::Vector3d::Zero();
  Eigen::Matrix3d eeRotCurrent = Eigen::Matrix3d::Identity();
  if (eeFkReady_ && eeFkSolver_) {
    KDL::JntArray q(static_cast<unsigned int>(eeChain_.getNrOfJoints()));
    q.data.setZero();
    for (size_t j = 0; j < eeChainJointNames_.size(); ++j) {
      auto it = jointNameToIndex_.find(eeChainJointNames_[j]);
      if (it == jointNameToIndex_.end()) {
        continue;
      }
      q(static_cast<unsigned int>(j)) = jointPos(static_cast<int>(it->second));
    }
    KDL::Frame eeFrame;
    if (eeFkSolver_->JntToCart(q, eeFrame) >= 0) {
      eePosCurrent << eeFrame.p.x(), eeFrame.p.y(), eeFrame.p.z();
      eeRotCurrent << eeFrame.M(0, 0), eeFrame.M(0, 1), eeFrame.M(0, 2), eeFrame.M(1, 0), eeFrame.M(1, 1),
          eeFrame.M(1, 2), eeFrame.M(2, 0), eeFrame.M(2, 1), eeFrame.M(2, 2);
    }
  }

  // EE pose (pos + x-axis + y-axis), for contactNet group.
  std::vector<scalar_t> eePoseB;
  eePoseB.reserve(9);
  eePoseB.push_back(eePosCurrent.x());
  eePoseB.push_back(eePosCurrent.y());
  eePoseB.push_back(eePosCurrent.z());
  // first column (x-axis)
  eePoseB.push_back(eeRotCurrent(0, 0));
  eePoseB.push_back(eeRotCurrent(1, 0));
  eePoseB.push_back(eeRotCurrent(2, 0));
  // second column (y-axis)
  eePoseB.push_back(eeRotCurrent(0, 1));
  eePoseB.push_back(eeRotCurrent(1, 1));
  eePoseB.push_back(eeRotCurrent(2, 1));

  // ---- EE target pose from RC command (base frame) ----
  vector3_t eeTargetPos, eeTargetRpy;
  eeTargetPos << rc_ee_cmd.ee_position[0] + ee_init_pos_[0], rc_ee_cmd.ee_position[1] + ee_init_pos_[1],
      rc_ee_cmd.ee_position[2] + ee_init_pos_[2];
  // rc_ee_cmd.ee_rpy is [roll, pitch, yaw] (see PointfootHW publisher /EEPose_cmd_rc).
  eeTargetRpy << ee_init_rpy_[0] + rc_ee_cmd.ee_rpy[0], ee_init_rpy_[1] + rc_ee_cmd.ee_rpy[1],
      ee_init_rpy_[2] + rc_ee_cmd.ee_rpy[2];

  // clamp ee target position
  eeTargetPos[0] = std::min(0.8, std::max(0.0, eeTargetPos[0]));
  eeTargetPos[1] = std::min(0.5, std::max(-0.5, eeTargetPos[1]));
  eeTargetPos[2] = std::min(0.5, std::max(-0.3, eeTargetPos[2]));

  // // if target changes little, treat arm as still (used for PD tuning)
  // if ((eeTargetPos - lastEePos_).norm() < 0.03 && (eeTargetRpy - lastEeRpy_).norm() < 0.01) {
  //   armHoldStill_ = true;
  // } else {
  //   armHoldStill_ = false;
  // }
  lastEePos_ = eeTargetPos;
  lastEeRpy_ = eeTargetRpy;

  Eigen::Quaternion<scalar_t> eeTargetQuat = getQuaternionFromRpy(eeTargetRpy);
  Eigen::Matrix<scalar_t, 3, 3> eeTargetRot = eeTargetQuat.toRotationMatrix();

  // ---- EE target pose in BASE frame (for EE_commands_b obs, matching training PolicyCfg) ----
  // Default: base-frame target from RC cmd (used in legacy/non-world mode).
  Eigen::Matrix<scalar_t, 3, 1> eeTargetPosB = eeTargetPos;
  Eigen::Matrix<scalar_t, 3, 3> eeTargetRotB = eeTargetRot;

  // ---- EE tracking error: pos in EE frame + rot6d(first two cols, row-major) ----
  // (kept for reference; policy obs now uses EE_commands_b instead)
  const scalar_t posScale = 10.0;
  const scalar_t ornScale = 1.5;

  Eigen::Matrix<scalar_t, 3, 1> relPos;
  Eigen::Matrix<scalar_t, 3, 3> relRot;
  bool usedWorldEeTarget = false;

  // If enabled in sim: compute error using world-frame target/current EE pose, but express it in EE frame.
  if (useWorldFrameEeTarget_ && gtBasePoseValid_.load(std::memory_order_relaxed)) {
    // Base pose in world.
    const Eigen::Vector3d p_wb(
        static_cast<double>(gtBasePosX_.load(std::memory_order_relaxed)),
        static_cast<double>(gtBasePosY_.load(std::memory_order_relaxed)),
        static_cast<double>(gtBasePosZ_.load(std::memory_order_relaxed)));
    Eigen::Quaterniond q_wb(
        static_cast<double>(gtBaseQuatW_.load(std::memory_order_relaxed)),
        static_cast<double>(gtBaseQuatX_.load(std::memory_order_relaxed)),
        static_cast<double>(gtBaseQuatY_.load(std::memory_order_relaxed)),
        static_cast<double>(gtBaseQuatZ_.load(std::memory_order_relaxed)));
    if (q_wb.norm() > 1.0e-9) {
      q_wb.normalize();
    } else {
      q_wb = Eigen::Quaterniond::Identity();
    }
    const Eigen::Matrix3d R_wb = q_wb.toRotationMatrix();  // base -> world

    // Current EE pose in world: T_we = T_wb * T_be
    const Eigen::Vector3d p_we = p_wb + R_wb * eePosCurrent;
    const Eigen::Matrix3d R_we = R_wb * eeRotCurrent;

    // Target pose in world.
    Eigen::Vector3d p_wt;
    Eigen::Matrix3d R_wt;

    // Allow toggling manual fallback at runtime (best-effort, throttled).
    static int manual_enable_update_count = 0;
    if ((manual_enable_update_count++ % 50) == 0) { 
      nh_.param<bool>("/PointfootCfg/ee_target/manual_enable", manualEeTargetWorldEnabled_, manualEeTargetWorldEnabled_);
    }

    if (eeTargetWorldValid_.load(std::memory_order_relaxed)) {
      p_wt = Eigen::Vector3d(
          static_cast<double>(eeTargetWorldPosX_.load(std::memory_order_relaxed)),
          static_cast<double>(eeTargetWorldPosY_.load(std::memory_order_relaxed)),
          static_cast<double>(eeTargetWorldPosZ_.load(std::memory_order_relaxed)));
      Eigen::Quaterniond q_wt(
          static_cast<double>(eeTargetWorldQuatW_.load(std::memory_order_relaxed)),
          static_cast<double>(eeTargetWorldQuatX_.load(std::memory_order_relaxed)),
          static_cast<double>(eeTargetWorldQuatY_.load(std::memory_order_relaxed)),
          static_cast<double>(eeTargetWorldQuatZ_.load(std::memory_order_relaxed)));
      if (q_wt.norm() > 1.0e-9) {
        q_wt.normalize();
      } else {
        q_wt = Eigen::Quaterniond::Identity();
      }
      R_wt = q_wt.toRotationMatrix();
    } else if (manualEeTargetWorldEnabled_) {
      // Allow tweaking manual target via rosparam without restarting (best-effort, throttled).
      static int manual_param_update_count = 0;
      if ((manual_param_update_count++ % 50) == 0) { 
        std::vector<double> pos;
        if (nh_.getParam("/PointfootCfg/ee_target/manual_set_pos", pos) && pos.size() == 3) {
          manualEeTargetWorldPos_ << pos[0], pos[1], pos[2];
        }
        std::vector<double> rpy;
        if (nh_.getParam("/PointfootCfg/ee_target/manual_set_rpy", rpy) && rpy.size() == 3) {
          manualEeTargetWorldRpy_ << rpy[0], rpy[1], rpy[2];
        }
      }
      p_wt = manualEeTargetWorldPos_;
      const Eigen::Quaternion<scalar_t> q_wt = getQuaternionFromRpy(manualEeTargetWorldRpy_);
      R_wt = q_wt.toRotationMatrix();
    } else if (anchorEeTargetToFirstBasePose_) {
      // If no world target is provided, anchor (world <- base) once, then map base-frame targets into a fixed world target.
      if (!eeTargetAnchorValid_) {
        eeTargetAnchorPosW_ = p_wb;
        eeTargetAnchorRotWb_ = R_wb;
        eeTargetAnchorValid_ = true;
        ROS_WARN_THROTTLE(1.0, "EE world target not received; anchoring base pose for world-frame EE target.");
      }
      p_wt = eeTargetAnchorPosW_ + eeTargetAnchorRotWb_ * eeTargetPos;
      R_wt = eeTargetAnchorRotWb_ * eeTargetRot;
    } else {
      // Fallback: world-frame target using current base pose.
      p_wt = p_wb + R_wb * eeTargetPos;
      R_wt = R_wb * eeTargetRot;
    }

    relPos = R_we.transpose() * (p_wt - p_we);
    relRot = R_we.transpose() * R_wt;
    usedWorldEeTarget = true;
    // Compute world target converted back to base frame: T_bt = R_wb^T * (p_wt - p_wb)
    eeTargetPosB = (R_wb.transpose() * (p_wt - p_wb)).cast<scalar_t>();
    eeTargetRotB = (R_wb.transpose() * R_wt).cast<scalar_t>();
  } else {
    // Legacy: target in base, error expressed in EE frame.
    relPos = eeRotCurrent.transpose() * (eeTargetPos - eePosCurrent);
    relRot = eeRotCurrent.transpose() * eeTargetRot;
    // eeTargetPosB / eeTargetRotB already initialized to base-frame target above.
  }

  // [OLD] EE tracking error in EE frame (replaced by EE_commands_b in policy obs):
  // relPos *= posScale;
  // std::vector<scalar_t> eeTrackingError;
  // eeTrackingError.reserve(9);
  // eeTrackingError.push_back(relPos.x());
  // eeTrackingError.push_back(relPos.y());
  // eeTrackingError.push_back(relPos.z());
  // eeTrackingError.push_back(relRot(0, 0) * ornScale);
  // eeTrackingError.push_back(relRot(0, 1) * ornScale);
  // eeTrackingError.push_back(relRot(1, 0) * ornScale);
  // eeTrackingError.push_back(relRot(1, 1) * ornScale);
  // eeTrackingError.push_back(relRot(2, 0) * ornScale);
  // eeTrackingError.push_back(relRot(2, 1) * ornScale);

  static int ee_tracking_log_count = 0;
  if (++ee_tracking_log_count % 100 == 0) {
    const Eigen::Vector3d pos_error = eeTargetPos.cast<double>() - eePosCurrent;
    const Eigen::Matrix3d rot_err_mat = eeTargetRot.cast<double>() * eeRotCurrent.transpose();
    const Eigen::Vector3d rpy_error =
        getRpyFromRotationMatrix<scalar_t>(rot_err_mat.cast<scalar_t>()).cast<double>();
    // ROS_INFO_STREAM("EE target (base frame) pos: ["
    //   << eeTargetPosB(0) << ", " << eeTargetPosB(1) << ", " << eeTargetPosB(2)
    //   << "], err xyz: ["
    //   << pos_error.x() << ", " << pos_error.y() << ", " << pos_error.z()
    //   << "], rpy_err: ["
    //   << rpy_error.x() << "(r), " << rpy_error.y() << "(p), " << rpy_error.z() << "(y)]"
    //   << (usedWorldEeTarget ? " | world=true" : " | world=false"));
  }


  // ---- Policy obs (actor input): 65 dims ----
  std::vector<scalar_t> policyObs;
  policyObs.reserve(observationSize_);
  // base_lin_vel: in sim, prefer gazebo ground truth if available; otherwise use GRU-estimated velocity.
  tensor_element_t vbx = baseLinVelObs_[0];
  tensor_element_t vby = baseLinVelObs_[1];
  tensor_element_t vbz = baseLinVelObs_[2];
  if (useGroundTruthBaseLinVel_ && gtBaseLinVelValid_.load(std::memory_order_relaxed)) {
    vbx = gtBaseLinVelX_.load(std::memory_order_relaxed);
    vby = gtBaseLinVelY_.load(std::memory_order_relaxed);
    vbz = gtBaseLinVelZ_.load(std::memory_order_relaxed);
  }
  // base_lin_vel: removed from policy obs
  // policyObs.push_back(static_cast<scalar_t>(vbx));
  // policyObs.push_back(static_cast<scalar_t>(vby));
  // policyObs.push_back(static_cast<scalar_t>(vbz));
  // base_ang_vel (3)
  policyObs.push_back(baseAngVel.x());
  policyObs.push_back(baseAngVel.y());
  policyObs.push_back(baseAngVel.z());
  // projected_gravity (3)
  policyObs.push_back(projectedGravity.x());
  policyObs.push_back(projectedGravity.y());
  policyObs.push_back(projectedGravity.z());
  // EE_pose_commands: target EE pose in base frame (pos + rot6d col0 + rot6d col1) = 9 dims
  // Matches training EE_commands_b obs term.
  policyObs.push_back(eeTargetPosB(0));
  policyObs.push_back(eeTargetPosB(1));
  policyObs.push_back(eeTargetPosB(2));
  policyObs.push_back(eeTargetRotB(0, 0));
  policyObs.push_back(eeTargetRotB(1, 0));
  policyObs.push_back(eeTargetRotB(2, 0));
  policyObs.push_back(eeTargetRotB(0, 1));
  policyObs.push_back(eeTargetRotB(1, 1));
  policyObs.push_back(eeTargetRotB(2, 1));
  // [OLD] ee tracking error in EE frame (replaced by EE_commands_b above):
  // policyObs.insert(policyObs.end(), eeTrackingError.begin(), eeTrackingError.end());
  // joint_pos_rel (exclude ankles)
  for (int i = 0; i < jointPosRelNoAnkleVec.size(); ++i) {
    policyObs.push_back(jointPosRelNoAnkleVec(i));
  }
  // joint_vel (all joints)
  for (size_t i = 0; i < numJoints; ++i) {
    policyObs.push_back(jointVel(static_cast<int>(i)));
  }
  // last actions
  for (int i = 0; i < lastActions_.size(); ++i) {
    policyObs.push_back(lastActions_(i));
  }
  // EE pose in base frame (pos + x-axis + y-axis) = 9 dims
  policyObs.insert(policyObs.end(), eePoseB.begin(), eePoseB.end());
  // EE SE3 distance reference (scalar, decays from initial error toward 0) = 1 dim
  policyObs.push_back(static_cast<scalar_t>(se3DistanceRef_));

  if (static_cast<int>(policyObs.size()) != observationSize_) {
    ROS_ERROR_THROTTLE(1.0, "Policy obs dim mismatch: got %zu expected %d", policyObs.size(), observationSize_);
    return;
  }

  observations_.resize(observationSize_);
  for (int i = 0; i < observationSize_; i++) {
    observations_[i] = static_cast<tensor_element_t>(policyObs[i]);
    obs_debug_msg_.data[i] = observations_[i];
  }
  obs_debug_pub_.publish(obs_debug_msg_);

  // ---- contactNet obs (history): 55 dims ----
  std::vector<scalar_t> cnObs;
  cnObs.reserve(contactNetObsSize_);
  // base_ang_vel
  cnObs.push_back(baseAngVel.x());
  cnObs.push_back(baseAngVel.y());
  cnObs.push_back(baseAngVel.z());
  // projected gravity
  cnObs.push_back(projectedGravity.x());
  cnObs.push_back(projectedGravity.y());
  cnObs.push_back(projectedGravity.z());
  // joint_pos_rel (exclude ankles)
  for (int i = 0; i < jointPosRelNoAnkleVec.size(); ++i) {
    cnObs.push_back(jointPosRelNoAnkleVec(i));
  }
  // joint_vel (all joints)
  for (size_t i = 0; i < numJoints; ++i) {
    cnObs.push_back(jointVel(static_cast<int>(i)));
  }
  // joint_torque (all joints)
  for (size_t i = 0; i < numJoints; ++i) {
    cnObs.push_back(jointTor(static_cast<int>(i)));
  }
  // EE pose (9)
  cnObs.insert(cnObs.end(), eePoseB.begin(), eePoseB.end());

  if (static_cast<int>(cnObs.size()) != contactNetObsSize_) {
    ROS_ERROR_THROTTLE(1.0, "contactNet obs dim mismatch: got %zu expected %d", cnObs.size(), contactNetObsSize_);
    return;
  }

  Eigen::Matrix<tensor_element_t, Eigen::Dynamic, 1> cnObsEigen(contactNetObsSize_);
  for (int i = 0; i < contactNetObsSize_; i++) {
    cnObsEigen(i) = static_cast<tensor_element_t>(cnObs[i]);
  }

  if (isfirstRecObs_) {
    proprioHistoryBuffer_.resize(obsHistoryLength_ * contactNetObsSize_);
    for (int t = 0; t < obsHistoryLength_; t++) {
      proprioHistoryBuffer_.segment(t * contactNetObsSize_, contactNetObsSize_) = cnObsEigen;
    }
    std::fill(gruHiddenState_.begin(), gruHiddenState_.end(), 0.0f);
    baseLinVelObs_ = {0.0f, 0.0f, 0.0f};
    // Initialize SE3 distance reference: 2*pos_err + rot_err (matching training reset behavior)
    {
      const double posErr = (eeTargetPos.cast<double>() - eePosCurrent).norm();
      const Eigen::Matrix3d relR = eeTargetRot.cast<double>() * eeRotCurrent.transpose();
      const double cosA = std::max(-1.0, std::min(1.0, (relR.trace() - 1.0) * 0.5));
      se3DistanceRef_ = static_cast<tensor_element_t>(2.0 * posErr + std::acos(cosA));
    }
    isfirstRecObs_ = false;
  } else {
    proprioHistoryBuffer_.head(proprioHistoryBuffer_.size() - contactNetObsSize_) =
        proprioHistoryBuffer_.tail(proprioHistoryBuffer_.size() - contactNetObsSize_);
    proprioHistoryBuffer_.tail(contactNetObsSize_) = cnObsEigen;
    // Decay SE3 distance reference toward 0 at ~1.0/s (training: vel~[0.5,1.4], step_dt=0.02s)
    se3DistanceRef_ = std::max(0.0f, se3DistanceRef_ - 1.0f * 0.02f);
  }

  // clip policy obs
  scalar_t obsMin = -soleRobotCfg_.soleRlCfg.clipObs;
  scalar_t obsMax = soleRobotCfg_.soleRlCfg.clipObs;
  std::transform(observations_.begin(), observations_.end(), observations_.begin(), [obsMin, obsMax](scalar_t x) {
    return std::max(obsMin, std::min(obsMax, x));
  });
}

void SolefootController::computeEncoder() {
  if (!encoderSessionPtr_ || !gruSessionPtr_) {
    ROS_ERROR_THROTTLE(1.0, "ONNX sessions are not ready.");
    return;
  }
  if (encoderInputNames_.size() != 1 || encoderOutputNames_.size() != 1) {
    ROS_ERROR_THROTTLE(1.0, "contactNet IO meta mismatch: inputs=%zu outputs=%zu",
                       encoderInputNames_.size(), encoderOutputNames_.size());
    return;
  }
  if (gruInputNames_.size() != 2 || gruOutputNames_.size() != 2) {
    ROS_ERROR_THROTTLE(1.0, "GRU IO meta mismatch: inputs=%zu outputs=%zu",
                       gruInputNames_.size(), gruOutputNames_.size());
    return;
  }

  // Best-effort: resolve dims from session meta if they weren't set (e.g. due to old binaries or failed init flow).
  auto resolveLastDim = [](const std::vector<int64_t>& shape) -> int {
    if (shape.empty()) {
      return 0;
    }
    const int64_t d = shape.back();
    if (d <= 0 || d > std::numeric_limits<int>::max()) {
      return 0;
    }
    return static_cast<int>(d);
  };

  if (contactNetObsSize_ <= 0) {
    try {
      if (encoderSessionPtr_->GetInputCount() > 0) {
        const auto shape = encoderSessionPtr_->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
        const int dim = resolveLastDim(shape);
        if (dim > 0) {
          contactNetObsSize_ = dim;
        }
      }
    } catch (...) {
    }
  }
  if (contactNetOutputSize_ < 0) {
    contactNetOutputSize_ = 0;
  }
  if (gruLatentSize_ <= 0) {
    try {
      const int outCount = static_cast<int>(gruSessionPtr_->GetOutputCount());
      for (int i = 0; i < outCount; ++i) {
        const auto shape = gruSessionPtr_->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        const int dim = resolveLastDim(shape);
        if (dim > gruLatentSize_) {
          gruLatentSize_ = dim;
        }
      }
    } catch (...) {
    }
  }
  if (nextObsLatentSize_ <= 0 && gruLatentSize_ > 0 && gruLatentSize_ >= 3 && (gruLatentSize_ - 3) % 2 == 0) {
    nextObsLatentSize_ = (gruLatentSize_ - 3) / 2;
  }

  if (contactNetObsSize_ <= 0 || gruLatentSize_ <= 0 || nextObsLatentSize_ <= 0) {
    ROS_ERROR_THROTTLE(1.0, "RFM model dims are not resolved: cn_obs=%d cn_out=%d gru_latent=%d next_obs_latent=%d",
                       contactNetObsSize_, contactNetOutputSize_, gruLatentSize_, nextObsLatentSize_);
    return;
  }
  const int expectedHistorySize = obsHistoryLength_ * contactNetObsSize_;
  if (static_cast<int>(proprioHistoryBuffer_.size()) != expectedHistorySize) {
    ROS_ERROR_THROTTLE(1.0, "contactNet history buffer size mismatch: %ld vs %d",
                       static_cast<long>(proprioHistoryBuffer_.size()), expectedHistorySize);
    return;
  }

  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
                                                         OrtMemType::OrtMemTypeDefault);

  // 1) contactNet(obs_history) -> cn_output
  std::array<int64_t, 3> cnInputShape = {1, static_cast<int64_t>(obsHistoryLength_),
                                        static_cast<int64_t>(contactNetObsSize_)};
  std::vector<Ort::Value> cnInputValues;
  cnInputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(
      memoryInfo, proprioHistoryBuffer_.data(), proprioHistoryBuffer_.size(), cnInputShape.data(), cnInputShape.size()));

  Ort::RunOptions runOptions;
  std::vector<Ort::Value> cnOutputValues =
      encoderSessionPtr_->Run(runOptions, encoderInputNames_.data(), cnInputValues.data(), 1,
                              encoderOutputNames_.data(), 1);
  if (cnOutputValues.empty()) {
    ROS_ERROR_THROTTLE(1.0, "contactNet returned empty output.");
    return;
  }
  const tensor_element_t* cnData = cnOutputValues[0].GetTensorMutableData<tensor_element_t>();
  const auto cnInfo = cnOutputValues[0].GetTensorTypeAndShapeInfo();
  const size_t cnElemCount = cnInfo.GetElementCount();
  const auto cnShape = cnInfo.GetShape();

  int cnOutDim = 0;
  if (!cnShape.empty()) {
    for (auto it = cnShape.rbegin(); it != cnShape.rend(); ++it) {
      const int64_t d = *it;
      if (d > 0 && d <= std::numeric_limits<int>::max()) {
        cnOutDim = static_cast<int>(d);
        break;
      }
    }
  }
  if (cnOutDim <= 0) {
    if (cnElemCount > 0 && cnElemCount <= static_cast<size_t>(std::numeric_limits<int>::max())) {
      cnOutDim = static_cast<int>(cnElemCount);
    }
  }
  if (cnOutDim <= 0) {
    ROS_ERROR_THROTTLE(1.0, "Failed to resolve contactNet output dim from runtime output.");
    return;
  }
  if (contactNetOutputSize_ != cnOutDim) {
    if (contactNetOutputSize_ > 0) {
      ROS_WARN_THROTTLE(1.0, "contactNetOutputSize_(%d) != runtime cn_out_dim(%d), overriding.", contactNetOutputSize_,
                        cnOutDim);
    }
    contactNetOutputSize_ = cnOutDim;
  }
  if (gruLatentSize_ > 0 && contactNetOutputSize_ != gruLatentSize_) {
    ROS_ERROR_THROTTLE(1.0, "contactNet output dim(%d) != GRU latent dim(%d)", contactNetOutputSize_, gruLatentSize_);
    return;
  }

  cnOutput_.assign(static_cast<size_t>(contactNetOutputSize_), 0.0f);
  size_t startIndex = 0;
  if (cnElemCount >= static_cast<size_t>(contactNetOutputSize_) && contactNetOutputSize_ > 0 &&
      cnElemCount % static_cast<size_t>(contactNetOutputSize_) == 0) {
    startIndex = cnElemCount - static_cast<size_t>(contactNetOutputSize_);
  }
  for (int i = 0; i < contactNetOutputSize_; i++) {
    cnOutput_[static_cast<size_t>(i)] = *(cnData + startIndex + static_cast<size_t>(i));
  }

  // 2) gru(cn_output, hidden_state) -> gru_latent, new_hidden_state
  std::array<int64_t, 2> gruIn0Shape = {1, static_cast<int64_t>(contactNetOutputSize_)};
  std::array<int64_t, 3> gruIn1Shape = {1, 1, static_cast<int64_t>(gruLatentSize_)};
  std::vector<Ort::Value> gruInputValues;
  gruInputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(
      memoryInfo, cnOutput_.data(), cnOutput_.size(), gruIn0Shape.data(), gruIn0Shape.size()));
  gruInputValues.push_back(Ort::Value::CreateTensor<tensor_element_t>(
      memoryInfo, gruHiddenState_.data(), gruHiddenState_.size(), gruIn1Shape.data(), gruIn1Shape.size()));

  std::array<const char*, 2> gruOutputNames = {gruOutputNames_.at(0), gruOutputNames_.at(1)};
  std::vector<Ort::Value> gruOutputValues =
      gruSessionPtr_->Run(runOptions, gruInputNames_.data(), gruInputValues.data(), 2, gruOutputNames.data(), 2);

  if (gruOutputValues.size() != 2) {
    ROS_ERROR_THROTTLE(1.0, "GRU output size mismatch: %zu", gruOutputValues.size());
    return;
  }

  const tensor_element_t* gruLatentData = nullptr;
  const tensor_element_t* newHiddenData = nullptr;
  size_t latentCount = 0;
  size_t hiddenCount = 0;

  auto out0Info = gruOutputValues[0].GetTensorTypeAndShapeInfo();
  auto out1Info = gruOutputValues[1].GetTensorTypeAndShapeInfo();
  const auto out0Shape = out0Info.GetShape();
  const auto out1Shape = out1Info.GetShape();

  // Identify which output is latent (rank=2) vs hidden_state (rank=3).
  if (out0Shape.size() == 2 && out1Shape.size() == 3) {
    gruLatentData = gruOutputValues[0].GetTensorMutableData<tensor_element_t>();
    newHiddenData = gruOutputValues[1].GetTensorMutableData<tensor_element_t>();
    latentCount = out0Info.GetElementCount();
    hiddenCount = out1Info.GetElementCount();
  } else if (out0Shape.size() == 3 && out1Shape.size() == 2) {
    gruLatentData = gruOutputValues[1].GetTensorMutableData<tensor_element_t>();
    newHiddenData = gruOutputValues[0].GetTensorMutableData<tensor_element_t>();
    latentCount = out1Info.GetElementCount();
    hiddenCount = out0Info.GetElementCount();
  } else {
    ROS_ERROR_THROTTLE(1.0, "Unexpected GRU output ranks: out0=%zu out1=%zu", out0Shape.size(), out1Shape.size());
    return;
  }
  if (latentCount < static_cast<size_t>(gruLatentSize_) || hiddenCount < static_cast<size_t>(gruLatentSize_)) {
    ROS_ERROR_THROTTLE(1.0, "GRU output element count mismatch: latent=%zu hidden=%zu expected=%d", latentCount,
                       hiddenCount, gruLatentSize_);
    return;
  }
  if (static_cast<int>(gruHiddenState_.size()) != gruLatentSize_) {
    ROS_ERROR_THROTTLE(1.0, "gruHiddenState_ size mismatch: %zu expected=%d", gruHiddenState_.size(), gruLatentSize_);
    return;
  }

  // update hidden state (batch=1)
  for (int i = 0; i < gruLatentSize_; i++) {
    gruHiddenState_[i] = *(newHiddenData + i);
  }

  // 3) sample next_obs_latent from (mu, logvar), then build next_gru_latent
  const int expectedActorLatentSize = 3 + nextObsLatentSize_;
  if (encoderOutputSize_ != expectedActorLatentSize) {
    ROS_WARN_THROTTLE(1.0, "encoderOutputSize_(%d) != (3 + nextObsLatentSize_=%d)",
                      encoderOutputSize_, expectedActorLatentSize);
  }
  std::vector<tensor_element_t> nextGruLatent(expectedActorLatentSize, 0.0f);
  for (int i = 0; i < 3; i++) {
    nextGruLatent[i] = *(gruLatentData + i);
    baseLinVelObs_[i] = nextGruLatent[i];
  }
  for (int i = 0; i < nextObsLatentSize_; i++) {
    const tensor_element_t mu = *(gruLatentData + 3 + i);
    const tensor_element_t logvar = *(gruLatentData + 3 + nextObsLatentSize_ + i);
    const tensor_element_t std = std::sqrt(std::exp(logvar) + static_cast<tensor_element_t>(1.0e-4));
    nextGruLatent[3 + i] = sampleNextObsLatent_ ? (mu + std * standardNormal_(rng_)) : mu;
  }

  encoderOut_.resize(encoderOutputSize_);
  for (int i = 0; i < encoderOutputSize_ && i < static_cast<int>(nextGruLatent.size()); i++) {
    encoderOut_[i] = nextGruLatent[i];
  }
}

// Computes actions using the policy model.
void SolefootController::computeActions() {
  if (!policySessionPtr_) {
    ROS_ERROR_THROTTLE(1.0, "Policy session is not ready.");
    return;
  }
  if (static_cast<int>(observations_.size()) != observationSize_ || static_cast<int>(encoderOut_.size()) != encoderOutputSize_) {
    ROS_ERROR_THROTTLE(1.0, "Policy input size mismatch: obs=%zu/%d latent=%zu/%d", observations_.size(),
                       observationSize_, encoderOut_.size(), encoderOutputSize_);
    return;
  }

  // create input tensor object
  Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator,
    OrtMemType::OrtMemTypeDefault);
  std::vector<Ort::Value> inputValues;
  std::vector<tensor_element_t> combined_obs;

  std::vector<tensor_element_t> latent;
  for (const auto &item : encoderOut_)
  {
    latent.push_back(item);
  }
  for (const auto &item : observations_)
  {
    combined_obs.push_back(item);
  }

  // Build input shapes robustly (avoid passing dynamic dims like -1 into CreateTensor).
  std::vector<int64_t> obsShape;
  if (policyInputShapes_.size() >= 1 && policyInputShapes_[0].size() == 1) {
    obsShape = {static_cast<int64_t>(combined_obs.size())};
  } else {
    obsShape = {1, static_cast<int64_t>(combined_obs.size())};
  }
  std::vector<int64_t> latentShape;
  if (policyInputShapes_.size() >= 2 && policyInputShapes_[1].size() == 1) {
    latentShape = {static_cast<int64_t>(latent.size())};
  } else {
    latentShape = {1, static_cast<int64_t>(latent.size())};
  }

  inputValues.push_back(
  Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, combined_obs.data(), combined_obs.size(),
  obsShape.data(), obsShape.size()));
  inputValues.push_back(
  Ort::Value::CreateTensor<tensor_element_t>(memoryInfo, latent.data(), latent.size(),
  latentShape.data(), latentShape.size()));

  // run inference
  Ort::RunOptions runOptions;
  std::vector<Ort::Value> outputValues = policySessionPtr_->Run(runOptions, policyInputNames_.data(),
            inputValues.data(), 2, policyOutputNames_.data(), 1);
  // vector_t action(8);
  for (int i = 0; i < actionsSize_; i++)
  {
    actions_[i] = *(outputValues[0].GetTensorMutableData<tensor_element_t>() + i);
  }
}

void SolefootController::cmdVelCallback(const geometry_msgs::TwistConstPtr &msg) {
  commands_(0) = std::min(0.5, std::max(msg->linear.x, -0.5));
  commands_(1) = std::min(0.5, std::max(msg->linear.y, -0.5));
  commands_(2) = std::min(0.3, std::max(msg->angular.z, -0.3));

  double command_z = std::min(1.0, std::max(msg->linear.z, -1.0));
  baseHeightCmd_ = 0.7 + command_z / 5; 

  imuOrientationOffset_[0] = msg->angular.x;
  imuOrientationOffset_[1] = msg->angular.y;
}

void SolefootController::clearData(){
  actions_.resize(actionsSize_);
  filtered_actions_.resize(2);
  observations_.resize(observationSize_);
  proprioHistoryVector_.resize(observationSize_ * obsHistoryLength_);
  encoderOut_.resize(encoderOutputSize_);
  oneHotEncoderOut_.resize(oneHotEncoderOutputSize_);
  lastActions_.resize(actionsSize_);
  gaitGeneratorOut_.resize(gaitGeneratorOutputSize_);
  lastActions_.setZero();
  commands_.setZero();
  extraCommands_.setZero();
  scaledCommandsSole_.setZero();
  baseLinVel_.setZero();
  basePosition_.setZero();
  baseLinVelObs_ = {0.0f, 0.0f, 0.0f};
  std::fill(gruHiddenState_.begin(), gruHiddenState_.end(), 0.0f);
  se3DistanceRef_ = 5.0f;
  isfirstRecObs_ = true;
}


vector_t SolefootController::handleGaitCommand(){
  vector_t gait(4);// 4
  // frequency, phase offset, contact duration, swing height
  gait << 1.3, 0.5, 0.5, 0.12;
  return gait;
}

vector_t SolefootController::handleGaitPhase(vector_t &gait){
  vector_t gait_clock(2);
  gaitIndex_ += 0.02 * gait(0);
  if (gaitIndex_ > 1.0)
  {
    gaitIndex_ = 0.0;
  }
  if(scaledCommandsSole_.head(3).norm() < 0.01){
    gaitIndex_ = 0.0;
  }

  gait_clock << sin(gaitIndex_ * 2 * M_PI), cos(gaitIndex_ * 2 * M_PI);
  return gait_clock;
}

double SolefootController::sliding_window(std::vector<double>& data, int window_size) {
  std::vector<double> window(window_size, 1.0 / window_size);
  std::vector<double> smooth_data;
  int len = data.size();
  for (int i = 0; i < std::min(window_size - 1, len); ++i) {
    double mean = std::accumulate(data.begin(), data.begin() + i + 1, 0.0) / (i + 1);
    if(std::abs(mean) < 0.01){
      mean = 0.0;
    }else{
      mean = round(mean * 100) / 100;
    }
    smooth_data.push_back(mean);
  }

  for (int i = window_size - 1; i < data.size(); ++i) {
    double sum = 0.0;
    for (int j = i - window_size + 1; j <= i; ++j) {
      sum += data[j] * window[i - j];
    }
    if(std::abs(sum) < 0.01){
      sum = 0.0;
    }else{
      sum = round(sum * 100) / 100;
    }
    smooth_data.push_back(sum);
  }
  return smooth_data.back();
}

void SolefootController::EEPoseCmdRCCallback(const std_msgs::Float32MultiArrayConstPtr &msg)
{
  if (msg->data.size() == 8 && msg->data[7] == -1) {
    mode_ = Mode::SOLE_STOP;
  }
  ee_pos_cmd_rc_delta_msg_ = *msg;
  rc_ee_cmd.ee_position[0] += msg->data[0];
  rc_ee_cmd.ee_position[1] += msg->data[1];
  rc_ee_cmd.ee_position[2] += msg->data[2];

  rc_ee_cmd.ee_position[0] = std::min(0.6-ee_init_pos_[0], std::max(0.0-ee_init_pos_[0], rc_ee_cmd.ee_position[0]));
  rc_ee_cmd.ee_position[1] = std::min(0.5-ee_init_pos_[1], std::max(-0.5-ee_init_pos_[1], rc_ee_cmd.ee_position[1]));
  rc_ee_cmd.ee_position[2] = std::min(0.5-ee_init_pos_[2], std::max(-0.3-ee_init_pos_[2], rc_ee_cmd.ee_position[2]));

  rc_ee_cmd.ee_rpy[0] += msg->data[3];
  rc_ee_cmd.ee_rpy[1] += msg->data[4];
  rc_ee_cmd.ee_rpy[2] += msg->data[5];

  rc_ee_cmd.gripper_cmd = msg->data[6];

  gripper_cmd_msg_.data = rc_ee_cmd.gripper_cmd;
  
  gripper_cmd_pub_.publish(gripper_cmd_msg_);
}

void SolefootController::EEPoseTargetWorldCallback(const geometry_msgs::PoseStampedConstPtr& msg) {
  eeTargetWorldPosX_.store(static_cast<tensor_element_t>(msg->pose.position.x), std::memory_order_relaxed);
  eeTargetWorldPosY_.store(static_cast<tensor_element_t>(msg->pose.position.y), std::memory_order_relaxed);
  eeTargetWorldPosZ_.store(static_cast<tensor_element_t>(msg->pose.position.z), std::memory_order_relaxed);
  eeTargetWorldQuatW_.store(static_cast<tensor_element_t>(msg->pose.orientation.w), std::memory_order_relaxed);
  eeTargetWorldQuatX_.store(static_cast<tensor_element_t>(msg->pose.orientation.x), std::memory_order_relaxed);
  eeTargetWorldQuatY_.store(static_cast<tensor_element_t>(msg->pose.orientation.y), std::memory_order_relaxed);
  eeTargetWorldQuatZ_.store(static_cast<tensor_element_t>(msg->pose.orientation.z), std::memory_order_relaxed);
  eeTargetWorldValid_.store(true, std::memory_order_relaxed);
}

void SolefootController::groundTruthCallback(const nav_msgs::OdometryConstPtr& msg) {
  // Store base pose in world (needed for world-frame EE target tracking).
  gtBasePosX_.store(static_cast<tensor_element_t>(msg->pose.pose.position.x), std::memory_order_relaxed);
  gtBasePosY_.store(static_cast<tensor_element_t>(msg->pose.pose.position.y), std::memory_order_relaxed);
  gtBasePosZ_.store(static_cast<tensor_element_t>(msg->pose.pose.position.z), std::memory_order_relaxed);
  gtBaseQuatW_.store(static_cast<tensor_element_t>(msg->pose.pose.orientation.w), std::memory_order_relaxed);
  gtBaseQuatX_.store(static_cast<tensor_element_t>(msg->pose.pose.orientation.x), std::memory_order_relaxed);
  gtBaseQuatY_.store(static_cast<tensor_element_t>(msg->pose.pose.orientation.y), std::memory_order_relaxed);
  gtBaseQuatZ_.store(static_cast<tensor_element_t>(msg->pose.pose.orientation.z), std::memory_order_relaxed);
  gtBasePoseValid_.store(true, std::memory_order_relaxed);

  // Take linear velocity from odometry twist. Some publishers express twist in world frame; others in body frame.
  Eigen::Vector3d v(msg->twist.twist.linear.x, msg->twist.twist.linear.y, msg->twist.twist.linear.z);
  if (groundTruthTwistInWorldFrame_) {
    // Rotate world-frame velocity into base frame using base orientation in world.
    const auto& q = msg->pose.pose.orientation;
    Eigen::Quaterniond q_wb(q.w, q.x, q.y, q.z);  // base -> world
    const Eigen::Matrix3d R_wb = q_wb.toRotationMatrix();
    v = R_wb.transpose() * v;
  }
  gtBaseLinVelX_.store(static_cast<tensor_element_t>(v.x()), std::memory_order_relaxed);
  gtBaseLinVelY_.store(static_cast<tensor_element_t>(v.y()), std::memory_order_relaxed);
  gtBaseLinVelZ_.store(static_cast<tensor_element_t>(v.z()), std::memory_order_relaxed);
  gtBaseLinVelValid_.store(true, std::memory_order_relaxed);
}
} // namespace

// Export the class as a plugin.
PLUGINLIB_EXPORT_CLASS(robot_controller::SolefootController, controller_interface::ControllerBase)
