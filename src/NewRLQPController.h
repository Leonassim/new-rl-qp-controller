#pragma once

#include <mc_control/fsm/Controller.h>
#include <mc_tasks/TorqueJointTask.h>

#include "api.h"

#include "RLPolicyInterface.h"
#include "utils.h"


struct NewRLQPController_DLLAPI NewRLQPController : public mc_control::fsm::Controller
{
  NewRLQPController(mc_rbdyn::RobotModulePtr rm, double dt, const mc_rtc::Configuration & config);

  bool run() override;
  void reset(const mc_control::ControllerResetData & reset_data) override;

  // Task
  std::shared_ptr<mc_tasks::TorqueJointTask> torqueJointTask;
  
  int dofNumber = 0;

  // Public RL related variables
  Eigen::VectorXd q_rl;
  Eigen::VectorXd q_zero;                      // Reference joint positions

  Eigen::VectorXd currentObservation;
  Eigen::VectorXd currentAction;

  double actionScale;
  double policyStepSize;

  size_t currentPolicyIndex = 0;
  std::unique_ptr<RLPolicyInterface> rlPolicy;
  utils utilsClass; // Utility functions for RL controller

  // observation data commented examples - Policy specific
  // Eigen::Vector3d baseAngVel; // Angular velocity of the base
  // Eigen::Vector3d rpy; // Roll, Pitch, Yaw angles of the base
  // Eigen::VectorXd jointPos, jointVel, jointAction; // Joint position, velocity and action
  // Eigen::Vector3d velCmdRL;                        // Command vector [vx, vy, yaw_rate]

private:
  mc_rtc::Configuration config_;
  // Add log entries readable by mc_log_ui 
  void addLog();
  // Add GUI elements to the mc_rtc GUI through Rviz and mc_mujoco
  void addGui();

  void initializeRobot();
  void configRL();
  void initializeRLPolicy();
  void switchPolicy(int policyIndex);  

  // Handle switching between Torque and Position control modes. Torque control is better for directly applying the RL torques, while position control is simulating the torque reference in high gains position control which is experimental. Except in simulation avoid switching between control modes during the execution on the real robot to prevent potential issues with the hardware.
  bool manageModeSwitching();
  // Directly use RL output without QP modifications (Torque Control only) 
  bool byPassQPControl(); 

  std::string robotName_;
  std::vector<std::string> jointNames_;

  // Mode switching
  bool useQP_ = true;
  bool isTorqueControl_ = true;
  bool controlModeChanged_ = false;

  // Constraint configuration
  double velPercent_ = 0.95; // Percentage of the max velocity taking account in the joint velocity constraint.
  double dsPercent_ = 0.01; // Percentage of the max joint range taking account in the joint position limit constraint.
  double diPercent_ = 0.1; // Doesn't matter since di > ds. This variable is not used in the constraint dynamics.

  // CBF Gains More details are explained in the paper cf. Readme.md. 
  // Must be tuned depending on the robot.
  double zeta_jointLimit_ = 1.2;
  double lambda_jointLimit_ = 100.0; // Same gain for joint position limits and velocity limits. 
  double zeta_selfCollision_ = 1.2;
  double lambda_selfCollision_ = 10.0; 

  // Gains
  double pdGainsRatio_ = 1.0;
  Eigen::VectorXd kp_;  // Gains set to the robot/simulator = pd_gains_ratio * kp_base
  Eigen::VectorXd kd_;  // Gains set to the robot/simulator = pd_gains_ratio * kd_base
  Eigen::VectorXd kpBase_; // Base RL PD gains from config
  Eigen::VectorXd kdBase_; // Base RL PD gains from config

  // RL
  std::vector<std::string> policyPaths_;
};
