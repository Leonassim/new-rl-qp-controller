#include "utils.h"
#include <Eigen/src/Core/Matrix.h>
#include <mc_rtc/logging.h>

#include "NewRLQPController.h"

void utils::start_rl_state(mc_control::fsm::Controller & ctl_, std::string state_name)
{
  auto & ctl = static_cast<NewRLQPController&>(ctl_);
  state_name_ = state_name;
  mc_rtc::log::info("[NewRLQPController::utils] {} state started", state_name);

  // Hold the measured posture for 100 ms (sim settling), then ramp to q_zero
  // before the first inference.
  syncTime_ = -0.1;
  warmupSteps_ = static_cast<int>(0.1 / ctl.timeStep);

  if(!ctl.rlPolicy || !ctl.rlPolicy->isLoaded())
  {
    mc_rtc::log::error("[NewRLQPController::utils] RL policy not loaded in {} state", state_name);
    return;
  }

  // Capture the measured posture so the PD target starts where the robot is
  // (the policy's q_zero may differ from the module stance the robot spawns in).
  auto & rr = ctl.realRobot(ctl.robots()[0].name());
  rampStartQ_ = ctl.q_zero;
  for(size_t i = 0; i < ctl.jointNames.size(); ++i)
  {
    const auto & name = ctl.jointNames[i];
    if(rr.hasJoint(name))
    {
      const int mcIdx = static_cast<int>(rr.jointIndexByName(name));
      if(!rr.mbc().q[mcIdx].empty()) { rampStartQ_(i) = rr.mbc().q[mcIdx][0]; }
    }
  }
  ctl.q_rl = rampStartQ_;

  // Ramp duration scales with the largest joint offset (~0.15 rad/s, capped
  // at 2 s); if the policy's q_zero matches the spawn posture (old policies)
  // the ramp is skipped entirely.
  const double maxOffset = (rampStartQ_ - ctl.q_zero).cwiseAbs().maxCoeff();
  const double rampDuration = maxOffset < 0.02 ? 0.0 : std::min(2.0, maxOffset / 0.15);
  rampTotalSteps_ = std::max(1, static_cast<int>(rampDuration / ctl.timeStep));
  rampSteps_ = rampDuration > 0.0 ? rampTotalSteps_ : 0;
  mc_rtc::log::info(
      "[NewRLQPController::utils] go-to-init ramp: max offset {:.3f} rad, duration {:.2f} s",
      maxOffset, rampDuration);

  ctl.initializeRLObservation();
  ctl.q_rl_prev_ = ctl.q_rl;

  mc_rtc::log::success("[NewRLQPController::utils] {} state initialization completed", state_name);
}

void utils::run_rl_state(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<NewRLQPController&>(ctl_);
  try
  {
    if(warmupSteps_ > 0)
    {
      warmupSteps_--;
      return; // hold q_rl = measured start posture while MuJoCo settles
    }
    if(rampSteps_ > 0)
    {
      rampSteps_--;
      const double alpha =
          1.0 - static_cast<double>(rampSteps_) / static_cast<double>(rampTotalSteps_);
      ctl.q_rl = (1.0 - alpha) * rampStartQ_ + alpha * ctl.q_zero;
      return; // go-to-init: reach q_zero before the first inference
    }
    syncTime_ += ctl.timeStep;
    if(syncTime_ >= ctl.policyStepSize)
    {
      ctl.currentObservation = getCurrentObservation(ctl);
      ctl.currentAction = ctl.rlPolicy->predict(ctl.currentObservation);
      for (int j = 0; j < ctl.currentAction.size(); ++j) {
          int i = ctl.actionToDofMap[j];
          ctl.currentActionScaled(i) = ctl.actionScale(i) * ctl.currentAction(j);
          ctl.q_rl(i) = ctl.currentActionScaled(i) + ctl.q_zero(i);
      }
      syncTime_ -= ctl.policyStepSize;
    }
  }
  catch(const std::exception & e)
  {
    mc_rtc::log::error("[NewRLQPController::utils] Error during RL state run: {}", e.what());
  }
}

Eigen::VectorXd utils::getCurrentObservation(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<NewRLQPController&>(ctl_);
  Eigen::VectorXd obs(ctl.rlPolicy->getObservationSize());
  obs = Eigen::VectorXd::Zero(ctl.rlPolicy->getObservationSize());

  // Observation examples - these should be adapted to match the expected observation of the loaded policy
  int offset = 0;
  auto appendToObs = [&](const Eigen::VectorXd& v) {
    obs.segment(offset, v.size()) = v;
    offset += v.size();
  };

  switch (ctl.currentPolicyIndex) {
    case 0: // RHPS1 velocity policies — V3 format (126 dims)
    case 1: // (index 1 = live training checkpoint, same V3 observation)
            // mjlab-rhps1 training 2026-07-10_13-52-54: history (length 5,
            // oldest first) on base_lin_vel and command only, all other terms
            // current-step: base_lin_vel[15], base_ang_vel[3],
            // projected_gravity[3], joint_pos[30], joint_vel[30], actions[30],
            // command[15].
    {
      auto & rr = ctl.realRobot(ctl.robots()[0].name());
      const std::string & baseName = rr.mb().body(0).name();

      // Rotation world → body (updated by Tilt observer)
      const Eigen::Matrix3d R_w2b = rr.bodyPosW(baseName).rotation();

      // Shift history buffers: drop oldest (index HISTORY_SIZE-1), push at index 0
      for(int i = ctl.HISTORY_SIZE - 1; i > 0; --i)
      {
        ctl.linVel_[i]    = ctl.linVel_[i-1];
        ctl.angVel_[i]    = ctl.angVel_[i-1];
        ctl.projGrav_[i]  = ctl.projGrav_[i-1];
        ctl.jointPos_[i]  = ctl.jointPos_[i-1];
        ctl.jointVel_[i]  = ctl.jointVel_[i-1];
        ctl.jointAct_[i]  = ctl.jointAct_[i-1];
        ctl.velCmd_[i]    = ctl.velCmd_[i-1];
      }

      // Fill index 0 with current state
      ctl.linVel_[0]   = R_w2b * rr.bodyVelW(baseName).linear();
      ctl.angVel_[0]   = R_w2b * rr.bodyVelW(baseName).angular();
      ctl.projGrav_[0] = R_w2b * Eigen::Vector3d(0, 0, -1);
      ctl.velCmd_[0]   = ctl.currentVelCmd_;

      // jointAct_[0] = raw NN output from previous step (before scaling)
      ctl.jointAct_[0] = ctl.currentAction;

      const int actionDim = static_cast<int>(ctl.refJointOrderRLAction.size());
      ctl.jointPos_[0] = Eigen::VectorXd::Zero(actionDim);
      ctl.jointVel_[0] = Eigen::VectorXd::Zero(actionDim);
      for(int j = 0; j < actionDim; ++j)
      {
        const int mcIdx = rr.jointIndexByName(ctl.refJointOrderRLAction[j]);
        ctl.jointPos_[0](j) = rr.mbc().q[mcIdx][0] - ctl.q_zero[ctl.actionToDofMap[j]];
        ctl.jointVel_[0](j) = rr.mbc().alpha[mcIdx][0];
      }

      // Build observation: histories stacked oldest first (index HISTORY_SIZE-1 → 0)
      auto write3 = [&](const Eigen::Vector3d & v)
      { obs.segment(offset, 3) = v; offset += 3; };

      for(int i = ctl.HISTORY_SIZE-1; i >= 0; --i) write3(ctl.linVel_[i]);
      write3(ctl.angVel_[0]);
      write3(ctl.projGrav_[0]);
      appendToObs(ctl.jointPos_[0]);
      appendToObs(ctl.jointVel_[0]);
      appendToObs(ctl.jointAct_[0]);
      for(int i = ctl.HISTORY_SIZE-1; i >= 0; --i) write3(ctl.velCmd_[i]);
      break;
    }
    default:
    {
      mc_rtc::log::error("[NewRLQPController::utils] Unknown policy index: {}", ctl.currentPolicyIndex);
      break;
    }
  }

  assert(offset == obs.size() && "[NewRLQPController::utils] Observation size mismatch: written bytes != allocated size");
  return obs;
}