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
      // Order matters: the raw-torque channel is measured on the target BEFORE
      // the projection (measuring it after makes every joint report exactly the
      // ratio the projection enforces -- the projection measuring itself), and it
      // must run in both QP and bypass because the V5 network reads it either way.
      ctl.updateRawTorqueRatio(ctl.q_rl);
      // Project onto the torque-feasible set, exactly as the training actuator
      // does. No-op unless the policy block sets torque_feasibility_ratio.
      ctl.q_rl = ctl.projectTorqueFeasible(ctl.q_rl);
      // Feed back the action as EXECUTED, not as requested: the V4 observation's
      // actions block is executed_action. Beyond the feasible window many raw
      // actions map to one execution, and a policy that only ever sees what it
      // asked for cannot tell them apart -- which is the whole reason the
      // training observation switched away from last_action.
      for (int j = 0; j < ctl.currentAction.size(); ++j) {
          int i = ctl.actionToDofMap[j];
          const double scale = ctl.actionScale(i);
          if(std::abs(scale) > 1e-12)
          {
            ctl.currentActionScaled(i) = ctl.q_rl(i) - ctl.q_zero(i);
            ctl.currentAction(j) = ctl.currentActionScaled(i) / scale;
          }
      }
      syncTime_ -= ctl.policyStepSize;
    }
  }
  catch(const std::exception & e)
  {
    mc_rtc::log::error("[NewRLQPController::utils] Error during RL state run: {}", e.what());
  }
}

// Les formats recents partagent un meme corps et ne different que par leurs
// blocs de queue, chacun optionnel :
//
//   246  corps seul                      index 1, 4
//   266  corps + gait_phase[20]          index 2   (V4)
//   566  corps + gait_phase + raw[300]   index 3   (V5)
//
// Enonce ici une fois, lu la ou les blocs sont ecrits. Ajouter un index a l'un
// de ces formats demande une etiquette de case ET une entree ici ; en oublier
// une leve sur la taille d'observation, ce qui est le role de ce controle.
//
// L'index 4 (run 2026-08-12_20-36-28, filtre de PostureTask modelise) a ete
// ajoute au yaml le 2026-08-13 SANS toucher a ce switch -- l'erreur exacte que
// ce commentaire previent depuis le debut. Résultat mesure le 2026-08-14 :
// aucune case ne correspond, l'observation reste a zero, "Wrote 0 expects
// 246". Case 4 ajoutee juste en dessous pour corriger.
bool utils::hasGaitPhase(int policyIndex)
{
  return policyIndex == 2 || policyIndex == 3;
}

bool utils::isV5(int policyIndex)
{
  return policyIndex == 3;
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
    case 1: // RHPS1 velocity policies — 246 dims
            // mjlab-rhps1 run 2026-08-07_15-40-43, checkpoint 7050 : le retour
            // a la base policy 0 (echelle x1.5, keyframe genou 0.622) avec les
            // armatures reelles et joint_vel par difference finie. C'est le
            // corps commun tout court : ni gait_phase, ni raw_torque.
            // 15+3+3+30+30+150+15 = 246.
            //
            // joint_vel ne demande rien de special ici : l'entrainement derive
            // desormais les positions, et sur le robot l'EncoderObserver fait
            // exactement pareil (mc_rtc ne recoit aucune vitesse articulaire).
            // Les deux cotes coincident sans code supplementaire.
    case 4: // RHPS1 velocity policies — 246 dims, meme format que l'index 1
            // mjlab-rhps1 run 2026-08-12_20-36-28 : premiere policy entrainee
            // avec le filtre de PostureTask modelise (posture_stiffness=1600
            // par policy, voir NewRLQPController::postureStiffness()). Le
            // corps d'observation lui-meme n'a pas change par rapport a
            // l'index 1 -- le filtre vit dans l'actionneur d'entrainement et
            // dans la PostureTask du QP ici, pas dans le vecteur d'observation.
            // hasGaitPhase(4) et isV5(4) valent false par construction (les
            // deux predicats testent explicitement 2 et 3), donc ce cas tombe
            // bien sur le corps seul, 246 dims.
    case 3: // RHPS1 velocity policies — V5 format (566 dims)
            // mjlab-rhps1 run 2026-08-05_11-17-44 ("abl15"). V5 is V4 with one
            // block appended and nothing else moved:
            //   raw_torque[300] = 10x30  (NEW)
            // 266 + 300 = 566. The block is |tau_raw| / effort_limit per joint,
            // measured on the target BEFORE the feasibility projection --
            // see NewRLQPController::updateRawTorqueRatio(). Its history is 10
            // deep, not 5 like every other block, which is why it has its own
            // buffer instead of sharing HISTORY_SIZE.
            //
            // Falls through the V4 body below: everything up to gait_phase is
            // byte-for-byte identical, and duplicating 60 lines to append one
            // block is how the two drift apart.
    case 2: // RHPS1 velocity policies — V4 format (266 dims)
            // mjlab-rhps1 run 2026-08-01_14-55-55 ("abl7"). Two blocks grew and
            // one is new relative to V3:
            //   base_lin_vel[15]  = 5x3   (unchanged)
            //   base_ang_vel[3], projected_gravity[3], joint_pos[30],
            //   joint_vel[30]                        (unchanged)
            //   actions[150]      = 5x30  (was 30)
            //   command[15]       = 5x3   (unchanged)
            //   gait_phase[20]    = 5x4   (NEW -- absent from V3 entirely)
            // 15+3+3+30+30+150+15+20 = 266.
            //
            // The actions block is executed_action in training, i.e. the target
            // AFTER the actuator's feasibility projection, not the raw network
            // output. Without that projection here the two coincide, so this is
            // faithful only for a controller that does not project. See the note
            // on the projection in NewRLQPController.h.
    {
      auto & rr = ctl.realRobot(ctl.robots()[0].name());
      const std::string & baseName = rr.mb().body(0).name();
      const Eigen::Matrix3d R_w2b = rr.bodyPosW(baseName).rotation();

      for(int i = ctl.HISTORY_SIZE - 1; i > 0; --i)
      {
        ctl.linVel_[i]    = ctl.linVel_[i-1];
        ctl.angVel_[i]    = ctl.angVel_[i-1];
        ctl.projGrav_[i]  = ctl.projGrav_[i-1];
        ctl.jointPos_[i]  = ctl.jointPos_[i-1];
        ctl.jointVel_[i]  = ctl.jointVel_[i-1];
        ctl.jointAct_[i]  = ctl.jointAct_[i-1];
        ctl.velCmd_[i]    = ctl.velCmd_[i-1];
        ctl.gaitPhase_[i] = ctl.gaitPhase_[i-1];
      }

      ctl.linVel_[0]   = R_w2b * rr.bodyVelW(baseName).linear();
      ctl.angVel_[0]   = R_w2b * rr.bodyVelW(baseName).angular();
      ctl.projGrav_[0] = R_w2b * Eigen::Vector3d(0, 0, -1);
      ctl.velCmd_[0]   = ctl.currentVelCmd_;
      ctl.jointAct_[0] = ctl.currentAction;
      // Advance the clock exactly once per inference, after velCmd_[0] is set
      // (the cadence depends on it) and before the block is written out.
      if(utils::hasGaitPhase(ctl.currentPolicyIndex)) { ctl.gaitPhaseStep(); }

      const int actionDim = static_cast<int>(ctl.refJointOrderRLAction.size());
      ctl.jointPos_[0] = Eigen::VectorXd::Zero(actionDim);
      ctl.jointVel_[0] = Eigen::VectorXd::Zero(actionDim);
      for(int j = 0; j < actionDim; ++j)
      {
        const int mcIdx = rr.jointIndexByName(ctl.refJointOrderRLAction[j]);
        ctl.jointPos_[0](j) = rr.mbc().q[mcIdx][0] - ctl.q_zero[ctl.actionToDofMap[j]];
        ctl.jointVel_[0](j) = rr.mbc().alpha[mcIdx][0];
      }

      auto write3 = [&](const Eigen::Vector3d & v)
      { obs.segment(offset, 3) = v; offset += 3; };
      auto write4 = [&](const Eigen::Vector4d & v)
      { obs.segment(offset, 4) = v; offset += 4; };

      for(int i = ctl.HISTORY_SIZE-1; i >= 0; --i) write3(ctl.linVel_[i]);
      write3(ctl.angVel_[0]);
      write3(ctl.projGrav_[0]);
      appendToObs(ctl.jointPos_[0]);
      appendToObs(ctl.jointVel_[0]);
      for(int i = ctl.HISTORY_SIZE-1; i >= 0; --i) appendToObs(ctl.jointAct_[i]);
      for(int i = ctl.HISTORY_SIZE-1; i >= 0; --i) write3(ctl.velCmd_[i]);
      if(utils::hasGaitPhase(ctl.currentPolicyIndex))
      {
        for(int i = ctl.HISTORY_SIZE-1; i >= 0; --i) write4(ctl.gaitPhase_[i]);
      }

      // V5 tail. rawTorque_ is pushed by updateRawTorqueRatio() at the END of the
      // previous policy step, so index 0 holds the demand of the action that has
      // just been executed -- the same alignment mjlab has, where the observation
      // reads a peak accumulated over the previous step's substeps.
      if(utils::isV5(ctl.currentPolicyIndex))
      {
        for(int i = ctl.RAW_TORQUE_HISTORY-1; i >= 0; --i) appendToObs(ctl.rawTorque_[i]);
      }
      break;
    }
    default:
    {
      mc_rtc::log::error("[NewRLQPController::utils] Unknown policy index: {}", ctl.currentPolicyIndex);
      break;
    }
  }

  // Hard error, not assert: the superbuild builds RelWithDebInfo, which defines
  // NDEBUG and compiles asserts out entirely. A short write would then leave the
  // tail of the vector at zero and the policy would run on a silently truncated
  // observation -- exactly the failure this check exists to catch, and the one
  // that is hardest to diagnose from behaviour alone.
  if(offset != obs.size())
  {
    mc_rtc::log::error_and_throw<std::runtime_error>(
        "[NewRLQPController::utils] Observation size mismatch for policy index {}: wrote {} "
        "values, the network expects {}. The observation layout in this switch does not match "
        "the loaded ONNX.",
        ctl.currentPolicyIndex, offset, obs.size());
  }
  return obs;
}