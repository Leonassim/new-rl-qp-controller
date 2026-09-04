#include "utils.h"
#include <algorithm>
#include <cmath>
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
    const bool newInference = syncTime_ >= ctl.policyStepSize;
    if(newInference)
    {
      ctl.currentObservation = getCurrentObservation(ctl);
      ctl.currentAction = ctl.rlPolicy->predict(ctl.currentObservation);

      // Raw policy output -> velocity feedforward, needed by both contracts
      // below (also feeds the "actions"/last_action observation term).
      for (int j = 0; j < ctl.currentAction.size(); ++j) {
          int i = ctl.actionToDofMap[j];
          ctl.currentActionScaled(i) = ctl.actionScale(i) * ctl.currentAction(j);
      }
      syncTime_ -= ctl.policyStepSize;
    }

      if(ctl.velocityAction_)
      {
        // velocity_action (policy 4, mjlab JointVelocityAction +
        // IdealPdActuator): free-running integral of the commanded velocity,
        //   self.pos_target[ids] = self.pos_target[ids] + cmd.velocity_target[ids] * dt
        // i.e. the target advances on its own rather than being reseeded from
        // the measurement each step. Both variants exist in mjlab's
        // pd_actuator.py; this one is what the QP can actually transmit.
        //
        // Why it matters: the QP emits ONE coupled (q, qdot) trajectory. A
        // target pinned to the measurement (q_meas + v*dt) holds a constant
        // position error whatever the joint is doing, so every velocity --
        // including zero -- satisfies it and the solver picks the cheapest;
        // the command simply is not in the position channel. Measured under
        // QP: qdot ~ 1/3 of commanded, and no lead/stiffness/feedback-mode
        // combination fixed both channels at once. A free-running integral is
        // a genuine ramp at v, and a critically damped task tracks a ramp with
        // zero steady-state velocity error, so qdot = v falls out on its own
        // -- no lead term, no refVel, nothing to calibrate.
        //
        // Integrated at ctl.timeStep (the QP's own tick, 5ms here), not
        // ctl.policyStepSize (the policy's decision rate, 10ms) -- inference
        // above still only refreshes currentActionScaled once per
        // policyStepSize, but the held value is now integrated in
        // policyStepSize/timeStep sub-steps instead of one single jump. Same
        // net displacement per policy step in the unclamped regime (two 5ms
        // steps sum to the same one 10ms step would have given), but the QP
        // gets a target that moves at its OWN resolution rather than a
        // staircase held flat for a tick then jumped, and the clamps below
        // re-check against the freshly measured position at each sub-step
        // instead of once after the full jump -- a checkpoint every QP tick.
        //
        // The two clamps below are not optional: they are what makes the
        // free-running integral usable at all.
        auto & rr = ctl.realRobot(ctl.robots()[0].name());
        for (int j = 0; j < ctl.currentAction.size(); ++j) {
            int i = ctl.actionToDofMap[j];
            ctl.q_rl(i) += ctl.currentActionScaled(i) * ctl.timeStep;

            const int mcIdx = static_cast<int>(rr.jointIndexByName(ctl.jointNames[i]));
            const double q = rr.mbc().q[mcIdx][0];

            // Anti-windup, IdealPdActuator's
            //   max_dev = force_limit / stiffness
            //   pos_target = clamp(pos_target, pos - max_dev, pos + max_dev)
            // Once a joint saturates it stops following the target, and the
            // integral would otherwise gain velocity*dt of error every step
            // with nothing to unwind it. Pinning it to the measured position
            // bounds that.
            double dev = ctl.maxTargetDev_(i);
            if(dev > 0.0 && ctl.useQP())
            {
              // Under the QP that bound has a side effect it does not have in
              // training, and it is disabling. The solver emits one coupled
              // (q, qdot), so the ONLY way it can produce velocity is the
              // position error: qdot = sqrt(K)*e/2. Bounding e therefore caps
              // the achievable velocity at sqrt(K)*max_dev/2 -- 0.055 rad/s on
              // CROTCH_Y against the ~0.9 the policy asks for. Measured on the
              // 2026-08-12 13:58 log: the per-joint cap/demand ratio predicts
              // the delivered velocity almost exactly (0.06 -> 0.23, 0.14 ->
              // 0.16, 0.52 -> 0.51, 1.01 -> 0.56), overall 0.43, and the robot
              // fell. Bypass is unaffected because alpha_ref is a separate
              // channel there, which is why the same clamp is harmless in
              // training and in bypass (ratio 1.000 in the same run).
              //
              // Widen it to the smallest bound that does not clip the current
              // command, 2*|qdot|/sqrt(K), keeping the training value as the
              // floor. Still bounded, still unwinds, but no longer a velocity
              // ceiling. Windup beyond this is additionally contained by the
              // QP's own joint-limit constraint, which the bypass path lacks --
              // that asymmetry is the whole reason the clamp is needed there.
              const double K = ctl.postureTaskStiffness();
              if(K > 1e-9)
              {
                dev = std::max(dev, 2.0 * std::abs(ctl.currentActionScaled(i)) / std::sqrt(K));
              }
            }
            if(dev > 0.0) { ctl.q_rl(i) = std::max(q - dev, std::min(q + dev, ctl.q_rl(i))); }

            // And the physical joint range, as the actuator also does.
            const auto & lo = rr.ql()[mcIdx];
            const auto & hi = rr.qu()[mcIdx];
            if(!lo.empty() && !hi.empty()) { ctl.q_rl(i) = std::max(lo[0], std::min(hi[0], ctl.q_rl(i))); }
        }
        // Defensive clamp (config key: vel_target_limit_per_joint, already
        // loaded for every policy but previously unused here): nothing
        // bounds the raw network output, and a log of the QP run showed it
        // is exactly this quantity that runs away -- stable (~0.4 max)
        // under bypass, then diverging past +/-2 within ~8s of switching to
        // QP, well before the robot's orientation visibly degrades.
        for(int i = 0; i < ctl.nbActuatedJoints; ++i)
        {
          const double lim = ctl.velTargetLimitPerJoint_(i);
          ctl.qdTarget_(i) = std::max(-lim, std::min(lim, ctl.currentActionScaled(i)));
        }
      }
      // position_action (RHPS1 policies 0-3: target rebuilt from q_zero every
      // step, upstream posture filter, raw-torque channel, velocity damper,
      // torque-feasibility projection) removed 2026-09-04: this branch is
      // RHP7-only and velocity_action is its only action contract. See git
      // history to bring position_action back for a future RHP7 policy --
      // applyPostureFilter()/applyVelocityDamper()/projectTorqueFeasible()/
      // projectionFeedsCommand()/updateRawTorqueRatio() and their state were
      // removed alongside it, since this was their only caller.
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
//   510  hist 5 sur les SEPT termes      obs_format 5
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
  // 2 was the V4 entry; it is the velocity-action policy now, which has none.
  return policyIndex == 3;
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

  switch (ctl.obsFormat_) {
    case 6: // RHP7 Kaleido, velocity-action -- le SEUL format de cette branche.
            // Sept termes (base_lin_vel, base_ang_vel, projected_gravity,
            // joint_pos, joint_vel, actions, command) a l'historique 40, sur
            // 32 joints actionnes : 40*(3+3+3+32+32+32+3) = 4320, la forme
            // d'entree des ONNX RHP7. Toutes les dimensions viennent de
            // refJointOrderRLAction.size(), jamais d'un 30 litteral.
            //
            // Les cas RHPS1 (0/1/3/4/5, et le 2 qui partageait ce corps) ont
            // ete retires le 2026-09-01 avec la separation des deux robots par
            // branche : ils vivent sur `real-robot-safe`. Le numero 6 est
            // garde tel quel plutot que remis a 0, pour que ce bloc reste
            // reconnaissable des deux cotes lors des merges -- le yaml y
            // route explicitement via obs_format: 6 sur ses trois blocs, donc
            // le numero d'index n'entre pas en jeu.
            //
            // Utilise ses propres buffers *Deep_ (V3_DEEP_HISTORY_SIZE), pas
            // les buffers HISTORY_SIZE partages que les formats RHPS1
            // utilisaient.
    {
      auto & rr = ctl.realRobot(ctl.robots()[0].name());
      const std::string & baseName = rr.mb().body(0).name();
      const Eigen::Matrix3d R_w2b = rr.bodyPosW(baseName).rotation();

      // Shift history buffers: drop oldest (index V3_DEEP_HISTORY_SIZE-1), push at index 0
      for(int i = ctl.V3_DEEP_HISTORY_SIZE - 1; i > 0; --i)
      {
        ctl.linVelDeep_[i]   = ctl.linVelDeep_[i-1];
        ctl.angVelDeep_[i]   = ctl.angVelDeep_[i-1];
        ctl.projGravDeep_[i] = ctl.projGravDeep_[i-1];
        ctl.jointPosDeep_[i] = ctl.jointPosDeep_[i-1];
        ctl.jointVelDeep_[i] = ctl.jointVelDeep_[i-1];
        ctl.jointActDeep_[i] = ctl.jointActDeep_[i-1];
        ctl.velCmdDeep_[i]   = ctl.velCmdDeep_[i-1];
      }

      // Fill index 0 with current state
      ctl.linVelDeep_[0]   = R_w2b * rr.bodyVelW(baseName).linear();
      ctl.angVelDeep_[0]   = R_w2b * rr.bodyVelW(baseName).angular();
      ctl.projGravDeep_[0] = R_w2b * Eigen::Vector3d(0, 0, -1);
      ctl.velCmdDeep_[0]   = ctl.currentVelCmd_;

      // jointActDeep_[0] = raw NN output from the previous step (before scaling)
      ctl.jointActDeep_[0] = ctl.currentAction;

      const int actionDim = static_cast<int>(ctl.refJointOrderRLAction.size());
      ctl.jointPosDeep_[0] = Eigen::VectorXd::Zero(actionDim);
      ctl.jointVelDeep_[0] = Eigen::VectorXd::Zero(actionDim);
      for(int j = 0; j < actionDim; ++j)
      {
        const int mcIdx = rr.jointIndexByName(ctl.refJointOrderRLAction[j]);
        // mjlab's joint_pos_rel observation term: joint_pos - default_joint_pos
        // (velocity_env_cfg_rhps1.py, "joint_pos": func=mdp.joint_pos_rel).
        ctl.jointPosDeep_[0](j) = rr.mbc().q[mcIdx][0] - ctl.q_zero[ctl.actionToDofMap[j]];
        ctl.jointVelDeep_[0](j) = rr.mbc().alpha[mcIdx][0];
      }

      // Build observation: every term stacked as history, oldest first
      // (index V3_DEEP_HISTORY_SIZE-1 -> 0)
      auto write3 = [&](const Eigen::Vector3d & v)
      { obs.segment(offset, 3) = v; offset += 3; };

      for(int i = ctl.V3_DEEP_HISTORY_SIZE-1; i >= 0; --i) write3(ctl.linVelDeep_[i]);
      for(int i = ctl.V3_DEEP_HISTORY_SIZE-1; i >= 0; --i) write3(ctl.angVelDeep_[i]);
      for(int i = ctl.V3_DEEP_HISTORY_SIZE-1; i >= 0; --i) write3(ctl.projGravDeep_[i]);
      for(int i = ctl.V3_DEEP_HISTORY_SIZE-1; i >= 0; --i) appendToObs(ctl.jointPosDeep_[i]);
      for(int i = ctl.V3_DEEP_HISTORY_SIZE-1; i >= 0; --i) appendToObs(ctl.jointVelDeep_[i]);
      for(int i = ctl.V3_DEEP_HISTORY_SIZE-1; i >= 0; --i) appendToObs(ctl.jointActDeep_[i]);
      for(int i = ctl.V3_DEEP_HISTORY_SIZE-1; i >= 0; --i) write3(ctl.velCmdDeep_[i]);
      break;
    }
    default:
    {
      mc_rtc::log::error("[NewRLQPController::utils] Unknown obs_format: {}", ctl.obsFormat_);
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
        ctl.obsFormat_, offset, obs.size());
  }
  return obs;
}