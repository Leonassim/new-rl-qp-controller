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
      else if(newInference)
      {
        // position_action (policies 0-3): the target is rebuilt from q_zero
        // every step, not integrated. Kept in this branch rather than in the
        // loop above because velocity_action integrates instead.
        for (int j = 0; j < ctl.currentAction.size(); ++j) {
            int i = ctl.actionToDofMap[j];
            ctl.q_rl(i) = ctl.currentActionScaled(i) + ctl.q_zero(i);
        }
        // The QP's PostureTask, reproduced upstream of the finite difference and
        // the projection because that is where training puts it
        // (finite_difference_pd_actuator.py:308) and the mc_rtc PostureTask is
        // downstream of both. No-op unless the policy declares
        // posture_filter_stiffness.
        ctl.q_rl = ctl.applyPostureFilter(ctl.q_rl);
        // Order matters: the raw-torque channel is measured on the target BEFORE
        // the projection (measuring it after makes every joint report exactly the
        // ratio the projection enforces -- the projection measuring itself), and it
        // must run in both QP and bypass because the V5 network reads it either way.
        ctl.updateRawTorqueRatio(ctl.q_rl);
        // Velocity damper, between the qd* estimate and the projection, as
        // finite_difference_pd_actuator.py:377 has it. It also clamps qd*, so it
        // must run before the projection reads it. No-op unless the policy block
        // sets velocity_damper_di.
        ctl.q_rl = ctl.applyVelocityDamper(ctl.q_rl);
        // Project onto the torque-feasible set, exactly as the training actuator
        // does. No-op unless the policy block sets torque_feasibility_ratio.
        const Eigen::VectorXd qProjected = ctl.projectTorqueFeasible(ctl.q_rl);
        // The projected target drives the command only on the bypass path, where
        // the plant really is the PD the projection was derived from. Under the QP
        // it would hand the PostureTask a torque encoding as if it were a pose --
        // see NewRLQPController::projectionFeedsCommand().
        if(ctl.projectionFeedsCommand()) { ctl.q_rl = qProjected; }
        // Feed back the action as EXECUTED, not as requested: the V4 observation's
        // actions block is executed_action. Beyond the feasible window many raw
        // actions map to one execution, and a policy that only ever sees what it
        // asked for cannot tell them apart -- which is the whole reason the
        // training observation switched away from last_action. Always the projected
        // target, whatever drives the command: that is the signal training feeds.
        for (int j = 0; j < ctl.currentAction.size(); ++j) {
            int i = ctl.actionToDofMap[j];
            const double scale = ctl.actionScale(i);
            if(std::abs(scale) > 1e-12)
            {
              ctl.currentActionScaled(i) = qProjected(i) - ctl.q_zero(i);
              ctl.currentAction(j) = ctl.currentActionScaled(i) / scale;
            }
      }
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
//   510  hist 5 sur les SEPT termes      obs_format 5
//   530  510 + gait_phase[4 x 5]         obs_format 7
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
  // 6 est le format 530 dims : le format 5 plus l'horloge en queue.
  // 6 est le RHP7 Kaleido (velocity-action, pas d'horloge).
  // 7 est le format 530 dims du RHPS1 : le format 5 plus l'horloge en queue.
  return policyIndex == 3 || policyIndex == 7;
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
    case 1: // 2026-08-21 : l'index 1 porte desormais une policy 126 dims
            // (run 2026-08-21_11-47-31, paquet `lift` sur p0+rand), donc le
            // meme corps que l'index 0 et non plus les 246 dims. L'ONNX le
            // confirme : obs [1,126], history 5 sur base_lin_vel et command
            // seulement. Rebasculer l'etiquette si on reinstalle une policy
            // 246 dims sur cet index.
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
    case 7: // 530 dims = le format 5 plus l'horloge de demarche (4 x 5 dims).
            // La politique DOIT voir la phase, sinon elle subit la penalite de
            // gait_phase_tracking sans pouvoir la satisfaire. gaitPhaseStep()
            // reproduit l'horloge : periode interpolee de 2.0 s au seuil de
            // commande a 1.1 s a la reference 0.7, avancee de dt/periode par pas
            // de politique, figee sous le seuil, et le bloc mis a l'echelle par
            // une amplitude qui monte de 0 a 1 sur [0, seuil].
    case 5: // RHPS1 velocity policies — 510 dims, historique 5 sur les SEPT termes
            // mjlab-rhps1 run 2026-08-27_09-31-27 (etape 0b : config policy 0 +
            // hist5 + biais capteurs + mirror loss + paires QP, sans la
            // randomisation large).
            //
            // Meme corps que V3, meme profondeur 5, mais l'historique porte sur
            // tous les termes au lieu de deux :
            //   base_lin_vel[15] base_ang_vel[15] projected_gravity[15]
            //   joint_pos[150]   joint_vel[150]   actions[150] command[15]
            // 3*15 + 3*150 + 15 = 510.
            //
            // Ordre : mjlab renvoie CircularBuffer.buffer, documente "index 0
            // is oldest and index -1 is newest", puis aplatit. Les tampons ici
            // ont l'index 0 le plus RECENT, d'ou la boucle descendante -- la
            // meme que V3 utilise deja pour ses deux termes historises.
            //
            // Les cinq creneaux sont amorces a l'init (NewRLQPController.cpp,
            // histInitialized_), donc aucun VectorXd de taille nulle ne peut
            // desaligner le vecteur.
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
        // Le format 6 porte un huitieme terme, decale comme les autres. Sans
        // cette ligne les cinq creneaux d'historique liraient la meme valeur.
        if(ctl.obsFormat_ == 7) { ctl.gaitPhase_[i] = ctl.gaitPhase_[i-1]; }
      }

      ctl.linVel_[0]   = R_w2b * rr.bodyVelW(baseName).linear();
      ctl.angVel_[0]   = R_w2b * rr.bodyVelW(baseName).angular();
      ctl.projGrav_[0] = R_w2b * Eigen::Vector3d(0, 0, -1);
      ctl.velCmd_[0]   = ctl.currentVelCmd_;
      ctl.jointAct_[0] = ctl.currentAction;
      // Avancer l'horloge exactement une fois par inference, APRES velCmd_[0]
      // dont depend la cadence, et AVANT l'ecriture du bloc. Le format 3 fait
      // le meme appel dans son propre bloc ; il n'est pas partage.
      if(ctl.obsFormat_ == 7) { ctl.gaitPhaseStep(); }

      const int actionDim = static_cast<int>(ctl.refJointOrderRLAction.size());
      ctl.jointPos_[0] = Eigen::VectorXd::Zero(actionDim);
      ctl.jointVel_[0] = Eigen::VectorXd::Zero(actionDim);
      for(int j = 0; j < actionDim; ++j)
      {
        const int mcIdx = rr.jointIndexByName(ctl.refJointOrderRLAction[j]);
        ctl.jointPos_[0](j) = rr.mbc().q[mcIdx][0] - ctl.q_zero[ctl.actionToDofMap[j]];
        ctl.jointVel_[0](j) = rr.mbc().alpha[mcIdx][0];
      }

      auto write3b = [&](const Eigen::Vector3d & v)
      { obs.segment(offset, 3) = v; offset += 3; };

      for(int i = ctl.HISTORY_SIZE-1; i >= 0; --i) write3b(ctl.linVel_[i]);
      for(int i = ctl.HISTORY_SIZE-1; i >= 0; --i) write3b(ctl.angVel_[i]);
      for(int i = ctl.HISTORY_SIZE-1; i >= 0; --i) write3b(ctl.projGrav_[i]);
      for(int i = ctl.HISTORY_SIZE-1; i >= 0; --i) appendToObs(ctl.jointPos_[i]);
      for(int i = ctl.HISTORY_SIZE-1; i >= 0; --i) appendToObs(ctl.jointVel_[i]);
      for(int i = ctl.HISTORY_SIZE-1; i >= 0; --i) appendToObs(ctl.jointAct_[i]);
      for(int i = ctl.HISTORY_SIZE-1; i >= 0; --i) write3b(ctl.velCmd_[i]);
      if(ctl.obsFormat_ == 7)
      {
        // 20 dims de plus : l'horloge de demarche, 4 canaux x 5 d'historique.
        // Ordre lu dans la metadonnee de l'ONNX et non suppose :
        //   base_lin_vel, base_ang_vel, projected_gravity, joint_pos,
        //   joint_vel, actions, command, gait_phase
        // donc le bloc de phase vient EN DERNIER, offsets 510 a 529.
        for(int i = ctl.HISTORY_SIZE-1; i >= 0; --i)
        {
          obs.segment(offset, 4) = ctl.gaitPhase_[i];
          offset += 4;
        }
      }
      break;
    }
    // Le bloc 246 dims ci-dessous : l'index 1 l'a quitte le 2026-08-21.
            // Format introduced by run 2026-08-07_15-40-43 : le retour a la base
            // policy 0 (echelle x1.5, keyframe genou 0.622) avec les armatures
            // reelles et joint_vel par difference finie. C'est le corps commun
            // tout court : ni gait_phase, ni raw_torque.
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
      if(utils::hasGaitPhase(ctl.obsFormat_)) { ctl.gaitPhaseStep(); }

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
      if(utils::hasGaitPhase(ctl.obsFormat_))
      {
        for(int i = ctl.HISTORY_SIZE-1; i >= 0; --i) write4(ctl.gaitPhase_[i]);
      }

      // V5 tail. rawTorque_ is pushed by updateRawTorqueRatio() at the END of the
      // previous policy step, so index 0 holds the demand of the action that has
      // just been executed -- the same alignment mjlab has, where the observation
      // reads a peak accumulated over the previous step's substeps.
      if(utils::isV5(ctl.obsFormat_))
      {
        for(int i = ctl.RAW_TORQUE_HISTORY-1; i >= 0; --i) appendToObs(ctl.rawTorque_[i]);
      }
      break;
    }
    case 2: // hippolyte's velocity-action run -- obs = 4080 dims.
            // Index 2 is this policy now: the list was condensed to v3 (0),
            // v9 (1) and this one (2), so the V4 label that used to sit on
            // case 2 was dropped from the shared body above. The only
            // velocity_action entry (see NewRLQPController::velocityAction_).
            // observation_names (ONNX metadata): base_lin_vel, base_ang_vel,
            // projected_gravity, joint_pos, joint_vel, actions, command --
            // ALL seven terms at history depth 40 (unlike V3/V4/V5's depth 5,
            // and V3's depth-5-on-two-terms-only). Uses its own *Deep_
            // buffers (V3_DEEP_HISTORY_SIZE), not the shared HISTORY_SIZE
            // ones case 0 uses, so this doesn't disturb any other index:
            // 40 * (3+3+3+30+30+30+3) = 40 * 102 = 4080.
    case 6: // RHP7 Kaleido -- same layout as case 2 (velocity-action, same
            // seven terms at history depth 40), on a different robot with 32
            // actuated joints instead of 30: 40*(3+3+3+32+32+32+3) = 4320.
            // Every dimension below already comes from
            // refJointOrderRLAction.size(), not a literal 30, so nothing in
            // this block changes for the extra 2 joints. Shares case 2's
            // block rather than a copy: same code path, only the config
            // (policies[6/7/8] in the yaml, obs_format: 6) and the loaded
            // robot differ. Was case 5 until the colleague's 2026-08-31
            // merge claimed case 5 for a genuinely new RHPS1 510-dim/hist5
            // format above -- moved here (6) instead, still free. Index 1,
            // 3 and 4 were NOT free either -- they are the RHPS1 V5/gait-phase
            // case above (case 1/3/4 shared body) -- picking any of them
            // would have silently written that layout instead of this one.
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