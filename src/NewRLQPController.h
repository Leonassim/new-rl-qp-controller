#pragma once

#include <array>
#include <mc_control/fsm/Controller.h>
#include <mc_rbdyn/SCHAddon.h>

#include "api.h"

#include "RLPolicyInterface.h"
#include "utils.h"


/**
 * @brief RL-QP Controller for deploying reinforcement learning policies on real robots.
 *
 * This controller bridges a trained RL policy (exported as ONNX) with mc_rtc's
 * whole-body QP framework augmented with Control Barrier Functions (CBFs).
 *
 * ## Control Pipeline
 *
 * At each policy step (typically 20ms):
 *   1. Build the observation vector from robot state (joint positions, velocities,
 *      IMU data, contact forces, velocity commands, etc.)
 *   2. Run inference: action = policy(observation)
 *   3. Compute position target: q* = action * action_scale + q_zero
 *   4. Compute desired torque via PD control: τ = Kp*(q* - q) - Kd*q̇
 *   5. Either:
 *      - (useQP=true)  Feed τ as input to TorqueJointTask inside the CBF-QP solver,
 *        which enforces joint limits, velocity limits, and self-collision constraints.
 *      - (useQP=false) Apply τ directly to the robot joints, bypassing the QP.
 *
 * ## Torque Equation
 *
 *   τ_i = clip(Kp_i * (q*_i - q_i) - Kd_i * q̇_i,  ±effort_limit_i)
 *
 * where:
 *   - q*_i  = action_i * action_scale_i + q_zero_i   (position target)
 *   - Kp_i  = pd_gains_ratio * kp_base_i
 *   - Kd_i  = sqrt(pd_gains_ratio) * kd_base_i
 *
 * ## Configuration
 *
 * All parameters are loaded from the YAML config file. See etc/NewRLQPController.in.yaml
 * for a fully documented example. Key sections per policy entry:
 *   - policy_path:       Path(s) to ONNX policy file(s)
 *   - action_scale:      Per-joint scale applied to raw policy output (map: joint -> scale)
 *   - q0:                Reference joint positions (default pose), in radians
 *   - kp / kd:           PD gains per joint
 *   - ref_joint_order:   Ordered list of joints controlled by the policy (action vector order)
 *   - use_QP:            Whether to route torques through the CBF-QP (true) or apply directly (false)
 *   - pd_gains_ratio:    Runtime gain scaling factor (1.0 = nominal gains)
 *   - policy_step_size:  Policy inference period in seconds (e.g. 0.02)
 *   - physics_step_size: Simulation/control timestep in seconds (e.g. 0.0025)
 *
 * ## Observation Vector
 *
 * The observation is built in utils::getCurrentObservation() and must exactly match
 * what the policy was trained on. Typical terms (history_length stacked, oldest first):
 *   - base_lin_vel:          Linear velocity of the floating base in body frame   [3]
 *   - base_ang_vel:          Angular velocity of the floating base in body frame   [3]
 *   - projected_gravity:     Unit gravity vector in body frame (from R_world_to_body * [0,0,-1]) [3]
 *   - joint_pos:             Joint positions relative to default pose: q - q_zero  [N_joints]
 *   - joint_vel:             Joint velocities                                      [N_joints]
 *   - foot_contact_forces:   Log-compressed world-frame contact forces             [N_feet * 3]
 *   - last_action:           Previous raw policy output (before scaling)           [N_action]
 *   - command:               Velocity command [vx, vy, yaw_rate]                  [3]
 *
 * Contact forces must be log-compressed before insertion:
 *   f_obs = sign(f) * log(1 + |f|)
 *
 * @see utils.h for the observation/action state machine helpers.
 * @see RLPolicyInterface.h for ONNX inference wrapper.
 */
struct NewRLQPController_DLLAPI NewRLQPController : public mc_control::fsm::Controller
{
  NewRLQPController(mc_rbdyn::RobotModulePtr rm, double dt, const mc_rtc::Configuration & config);

  bool run() override;
  void reset(const mc_control::ControllerResetData & reset_data) override;

  /** @brief Enable or disable the CBF-QP layer at runtime. */
  void activateQPControl(bool activate);

  /**
   * @brief Populate the history buffers with the current robot state.
   *
   * Called once at startup and at the beginning of each RL FSM state.
   * Must be overridden (filled in) by the user to match the specific
   * observation structure expected by the loaded policy.
   *
   * Typical implementation:
   * @code
   *   auto & robot = realRobot(robots()[0].name());
   *   // ... fill linVel[0], angVel[0], jointPos[0], etc. ...
   * @endcode
   */
  void initializeRLObservation();

  /** @brief PostureTask stiffness: `posture_stiffness` from the current
   *  policy's entry if it declares one, else from the global configuration,
   *  else the historical 0.2/(policyDt*timeStep).
   *
   *  A policy trained against a modelled PostureTask filter must run under the
   *  same stiffness: the lag it learned is 1/sqrt(K) and does not depend on the
   *  control rate -- see the note in the implementation. */
  double postureStiffness() const;

  /** @brief Load another policy index at runtime, from the GUI.
   *
   *  Refuses while ARMED: every per-policy plant parameter changes underneath
   *  the running loop otherwise. Replays initializeRobot() + initializeRLPolicy(),
   *  which is what startup does, then clears the projection, posture-filter and
   *  observation-history state so nothing carries over from the previous policy.
   *  The commanded posture is preserved across the switch so the PostureTask
   *  target does not jump, and the operator's QP toggle is preserved too --
   *  switching a policy is not a reason to silently re-enter the QP.
   *
   *  @return false and logs an error if armed or the index is out of range. */
  bool switchPolicy(size_t index);

  // =========================================================================
  // Robot state
  // =========================================================================

  /** @brief Total number of actuated joints (from robot().refJointOrder()). */
  int nbActuatedJoints = 0;

  /**
   * @brief Joint names in mc_rtc's reference order (robot().refJointOrder()).
   *
   * This order is used as the canonical ordering for all Eigen vectors
   * in this controller (kp_, kd_, q_zero, q_rl, etc.).
   */
  std::vector<std::string> jointNames;

  // Indices (into jointNames/nbActuatedJoints space) of L_HAND/R_HAND, -1 if
  // the robot has none. RHP7 registers them as mc_rbdyn::Gripper ("l_gripper"
  // /"r_gripper", rhp7.cpp), which unconditionally overwrites robot.mbc().q
  // for these two joints every tick with its own internal target
  // (generic_gripper.cpp: `robot.mbc().q[...] = {_q[i]};`, no guard) --
  // whatever the RL/QP computes for them is silently discarded, and the
  // ever-growing mismatch between our evolving q_rl and the Gripper's own
  // (unrelated, frozen-since-init) target is exactly what trips its safety
  // (see git history, 2026-09-06/07: "Gripper safety triggered on L_HAND").
  // Rather than fight that second control path, utils.cpp forces the RL
  // action to zero for these two indices every step, so both pos target and
  // vel target hold flat at q_zero/zero regardless of what the network
  // outputs.
  int lHandIdx_ = -1;
  int rHandIdx_ = -1;

  // =========================================================================
  // RL action
  // =========================================================================

  /**
   * @brief Current joint position targets sent to the PD controller.
   *
   * Computed as: q_rl = currentActionScaled + q_zero
   * Size: nbActuatedJoints (uncontrolled joints keep their q_zero value).
   */
  Eigen::VectorXd q_rl;

  /** @brief Previous q_rl, used to compute finite-difference velocity in byPassQPControl(). */
  Eigen::VectorXd q_rl_prev_;

   /**
   * @brief Default/reference joint positions loaded from config (q0).
   *
   * Corresponds to the robot's nominal standing pose used during RL training.
   * A zero policy output produces q_rl = q_zero.
   * Units: radians. Size: nbActuatedJoints.
   */
  Eigen::VectorXd q_zero;                      // Reference joint positions

  /**
   * @brief Raw output from the RL policy (before scaling).
   *
   * Directly returned by rlPolicy->predict(). Typically in [-1, 1] if the
   * policy uses a tanh output activation. Size: action space dimension.
   */
  Eigen::VectorXd currentAction;

  /**
   * @brief Scaled policy output: currentActionScaled = currentAction * actionScale.
   *
   * Has the same size as q_rl (nbActuatedJoints). Joints not controlled by
   * the policy have value 0.
   */
  Eigen::VectorXd currentActionScaled;

  /**
   * @brief Per-joint action scaling factors loaded from config.
   *
   * Typically computed as effort_limit / Kp to keep the policy output
   * in a physically meaningful range. Size: nbActuatedJoints.
   */
  Eigen::VectorXd actionScale;

  /** @brief N*Kt par joint, pour convertir en couple ce que le robot mesure.
   *
   * Sur RHPS1, ce que RobotHardware publie sur `tau` (et que setJointTorques
   * range dans robot().jointTorques()) n'est PAS un couple : le VRML donne
   * gearRatio = 1 et torqueConst = 1, donc la valeur vaut ratedCurrent *
   * 0x6077/1000, c'est-a-dire un COURANT en amperes.
   *
   * Rempli depuis `joint_torque_scale` du yaml, et laisse a 0 pour les joints
   * absents de cette table : les paires differentielles n'ont pas de Kt par
   * articulation, et les articulations a verins ont un bras de levier variable
   * qui demande CylinderToAngle. Size: nbActuatedJoints.
   */
  Eigen::VectorXd jointTorqueScale_;

  /** @brief Couples mesures en N.m, = jointTorques() * jointTorqueScale_.
   *  Zero sur les joints sans facteur connu. Size: nbActuatedJoints. */
  Eigen::VectorXd jointTorqueNm_;

  /** @brief Pour chaque joint pilote, son indice dans le refJointOrder COMPLET.
   *
   * robot().jointTorques() est indexe sur le refJointOrder complet, mains
   * comprises, alors que jointNames est filtre (30 joints). Sans cette
   * table, tout ce qui suit L_HAND est decale d'un cran et le bras droit
   * lit les valeurs du joint voisin. */
  std::vector<int> refIdx_;

  /** @brief Courants mesures en amperes, tels que publies par le robot.
   *  Tampon distinct de jointTorqueNm_ : les deux entrees de log sont
   *  evaluees dans le meme cycle. Size: nbActuatedJoints. */
  Eigen::VectorXd jointCurrentA_;

  /** @brief True quand le controleur tourne sous mc_mujoco plutot que sur le
   *  robot reel.
   *
   *  jointTorqueScale_ (voir plus haut) convertit le courant que publie
   *  RobotHardware sur le VRAI robot en couple. mc_mujoco publie deja un
   *  couple physique sur robot().jointTorques() -- lui reappliquer
   *  jointTorqueScale_ double-convertit et gonfle le log d'un facteur egal a
   *  ce scale (observe : L_KNEE_P sature a 45 N.m via le forcerange MuJoCo
   *  mais torque_L_KNEE_P affichait -954.45 = -45 * 21.21).
   *
   *  Detecte via le datastore : mc_mujoco (MjSimImpl::makeDatastoreCalls,
   *  mj_sim.cpp) enregistre un call "<robot>::SetPosW" pour chaque robot
   *  qu'il simule ; RobotHardware/CB sur le vrai robot n'a pas d'equivalent.
   *  Non mis en cache -- addLog() tourne dans le constructeur, avant que
   *  mc_mujoco n'ait forcement deja enregistre ce call, donc verifie a
   *  chaque appel des lambdas de log (executees bien plus tard, une fois la
   *  sim demarree) plutot qu'une fois a la construction. */
  bool isSimulated() const;

  // =========================================================================
  // Self-collision distance monitoring
  // =========================================================================

  /** @brief sch distance pairs mirroring the module minimalSelfCollisions,
   * evaluated on the realRobot for logging (NewRLQPController_selfcol_dist). */
  std::vector<std::tuple<std::string, std::string, std::shared_ptr<sch::CD_Pair>>> selfColPairs_;
  Eigen::VectorXd selfColDists_;
  void updateSelfCollisionDistances();

  // =========================================================================
  // Observation
  // =========================================================================

  /** @brief Full observation vector fed to the policy at each inference step. */
  Eigen::VectorXd currentObservation;

  // RHPS1 policy indices 0-3 (V3/V4/V5) used to share a depth-5 history here
  // (HISTORY_SIZE, linVel_/angVel_/projGrav_/jointPos_/jointVel_/jointAct_/
  // velCmd_/gaitPhase_/histInitialized_, plus the "V4 only" gait-phase clock
  // below it). Removed 2026-09-04: those cases no longer exist in the
  // switch on this branch, and every one of those buffers was write-only
  // (filled at reset(), never read by anything) once they went. See git
  // history if RHPS1-branch parity is ever needed again.
  //
  // updateRawTorqueRatio()/rawTorqueRatio_/rawTorque_/RAW_TORQUE_HISTORY
  // (the V5-only raw-torque channel, depth 10) lived here too, along with
  // applyPostureFilter()/applyVelocityDamper()/projectTorqueFeasible()/
  // projectionFeedsCommand() elsewhere in this file. All position_action-only
  // (utils.cpp's `else if(newInference)` branch, removed alongside them):
  // RHP7 will only ever run velocity_action. See git history to bring any
  // of it back if that ever changes.

  // Deep observation history for velocity_action policies (case 6 on this
  // branch, RHP7 Kaleido). V3_DEEP_HISTORY_SIZE = 10 -> with the 7 terms
  // `case 6:` stacks (base_lin_vel, base_ang_vel, projected_gravity,
  // joint_pos, joint_vel, actions, command) over 32 actuated joints, that is
  // 10*(3+3+3+32+32+32+3) = 1080, matching every RHP7 ONNX's declared input
  // ("ONNX policy loaded successfully (input: 1080, ...)" at load time).
  // Originally added for an RHPS1 policy (index 4, "hippolyte's run") at
  // history 40 (obs 4080) -- kept in its own buffers rather than sharing an
  // array size with any other case's declarations, so bumping it for one
  // policy cannot silently break the observation size of another.
  static constexpr int V3_DEEP_HISTORY_SIZE = 10;
  bool histInitializedV3Deep_ = false;
  std::array<Eigen::Vector3d, V3_DEEP_HISTORY_SIZE> linVelDeep_;
  std::array<Eigen::Vector3d, V3_DEEP_HISTORY_SIZE> angVelDeep_;
  std::array<Eigen::Vector3d, V3_DEEP_HISTORY_SIZE> projGravDeep_;
  std::array<Eigen::VectorXd, V3_DEEP_HISTORY_SIZE> jointPosDeep_;
  std::array<Eigen::VectorXd, V3_DEEP_HISTORY_SIZE> jointVelDeep_;
  std::array<Eigen::VectorXd, V3_DEEP_HISTORY_SIZE> jointActDeep_;
  std::array<Eigen::Vector3d, V3_DEEP_HISTORY_SIZE> velCmdDeep_;

  Eigen::Vector3d currentVelCmd_ = Eigen::Vector3d::Zero();

  // A gait-phase clock (open-loop, mirroring mjlab's mdp.rewards.gait_phase_
  // tracking) used to live here: gaitPhaseStep(), gaitPhase_value_,
  // gaitPeriodSlow_/Fast_, gaitCommandThreshold_, gaitCommandRef_, feeding a
  // gaitPhase_[HISTORY_SIZE] buffer. Removed 2026-09-04 along with the
  // depth-5 history buffers above: it fed the "V4 observation only" RHPS1
  // policy, gaitPhaseStep() was declared and defined but never called once
  // that case left the switch, and none of the doubles above had any other
  // reader either. See git history to bring it back.

  /**
   * @brief Config key: velocity_action (default false).
   *
   * Selects which mjlab action/actuator contract this policy was trained
   * against -- they are NOT interchangeable:
   *
   * - false (default, RHPS1 policies 0-3): position action (mjlab
   *   FiniteDifferencePdActuator). q_rl = q_zero + currentActionScaled,
   *   recomputed fresh every policy step, with the velocity feedforward
   *   reconstructed by finite-differencing consecutive q_rl targets exactly
   *   as the training actuator does. NOT IMPLEMENTED on this branch: RHP7
   *   only ever trains velocity_action, and the position_action code
   *   (updateRawTorqueRatio(), projectTorqueFeasible(), and the rest of
   *   utils.cpp's `else if(newInference)` branch) was removed 2026-09-04.
   *   See git history if a future RHP7 policy needs it back.
   * - true (policy 4, "hippolyte's velocity policy"): velocity action
   *   (mjlab JointVelocityAction + IdealPdActuator). q_rl = measured joint
   *   position + currentActionScaled * policyStepSize -- the actuator's own
   *   pos_target update, reseeded from the real robot state every policy
   *   step rather than integrated from its own previous value. qdTarget_ is
   *   currentActionScaled directly: the actuator's vel_error uses
   *   cmd.velocity_target raw and independently of how pos_target is built,
   *   so no finite difference and no torque-feasibility projection (a
   *   position-action-only concept) apply here.
   */
  bool velocityAction_ = false;

  /** @brief Is the CBF-QP layer active? (read by utils::run_rl_state) */
  bool useQP() const noexcept { return useQP_; }

  /** @brief Live PostureTask stiffness (tunable from the GUI), 0 if absent. */
  double postureTaskStiffness();


  // =========================================================================
  // Torque-feasibility projection (mirrors mjlab FiniteDifferencePdActuator)
  // =========================================================================
  //
  // Clamps the position target so the PD it feeds can never demand more than
  // ratio * effort_limit:
  //
  //   v_term = kd * (qd* - qdot)
  //   q* in [ q + (-budget - v_term)/kp , q + (budget - v_term)/kp ]
  //
  // The interval always has width 2*budget/kp > 0; the velocity term shifts it
  // rather than shrinking it, and when the velocity error alone would blow the
  // budget the whole interval sits on one side of q -- the projection then
  // commands the joint *back*, which is what the effort clamp was doing
  // implicitly anyway.
  //
  // Why it belongs here and not only in training: the policy learned on a plant
  // that projects. Without it the controller is a different plant, and the gap
  // shows up in transients, where demand peaks (a deterministic rollout of the
  // abl7 checkpoint read a max ratio of 234 against a steady-state mean of 0.27).
  //
  // qd* is NOT the measured velocity: it is the EMA-filtered finite difference
  // of the position target, exactly as the actuator computes it --
  //   raw   = clamp((q*_k - q*_{k-1}) / dt, +/- vel_target_limit)
  //   qd*_k = alpha * qd*_{k-1} + (1 - alpha) * raw
  // With alpha = 0.8 this carries state, which is why the policy is also given
  // an action history: nothing else in the observation reveals it.
  double torqueFeasibilityRatio_ = -1.0; ///< <= 0 disables the projection.
  double velTargetFilterAlpha_ = 0.0;
  Eigen::VectorXd effortLimit_;
  Eigen::VectorXd velTargetLimitPerJoint_; ///< Per-joint clamp used by the projection only (the scalar velTargetLimit_ below is the older feedforward clamp, kept separate on purpose).

  /**
   * @brief Anti-windup bound on the integrated position target, effort_limit/kp.
   *
   * Mirrors IdealPdActuator's `max_dev = force_limit / stiffness` clamp. A
   * free-running integral of the velocity command drifts without bound once the
   * joint saturates and stops tracking it, so the training actuator pins
   * pos_target to within this distance of the MEASURED position. That bound is
   * what makes the integral form usable at all -- without it the target runs
   * away (observed as the waddle, then divergence).
   *
   * It is the position error whose proportional term alone spends the whole
   * torque budget: 120/20000 = 6 mrad on CROTCH_P, 70/20000 = 3.5 mrad on the
   * knee, against 2 mrad of travel per policy step at 0.2 rad/s. Below
   * saturation it is inactive, so consecutive targets still differ by exactly
   * velocity*policyStepSize and the commanded velocity stays recoverable by
   * differencing -- which is what the QP path depends on.
   *
   * Zero (or negative) entries mean the policy declared no effort_limit for
   * that joint; the clamp is then skipped rather than pinning the target to
   * the measurement.
   */
  Eigen::VectorXd maxTargetDev_;
  Eigen::VectorXd qdTarget_;    ///< Filtered velocity target (qd*).

  // The upstream PostureTask filter (postureFilterK_/postureQ_/postureQd_/
  // postureFilterInit_/applyPostureFilter()), the velocity damper (damperDi_/
  // Ds_/VelPercent_/jointLower_/jointUpper_/velLimit_/applyVelocityDamper()),
  // the torque-feasibility projection (projectTorqueFeasible()/
  // projectionFeedsCommand()), and the raw-torque channel that fed it
  // (updateRawTorqueRatio()/qTargetPrev_/projInitialized_/rawTorqueRatio_)
  // used to live here. All position_action-only, removed 2026-09-04: RHP7
  // will only ever run velocity_action. See git history to bring any of it
  // back if that ever changes.

  /** @brief Apply the posture task gains for the current policy. */
  void applyPostureMode();

  // Posture pass-through, `posture_passthrough` per policy. The QP here is
  // kinematic, so the task's 2nd order is a stage training does not have:
  // q* -> [task + QP] -> q_out -> PD, against q* -> PD.
  //
  // Under TVM, PostureFunction is an IdentityFunction over qJoints and refAccel
  // sets normalAcceleration_ = -acc, so the QP row reads
  // q̈ = -Kp f - Kv ḟ + refAccel: additive, and zero gains leave q̈ = refAccel.
  //
  // SIZE TRAP: mc_tasks::PostureTask::refAccel asserts nrDof, the TVM function
  // asserts its own size, and on a floating base those differ by 6. Both are
  // compiled out in RelWithDebInfo, so a nrDof vector silently resizes
  // refAccel_ and corrupts the function -- that is what blew the robot up on
  // 2026-08-18. The function size is the correct one; a Debug build will trip
  // the mc_tasks assert, which is an mc_rtc inconsistency worth reporting.
  bool posturePassthrough_ = false;
  double postureAccelMax_ = 200.0;
  // Despite the name, this now guards THREE writers of refVel/refAccel --
  // setPostureRefAccel (posturePassthrough_), setPostureFeedforward
  // (postureFeedforward_), and setPostureRefVel (the plain default path,
  // refVel only) -- so that whichever one ran, the disarm branch in run()
  // and the mode switch in applyPostureMode() know there is something to
  // zero back out.
  bool postureRefAccelWritten_ = false;

  // Feedforward on the 2nd-order task. Gains alone are pure feedback, so the
  // task lags any moving target by 1/sqrt(K) and, worse, low-passes it at
  // sqrt(K) rad/s -- 40 here. Feeding the commanded trajectory's own qd* and
  // qdd* alongside makes the QP row q_ddot = qdd* + K(q* - q) + Kv(qd* - qd):
  // the gains only correct error, and a target the policy slews smoothly is
  // tracked without lag. mc_tvm::PostureFunction already composes it that way
  // (velocity_ -= refVel_, normalAcceleration_ = -refAccel).
  //
  // Off by default: qd*/qdd* are finite differences, so they amplify tremor by
  // 1/dt and 1/dt^2. Worth having only for a policy trained with the smoothness
  // penalties -- on a shaky one it makes things worse, not better.
  bool postureFeedforward_ = false;
  bool ffInit_ = false;
  Eigen::VectorXd ffPrevQ_, ffVel_, ffPrevVel_, ffAcc_;

  // Experimental ablation: hide the QP's own computed velocity from
  // mc_mujoco, sending position only (alpha_ref = 0 in MjRobot::sendControl's
  // PD). NOT a straight zero-out of robot().mbc().alpha -- the solver's own
  // OpenLoop integration uses that same storage as its state from one tick to
  // the next, so restoreQPVelocity()/zeroAlphaOut() shuttle the true value
  // through alphaOutShadow_ across ticks instead. See run(). Off by default,
  // config key qp_zero_vel_out, simulation-only for now.
  bool qpZeroVelOut_ = false;
  Eigen::VectorXd alphaOutShadow_;

  /** @brief Put back the velocity zeroAlphaOut() hid from mc_mujoco last
   *  tick, before the solver integrates from it again. Must run before
   *  mc_control::fsm::Controller::run(). */
  void restoreQPVelocity();

  /** @brief Stash this tick's true QP-computed velocity into
   *  alphaOutShadow_ and zero robot().mbc().alpha so mc_mujoco's PD sees no
   *  velocity feedforward. Must run after
   *  mc_control::fsm::Controller::run(). */
  void zeroAlphaOut();

  // -1 = auto (whatever stiffness() derives as 2*sqrt(K)); >= 0 = explicit
  // override, config key posture_damping. Kept as a member, not applied once
  // and forgotten, because stiffness() resets damping as a side effect every
  // time it runs -- see applyPostureDampingOverride().
  double postureDampingOverride_ = -1.0;

  /** @brief Reapply postureDampingOverride_ after any pt->stiffness(...)
   *  call, which resets damping to 2*sqrt(K) as a side effect and would
   *  otherwise silently drop the override. No-op if postureDampingOverride_
   *  is the -1 sentinel (auto). */
  void applyPostureDampingOverride(mc_tasks::PostureTaskPtr & pt);

  /** @brief Dofs the posture function spans: nrDof minus the floating base. */
  int postureDofOffset() const;

  /** @brief Write qdTarget_ as refVel, every tick, unconditionally (the plain
   *  default path -- no posture_passthrough/posture_feedforward involved).
   *  qdTarget_ is already the exact commanded velocity for velocity_action
   *  policies (utils.cpp run_rl_state, no finite difference -- the only
   *  contract RHP7 runs). Harmless under posturePassthrough_ (damping is
   *  zeroed there, so the Kv*(refVel-qdot) term it feeds is inert) and
   *  overwritten by setPostureFeedforward() when postureFeedforward_ is on
   *  (that path owns refVel with its own, richer qd-star / qdd-star pair).
   *  Sets postureRefAccelWritten_ so the existing disarm path zeroes it back
   *  out -- see run(). */
  void setPostureRefVel(mc_tasks::PostureTaskPtr & pt);

  /** @brief refAccel = 2 (q* - q_out - dq_out T) / T^2, clamped, T floored at
   *  3 ticks -- the receding-horizon deadbeat diverges at T == dt. */
  void setPostureRefAccel(mc_tasks::PostureTaskPtr & pt);

  /** @brief Refresh the velocity and acceleration feedforward from finite
   *  differences of q_rl, once per policy step (q_rl is a staircase when
   *  policyStepSize > timeStep). */
  void updatePostureFeedforward();

  /** @brief Write the feedforward as refVel and refAccel, gains kept. */
  void setPostureFeedforward(mc_tasks::PostureTaskPtr & pt);

  // =========================================================================
  // Policy / timing
  // =========================================================================

  /**
   * @brief Policy inference period in seconds.
   *
   * The policy runs once every policyStepSize seconds. Between inference steps,
   * q_rl is held constant while the PD torque is recomputed at every controller
   * timestep using fresh joint state (equivalent to a real onboard PD loop).
   */
  double policyStepSize;

  /**
   * @brief Ordered joint names that correspond to the policy's action vector.
   *
   * Loaded from config ref_joint_order. May be a subset of jointNames if the
   * policy does not control all joints (e.g. fingers excluded).
   */
  std::vector<std::string> refJointOrderRLAction;

  /**
   * @brief Maps action vector index → mc_rtc joint index (jointNames).
   *
   * actionToDofMap[j] = i means policy output j controls jointNames[i].
   * Size: action space dimension.
   */
  std::vector<int> actionToDofMap;

  /**
   * @brief Maps mc_rtc joint index → RL framework joint index.
   *
   * mcRtcToRLFrameworkJointMap[i] = j means jointNames[i] corresponds to
   * position j in the RL observation/action joint ordering (from q0 config keys).
   * Size: nbActuatedJoints.
   */
  std::vector<int> mcRtcToRLFrameworkJointMap;

  // =========================================================================
  // Policy management
  // =========================================================================

  /** @brief Index of the currently active policy (indexes into policy_path list). */
  size_t currentPolicyIndex = 0;
  /// Observation layout, from `obs_format`. Defaults to currentPolicyIndex, so
  /// the switch in utils.cpp keeps its historical meaning; declare it when a
  /// slot reuses another slot's network format.
  size_t obsFormat_ = 0;
  /// Stiffness typed into the GUI, -1 when untouched. Wins over the yaml so
  /// applyPostureMode() cannot silently undo it mid-experiment.
  double postureStiffnessOverride_ = -1.0;
  /// Auto-disarm above this measured joint velocity (rad/s), 0 = off. A runaway
  /// reaches saturation in ~400 ms, faster than an operator can react.
  double runawayDisarmVel_ = 0.0;

  /** @brief ONNX runtime wrapper for running policy inference. */
  std::unique_ptr<RLPolicyInterface> rlPolicy;

  /** @brief Utility helpers for FSM state lifecycle and observation building. */
  utils utilsClass;

  /** @brief Policy engaged? False until the operator arms it from the GUI.
   *
   * Loading the controller and running the policy are deliberately separate on
   * real hardware. While disarmed nothing calls the policy and nothing writes
   * the posture target, so the PostureTask keeps holding the posture mc_rtc
   * captured at reset -- the robot stands where the operator left it, under QP,
   * for as long as they want. Arming is what runs start_rl_state (measured-
   * posture hold, then ramp to q_zero) and starts inference.
   *
   * Public because the Initial state reads and sets it.
   */
  bool policyArmed_ = false;

private:
  mc_rtc::Configuration config_;

  /** @brief Register data entries visible in mc_log_ui. */
  void addLog();
  /** @brief Register GUI elements visible in RViz and mc_mujoco. */
  void addGui();

  /** @brief Abort if no floating-base observer made it into the pipeline. */
  void checkFloatingBaseObserver();

  /** @brief Load robot parameters (gains, action scale, q0) from config. */
  void initializeRobot();

  /** @brief Install the constraints needed by the QP backend. */
  void configureSolverConstraints();

  /** @brief Load and validate the ONNX policy, build joint mappings. */
  void configRL();

  /** @brief Instantiate rlPolicy and initialize observation buffers. */
  void initializeRLPolicy();

   /**
   * @brief Apply RL torques directly, bypassing the QP (useQP=false mode).
   *
   * Computes τ = Kp*(q_rl - q) - Kd*q̇ and writes it to robot().mbc().jointTorque.
   * @return true if bypass was applied, false if QP should run instead.
   */
  bool byPassQPControl();
  bool useQP_ = true; ///< Route torques through CBF-QP (true) or apply directly (false)











  /** @brief Read joystick via datastore and apply velocity ramp to currentVelCmd_. */
  void updateVelocityCommand();

  // --- Joystick / velocity command ---
  double maxVelX_          = 0.6;
  double maxVelY_          = 0.4;
  double maxYawCmd_        = 0.7;
  double joystickDeadZone_ = 0.05;
  double velRampRate_      = 0.5; ///< m/s per second rate limit on velocity command
  double velTargetLimit_   = 8.0; ///< rad/s clamp on the finite-difference velocity feedforward (training actuator velocity_target_limit)
  bool   useJoystick_      = true;

  /** @brief Detect foot touchdown from the ankle F/T sensors (RHPS1_MuJoCo's
   * rf_force/lf_force sites, see RHPS1main.xml) and record the ankle body
   * velocity at the instant of impact. Config key log_impact_vel -- present
   * in the yaml since the project's first commit but never wired up until
   * now. Rising-edge detection on Fz with hysteresis so noise around the
   * threshold does not retrigger it mid-stance. */
  void updateImpactVelocity();

  // --- Impact velocity logging ---
  bool   logImpactVel_          = false; ///< config: log_impact_vel
  double impactForceThreshold_  = 30.0;  ///< N on Fz, rising edge = touchdown. config: impact_force_threshold
  double impactForceHysteresis_ = 0.5;   ///< falling-edge threshold = ratio * impactForceThreshold_. config: impact_force_hysteresis_ratio
  bool   leftFootContact_       = false;
  bool   rightFootContact_      = false;
  double leftFootForceZ_        = 0.0;   ///< last-read Fz, logged raw to help pick impactForceThreshold_
  double rightFootForceZ_       = 0.0;
  Eigen::Vector3d leftFootImpactVel_  = Eigen::Vector3d::Zero(); ///< world-frame linear velocity of L_ANKLE_P_LINK at last touchdown; holds between impacts
  Eigen::Vector3d rightFootImpactVel_ = Eigen::Vector3d::Zero();

  /** @brief Log warnings when joint position/velocity/torque limits are exceeded. */
  void computeLimits();
  bool printLimits_ = true;

  std::string robotName_;
  
  // --- CBF-QP constraint parameters ---
  double velPercent_ = 0.95; // Percentage of the max velocity taking account in the joint velocity constraint.
  double dsPercent_ = 0.01; // Percentage of the max joint range taking account in the joint position limit constraint.
  double diPercent_ = 0.1; // Doesn't matter since di > ds. This variable is not used in the constraint dynamics.

  // --- CBF Gains ---
  // More details are explained in the paper cf. Readme.md.
  // Must be tuned depending on the robot.
  //
  // UNUSED on this branch: RHP7's real mc_rtc predates commit 2957e17c
  // (2026-05-05, bastien-muraccioli/mc_rtc), which is what added CBF support
  // to CollisionsConstraint/KinematicsConstraint. NewRLQPController.cpp's
  // constructor falls back to the plain velocity-level damper instead. Kept
  // here (not deleted) so re-enabling CBF is a one-line revert once the
  // robot's mc_rtc catches up.
  double zeta_jointLimit_ = 1.2;
  double lambda_jointLimit_ = 100.0; // Same gain for joint position limits and velocity limits.
  double zeta_selfCollision_ = 1.2;
  double lambda_selfCollision_ = 10.0;

  // --- PD gains ---
  double pdGainsRatio_ = 1.0;
  Eigen::VectorXd kp_;      ///< Active proportional gains = pdGainsRatio_ * kpBase_
  Eigen::VectorXd kd_;      ///< Active derivative gains   = sqrt(pdGainsRatio_) * kdBase_
  Eigen::VectorXd kpBase_;  ///< Nominal Kp from config
  Eigen::VectorXd kdBase_;  ///< Nominal Kd from config

  // --- Policy ---
  std::vector<std::string> policyPaths_; ///< Paths to ONNX files (one per policy index)
  std::map<std::string, double> q0_map_; ///< Raw q0 map used to build mcRtcToRLFrameworkJointMap
};
