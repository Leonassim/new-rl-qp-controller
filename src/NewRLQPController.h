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

  /** @brief Whether the projected target drives the command, or only the
   *  observation.
   *
   *  The projection encodes a torque, not a pose: its correctness is the
   *  identity "clamping tau and projecting q* onto tau's preimage are the same
   *  operation", which holds only for the PD it was derived from. Measured in
   *  mc_mujoco, the projected ankle-roll target sits 26 window widths from the
   *  measurement (the v_term shift, by design) -- fine for a PD that turns it
   *  into exactly effort_limit, absurd for a PostureTask at K=40000, which reads
   *  it as a ~4700 rad/s^2 demand. So under the QP the command keeps the
   *  filtered physical target and the QP's own torque bounds do the clamping;
   *  the observation still reads the projected target, as training does. */
  bool projectionFeedsCommand() const { return !useQP_; }

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

  // Observation history buffers.
  // Index 0 = most recent, index HISTORY_SIZE-1 = oldest.
  // Stacked oldest-first into the observation vector to match mjlab ordering.
  static constexpr int HISTORY_SIZE = 5;
  bool histInitialized_ = false;
  std::array<Eigen::Vector3d, HISTORY_SIZE> linVel_;
  std::array<Eigen::Vector3d, HISTORY_SIZE> angVel_;
  std::array<Eigen::Vector3d, HISTORY_SIZE> projGrav_;
  std::array<Eigen::VectorXd, HISTORY_SIZE> jointPos_;
  std::array<Eigen::VectorXd, HISTORY_SIZE> jointVel_;
  std::array<Eigen::VectorXd, HISTORY_SIZE> jointAct_;
  std::array<Eigen::Vector3d, HISTORY_SIZE> velCmd_;
  // V4 only: [sin(2pi*phi_L), cos(2pi*phi_L), sin(2pi*phi_R), cos(2pi*phi_R)]
  // scaled by the clock amplitude. See gaitPhaseStep().
  std::array<Eigen::Vector4d, HISTORY_SIZE> gaitPhase_;

  // V5 only: |tau_raw| / effort_limit per joint, history 10 (deeper than the
  // other blocks, which is why it does not share HISTORY_SIZE). See
  // updateRawTorqueRatio().
  static constexpr int RAW_TORQUE_HISTORY = 10;
  std::array<Eigen::VectorXd, RAW_TORQUE_HISTORY> rawTorque_;

  Eigen::Vector3d currentVelCmd_ = Eigen::Vector3d::Zero();

  // =========================================================================
  // Gait phase clock (V4 observation only)
  // =========================================================================
  //
  // mjlab drives the gait with an open-loop clock and feeds its phase to the
  // policy, so this has to reproduce that clock exactly or the network sees a
  // channel it never trained against. Mirrors mdp.rewards.gait_phase_tracking:
  // phase advances by dt/period each policy step and wraps into [0, 1); the
  // right foot is the left shifted by 0.5; the clock FREEZES while the
  // commanded speed is below gaitCommandThreshold_; the period interpolates
  // linearly from gaitPeriodSlow_ at that threshold to gaitPeriodFast_ at
  // gaitCommandRef_; and the block is scaled by an amplitude ramping 0 -> 1
  // over [0, gaitCommandThreshold_], so it goes flat at zero command instead of
  // freezing on whatever encoding the clock happened to stop on. A frozen but
  // nonzero encoding is indistinguishable from an active gait to the network.
  double gaitPhase_value_ = 0.0;
  double gaitPeriodSlow_ = 2.0;
  double gaitPeriodFast_ = 1.1;
  double gaitCommandThreshold_ = 0.1;
  double gaitCommandRef_ = 0.7;

  /** @brief Advance the gait clock one policy step and refresh gaitPhase_[0]. */
  void gaitPhaseStep();

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
  Eigen::VectorXd qdTarget_;    ///< Filtered velocity target (qd*).
  Eigen::VectorXd qTargetPrev_; ///< Previous *raw* position target, for the finite difference.
  bool projInitialized_ = false; ///< False until the first target seeds qTargetPrev_.
  Eigen::VectorXd rawTorqueRatio_; ///< |tau_raw| / effort_limit, current step.

  // Upstream PostureTask filter, `posture_filter_stiffness` per policy.
  //
  // Training puts the QP's PostureTask BEFORE the finite difference and the
  // projection (finite_difference_pd_actuator.py:308) -- the mc_rtc PostureTask
  // is downstream of both, so that ordering only exists here if we integrate it
  // ourselves. Without it qd* is the finite difference of the RAW target, which
  // for this policy jumps ~20 action units a step, saturates
  // vel_target_limit_per_joint and shoves the projection window ~0.3 rad off the
  // measurement -- the runaway described on qTargetPrev_ below, reached from the
  // other side. 0 disables it: index 0 trained with no filter at all.
  double postureFilterK_ = 0.0;
  Eigen::VectorXd postureQ_;  ///< Filter state, position.
  Eigen::VectorXd postureQd_; ///< Filter state, velocity.
  bool postureFilterInit_ = false;

  /** @brief Second-order posture filter, semi-implicit, `n` substeps of
   *  policyStepSize/n. Returns qCmd unchanged when the policy declares no
   *  posture_filter_stiffness. */
  Eigen::VectorXd applyPostureFilter(const Eigen::VectorXd & qCmd);

  // Velocity damper, `velocity_damper_di` per policy. mjlab runs it between the
  // qd* estimate and the torque projection (finite_difference_pd_actuator.py:377)
  // and it is NOT optional for the observation: _executed_position_target is
  // allocated unconditionally, so executed_action reads the POST-damper target
  // even when the projection is off. Under the QP mc_rtc's KinematicsConstraint
  // covers the command side, but nothing covered the observation, and the bypass
  // path had no damper at all.
  //
  // joint_limits must be mjlab's mj_model.jnt_range, not mc_rtc's -- the damper's
  // zone is a fraction of the range, so a different range is a different plant.
  // Declared in the yaml rather than read from the robot so the two cannot drift
  // apart silently.
  double damperDi_ = 0.0;
  double damperDs_ = 0.0;
  double damperVelPercent_ = 0.9;
  Eigen::VectorXd jointLower_, jointUpper_, velLimit_;

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

  /** @brief Dofs the posture function spans: nrDof minus the floating base. */
  int postureDofOffset() const;

  /** @brief refAccel = 2 (q* - q_out - dq_out T) / T^2, clamped, T floored at
   *  3 ticks -- the receding-horizon deadbeat diverges at T == dt. */
  void setPostureRefAccel(mc_tasks::PostureTaskPtr & pt);

  /** @brief Refresh the velocity and acceleration feedforward from finite
   *  differences of q_rl, once per policy step (q_rl is a staircase when
   *  policyStepSize > timeStep). */
  void updatePostureFeedforward();

  /** @brief Write the feedforward as refVel and refAccel, gains kept. */
  void setPostureFeedforward(mc_tasks::PostureTaskPtr & pt);

  /** @brief Clamp qd* to vel_percent * velocity_limits and project the target
   *  into the joint-limit safe region, as mjlab's _apply_velocity_damper does.
   *  No-op when the policy declares no velocity_damper_di. */
  Eigen::VectorXd applyVelocityDamper(const Eigen::VectorXd & qTarget);

  /**
   * @brief Advance qd*, compute the raw-torque ratio and push it into rawTorque_.
   *
   * Must run once per inference, on the RAW target, BEFORE projectTorqueFeasible
   * and regardless of use_QP -- the V5 network was trained with this channel and
   * will read garbage without it. Splitting it out of the projection is also what
   * keeps qTargetPrev_ fresh across a QP interlude: the projection used to be the
   * only writer, so it had to drop its seed on the way out.
   *
   * The ratio is the pre-clamp demand of the training PD:
   *   ratio_i = |kp_i (q*_i - q_i) + kd_i (qd*_i - qdot_i)| / effort_limit_i
   * evaluated on the raw target, which is what mjlab's actuator peak-holds over
   * the substeps between two policy steps.
   *
   * KNOWN APPROXIMATION: abl15 trained with decimation 2 (sim 0.0025 s, policy
   * 0.005 s), so its channel is the max over TWO substeps; this is a single
   * evaluation at the policy boundary. A max over two samples is >= either one,
   * so the controller feeds a slightly LOW ratio -- conservative in the sense
   * that it never invents demand, but it is not bit-exact. On the real robot
   * mc_rtc's dt equals the policy step, so there is no second sample to take.
   */
  void updateRawTorqueRatio(const Eigen::VectorXd & qTarget);

  /**
   * @brief Project @p qTarget onto the torque-feasible interval.
   * @return the projected target; unchanged when the projection is disabled.
   * @pre updateRawTorqueRatio() ran this step: qd* is read, not recomputed.
   */
  Eigen::VectorXd projectTorqueFeasible(const Eigen::VectorXd & qTarget);

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
