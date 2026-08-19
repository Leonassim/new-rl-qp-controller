#pragma once

#include <array>
#include <RBDyn/FD.h>
#include <mc_control/fsm/Controller.h>
#include <mc_rbdyn/SCHAddon.h>
#include <mc_tasks/TorqueJointTask.h>

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
  // Shared by policy indices 0/1/2/3 (V3/V4/V5), all trained with a 5-step
  // history. Do NOT bump this for a policy that wants a different depth --
  // it is a compile-time array size shared by every case in the switch, so
  // changing it silently breaks every other case's observation size (see
  // the dedicated V3-deep-history buffers below for how a case with its own
  // depth is meant to be added instead).
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

  // Policy index 4 only (hippolyte's run, obs = 4080 = 40 * 102): history 40
  // on every term (base_lin_vel, base_ang_vel, projected_gravity, joint_pos,
  // joint_vel, actions, command), unlike V3/V4/V5's depth-5 history. Kept in
  // its own buffers rather than resizing HISTORY_SIZE, for the same reason
  // rawTorque_ has its own RAW_TORQUE_HISTORY: one array size is shared by
  // every case's declarations, so bumping it for one policy silently breaks
  // the observation size of all the others.
  static constexpr int V3_DEEP_HISTORY_SIZE = 40;
  bool histInitializedV3Deep_ = false;
  std::array<Eigen::Vector3d, V3_DEEP_HISTORY_SIZE> linVelDeep_;
  std::array<Eigen::Vector3d, V3_DEEP_HISTORY_SIZE> angVelDeep_;
  std::array<Eigen::Vector3d, V3_DEEP_HISTORY_SIZE> projGravDeep_;
  std::array<Eigen::VectorXd, V3_DEEP_HISTORY_SIZE> jointPosDeep_;
  std::array<Eigen::VectorXd, V3_DEEP_HISTORY_SIZE> jointVelDeep_;
  std::array<Eigen::VectorXd, V3_DEEP_HISTORY_SIZE> jointActDeep_;
  std::array<Eigen::Vector3d, V3_DEEP_HISTORY_SIZE> velCmdDeep_;

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

  /**
   * @brief Config key: velocity_action (default false).
   *
   * Selects which mjlab action/actuator contract this policy was trained
   * against -- they are NOT interchangeable:
   *
   * - false (default, policies 0-3): position action (mjlab
   *   FiniteDifferencePdActuator). q_rl = q_zero + currentActionScaled is
   *   recomputed fresh every policy step; updateRawTorqueRatio()/
   *   projectTorqueFeasible() reconstruct the velocity feedforward by
   *   finite-differencing consecutive q_rl targets, exactly as the training
   *   actuator does.
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
  Eigen::VectorXd qTargetPrev_; ///< Previous *raw* position target, for the finite difference.
  bool projInitialized_ = false; ///< False until the first target seeds qTargetPrev_.
  Eigen::VectorXd rawTorqueRatio_; ///< |tau_raw| / effort_limit, current step.

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



  /** @brief PostureTask stiffness, 0.2/(policy_step_size*timeStep). */
  double postureStiffness();

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

  /**
   * @brief Acceleration feedforward on the PostureTask (config: qp_accel_task).
   *
   * Computes the training actuator's torque explicitly and hands the solver the
   * matching acceleration:
   *   tau  = kp*(q_rl - q_meas) + kd*(qdTarget_ - qdot_meas)
   *   qddot = tau / m,            m = diag(H + HIr)
   *
   * Why: with only pt->target(q_rl) the commanded velocity survives merely as
   * the slope of q_rl and the solver has to re-derive it by tracking the ramp.
   * It recovers 85.6% (measured 2026-08-12 14:06, bypass 1.000 in the same
   * run). The shortfall is NOT bandwidth -- 99.8% of the command's variance is
   * below the task's 10 Hz -- but the anti-windup clamp biting on transients:
   * it bounds the position error at the steady-state value 2|qdot|/sqrt(K),
   * while *changing* velocity transiently needs more, and the two joints
   * clamped ~33% of the time are exactly the two delivering the least
   * (L_ANKLE_P 0.644, R_CROTCH_Y 0.725). A feedforward supplies that
   * acceleration directly instead of extracting it from a bounded error.
   *
   * refAccel is a native task input, so this needs NO contacts: the QP stays
   * kinematic. TorqueJointTask, by contrast, needs dynamicsConstraint and
   * produced 3064 Nm with no contact set defined. The output remains entirely
   * the solver's, so the CBF joint limits and self-collisions stay guaranteed.
   *
   * CAVEAT: mc_mujoco still applies its own PD (kp=20000, kd=400) to the QP's
   * (q, alpha). The torque the robot feels is kp(q_QP - q_meas) +
   * kd(alpha_QP - qdot_meas), not tau. This transmits the intent; it does not
   * reproduce tau exactly -- which is why the position feedback is kept by
   * default rather than running pure feedforward.
   */
  bool qpAccelTask_ = false;

  /**
   * @brief Task gains used while qpAccelTask_ is on (qp_accel_stiffness/damping).
   *
   * Default to the existing values (postureStiffness() and its 2*sqrt(K)), so
   * the feedforward is *added* to the current behaviour rather than replacing
   * it. Set both to 0 for the literal "qddot = tau/m and nothing else" variant
   * -- but note the reference then has no pull back towards q_rl, while tau is
   * computed from the MEASURED state and the QP integrates its own
   * (FeedbackType::OpenLoop): the two can drift apart with nothing to rejoin
   * them. Watch ctrl_q against qIn if testing that.
   */
  double qpAccelStiffness_ = -1.0; ///< <0 means "use postureStiffness()"
  double qpAccelDamping_ = -1.0;   ///< <0 means "use 2*sqrt(stiffness)"

  /**
   * @brief Rotor inertia added to diag(H) (config key: joint_armature).
   *
   * The mc_rtc model comes from the URDF, which carries no rotor inertia --
   * fd_.HIr() is 0.00000 on every joint. The plant actually being controlled
   * has one: mc_mujoco's RHPS1main.xml sets armature="1" in its <joint>
   * default, and mjlab's rhps1_constants.py sets armature=1.0 on every
   * actuator, so training and simulation both run with it.
   *
   * Ignoring it is not a detail, it is the difference between stable and
   * divergent. diag(H) alone spans 13000x across the robot (L_WRIST_Y 0.0003,
   * CHEST_P 4.26), so kp/m explodes on the light distal segments: the ankles
   * sit at sqrt(kp/m)*dt = 3.9-4.6, far past the ~1 stability limit, and the
   * first attempt diverged there -- qddot doubling every ~3 ticks on
   * R_ANKLE_P, 3235 -> 17564 rad/s^2 in nine ticks. Adding 1.0 compresses the
   * spread to 5.3x and puts every joint between 0.42 and 0.68.
   */
  double jointArmature_ = 0.0;

  /** @brief Inertia model, reused across ticks (HIr is configuration-independent). */
  rbd::ForwardDynamics fd_;

  /** @brief diag(H + HIr) at the last tick, indexed like jointNames. Logged. */
  Eigen::VectorXd jointInertia_;

  /** @brief Last commanded qddot, indexed like jointNames. Logged. */
  Eigen::VectorXd qddotRef_;

  /**
   * @brief Torque-control mode (config key: use_torque_task).
   *
   * The PostureTask route puts TWO regulators in series -- the task, then
   * mc_mujoco's own joint PD acting on the (q, qdot) the QP produced -- while
   * training had exactly one. That cascade is why the QP path could never
   * reproduce the bypass: the QP's single coupled output cannot feed the two
   * independent channels (q_ref, alpha_ref) that PD consumes, and every mode
   * trades one against the other (measured: velocity ratio 1.00 but position
   * feedforward 61%, or full position bias and no holding torque at all).
   *
   * TorqueJointTask removes the cascade. It computes
   *   tau = gainsRatio*kp*(qd - q) + sqrt(gainsRatio)*kd*(qd_dot - qdot) + tau_ff
   * -- the training law verbatim, same gains convention as kp_/kd_ -- and the
   * QP solves for the accelerations realising it under the constraints. The
   * resulting torques go straight to the robot, so mc_mujoco's PD is out of
   * the loop entirely.
   *
   * REQUIRES: dynamicsConstraint in the solver (TorqueTask needs the dynamic
   * model to relate accelerations to torques), and mc_mujoco started with
   * --torque-control, otherwise sendControl() falls back to its PD on
   * (q_ref, alpha_ref) and the torques are ignored.
   */
  bool useTorqueTask_ = false;
  std::shared_ptr<mc_tasks::TorqueJointTask> torqueTask_;

  /** @brief Build the TorqueJointTask, size its gains, and add it to the solver. */
  void setupTorqueTask();

  /** @brief Set the posture task's refAccel to tau/diag(H+HIr). See qpAccelTask_. */
  void applyAccelFeedforward(mc_tasks::PostureTask & pt);

  /** @brief One-shot startup dump of the inertia model, with an L/R symmetry check. */
  void logInertiaModel();

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
