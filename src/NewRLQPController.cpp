#include "NewRLQPController.h"

#include <algorithm>

#include <mc_rtc/io_utils.h>

#include <RBDyn/MultiBodyConfig.h>
#include <mc_joystick_plugin/joystick_inputs.h>

NewRLQPController::NewRLQPController(mc_rbdyn::RobotModulePtr rm, double dt, const mc_rtc::Configuration & config)
: mc_control::fsm::Controller(rm, dt, config, Backend::TVM)
{
  config_ = config;
  currentPolicyIndex = size_t(config_("default_policy_index", 0));

  // Full module minimalSelfCollisions kept: the RL policy is trained to stay
  // out of the dampers' braking zones (proximity penalties in mjlab-rhps1),
  // so the QP acts as a pure safety net. One module value is corrected: the
  // thigh pair's iDist (0.06) exceeds the thighs' *standing* sch distance
  // (0.028), so its damper braked permanently and killed lateral stepping.
  // 0.025 turns it back into a protection (sDist stays 0.01). Edited in the
  // stored `cols` because setCollisionsDampers re-creates every pair from it.
  for(auto & col : selfCollisionConstraint->cols)
  {
    const bool thighPair = (col.body1 == "L_CROTCH_P_LINK" && col.body2 == "R_CROTCH_P_LINK")
                           || (col.body1 == "R_CROTCH_P_LINK" && col.body2 == "L_CROTCH_P_LINK");
    if(thighPair)
    {
      col.iDist = 0.025;
      mc_rtc::log::info("[NewRLQPController] Thigh self-collision iDist corrected to {} (sDist {})", col.iDist,
                        col.sDist);
    }
  }
  selfCollisionConstraint->setCollisionsDampers(solver(), {zeta_selfCollision_, lambda_selfCollision_});
  solver().removeConstraintSet(dynamicsConstraint);
  kinematicsConstraint = mc_rtc::unique_ptr<mc_solver::KinematicsConstraint>(
    new mc_solver::KinematicsConstraint(robots(), 0,
      {diPercent_, dsPercent_, 0.0, zeta_jointLimit_, lambda_jointLimit_}, velPercent_));
  solver().addConstraintSet(kinematicsConstraint);

  initializeRobot();

  datastore().make_call("KinematicAnchorFrame::" + robot().name(),
    [](const mc_rbdyn::Robot & robot) -> std::pair<sva::PTransformd, Eigen::Vector3d> {
      return {sva::interpolate(robot.surfacePose("RightFootCenter"),
                              robot.surfacePose("LeftFootCenter"), 0.5),
              Eigen::Vector3d::Zero()};
    });

  initializeRLPolicy();

  addGui();
  addLog();
  mc_rtc::log::success("NewRLQPController init done");
}

bool NewRLQPController::run()
{
  updateVelocityCommand();
  if(printLimits_) computeLimits();

  // Disarmed: never write the posture target. The PostureTask then keeps the
  // posture mc_rtc captured at reset, so the robot holds its stance under the
  // QP until the operator arms the policy from the GUI.
  if(useQP_ && policyArmed_)
  {
    auto pt = getPostureTask(robot().name());
    std::map<std::string, std::vector<double>> q_target;
    for(int i = 0; i < nbActuatedJoints; ++i)
      q_target[jointNames[i]] = {q_rl(i)};
    pt->target(q_target);
  }

  bool ret = mc_control::fsm::Controller::run(mc_solver::FeedbackType::OpenLoop);
  // Same gate on the bypass path: disarmed means no policy output reaches the
  // robot, whichever mode is configured.
  if(!useQP_ && policyArmed_) byPassQPControl();

  return ret;
}

void NewRLQPController::reset(const mc_control::ControllerResetData & reset_data)
{
  mc_control::fsm::Controller::reset(reset_data);

  double policyDt = config_("policies")[currentPolicyIndex]("policy_step_size", 0.005);
  double K = 0.2 / (policyDt * timeStep);
  auto pt = getPostureTask(robot().name());
  if(pt) pt->stiffness(K);

  q_rl       = q_zero;
  q_rl_prev_ = q_zero;
  // Drop the projection's history rather than guessing it: the next call seeds
  // qTargetPrev_ with the first real target, so the first finite difference is
  // zero. Seeding with q_zero here would be a guess, and a wrong one whenever
  // the go-to-init ramp has not finished.
  projInitialized_ = false;
  qTargetPrev_ = q_zero;
  qdTarget_.setZero();
  currentVelCmd_.setZero();
  histInitialized_ = false;
  // Never resume armed across a reset: a controller switch or a re-reset must
  // put the robot back in the held state and require a deliberate re-arm.
  policyArmed_ = false;

  checkFloatingBaseObserver();

  mc_rtc::log::success("NewRLQPController reset completed");
}

void NewRLQPController::checkFloatingBaseObserver()
{
  // Why this exists: running without a floating-base observer is silent and
  // fatal. realRobot's base pose and velocity are then never updated, so
  // projected_gravity is a constant (the policy believes itself perfectly
  // upright forever) and base_lin_vel / base_ang_vel are dead -- no balance
  // feedback at all. In simulation that is a fall; on hardware it is a fall
  // with nothing to catch it. Nothing in mc_rtc warns about it.
  //
  // Read pipelineObservers_, not mc_rtc's printed pipeline description. That
  // description is what produced the 2026-07-29 "MCWaiko silently dropped"
  // diagnosis, and it was wrong: ObserverPipeline::reset() never removes an
  // observer, it only concatenates observer.desc(), and MCWaiko never assigns
  // desc_ -- so a present, updating MCWaiko renders as "Encoder (...) -> "
  // with nothing after the arrow. The observer was there the whole time.
  std::vector<std::string> found;
  for(const auto & pipeline : observerPipelines())
    for(const auto & obs : pipeline.observers()) found.push_back(obs.observer().name());

  mc_rtc::log::info("[NewRLQPController] observer pipeline: [{}]", mc_rtc::io::to_string(found));

  const bool hasFloatingBase =
      std::any_of(found.begin(), found.end(),
                  [](const std::string & n) { return n == "MCWaiko" || n == "BodySensor" || n == "KinematicInertial"; });
  if(!hasFloatingBase)
    mc_rtc::log::error_and_throw(
        "[NewRLQPController] no floating-base observer in the pipeline (found: [{}]). The policy would run on a "
        "constant projected_gravity and zero base velocity, and fall. Check ObserverPipelines in the controller yaml.",
        mc_rtc::io::to_string(found));
}

void NewRLQPController::initializeRobot()
{
  useQP_    = config_("policies")[currentPolicyIndex]("use_QP", true);
  robotName_ = robot().name();
  jointNames = robot().refJointOrder();
  // refJointOrder may name joints this robot does not actually carry, and
  // multi-DoF ones we cannot drive with a scalar PD. The RHPS1 module is the
  // case in point: its non-mujoco branch lists L_HAND/R_HAND (plus 16 finger
  // joints per hand for the leap end effector), so MainRobot: RHPS1 threw
  // std::out_of_range on kp_map.at(jointNames[i]) below, while MainRobot:
  // RHPS1_MuJoCo, whose branch lists 30, was fine. Filtering here also fixes
  // the q0-size check in configRL(): nbActuatedJoints becomes 30, matching
  // the 30-key q0 already in the yaml.
  jointNames.erase(std::remove_if(jointNames.begin(), jointNames.end(),
                                  [this](const std::string & j) {
                                    return !robot().hasJoint(j)
                                           || robot().mb().joint(robot().jointIndexByName(j)).dof() != 1;
                                  }),
                   jointNames.end());
  nbActuatedJoints = jointNames.size();
  mc_rtc::log::info("[NewRLQPController] {} actuated joints retained out of {} in refJointOrder",
                    nbActuatedJoints, robot().refJointOrder().size());

  q_rl              = Eigen::VectorXd::Zero(nbActuatedJoints);
  q_rl_prev_        = Eigen::VectorXd::Zero(nbActuatedJoints);
  q_zero            = Eigen::VectorXd::Zero(nbActuatedJoints);
  actionScale       = Eigen::VectorXd::Zero(nbActuatedJoints);
  currentActionScaled = Eigen::VectorXd::Zero(nbActuatedJoints);
  kp_    = Eigen::VectorXd::Zero(nbActuatedJoints);
  kd_    = Eigen::VectorXd::Zero(nbActuatedJoints);
  kpBase_ = Eigen::VectorXd::Zero(nbActuatedJoints);
  kdBase_ = Eigen::VectorXd::Zero(nbActuatedJoints);

  pdGainsRatio_ = config_("policies")[currentPolicyIndex]("pd_gains_ratio", 1.0);
  std::map<std::string, double> actionScale_map = config_("policies")[currentPolicyIndex]("action_scale");
  std::map<std::string, double> kp_map = config_("policies")[currentPolicyIndex]("kp");
  std::map<std::string, double> kd_map = config_("policies")[currentPolicyIndex]("kd");
  q0_map_ = config_("policies")[currentPolicyIndex]("q0");

  auto updateIfExists = [&](auto & target, const auto & map, const std::string & joint_name)
  {
    if(auto it = map.find(joint_name); it != map.end()) target = it->second;
  };

  for(int i = 0; i < nbActuatedJoints; ++i)
  {
    kpBase_[i]  = kp_map.at(jointNames[i]);
    kdBase_[i]  = kd_map.at(jointNames[i]);
    q_zero[i]   = q0_map_.at(jointNames[i]);
    updateIfExists(actionScale[i], actionScale_map, jointNames[i]);
  }

  kp_ = pdGainsRatio_ * kpBase_;
  kd_ = sqrt(pdGainsRatio_) * kdBase_;

  // Torque-feasibility projection. Off unless the policy block asks for it, so
  // indices that trained without it are untouched.
  torqueFeasibilityRatio_ = config_("policies")[currentPolicyIndex]("torque_feasibility_ratio", -1.0);
  velTargetFilterAlpha_   = config_("policies")[currentPolicyIndex]("vel_target_filter_alpha", 0.0);
  effortLimit_    = Eigen::VectorXd::Zero(nbActuatedJoints);
  velTargetLimitPerJoint_ = Eigen::VectorXd::Constant(nbActuatedJoints, 1e9);
  qdTarget_       = Eigen::VectorXd::Zero(nbActuatedJoints);
  qTargetPrev_    = Eigen::VectorXd::Zero(nbActuatedJoints);
  // The projection is only meaningful when the plant downstream really is the PD
  // it was derived from. Its correctness proof in training is an identity:
  // tau(q*) = kp*(q* - q) + kd*(qd* - qdot) is affine and increasing in q*, so
  // clamping tau to +/-e and projecting q* onto tau's preimage of [-e, e] are the
  // same operation -- same torque, same dynamics, no new constraint.
  //
  // Under use_QP that PD does not exist. q_rl becomes a PostureTask target and the
  // QP integrates it in OpenLoop, so nothing anywhere applies kp = 20000. The
  // projection then stops being an identity and becomes a hard constraint pinning
  // the target within budget/kp of the *measured* position -- 0.0018 rad on
  // CROTCH_Y, 0.007 rad on CROTCH_P. The QP's tracking error is an order of
  // magnitude larger than that, so essentially every command collapses onto the
  // measurement, the posture task sees no error, and the robot goes limp.
  //
  // Nothing is lost by skipping it here: the deterministic rollout of this
  // checkpoint left only 3.0% of joint-steps outside the window (mean demand
  // 0.269), i.e. the projection had already done its work at training time by
  // shaping what the policy learned to emit. The QP enforces its own torque and
  // velocity limits for its own plant.
  //
  // The QP check is made per call in projectTorqueFeasible, NOT here. Zeroing the
  // ratio at init looked equivalent and is not: "Toggle QP Control" flips useQP_
  // at runtime, so a config loaded with use_QP true would reach the bypass path
  // with the projection permanently destroyed -- i.e. exactly the raw, unclamped
  // behaviour this exists to prevent, on the one path where it is valid.
  if(torqueFeasibilityRatio_ > 0.0)
  {
    std::map<std::string, double> eff_map = config_("policies")[currentPolicyIndex]("effort_limit");
    std::map<std::string, double> vtl_map = config_("policies")[currentPolicyIndex]("vel_target_limit_per_joint");
    for(int i = 0; i < nbActuatedJoints; ++i)
    {
      effortLimit_[i] = eff_map.at(jointNames[i]);
      updateIfExists(velTargetLimitPerJoint_[i], vtl_map, jointNames[i]);
    }
    mc_rtc::log::info("[NewRLQPController] Torque-feasibility projection ARMED (ratio {}, "
                      "vel_target_filter_alpha {}) -- applied only while QP control is "
                      "bypassed; currently useQP={}",
                      torqueFeasibilityRatio_, velTargetFilterAlpha_, useQP_);
  }

  maxVelX_          = config_("policies")[currentPolicyIndex]("max_vel_x",         0.6);
  maxVelY_          = config_("policies")[currentPolicyIndex]("max_vel_y",         0.4);
  maxYawCmd_        = config_("policies")[currentPolicyIndex]("max_yaw",           0.7);
  joystickDeadZone_ = config_("policies")[currentPolicyIndex]("joystick_deadzone", 0.05);
  velRampRate_      = config_("policies")[currentPolicyIndex]("vel_ramp_rate",     0.5);
  // Matches the training actuator's velocity_target_limit (rad/s): clamp on
  // the finite-difference velocity feedforward in byPassQPControl().
  velTargetLimit_   = config_("policies")[currentPolicyIndex]("vel_target_limit",  8.0);

  double policyDt = config_("policies")[currentPolicyIndex]("policy_step_size", 0.005);
  double K = 0.2 / (policyDt * timeStep);
  auto pt = getPostureTask(robot().name());
  pt->stiffness(K);
  mc_rtc::log::info("[NewRLQPController] useQP={} PostureTask stiffness={:.0f}", useQP_, K);
}

void NewRLQPController::initializeRLPolicy()
{
  policyPaths_ = config_("policy_path", std::vector<std::string>{"walking_better_h1.onnx"});
  configRL();

  currentObservation = Eigen::VectorXd::Zero(rlPolicy->getObservationSize());
  currentAction      = Eigen::VectorXd::Zero(rlPolicy->getActionSize());
  q_rl = q_zero;

  initializeRLObservation();
}

void NewRLQPController::initializeRLObservation()
{
  auto & rr = realRobot(robots()[0].name());
  const std::string & baseName = rr.mb().body(0).name();
  const Eigen::Matrix3d R_w2b = rr.bodyPosW(baseName).rotation();

  const Eigen::Vector3d lv = R_w2b * rr.bodyVelW(baseName).linear();
  const Eigen::Vector3d av = R_w2b * rr.bodyVelW(baseName).angular();
  const Eigen::Vector3d pg = R_w2b * Eigen::Vector3d(0, 0, -1);

  const int actionDim = rlPolicy->getActionSize();
  Eigen::VectorXd jp = Eigen::VectorXd::Zero(actionDim);
  Eigen::VectorXd jv = Eigen::VectorXd::Zero(actionDim);
  for(int j = 0; j < actionDim; ++j)
  {
    int mcIdx = rr.jointIndexByName(refJointOrderRLAction[j]);
    jp(j) = rr.mbc().q[mcIdx][0] - q_zero[actionToDofMap[j]];
    jv(j) = rr.mbc().alpha[mcIdx][0];
  }
  const Eigen::VectorXd ja = Eigen::VectorXd::Zero(actionDim);

  for(int i = 0; i < HISTORY_SIZE; ++i)
  {
    linVel_[i]   = lv;
    angVel_[i]   = av;
    projGrav_[i] = pg;
    jointPos_[i] = jp;
    jointVel_[i] = jv;
    jointAct_[i] = ja;
    velCmd_[i]   = currentVelCmd_;
    gaitPhase_[i].setZero();
  }
  gaitPhase_value_ = 0.0;

  // Zero, not the current demand: at init the target is the measured posture, so
  // the true ratio is ~0 anyway, and mjlab's buffer starts zeroed as well.
  for(int i = 0; i < RAW_TORQUE_HISTORY; ++i)
  {
    rawTorque_[i] = Eigen::VectorXd::Zero(actionDim);
  }
  rawTorqueRatio_ = Eigen::VectorXd::Zero(nbActuatedJoints);

  histInitialized_ = true;
}

void NewRLQPController::updateRawTorqueRatio(const Eigen::VectorXd & qTarget)
{
  auto & rr = realRobot(robots()[0].name());

  // First call: seed the history with this very target so the finite difference
  // is exactly zero, as the actuator's `uninitialized` branch does. Seeding with
  // q_zero instead injects a spurious first-step velocity of (qTarget - q_zero)/dt.
  if(!projInitialized_)
  {
    qTargetPrev_ = qTarget;
    qdTarget_.setZero();
    projInitialized_ = true;
  }

  for(int i = 0; i < nbActuatedJoints; ++i)
  {
    // qd*: EMA-filtered finite difference of the *target*, not of the measured
    // position. Order matters -- estimate, clamp, then filter, as the actuator does.
    double raw = (qTarget(i) - qTargetPrev_(i)) / policyStepSize;
    raw = std::clamp(raw, -velTargetLimitPerJoint_(i), velTargetLimitPerJoint_(i));
    qdTarget_(i) = velTargetFilterAlpha_ > 0.0
                       ? velTargetFilterAlpha_ * qdTarget_(i) + (1.0 - velTargetFilterAlpha_) * raw
                       : raw;

    const int mcIdx = rr.jointIndexByName(jointNames[i]);
    const double q = rr.mbc().q[mcIdx][0];
    const double qdot = rr.mbc().alpha[mcIdx][0];

    // kpBase_/kdBase_, not kp_/kd_: this channel has to reproduce what the
    // training actuator measured, and those gains are fixed in mjlab. kp_ carries
    // the runtime pd_gains_ratio, so using it would feed the network a rescaled
    // version of a quantity it learned at ratio 1.
    const double tau = kpBase_(i) * (qTarget(i) - q) + kdBase_(i) * (qdTarget_(i) - qdot);
    // effortLimit_ is only filled when the policy block declares one. Guard on it
    // rather than clamping the divisor: a policy with no effort_limit would
    // otherwise get tau/1e-6, i.e. ~1e9, silently poured into a channel that
    // happens to be unread today and would be read the moment someone adds a V5
    // sibling. Zero says "not measured" instead of lying with a huge number.
    rawTorqueRatio_(i) = effortLimit_(i) > 0.0 ? std::abs(tau) / effortLimit_(i) : 0.0;
  }

  // The finite difference is taken between two *raw* targets, never against the
  // projected one. FiniteDifferencePdActuator stores `_last_position_target`
  // before `_apply_torque_feasibility` runs, so qd* measures how fast the policy
  // is moving its command, not how far the projection had to pull it back.
  //
  // Storing the projected target here is a runaway, and it is what blew the robot
  // up: the projected target sits within budget/kp of the measurement (7e-3 rad on
  // CROTCH_P), so the next raw target is a whole action away from it, the
  // difference saturates the per-joint clamp, and v_term = kd*(qd* - qdot) reaches
  // 400*8 = 3200 Nm. Divided by kp that shoves the window ~0.16 rad clear of q --
  // more than 20 window widths -- so the projection commands a large step in the
  // direction of its own saturated velocity estimate, every tick, and diverges.
  qTargetPrev_ = qTarget;

  // Shift the ratio history (index 0 = most recent) and publish the new value,
  // reordered into the policy's joint order. rawTorqueRatio_ is indexed by
  // jointNames (mc_rtc order); the network expects ref_joint_order, and the two
  // are not the same list -- actionToDofMap[j] = i is the bridge, as it is for
  // jointPos_/jointVel_.
  for(int i = RAW_TORQUE_HISTORY - 1; i > 0; --i) { rawTorque_[i] = rawTorque_[i - 1]; }
  const int actionDim = static_cast<int>(refJointOrderRLAction.size());
  rawTorque_[0] = Eigen::VectorXd::Zero(actionDim);
  for(int j = 0; j < actionDim; ++j) { rawTorque_[0](j) = rawTorqueRatio_(actionToDofMap[j]); }
}

Eigen::VectorXd NewRLQPController::projectTorqueFeasible(const Eigen::VectorXd & qTarget)
{
  if(torqueFeasibilityRatio_ <= 0.0) { return qTarget; }

  // Under the QP, q_rl is a PostureTask target integrated in OpenLoop and nothing
  // applies kp = 20000, so the projection stops being the identity it is proved to
  // be and becomes a hard constraint pinning the target within budget/kp of the
  // measurement -- 0.0018 rad on CROTCH_Y. On the bypass path q_rl goes straight
  // to mc_mujoco's servo, whose gains ARE these kp/kd (verified against
  // share/mc_mujoco/RHPS1/pdgains/RHPS1main/PDgains_sim.dat: 20000/400 legs,
  // 10000/300 ankles, 44000/440 chest), so there the window is correctly sized.
  //
  // No seed to drop on the way out any more: updateRawTorqueRatio() runs every
  // policy step in both modes, so qTargetPrev_ and qd* are never stale when
  // bypass is re-entered. That staleness is what blew the robot up on 2026-08-02.
  if(useQP_) { return qTarget; }

  auto & rr = realRobot(robots()[0].name());
  Eigen::VectorXd out = qTarget;

  for(int i = 0; i < nbActuatedJoints; ++i)
  {
    // qd* was computed on the raw target by updateRawTorqueRatio() this step.
    const int mcIdx = rr.jointIndexByName(jointNames[i]);
    const double q = rr.mbc().q[mcIdx][0];
    const double qdot = rr.mbc().alpha[mcIdx][0];

    const double budget = torqueFeasibilityRatio_ * effortLimit_(i);
    const double vTerm = kd_(i) * (qdTarget_(i) - qdot);
    const double kp = std::max(kp_(i), 1e-6);
    const double lo = q + (-budget - vTerm) / kp;
    const double hi = q + (budget - vTerm) / kp;
    out(i) = std::clamp(qTarget(i), lo, hi);
  }

  // qTargetPrev_ is NOT written here: updateRawTorqueRatio() owns it and has
  // already stored the raw target. Storing `out` instead would be the runaway
  // documented there.
  return out;
}

void NewRLQPController::gaitPhaseStep()
{
  // Commanded planar speed drives both the cadence and the amplitude. Yaw is
  // deliberately excluded: mjlab's clock keys off the linear command only.
  const double speed = currentVelCmd_.head<2>().norm();

  // Amplitude ramps 0 -> 1 over [0, threshold]; the clock itself only advances
  // above the threshold. Below it the phase is held, but the amplitude scaling
  // takes the whole block to zero, so a held phase is never visible.
  const double amplitude = std::clamp(speed / gaitCommandThreshold_, 0.0, 1.0);

  if(speed >= gaitCommandThreshold_)
  {
    // Period interpolates linearly between the slow point (at the threshold)
    // and the fast point (at the reference speed), clamped outside that span.
    const double t = std::clamp(
        (speed - gaitCommandThreshold_) / std::max(gaitCommandRef_ - gaitCommandThreshold_, 1e-9),
        0.0, 1.0);
    const double period = gaitPeriodSlow_ + t * (gaitPeriodFast_ - gaitPeriodSlow_);
    gaitPhase_value_ += policyStepSize / std::max(period, 1e-6);
    gaitPhase_value_ -= std::floor(gaitPhase_value_); // wrap into [0, 1)
  }

  const double phiL = 2.0 * M_PI * gaitPhase_value_;
  // Right foot is the left shifted by half a period; sin/cos are pi-antiperiodic,
  // so the right block is exactly the negation of the left.
  gaitPhase_[0] << amplitude * std::sin(phiL), amplitude * std::cos(phiL),
      -amplitude * std::sin(phiL), -amplitude * std::cos(phiL);
}

void NewRLQPController::updateVelocityCommand()
{
  if(!useJoystick_) return;
  if(!datastore().has("Joystick::connected") || !datastore().get<bool>("Joystick::connected")) return;
  if(!datastore().has("Joystick::Stick")) return;

  Eigen::Vector3d targetCmd = Eigen::Vector3d::Zero();

  auto leftStick = datastore().call<Eigen::Vector2d>("Joystick::Stick", joystickAnalogicInputs::L_STICK);
  if(std::abs(leftStick(0) - 0.5) > joystickDeadZone_)
    targetCmd(0) = (leftStick(0) - 0.5) * 2.0 * maxVelX_;
  if(std::abs(leftStick(1) - 0.5) > joystickDeadZone_)
    targetCmd(1) = std::clamp((leftStick(1) - 0.5) * 2.0 * maxVelY_, -maxVelY_, maxVelY_);

  auto rightStick = datastore().call<Eigen::Vector2d>("Joystick::Stick", joystickAnalogicInputs::R_STICK);
  if(std::abs(rightStick(1) - 0.5) > joystickDeadZone_)
    targetCmd(2) = (rightStick(1) - 0.5) * 2.0 * maxYawCmd_;

  const double maxDelta = velRampRate_ * timeStep;
  for(int i = 0; i < 3; ++i)
  {
    const double diff = targetCmd(i) - currentVelCmd_(i);
    currentVelCmd_(i) += std::abs(diff) > maxDelta ? std::copysign(maxDelta, diff) : diff;
  }
}

bool NewRLQPController::byPassQPControl()
{
  if(useQP_) return false;

  for(int i = 0; i < nbActuatedJoints; ++i)
  {
    const int idx = robot().jointIndexByName(jointNames[i]);
    robot().mbc().q[idx][0] = q_rl(i);
    // Velocity feedforward from finite differences of the position target.
    // Clamped like the training actuator (FiniteDifferencePdActuator
    // velocity_target_limit): a policy-step target jump of 0.1 rad otherwise
    // becomes a 20+ rad/s velocity target and the kd term injects torque
    // kicks the policy never experienced in training (observed: hip-yaw
    // blow-up at the first inference of the 2026-07-16 checkpoint).
    const double alphaRaw = (q_rl(i) - q_rl_prev_(i)) / timeStep;
    robot().mbc().alpha[idx][0] =
        std::max(-velTargetLimit_, std::min(velTargetLimit_, alphaRaw));
  }
  q_rl_prev_ = q_rl;
  return true;
}

void NewRLQPController::updateSelfCollisionDistances()
{
  auto & rr = realRobot(robots()[0].name());
  for(size_t i = 0; i < selfColPairs_.size(); ++i)
  {
    const auto & [b1, b2, pair] = selfColPairs_[i];
    sch::mc_rbdyn::transform(*rr.convex(b1).second, rr.collisionTransform(b1) * rr.bodyPosW(b1));
    sch::mc_rbdyn::transform(*rr.convex(b2).second, rr.collisionTransform(b2) * rr.bodyPosW(b2));
    Eigen::Vector3d p1, p2;
    // sch returns the signed *squared* distance.
    const double d2 = sch::mc_rbdyn::distance(*pair, p1, p2);
    selfColDists_(static_cast<int>(i)) = d2 >= 0 ? std::sqrt(d2) : -std::sqrt(-d2);
  }
}

void NewRLQPController::addLog()
{
  // Ground-truth sch distances of the module's minimalSelfCollisions pairs,
  // measured on the realRobot with the same convex objects as the QP.
  auto & rr = realRobot(robots()[0].name());
  for(const auto & col : robot().module().minimalSelfCollisions())
  {
    if(!rr.hasConvex(col.body1) || !rr.hasConvex(col.body2))
    {
      mc_rtc::log::warning("[NewRLQPController] selfcol_dist: no convex for {}/{}", col.body1, col.body2);
      continue;
    }
    selfColPairs_.emplace_back(col.body1, col.body2,
                               std::make_shared<sch::CD_Pair>(rr.convex(col.body1).second.get(),
                                                              rr.convex(col.body2).second.get()));
    mc_rtc::log::info("[NewRLQPController] selfcol_dist[{}] = {}/{}", selfColPairs_.size() - 1, col.body1,
                      col.body2);
  }
  selfColDists_ = Eigen::VectorXd::Zero(static_cast<int>(selfColPairs_.size()));
  logger().addLogEntry("NewRLQPController_selfcol_dist", [this]() -> const Eigen::VectorXd & {
    const_cast<NewRLQPController *>(this)->updateSelfCollisionDistances();
    return selfColDists_;
  });
  logger().addLogEntry("NewRLQPController_kp_base",          [this]() { return kpBase_; });
  logger().addLogEntry("NewRLQPController_kd_base",          [this]() { return kdBase_; });
  logger().addLogEntry("NewRLQPController_kp_current",       [this]() { return kp_; });
  logger().addLogEntry("NewRLQPController_kd_current",       [this]() { return kd_; });
  logger().addLogEntry("NewRLQPController_pd_gains_ratio",   [this]() { return pdGainsRatio_; });
  logger().addLogEntry("NewRLQPController_RL_q",             [this]() { return q_rl; });
  logger().addLogEntry("NewRLQPController_RL_qZero",         [this]() { return q_zero; });
  logger().addLogEntry("NewRLQPController_RL_currentObservation", [this]() { return currentObservation; });
  logger().addLogEntry("NewRLQPController_RL_currentAction", [this]() { return currentAction; });
  logger().addLogEntry("NewRLQPController_RL_actionScale",   [this]() { return actionScale; });
  logger().addLogEntry("NewRLQPController_useQP",            [this]() { return useQP_; });
  logger().addLogEntry("NewRLQPController_velCmd",           [this]() { return currentVelCmd_; });
  // The whole point of the raw-torque campaign: what the policy would demand of
  // each joint if nothing clipped it, in units of effort_limit. 1.0 = at the
  // training limit. Reading the max here is the deployment-side counterpart of
  // Metrics/raw_torque_peak_max, and the knee is the one to watch -- its
  // effort_limit is 70 N.m against a real continuous rating of 21.4.
  logger().addLogEntry("NewRLQPController_rawTorqueRatio",   [this]() { return rawTorqueRatio_; });
  logger().addLogEntry("NewRLQPController_rawTorqueRatioMax", [this]() {
    return rawTorqueRatio_.size() ? rawTorqueRatio_.maxCoeff() : 0.0;
  });
  // Per-joint gap between the RL target and the measured position (refJointOrder).
  // With QP on, a joint whose gap grows/saturates is being clamped by a QP
  // constraint (joint-limit/velocity damper or collision damper).
  logger().addLogEntry("NewRLQPController_RL_q_tracking_error", [this]() {
    const auto & enc = robot().encoderValues();
    Eigen::VectorXd err = q_rl;
    for(int i = 0; i < nbActuatedJoints && i < static_cast<int>(enc.size()); ++i)
      err(i) -= enc[i];
    return err;
  });
}

void NewRLQPController::addGui()
{
  gui()->addElement({"NewRLQPController", "Policy"},
    mc_rtc::gui::Label("Current policy",   [this]() -> const std::string & { return policyPaths_[currentPolicyIndex]; }),
    mc_rtc::gui::Label("Policy Loaded",    [this]() { return rlPolicy->isLoaded() ? "Yes" : "No"; }),
    mc_rtc::gui::Label("Observation Size", [this]() { return std::to_string(rlPolicy->getObservationSize()); }),
    mc_rtc::gui::Label("Action Size",      [this]() { return std::to_string(rlPolicy->getActionSize()); })
  );

  // Arming lives at the top level, not under a sub-category, so it is the
  // first thing an operator sees. "Hold" freezes the posture target at the
  // policy's last output rather than snapping anywhere, so disarming mid-
  // motion is a stop, not a jump.
  gui()->addElement({"NewRLQPController"},
    mc_rtc::gui::Label("Policy state", [this]() { return policyArmed_ ? "ARMED" : "held (disarmed)"; }),
    mc_rtc::gui::Button("ARM policy", [this]() {
      if(policyArmed_) return;
      policyArmed_ = true;
      mc_rtc::log::warning("[NewRLQPController] policy ARMED from the GUI");
    }),
    mc_rtc::gui::Button("HOLD (disarm)", [this]() {
      if(!policyArmed_) return;
      policyArmed_ = false;
      mc_rtc::log::warning("[NewRLQPController] policy DISARMED -- holding last posture");
    })
  );

  gui()->addElement({"NewRLQPController", "PostureTask"},
    mc_rtc::gui::NumberInput("Stiffness K",
      [this]() {
        auto pt = getPostureTask(robot().name());
        return pt ? pt->stiffness() : 0.0;
      },
      [this](double v) {
        auto pt = getPostureTask(robot().name());
        if(pt) pt->stiffness(v);
      })
  );

  gui()->addElement({"ControlMode"},
    mc_rtc::gui::Button("Toggle QP Control",       [this]() { useQP_ = !useQP_; }),
    mc_rtc::gui::Label("QP Control",               [this]() { return useQP_ ? "Enforced" : "Bypassed"; }),
    mc_rtc::gui::Button("Toggle print limits",     [this]() { printLimits_ = !printLimits_; }),
    mc_rtc::gui::Label("Print joint limits",       [this]() { return printLimits_ ? "Enabled" : "Disabled"; })
  );

  gui()->addElement({"NewRLQPController", "Velocity Command"},
    mc_rtc::gui::Checkbox("Use Joystick", [this]() { return useJoystick_; }, [this]() { useJoystick_ = !useJoystick_; }),
    mc_rtc::gui::NumberSlider("vx  (m/s)",   [this]() { return currentVelCmd_.x(); }, [this](double v) { currentVelCmd_.x() = v; }, -1.0, 1.0),
    mc_rtc::gui::NumberSlider("vy  (m/s)",   [this]() { return currentVelCmd_.y(); }, [this](double v) { currentVelCmd_.y() = v; }, -0.5, 0.5),
    mc_rtc::gui::NumberSlider("yaw (rad/s)", [this]() { return currentVelCmd_.z(); }, [this](double v) { currentVelCmd_.z() = v; }, -1.0, 1.0),
    mc_rtc::gui::Button("Stop", [this]() { currentVelCmd_.setZero(); }),
    mc_rtc::gui::NumberInput("Max Vx (m/s)",    [this]() { return maxVelX_; },     [this](double v) { maxVelX_     = v; }),
    mc_rtc::gui::NumberInput("Max Vy (m/s)",    [this]() { return maxVelY_; },     [this](double v) { maxVelY_     = v; }),
    mc_rtc::gui::NumberInput("Max Yaw (rad/s)", [this]() { return maxYawCmd_; },   [this](double v) { maxYawCmd_   = v; }),
    mc_rtc::gui::NumberInput("Ramp (m/s²)",     [this]() { return velRampRate_; }, [this](double v) { velRampRate_ = v; })
  );
}

void NewRLQPController::configRL()
{
  mc_rtc::log::info("[NewRLQPController] Loading RL policy [{}]: {}", currentPolicyIndex, policyPaths_[currentPolicyIndex]);
  try
  {
    rlPolicy = std::make_unique<RLPolicyInterface>(policyPaths_[currentPolicyIndex]);
    mc_rtc::log::success("[NewRLQPController] RL policy loaded (obs={} act={})",
                         rlPolicy->getObservationSize(), rlPolicy->getActionSize());
  }
  catch(const std::exception & e)
  {
    mc_rtc::log::error_and_throw("[NewRLQPController] Failed to load RL policy: {}", e.what());
  }

  policyStepSize = config_("policies")[currentPolicyIndex]("policy_step_size", 0.01);
  const double physicsStepSize = config_("policies")[currentPolicyIndex]("physics_step_size", 0.001);
  if(physicsStepSize - timeStep > 1e-6)
    mc_rtc::log::warning("[NewRLQPController] physics_step_size ({:.3f}s) > controller timeStep ({:.3f}s)",
                         physicsStepSize, timeStep);

  refJointOrderRLAction = config_("policies")[currentPolicyIndex]("ref_joint_order", std::vector<std::string>{});
  if(refJointOrderRLAction.size() != size_t(rlPolicy->getActionSize()))
    mc_rtc::log::error_and_throw("[NewRLQPController] ref_joint_order size ({}) != policy action size ({})",
                                 refJointOrderRLAction.size(), rlPolicy->getActionSize());

  actionToDofMap.assign(refJointOrderRLAction.size(), -1);
  for(size_t j = 0; j < refJointOrderRLAction.size(); ++j)
    for(int i = 0; i < nbActuatedJoints; ++i)
      if(jointNames[i] == refJointOrderRLAction[j]) { actionToDofMap[j] = i; break; }

  auto q0_map_cfg = config_("policies")[currentPolicyIndex]("q0");
  std::vector<std::string> keys = q0_map_cfg.keys();
  if(keys.size() != static_cast<size_t>(nbActuatedJoints))
    mc_rtc::log::error_and_throw("[NewRLQPController] q0 size ({}) != robot dof ({})",
                                 keys.size(), nbActuatedJoints);

  mcRtcToRLFrameworkJointMap.assign(nbActuatedJoints, -1);
  for(int j = 0; j < static_cast<int>(keys.size()); ++j)
    for(int i = 0; i < nbActuatedJoints; ++i)
      if(jointNames[i] == keys[j]) { mcRtcToRLFrameworkJointMap[i] = j; break; }

  for(int i = 0; i < nbActuatedJoints; ++i)
    if(mcRtcToRLFrameworkJointMap[i] == -1)
      mc_rtc::log::error_and_throw("[NewRLQPController] Joint '{}' not mapped!", jointNames[i]);
}

void NewRLQPController::computeLimits()
{
  constexpr double eps = 1e-5;
  auto & rr = realRobot(robots()[0].name());

  // jointNames, not refJointOrder(): the latter names joints the robot does
  // not carry at all (L_HAND on MainRobot: RHPS1), and jointIndexByName throws
  // std::out_of_range on those before any bounds check can run. These are also
  // the only joints this controller drives, so they are the only ones whose
  // limits are worth reporting.
  for(const auto & joint : jointNames)
  {
    int i = robot().jointIndexByName(joint);
    // Some DOFs (e.g. connector joints whose child link is in the robot
    // module's filtered_links) keep a real index in the model but carry no
    // parsed position/velocity/torque bounds -- skip rather than dereference
    // an empty bound vector.
    if(!rr.hasJoint(joint) || rr.ql()[i].empty() || rr.qu()[i].empty() || rr.vl()[i].empty()
       || rr.vu()[i].empty() || rr.tl()[i].empty() || rr.tu()[i].empty())
      continue;
    const double ds = dsPercent_ * (rr.qu()[i][0] - rr.ql()[i][0]);

    if(rr.q()[i][0]  > rr.qu()[i][0] - ds + eps)
      mc_rtc::log::warning("[limits] {} pos upper: {:.3f} > {:.3f}", joint, rr.q()[i][0],  rr.qu()[i][0] - ds);
    if(rr.q()[i][0]  < rr.ql()[i][0] + ds - eps)
      mc_rtc::log::warning("[limits] {} pos lower: {:.3f} < {:.3f}", joint, rr.q()[i][0],  rr.ql()[i][0] + ds);
    if(rr.alpha()[i][0] > velPercent_ * rr.vu()[i][0] + eps)
      mc_rtc::log::warning("[limits] {} vel upper: {:.3f} > {:.3f}", joint, rr.alpha()[i][0], velPercent_ * rr.vu()[i][0]);
    if(rr.alpha()[i][0] < velPercent_ * rr.vl()[i][0] - eps)
      mc_rtc::log::warning("[limits] {} vel lower: {:.3f} < {:.3f}", joint, rr.alpha()[i][0], velPercent_ * rr.vl()[i][0]);
    if(rr.jointTorque()[i][0] > rr.tu()[i][0] + eps)
      mc_rtc::log::warning("[limits] {} tau upper: {:.1f} > {:.1f}", joint, rr.jointTorque()[i][0], rr.tu()[i][0]);
    if(rr.jointTorque()[i][0] < rr.tl()[i][0] - eps)
      mc_rtc::log::warning("[limits] {} tau lower: {:.1f} < {:.1f}", joint, rr.jointTorque()[i][0], rr.tl()[i][0]);
  }
}
