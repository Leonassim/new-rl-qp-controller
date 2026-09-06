#include "NewRLQPController.h"

#include <algorithm>
#include <cmath>
#include <map>

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
  // 0.025 turns it back into a protection (sDist stays 0.01).
  //
  // The correction MUST go through MCController::remove/addCollisions, which
  // act on `collision_constraints_` -- the constraint built from the yaml's
  // `collisions:` block and, crucially, the only one `MCController` ever
  // passes to `solver().addConstraintSet()` (MCController.cpp:296-308).
  // The `selfCollisionConstraint` member is built at MCController.cpp:202-203
  // and then never added to any solver, so editing its `cols` (what this code
  // used to do) changed a vector nothing reads: the pairs actually in the QP
  // are copies held in the loaded constraint's own CollisionData. Verified in
  // both mc_rtc 655c7174 and 542a884, so this was silently a no-op on
  // real-robot-safe too, where setCollisionsDampers had the same target.
  // Calling removeCollisions on the unused constraint additionally segfaults
  // here: it dereferences gui_, which is only set by addToSolverImpl.
  //
  // remove-then-add because __createCollId returns -1 for a pair already
  // present and __addCollision then returns silently -- an add alone cannot
  // override iDist.
  const auto & rName = robot().name();
  for(const auto & col : selfCollisionConstraint->cols)
  {
    const bool thighPair = (col.body1 == "L_CROTCH_P_LINK" && col.body2 == "R_CROTCH_P_LINK")
                           || (col.body1 == "R_CROTCH_P_LINK" && col.body2 == "L_CROTCH_P_LINK");
    if(thighPair)
    {
      auto corrected = col;
      corrected.iDist = 0.025;
      removeCollisions(rName, rName, {corrected});
      addCollisions(rName, rName, {corrected});
      mc_rtc::log::info("[NewRLQPController] Thigh self-collision iDist corrected to {} (sDist {})", corrected.iDist,
                        corrected.sDist);
    }
  }
  // use_torque_task needs the dynamics: TorqueTask minimises the error between
  // the requested torque and the one the dynamic model produces, so without
  // dynamicsConstraint there is no relation between alphaD and tau to solve
  // against. The kinematic (position-target) route does not want it.
  solver().removeConstraintSet(dynamicsConstraint);
  // Replace the base class's joint-limit constraint before adding it. Leaving
  // this line out does NOT disable anything -- MCController has already built a
  // kinematicsConstraint of its own, and the addConstraintSet below would then
  // install *that* one instead. The two are different constraints entirely:
  //
  //                     this one (no CBF)     MCController's default
  //   constructor       3-element damper      3-element damper + timeStep
  //   level              velocity              velocity
  //   gains             none (di/ds/offset)   none
  //   velocityPercent   0.95                  0.50
  //
  // On mc_rtc with the CBF commit (2957e17c+, not this robot -- see above),
  // this used the 5-element acceleration-level damper (zeta=1.2, lambda=100)
  // instead: same hard bound, better-tuned braking, measured at ~1.6x less
  // roll than this fallback. The velocity ceiling is still the operational
  // win either way: the default caps every joint at half its maximum speed,
  // which brakes exactly the fast lateral corrections this policy relies on
  // -- it already runs ANKLE_R near saturation.
  kinematicsConstraint = mc_rtc::unique_ptr<mc_solver::KinematicsConstraint>(
    new mc_solver::KinematicsConstraint(robots(), 0, dt,
      {diPercent_, dsPercent_, 0.0}, velPercent_));
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

void NewRLQPController::restoreQPVelocity()
{
  // Undo the previous tick's zeroAlphaOut(). The solver's own OpenLoop
  // integration (TVMQPSolver::runOpenLoop -> updateRobot -> rbd::integration,
  // RBDyn/NumericalIntegration.cpp:358-370) reads robot().mbc().alpha as ITS
  // OWN integration state, not a separate copy: jointIntegration() advances q
  // using the alpha found there, then alpha += alphaD*step updates it in
  // place, both before the next solve even starts. Leaving it zeroed after
  // zeroAlphaOut() would make the QP forget its own commanded velocity every
  // tick -- not just hide it from mc_mujoco -- so this MUST run before
  // mc_control::fsm::Controller::run(), never after.
  for(int i = 0; i < nbActuatedJoints; ++i)
  {
    const int jIdx = robot().jointIndexByName(jointNames[i]);
    if(!robot().mbc().alpha[jIdx].empty()) { robot().mbc().alpha[jIdx][0] = alphaOutShadow_(i); }
  }
}

void NewRLQPController::zeroAlphaOut()
{
  // Stash the QP's true just-solved velocity (restoreQPVelocity() puts it
  // back next tick, before the solve), then zero what actually reaches
  // mc_mujoco: MjRobot::updateControl() (mc_mujoco/src/mj_sim.cpp:778) reads
  // controller->robots().robot(name) DIRECTLY, not outputRobots() -- unlike
  // grippers, there is no separate output copy to zero on for this, so this
  // has to touch the same robot() the solver itself uses. alpha_ref then
  // reaches MjRobot::sendControl()'s PD (mj_sim.cpp:30-36) as zero, i.e. pure
  // position tracking with no velocity feedforward from the QP.
  for(int i = 0; i < nbActuatedJoints; ++i)
  {
    const int jIdx = robot().jointIndexByName(jointNames[i]);
    if(robot().mbc().alpha[jIdx].empty()) { continue; }
    alphaOutShadow_(i) = robot().mbc().alpha[jIdx][0];
    robot().mbc().alpha[jIdx][0] = 0.0;
  }
}

bool NewRLQPController::run()
{
  // Must run before anything that could trigger a solve this tick -- see
  // restoreQPVelocity()'s own comment for why the ordering matters.
  if(useQP_ && qpZeroVelOut_) { restoreQPVelocity(); }

  updateVelocityCommand();
  if(printLimits_) computeLimits();
  if(logImpactVel_) updateImpactVelocity();

  // Runaway guard. The feedforward/servo instability grows ~1.38 per tick and
  // saturates in under 400 ms, so no operator can catch it; disarm on the
  // measured velocity, which is the quantity that actually breaks the hardware.
  if(policyArmed_ && runawayDisarmVel_ > 0.0)
  {
    const auto & rr = realRobot(robots()[0].name());
    double vmax = 0.0;
    for(int i = 0; i < nbActuatedJoints; ++i)
    {
      const int k = rr.jointIndexByName(jointNames[i]);
      if(!rr.mbc().alpha[k].empty()) { vmax = std::max(vmax, std::abs(rr.mbc().alpha[k][0])); }
    }
    if(vmax > runawayDisarmVel_)
    {
      policyArmed_ = false;
      mc_rtc::log::error("[NewRLQPController] RUNAWAY: |alpha| {:.2f} > {:.2f} rad/s, policy DISARMED",
                         vmax, runawayDisarmVel_);
    }
  }

  // Disarmed: never write the target. Whichever task is driving then keeps
  // whatever mc_rtc captured at reset, so the robot holds its stance under the
  // QP until the operator arms the policy from the GUI.
  if(useQP_ && policyArmed_)
  {
    auto pt = getPostureTask(robot().name());
    std::map<std::string, std::vector<double>> q_target;
    for(int i = 0; i < nbActuatedJoints; ++i)
      q_target[jointNames[i]] = {q_rl(i)};
    pt->target(q_target);
    // Give the QP the velocity target alongside the position target, always
    // -- not just under posture_feedforward. qdTarget_ already holds it
    // (exactly, for velocity_action; EMA-filtered, for position_action), and
    // the plain path used to leave refVel untouched, relying only on
    // "critically damped tracks a ramp with zero steady-state velocity
    // error" (utils.cpp run_rl_state) to get qdot right -- true in steady
    // state, but at the cost of the position lag documented on
    // postureFeedforward_ above. Setting refVel here removes that lag without
    // the finite-difference noise postureFeedforward_ risks: qdTarget_ is the
    // network's own action for velocity_action, not a derivative of q_rl.
    setPostureRefVel(pt);
    if(posturePassthrough_) { setPostureRefAccel(pt); }
    else if(postureFeedforward_)
    {
      updatePostureFeedforward();
      setPostureFeedforward(pt);
    }
  }
  else if(postureRefAccelWritten_)
  {
    // Holding must mean zero. At zero gains refAccel IS the objective; with
    // gains it is a feedforward that would keep pushing a target nobody updates.
    auto pt = getPostureTask(robot().name());
    if(pt)
    {
      const Eigen::VectorXd z = Eigen::VectorXd::Zero(robot().mb().nrDof() - postureDofOffset());
      pt->refAccel(z);
      pt->refVel(z);
      postureRefAccelWritten_ = false;
      ffInit_ = false;
    }
  }

  bool ret = mc_control::fsm::Controller::run(mc_solver::FeedbackType::OpenLoop);
  // Same gate on the bypass path: disarmed means no policy output reaches the
  // robot, whichever mode is configured.
  if(!useQP_ && policyArmed_) byPassQPControl();

  // After the solve: robot().mbc().alpha now holds this tick's true
  // QP-computed velocity. Stash it and zero what mc_mujoco reads -- see
  // zeroAlphaOut()'s own comment.
  if(useQP_ && qpZeroVelOut_) { zeroAlphaOut(); }

  return ret;
}

double NewRLQPController::postureStiffness() const
{
  // posture_stiffness is read from the GLOBAL configuration, not from this
  // controller's yaml: load_config() uses the global one as the base and lets
  // the controller yaml override it, so leaving the key out of
  // NewRLQPController.yaml is what allows mc_rtc_superbuild.yaml (real robot)
  // and mc_rtc_superbuild_mujoco.yaml (simulation) to carry different values.
  //
  // Without it, the fallback is the historical 0.2/(policyDt*timeStep). That
  // formula scales with the control rate, so it yields 8000 on the robot at
  // 200 Hz and 40000 in mc_mujoco at 1 kHz -- two very different plants.
  // Typed into the GUI: wins over everything, including applyPostureMode()'s
  // own re-apply, which would otherwise wipe it on the next mode toggle.
  if(postureStiffnessOverride_ > 0.0) { return postureStiffnessOverride_; }

  const auto & pol = config_("policies")[currentPolicyIndex];
  const double policyDt = pol("policy_step_size", 0.005);

  // Per-policy override, checked first (2026-08-13). Training now reproduces
  // this very filter (FiniteDifferencePdActuator's posture_task_stiffness), so
  // the stiffness stopped being a property of the environment and became a
  // property of the policy: it is part of the plant the network learned on, and
  // running it under a different one is a sim-to-real gap we put there
  // ourselves.
  //
  // Why this matters concretely, and why the older reasoning was incomplete:
  // 1600 at 200 Hz and 40000 at 1 kHz were called "the same setting" because
  // both give sqrt(K)*dt = 0.20. That equality is about NUMERICAL STABILITY
  // MARGIN, not about the physical response. The lag is 1/sqrt(K), independent
  // of dt: 25 ms at K=1600, 5 ms at K=40000. Five times apart. A policy trained
  // against a 25 ms lag must see 25 ms in mc_mujoco too, so it needs K=1600
  // there -- which is perfectly stable at 1 kHz (sqrt(K)*dt = 0.04).
  //
  // The global value stays the fallback so the older indices, trained with no
  // filter at all, keep whatever their environment yaml says.
  if(pol.has("posture_stiffness"))
  {
    return pol("posture_stiffness");
  }
  return config_("posture_stiffness", 0.2 / (policyDt * timeStep));
}

void NewRLQPController::reset(const mc_control::ControllerResetData & reset_data)
{
  mc_control::fsm::Controller::reset(reset_data);

  applyPostureMode();

  q_rl       = q_zero;
  q_rl_prev_ = q_zero;
  qdTarget_.setZero();
  currentVelCmd_.setZero();
  ffInit_ = false;
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
  velocityAction_ = config_("policies")[currentPolicyIndex]("velocity_action", false);
  obsFormat_      = size_t(int(config_("policies")[currentPolicyIndex]("obs_format", int(currentPolicyIndex))));
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
  ffPrevQ_          = Eigen::VectorXd::Zero(nbActuatedJoints);
  ffVel_            = Eigen::VectorXd::Zero(nbActuatedJoints);
  ffPrevVel_        = Eigen::VectorXd::Zero(nbActuatedJoints);
  ffAcc_            = Eigen::VectorXd::Zero(nbActuatedJoints);
  ffInit_           = false;
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

  // N*Kt par joint. Table optionnelle et volontairement incomplete : seuls les
  // joints a moteur unique y figurent, les autres restent a 0 et ne sont pas
  // convertis. Voir joint_torque_scale dans le yaml.
  std::map<std::string, double> torqueScale_map;
  if(config_.has("joint_torque_scale")) { torqueScale_map = config_("joint_torque_scale"); }
  jointTorqueScale_ = Eigen::VectorXd::Zero(nbActuatedJoints);
  jointTorqueNm_    = Eigen::VectorXd::Zero(nbActuatedJoints);
  jointCurrentA_    = Eigen::VectorXd::Zero(nbActuatedJoints);

  // Correspondance indice filtre -> indice dans le refJointOrder complet.
  {
    const auto & rjo = robot().refJointOrder();
    refIdx_.assign(nbActuatedJoints, -1);
    for(int i = 0; i < nbActuatedJoints; ++i)
    {
      auto it = std::find(rjo.begin(), rjo.end(), jointNames[i]);
      if(it != rjo.end()) refIdx_[i] = static_cast<int>(std::distance(rjo.begin(), it));
    }
  }

  for(int i = 0; i < nbActuatedJoints; ++i)
  {
    kpBase_[i]  = kp_map.at(jointNames[i]);
    kdBase_[i]  = kd_map.at(jointNames[i]);
    q_zero[i]   = q0_map_.at(jointNames[i]);
    updateIfExists(actionScale[i], actionScale_map, jointNames[i]);
    updateIfExists(jointTorqueScale_[i], torqueScale_map, jointNames[i]);
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

  posturePassthrough_ = config_("policies")[currentPolicyIndex]("posture_passthrough", false);
  postureAccelMax_    = config_("policies")[currentPolicyIndex]("posture_accel_max", 200.0);
  postureFeedforward_ = config_("policies")[currentPolicyIndex]("posture_feedforward", false);
  runawayDisarmVel_   = config_("policies")[currentPolicyIndex]("runaway_disarm_vel", 0.0);

  // Experimental: see restoreQPVelocity()/zeroAlphaOut() and run(). Off by
  // default -- untested on hardware, simulation-only ablation for now.
  qpZeroVelOut_  = config_("policies")[currentPolicyIndex]("qp_zero_vel_out", false);
  alphaOutShadow_ = Eigen::VectorXd::Zero(nbActuatedJoints);

  const auto & pol = config_("policies")[currentPolicyIndex];
  // applyPostureFilter()/applyVelocityDamper()/projectTorqueFeasible()/
  // updateRawTorqueRatio() and the config keys/state that fed only them
  // (posture_filter_stiffness, velocity_damper_di/ds/vel_percent,
  // joint_limits, velocity_limits, qTargetPrev_, rawTorqueRatio_,
  // rawTorque_) were removed 2026-09-04: RHP7's only case is
  // velocity_action, and that was their sole caller. See git history to
  // bring position_action back for a future RHP7 policy.
  //
  // torqueFeasibilityRatio_/effortLimit_ below stay: velocity_action reads
  // them directly for its own anti-windup (maxTargetDev_, utils.cpp) and
  // vel_target_limit_per_joint clamp, independently of the projection.
  //
  // vel_target_limit_per_joint is loaded unconditionally here, not gated by
  // torqueFeasibilityRatio_ or useQP_: it feeds qdTarget_ (via the clamp in
  // utils.cpp's velocity_action branch), and the PostureTask feedforward
  // reads qdTarget_ every step regardless of which policy or QP mode is
  // active. Loading it only when some other condition holds left qdTarget_
  // at the 1e9 default on every policy that did not happen to set that
  // condition -- unclamped, and invisible until something read it.
  if(pol.has("vel_target_limit_per_joint"))
  {
    std::map<std::string, double> vtl_map = pol("vel_target_limit_per_joint");
    for(int i = 0; i < nbActuatedJoints; ++i) { updateIfExists(velTargetLimitPerJoint_[i], vtl_map, jointNames[i]); }
  }

  if(torqueFeasibilityRatio_ > 0.0)
  {
    std::map<std::string, double> eff_map = config_("policies")[currentPolicyIndex]("effort_limit");
    for(int i = 0; i < nbActuatedJoints; ++i) { effortLimit_[i] = eff_map.at(jointNames[i]); }
    mc_rtc::log::info("[NewRLQPController] Torque-feasibility projection ARMED (ratio {}, "
                      "vel_target_filter_alpha {}) -- applied only while QP control is "
                      "bypassed; currently useQP={}",
                      torqueFeasibilityRatio_, velTargetFilterAlpha_, useQP_);
  }

  // effort_limit/kp, the training actuator's anti-windup bound. kpBase_, not
  // kp_: the actuator clamps with its own nominal stiffness, which does not
  // carry the runtime pd_gains_ratio. Left at 0 (clamp skipped) for joints
  // whose policy block declares no effort_limit.
  maxTargetDev_ = Eigen::VectorXd::Zero(nbActuatedJoints);
  for(int i = 0; i < nbActuatedJoints; ++i)
  {
    if(effortLimit_(i) > 0.0 && kpBase_(i) > 1e-9) { maxTargetDev_(i) = effortLimit_(i) / kpBase_(i); }
  }
  // Explicit per-joint override, applied on top of the formula above.
  // Needed for L_HAND/R_HAND: effort_limit/kp = 3/1 = 3 rad for them, a
  // window wide enough that the anti-windup never actually engages, and
  // q_rl's free-running integral (velocity_action) was observed to drift
  // unbounded over a run (no training reward on the hands, so nothing pulls
  // the raw action back toward zero) until mc_rtc's OWN Gripper safety
  // (generic_gripper.cpp, comparing commanded vs measured position)
  // stepped in and froze the command -- see the 2026-09-06 log
  // (mc-control-NewRLQPController-2026-09-06-23-47-30.bin) where qOut_L_HAND
  // visibly latches at t=8s while RL_q_L_HAND keeps climbing. Kept separate
  // from effort_limit itself, which also feeds torque-feasibility and must
  // stay at the trained actuator's real value.
  if(pol.has("max_target_dev_override"))
  {
    std::map<std::string, double> mtd_map = pol("max_target_dev_override");
    for(int i = 0; i < nbActuatedJoints; ++i) { updateIfExists(maxTargetDev_[i], mtd_map, jointNames[i]); }
  }

  maxVelX_          = config_("policies")[currentPolicyIndex]("max_vel_x",         0.6);
  maxVelY_          = config_("policies")[currentPolicyIndex]("max_vel_y",         0.4);
  maxYawCmd_        = config_("policies")[currentPolicyIndex]("max_yaw",           0.7);
  joystickDeadZone_ = config_("policies")[currentPolicyIndex]("joystick_deadzone", 0.05);
  velRampRate_      = config_("policies")[currentPolicyIndex]("vel_ramp_rate",     0.5);
  // Matches the training actuator's velocity_target_limit (rad/s): clamp on
  // the finite-difference velocity feedforward in byPassQPControl().
  velTargetLimit_   = config_("policies")[currentPolicyIndex]("vel_target_limit",  8.0);

  logImpactVel_          = config_("log_impact_vel", false);
  impactForceThreshold_  = config_("impact_force_threshold", 30.0);
  impactForceHysteresis_ = config_("impact_force_hysteresis_ratio", 0.5);

  // postureStiffness(), singular. There used to be a non-const overload
  // returning the bare 0.2/(policyDt*timeStep) formula, and because both call
  // sites live in non-const members it won overload resolution every time --
  // so posture_stiffness was dead everywhere it was written: the global key in
  // mc_rtc_superbuild.yaml and its mujoco overlay, the per-policy overrides,
  // and the GUI input. Every measurement taken "at 1600" was taken at 8000.

  // Sentinel -1: auto, i.e. whatever stiffness() just set as a side effect
  // (2*sqrt(K), TrajectoryTaskGeneric.cpp:227 -- critically damped). >= 0
  // overrides it. Read here, applied below and everywhere else stiffness()
  // is called (applyPostureMode(), the GUI slider) -- see
  // applyPostureDampingOverride()'s own comment for why it has to be
  // reapplied at every one of those, not just here.
  postureDampingOverride_ = config_("policies")[currentPolicyIndex]("posture_damping", -1.0);

  const double K = postureStiffness();
  auto pt = getPostureTask(robot().name());
  pt->stiffness(K);
  applyPostureDampingOverride(pt);
  mc_rtc::log::info("[NewRLQPController] useQP={} PostureTask stiffness={:.0f} damping={:.1f}", useQP_, K,
                    pt->damping());
}

void NewRLQPController::applyPostureDampingOverride(mc_tasks::PostureTaskPtr & pt)
{
  // stiffness(K) resets damping to 2*sqrt(K) as a side effect every time it
  // is called (TrajectoryTaskGeneric.cpp:227) -- initializeRobot(), the
  // non-passthrough branch of applyPostureMode(), and the "Stiffness K" GUI
  // slider all call it. Without reapplying the override right after each of
  // those, moving that slider (or a reset) would silently wipe out an active
  // posture_damping and nothing would say so -- the same class of bug as the
  // dead posture_stiffness override found earlier, avoided here by never
  // letting stiffness() be the last word on damping when an override is set.
  if(pt && postureDampingOverride_ >= 0.0) { pt->damping(postureDampingOverride_); }
}

double NewRLQPController::postureTaskStiffness()
{
  auto pt = getPostureTask(robot().name());
  return pt ? pt->stiffness() : 0.0;
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

  // Deep history for velocity_action (case 6, RHP7 -- see V3_DEEP_HISTORY_SIZE in the
  // header): seeded with the current state so the first observation has no
  // spurious jump.
  for(int i = 0; i < V3_DEEP_HISTORY_SIZE; ++i)
  {
    linVelDeep_[i]   = lv;
    angVelDeep_[i]   = av;
    projGravDeep_[i] = pg;
    jointPosDeep_[i] = jp;
    jointVelDeep_[i] = jv;
    jointActDeep_[i] = ja;
    velCmdDeep_[i]   = currentVelCmd_;
  }
  histInitializedV3Deep_ = true;
}

bool NewRLQPController::switchPolicy(size_t index)
{
  if(policyArmed_)
  {
    mc_rtc::log::error("[NewRLQPController] refusing to switch policy while ARMED -- disarm first");
    return false;
  }
  if(index >= policyPaths_.size())
  {
    mc_rtc::log::error("[NewRLQPController] policy index {} out of range (have {})", index,
                       policyPaths_.size());
    return false;
  }
  if(index == currentPolicyIndex)
  {
    mc_rtc::log::info("[NewRLQPController] policy {} already loaded", index);
    return true;
  }

  const size_t previous = currentPolicyIndex;
  // Hold the commanded posture and the operator's QP choice across the reload:
  // initializeRobot() re-reads use_QP from the policy block, and a switch is not
  // a reason to put the QP back in the loop behind the operator's back.
  const Eigen::VectorXd qHold = q_rl;
  const bool qpChoice = useQP_;

  currentPolicyIndex = index;
  try
  {
    initializeRobot();     // per-policy kp/kd/q0/action_scale/effort_limit/ratio/filter
    initializeRLPolicy();  // ONNX + mappings + observation buffers
  }
  catch(const std::exception & e)
  {
    mc_rtc::log::error("[NewRLQPController] policy {} failed to load ({}), reverting to {}", index,
                       e.what(), previous);
    currentPolicyIndex = previous;
    initializeRobot();
    initializeRLPolicy();
    useQP_ = qpChoice;
    q_rl = qHold;
    return false;
  }

  useQP_ = qpChoice;
  // Same clearing as reset(): the observation history is re-seeded by
  // initializeRLObservation() above. Carrying any of it over would mix two
  // different plants.
  q_rl = qHold;
  q_rl_prev_ = qHold;
  qdTarget_.setZero();
  ffInit_ = false;

  applyPostureMode();

  mc_rtc::log::success("[NewRLQPController] policy {} -> {} loaded ({}), QP {}, disarmed", previous,
                       index, policyPaths_[index], useQP_ ? "enforced" : "bypassed");
  return true;
}

int NewRLQPController::postureDofOffset() const
{
  return robot().mb().joint(0).type() == rbd::Joint::Free ? 6 : 0;
}

void NewRLQPController::applyPostureMode()
{
  auto pt = getPostureTask(robot().name());
  if(!pt) { return; }
  if(posturePassthrough_) { pt->stiffness(0.0); pt->damping(0.0); }
  else
  {
    if(postureRefAccelWritten_ && !postureFeedforward_)
    {
      const Eigen::VectorXd z = Eigen::VectorXd::Zero(robot().mb().nrDof() - postureDofOffset());
      pt->refAccel(z);
      pt->refVel(z);
      postureRefAccelWritten_ = false;
    }
    pt->stiffness(postureStiffness());
    applyPostureDampingOverride(pt);
  }
}

void NewRLQPController::setPostureRefVel(mc_tasks::PostureTaskPtr & pt)
{
  const auto & mb = robot().mb();
  const int off = postureDofOffset();
  const int n = mb.nrDof() - off;
  Eigen::VectorXd v = Eigen::VectorXd::Zero(n);
  for(int i = 0; i < nbActuatedJoints; ++i)
  {
    const int dof = mb.jointPosInDof(robot().jointIndexByName(jointNames[i])) - off;
    if(dof < 0 || dof >= n) { continue; }
    // Defense in depth, matching setPostureFeedforward's own clamp: qdTarget_
    // is already bounded upstream (the vel_target_limit_per_joint clamp in
    // utils.cpp's velocity_action branch), but nothing re-checks it here
    // otherwise.
    v(dof) = std::clamp(qdTarget_(i), -velTargetLimit_, velTargetLimit_);
  }
  pt->refVel(v);
  postureRefAccelWritten_ = true;
}

void NewRLQPController::setPostureRefAccel(mc_tasks::PostureTaskPtr & pt)
{
  const auto & mb = robot().mb();
  const auto & mbc = robot().mbc();
  const int off = postureDofOffset();
  // T floored at 3 ticks: the deadbeat has |lambda| 2.41 at T == dt (diverges),
  // 0.58 at 3 dt. On the robot timeStep and policy_step_size are both 0.005.
  const double T = std::max(policyStepSize, 3.0 * timeStep);
  Eigen::VectorXd a = Eigen::VectorXd::Zero(mb.nrDof() - off);
  for(int i = 0; i < nbActuatedJoints; ++i)
  {
    const int jIdx = robot().jointIndexByName(jointNames[i]);
    const int dof = mb.jointPosInDof(jIdx) - off; // index into qJoints, not nrDof
    if(dof < 0 || dof >= a.size()) { continue; }
    const double q = mbc.q[jIdx][0];
    const double dq = mbc.alpha[jIdx][0];
    const double acc = 2.0 * (q_rl(i) - q - dq * T) / (T * T);
    a(dof) = std::clamp(acc, -postureAccelMax_, postureAccelMax_);
  }
  pt->refAccel(a);
  postureRefAccelWritten_ = true;
}

void NewRLQPController::updatePostureFeedforward()
{
  if(!ffInit_)
  {
    ffPrevQ_ = q_rl;
    ffVel_.setZero();
    ffPrevVel_.setZero();
    ffAcc_.setZero();
    ffInit_ = true;
    return;
  }
  // Hold across substeps: q_rl only moves on a policy step, so differentiating
  // every tick would give one spike in policyStepSize/timeStep ticks and zeros
  // in between.
  if(q_rl == ffPrevQ_) { return; }
  const double dt = std::max(policyStepSize, timeStep);
  ffVel_     = (q_rl - ffPrevQ_) / dt;
  ffAcc_     = (ffVel_ - ffPrevVel_) / dt;
  ffPrevVel_ = ffVel_;
  ffPrevQ_   = q_rl;
}

void NewRLQPController::setPostureFeedforward(mc_tasks::PostureTaskPtr & pt)
{
  const auto & mb = robot().mb();
  const int off = postureDofOffset();
  const int n = mb.nrDof() - off;
  Eigen::VectorXd v = Eigen::VectorXd::Zero(n);
  Eigen::VectorXd a = Eigen::VectorXd::Zero(n);
  for(int i = 0; i < nbActuatedJoints; ++i)
  {
    const int dof = mb.jointPosInDof(robot().jointIndexByName(jointNames[i])) - off;
    if(dof < 0 || dof >= n) { continue; }
    // qd* itself, not a second derivation of it: qdTarget_ is refreshed every
    // policy step by the velocity_action clamp in utils.cpp (currentActionScaled
    // clamped to vel_target_limit_per_joint). ffVel_ re-derived the same q_rl
    // without that clamp, so the task would have been fed a different velocity
    // than the policy ever trained against -- the same kind of mismatch the
    // comment in setSimulationTargets() records having already fixed once for
    // mbc().alpha.
    //
    // The acceleration keeps its own derivation: training's PD has no
    // acceleration term at all, so there is no qd*-equivalent to match, and
    // refAccel here is purely mc_rtc-side feedforward.
    v(dof) = std::clamp(qdTarget_(i), -velTargetLimit_, velTargetLimit_);
    a(dof) = std::clamp(ffAcc_(i), -postureAccelMax_, postureAccelMax_);
  }
  pt->refVel(v);
  pt->refAccel(a);
  postureRefAccelWritten_ = true;
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

void NewRLQPController::updateImpactVelocity()
{
  // realRobot(), not robot(): a force reading only means something against
  // the measured plant. robot() is the QP's own kinematic plan (see the
  // ctrl_q/ctrl_alpha comment above) and carries no contact information of
  // its own.
  auto & rr = realRobot(robots()[0].name());

  auto processFoot = [&](const std::string & sensorName, const std::string & bodyName, bool & contact,
                         double & forceZ, Eigen::Vector3d & impactVel)
  {
    if(!rr.hasForceSensor(sensorName)) return;
    forceZ = rr.forceSensor(sensorName).wrench().force().z();

    if(!contact && forceZ > impactForceThreshold_)
    {
      // Rising edge: touchdown. Body velocity read this same tick, i.e. at
      // (or a control period after) the instant the force crossed the
      // threshold -- the closest available approximation of the pre-impact
      // velocity without sub-stepping the detection.
      contact = true;
      impactVel = rr.bodyVelW(bodyName).linear();
    }
    else if(contact && forceZ < impactForceHysteresis_ * impactForceThreshold_)
    {
      // Falling edge, with hysteresis so noise around the threshold mid-stance
      // does not flip contact_ back and forth and refire the impact.
      contact = false;
    }
  };

  processFoot("LeftFootForceSensor", "L_ANKLE_P_LINK", leftFootContact_, leftFootForceZ_, leftFootImpactVel_);
  processFoot("RightFootForceSensor", "R_ANKLE_P_LINK", rightFootContact_, rightFootForceZ_, rightFootImpactVel_);
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
    //
    // qd* already exists: it is refreshed every policy step by the
    // velocity_action clamp in utils.cpp (currentActionScaled clamped to
    // vel_target_limit_per_joint). Use it rather than re-deriving one here.
    // The old derivation differed three ways from what training assumes:
    // dt was timeStep (1 ms in mc_mujoco, 5x too large, and a one-tick-in-five
    // spike since q_rl only moves on a policy step), no EMA, and a legacy
    // scalar clamp instead of the per-joint one.
    if(velTargetFilterAlpha_ > 0.0) { robot().mbc().alpha[idx][0] = qdTarget_(i); }
    else
    {
      const double alphaRaw = (q_rl(i) - q_rl_prev_(i)) / timeStep;
      robot().mbc().alpha[idx][0] =
          std::max(-velTargetLimit_, std::min(velTargetLimit_, alphaRaw));
    }
  }
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

bool NewRLQPController::isSimulated() const
{
  // See the doc comment on the declaration (NewRLQPController.h) for why this
  // check exists and why it is not cached.
  return datastore().has(robot().name() + "::SetPosW");
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
  logger().addLogEntry("NewRLQPController_qdTarget",         [this]() { return qdTarget_; });
  // What mc_mujoco's PD actually receives. The stock qOut/alphaOut entries
  // read outputRobot() (the canonical robot), NOT the control robot mujoco
  // reads -- in bypass they showed alpha=0.0003 while the commanded value was
  // -0.044, which is what made the QP-vs-bypass difference invisible in the
  // 2026-08-07 logs. These two read the same mbc() mujoco does.
  logger().addLogEntry("NewRLQPController_ctrl_q", [this]() -> Eigen::VectorXd {
    Eigen::VectorXd v = Eigen::VectorXd::Zero(nbActuatedJoints);
    for(int i = 0; i < nbActuatedJoints; ++i)
      v(i) = robot().mbc().q[robot().jointIndexByName(jointNames[i])][0];
    return v;
  });
  logger().addLogEntry("NewRLQPController_ctrl_alpha", [this]() -> Eigen::VectorXd {
    Eigen::VectorXd v = Eigen::VectorXd::Zero(nbActuatedJoints);
    for(int i = 0; i < nbActuatedJoints; ++i)
      v(i) = robot().mbc().alpha[robot().jointIndexByName(jointNames[i])][0];
    return v;
  });
  logger().addLogEntry("NewRLQPController_RL_currentObservation", [this]() { return currentObservation; });
  logger().addLogEntry("NewRLQPController_RL_currentAction", [this]() { return currentAction; });
  logger().addLogEntry("NewRLQPController_RL_actionScale",   [this]() { return actionScale; });
  logger().addLogEntry("NewRLQPController_useQP",            [this]() { return useQP_; });
  logger().addLogEntry("NewRLQPController_qpZeroVelOut",     [this]() { return qpZeroVelOut_; });
  logger().addLogEntry("NewRLQPController_velCmd",           [this]() { return currentVelCmd_; });

  if(logImpactVel_)
  {
    // Raw Fz logged unconditionally: needed to pick impactForceThreshold_ in
    // the first place, from a run with the default value.
    logger().addLogEntry("NewRLQPController_leftFootForceZ",   [this]() { return leftFootForceZ_; });
    logger().addLogEntry("NewRLQPController_rightFootForceZ",  [this]() { return rightFootForceZ_; });
    logger().addLogEntry("NewRLQPController_leftFootContact",  [this]() { return leftFootContact_; });
    logger().addLogEntry("NewRLQPController_rightFootContact", [this]() { return rightFootContact_; });
    // World-frame linear velocity of the ankle body, captured on the tick of
    // the last rising edge. Held between impacts -- find the value at each
    // touchdown by reading it where *FootContact_ steps 0 -> 1.
    logger().addLogEntry("NewRLQPController_leftFootImpactVel",  [this]() { return leftFootImpactVel_; });
    logger().addLogEntry("NewRLQPController_rightFootImpactVel", [this]() { return rightFootImpactVel_; });
  }

  // Ce que les drives mesurent, en amperes : sur RHPS1 le "tau" de
  // RobotHardware est un courant, gearRatio et torqueConst valant 1 dans le
  // VRML. Journalise tel quel, sans conversion, pour pouvoir le comparer
  // directement aux limites CL/PL des drives.
  // tauIn de mc_rtc est laisse en place (donnee brute, indexee sur le
  // refJointOrder complet). Les entrees ci-dessous la doublent avec un nom
  // explicite et une unite, une par joint pour etre lisibles dans mc_log_ui.
  logger().addLogEntry("NewRLQPController_joint_current_A", [this]() -> const Eigen::VectorXd &
  {
    const auto & cur = robot().jointTorques();
    for(int i = 0; i < nbActuatedJoints; ++i)
    {
      const int k = refIdx_[i];
      jointCurrentA_(i) = (k >= 0 && k < static_cast<int>(cur.size())) ? cur[k] : 0.0;
    }
    return jointCurrentA_;
  });

  // Le meme signal converti en N.m, la ou on sait le faire : tau = I * N * Kt.
  // Reste a 0 sur les joints absents de joint_torque_scale (paires
  // differentielles, articulations a verins).
  //
  // Sauf sous mc_mujoco (isSimulated(), cf. son commentaire) : le "courant"
  // qu'il publie sur jointTorques() EST deja le couple en N.m, donc y
  // appliquer jointTorqueScale_ double-convertirait et gonflerait le log
  // d'un facteur ~10-20x selon le joint, sans rapport avec le couple reel
  // applique/clampe.
  logger().addLogEntry("NewRLQPController_joint_torque_Nm", [this]() -> const Eigen::VectorXd &
  {
    const auto & cur = robot().jointTorques();
    const bool sim = isSimulated();
    for(int i = 0; i < nbActuatedJoints; ++i)
    {
      const int k = refIdx_[i];
      const double s = sim ? 1.0 : jointTorqueScale_(i);
      jointTorqueNm_(i) = (k >= 0 && k < static_cast<int>(cur.size())) ? cur[k] * s : 0.0;
    }
    return jointTorqueNm_;
  });

  // Une entree par joint, nommee, pour pouvoir la retrouver dans mc_log_ui
  // sans compter les indices. current_* existe pour les 30 joints ; torque_*
  // seulement pour ceux dont on connait N*Kt, sinon la courbe serait un zero
  // trompeur.
  for(int i = 0; i < nbActuatedJoints; ++i)
  {
    logger().addLogEntry("current_" + jointNames[i], [this, i]() -> double
    {
      const auto & cur = robot().jointTorques();
      const int k = refIdx_[i];
      return (k >= 0 && k < static_cast<int>(cur.size())) ? cur[k] : 0.0;
    });
    if(jointTorqueScale_(i) != 0.0)
    {
      // isSimulated(): see its doc -- mc_mujoco already reports N.m here, so
      // jointTorqueScale_ (the real robot's current->torque conversion) must
      // not be applied a second time under mc_mujoco.
      logger().addLogEntry("torque_" + jointNames[i], [this, i]() -> double
      {
        const auto & cur = robot().jointTorques();
        const int k = refIdx_[i];
        const double s = isSimulated() ? 1.0 : jointTorqueScale_(i);
        return (k >= 0 && k < static_cast<int>(cur.size())) ? cur[k] * s : 0.0;
      });
    }
  }
  // NewRLQPController_rawTorqueRatio(Max) used to log what the policy would
  // demand of each joint if nothing clipped it (units of effort_limit),
  // computed by updateRawTorqueRatio() for position_action. Removed
  // 2026-09-04 along with that function: RHP7's only case is
  // velocity_action, which never fed it -- these entries always logged 0
  // on this branch already.
  // Per-joint gap between the RL target and the measured position (refJointOrder).
  // With QP on, a joint whose gap grows/saturates is being clamped by a QP
  // constraint (joint-limit/velocity damper or collision damper).
  //
  // refIdx_, pas i : encoderValues() est indexe sur le refJointOrder COMPLET,
  // ou le module RHPS1 insere L_HAND et l-one-finger entre le poignet gauche et
  // l'epaule droite (rhps1.cpp, construction de _ref_joint_order). jointNames
  // les a filtres, donc a partir de l'index 23 tout le bras droit se comparait
  // au voisin -- meme decalage que celui corrige sur les courants en f82bfba.
  //
  // Ce n'etait pas visible en simulation : RHPS1_MuJoCo ne liste que les 30
  // joints pilotes, les deux ordres coincident et l'erreur est nulle. Sur le
  // robot le run du 2026-08-06 rapportait R_ELBOW_P a -37.29 deg bloque a son
  // maximum 100 % du temps, ce qui a ete lu comme une anomalie du bras droit.
  logger().addLogEntry("NewRLQPController_RL_q_tracking_error", [this]() {
    const auto & enc = robot().encoderValues();
    Eigen::VectorXd err = q_rl;
    for(int i = 0; i < nbActuatedJoints; ++i)
    {
      const int k = i < static_cast<int>(refIdx_.size()) ? refIdx_[i] : -1;
      if(k >= 0 && k < static_cast<int>(enc.size())) { err(i) -= enc[k]; }
      else { err(i) = 0.0; }
    }
    return err;
  });
}

void NewRLQPController::addGui()
{
  gui()->addElement({"NewRLQPController", "Policy"},
    // Switching is refused while ARMED, so this is safe to leave in the GUI.
    mc_rtc::gui::IntegerInput("Policy index (disarm to change)",
                              [this]() { return static_cast<int>(currentPolicyIndex); },
                              [this](int i) { if(i >= 0) switchPolicy(static_cast<size_t>(i)); }),
    mc_rtc::gui::Label("Policies available", [this]() { return std::to_string(policyPaths_.size()); }),
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
        postureStiffnessOverride_ = v;
        auto pt = getPostureTask(robot().name());
        if(pt)
        {
          pt->stiffness(v);
          // Without this, moving K would silently reset damping to
          // 2*sqrt(K) and drop whatever posture_damping/the Damping slider
          // below had set -- see applyPostureDampingOverride()'s comment.
          applyPostureDampingOverride(pt);
        }
        mc_rtc::log::warning("[NewRLQPController] posture stiffness K = {} (GUI override)", v);
      }),
    // Sentinel -1 (postureDampingOverride_) shows as whatever stiffness()
    // last derived automatically (2*sqrt(K)); any value entered here is
    // remembered and reapplied every time K changes, from any source.
    mc_rtc::gui::NumberInput("Damping",
      [this]() {
        auto pt = getPostureTask(robot().name());
        return pt ? pt->damping() : 0.0;
      },
      [this](double v) {
        postureDampingOverride_ = v;
        auto pt = getPostureTask(robot().name());
        if(pt) pt->damping(v);
      })
  );

  gui()->addElement({"ControlMode"},
    mc_rtc::gui::Button("Toggle QP Control",       [this]() { useQP_ = !useQP_; }),
    mc_rtc::gui::Button("Toggle posture pass-through", [this]() {
      posturePassthrough_ = !posturePassthrough_;
      applyPostureMode();
      mc_rtc::log::warning("[NewRLQPController] posture {}",
                           posturePassthrough_ ? "PASS-THROUGH" : "2nd-order task");
    }),
    mc_rtc::gui::Button("Toggle posture feedforward", [this]() {
      postureFeedforward_ = !postureFeedforward_;
      ffInit_ = false;
      applyPostureMode();
      mc_rtc::log::warning("[NewRLQPController] posture feedforward {}",
                           postureFeedforward_ ? "ON" : "OFF");
    }),
    mc_rtc::gui::Label("Posture mode", [this]() {
      return posturePassthrough_ ? "pass-through (refAccel)"
             : postureFeedforward_ ? "2nd-order task + feedforward"
                                   : "2nd-order task"; }),
    mc_rtc::gui::Label("QP Control",               [this]() { return useQP_ ? "Enforced" : "Bypassed"; }),
    // Experimental: see qpZeroVelOut_'s doc comment in the header. Turning it
    // off restores the true velocity immediately rather than waiting for the
    // next tick's (now-skipped) restoreQPVelocity() call, so the QP's state
    // never sits on a stale zero longer than the tick that is already in flight.
    mc_rtc::gui::Button("Toggle QP zero-vel-out", [this]() {
      qpZeroVelOut_ = !qpZeroVelOut_;
      if(!qpZeroVelOut_) { restoreQPVelocity(); }
      mc_rtc::log::warning("[NewRLQPController] QP velocity output to mc_mujoco: {}",
                           qpZeroVelOut_ ? "ZEROED (position only)" : "restored (position + velocity)");
    }),
    mc_rtc::gui::Label("QP vel-out", [this]() { return qpZeroVelOut_ ? "zeroed" : "normal"; }),
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
