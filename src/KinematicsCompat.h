#pragma once

#include <mc_rbdyn/Robots.h>
#include <mc_rtc/logging.h>
#include <mc_solver/ConstraintSet.h>
#include <mc_solver/KinematicsConstraint.h>
#include <mc_solver/TVMQPSolver.h>
#include <mc_tvm/Robot.h>

#include <tvm/ControlProblem.h>
#include <tvm/requirements/PriorityLevel.h>
#include <tvm/task_dynamics/Proportional.h>

#include <array>
#include <utility>

/** The CBF joint-limit constructor differs between the two mc_rtc forks in use.
 *  - bastien-muraccioli (superbuild): (robots, index, {di, ds, offset, m, lambda}, vp); when m >= 1 the velocity
 *    limits use Proportional(lambda).
 *  - mathieu-celerier (rhps1c, mc_rtc 2.14): (robots, index, dt, {di, ds, offset}, {m, lambda}, vp); same tvm
 *    VelocityDamper on the joint limits, but the velocity limits keep Proportional(1/dt).
 * makeCbfKinematics picks whichever exists; CbfVelocityLimits restores Proportional(lambda) on the second. */
namespace rlqp_compat
{

template<typename KC>
auto makeCbfKinematics(const mc_rbdyn::Robots & robots, double, const std::array<double, 5> & d, double vp, int)
    -> decltype(new KC(robots, 0u, d, vp))
{
  return new KC(robots, 0u, d, vp);
}

template<typename KC>
KC * makeCbfKinematics(const mc_rbdyn::Robots & robots, double dt, const std::array<double, 5> & d, double vp, long)
{
  return new KC(robots, 0u, dt, std::array<double, 3>{{d[0], d[1], d[2]}}, std::array<double, 2>{{d[3], d[4]}}, vp);
}

template<typename KC>
constexpr auto velocityRateBuiltIn(int)
    -> decltype(new KC(std::declval<const mc_rbdyn::Robots &>(), 0u, std::array<double, 5>{}, 0.5), bool())
{
  return true;
}

template<typename KC>
constexpr bool velocityRateBuiltIn(long)
{
  return false;
}

constexpr bool kVelocityRateBuiltIn = velocityRateBuiltIn<mc_solver::KinematicsConstraint>(0);

/** vp * vl <= dq <= vp * vu with Proportional(lambda), TVM only. Added on top of the fork's Proportional(1/dt) bound,
 *  so the tighter one applies: lambda below the limit, 1/dt above it. */
struct CbfVelocityLimits : public mc_solver::ConstraintSet
{
  CbfVelocityLimits(const mc_rbdyn::Robot & robot, double vp, double lambda) : robot_(robot), vp_(vp), lambda_(lambda) {}

  void addToSolverImpl(mc_solver::QPSolver & solver) override
  {
    if(solver.backend() != mc_solver::QPSolver::Backend::TVM)
    {
      mc_rtc::log::error_and_throw("[CbfVelocityLimits] TVM backend only");
    }
    auto & r = robot_.tvmRobot();
    const auto start = r.qFloatingBase()->space().tSize();
    const auto n = r.qJoints()->space().tSize();
    Eigen::VectorXd vl = r.limits().vl.segment(start, n) * vp_;
    Eigen::VectorXd vu = r.limits().vu.segment(start, n) * vp_;
    c_ = mc_solver::tvm_solver(solver).problem().add(vl <= tvm::dot(r.qJoints()) <= vu,
                                                     tvm::task_dynamics::Proportional(lambda_),
                                                     {tvm::requirements::PriorityLevel(0)});
  }

  void removeFromSolverImpl(mc_solver::QPSolver & solver) override
  {
    if(c_) { mc_solver::tvm_solver(solver).problem().remove(*c_); }
    c_.reset();
  }

private:
  const mc_rbdyn::Robot & robot_;
  double vp_;
  double lambda_;
  tvm::TaskWithRequirementsPtr c_;
};

} // namespace rlqp_compat
