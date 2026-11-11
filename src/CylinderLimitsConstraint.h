#pragma once

#include <mc_rbdyn/CompoundJointConstraintDescription.h>
#include <mc_rbdyn/Robots.h>
#include <mc_rtc/logging.h>
#include <mc_solver/ConstraintSet.h>
#include <mc_solver/TVMQPSolver.h>
#include <mc_solver/TasksQPSolver.h>
#include <mc_tvm/Robot.h>

#include <Tasks/QPSolver.h>

#include <tvm/function/BasicLinearFunction.h>
#include <tvm/requirements/PriorityLevel.h>
#include <tvm/task_dynamics/VelocityDamper.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <vector>

/** Half-planes on two joints, g(q) = a q1 + b q2 + c <= 0 with (a, b) unit so g is the signed distance to the line
 * [rad], plus a velocity damper: the QP starts braking di before the line and stops ds short of it, like the joint
 * limits. Built from mc_rtc CompoundJointConstraintDescription ((q - p1) x (p2 - p1) <= 0).
 *
 * Both mc_rtc backends:
 *  - TVM: tvm::task_dynamics::VelocityDamper on the position-level function (same CBF as KinematicsConstraint).
 *  - Tasks: first-order damper with the formula of tasks::qp::DamperJointLimitsConstr.
 * Not mc_solver::CompoundJointConstraint: it has no damper (one-step bound, brakes at the last tick), and under TVM
 * it ignores the descriptions passed to it and mc_tvm::CompoundJointFunction::dt() throws. */
struct CylinderLimitsConstraint : public mc_solver::ConstraintSet
{
  struct Damper
  {
    double diPercent;  // interaction distance, fraction of the smaller joint range
    double dsPercent;  // safety distance, same
    double xsiOff;     // Tasks: damping offset
    double m;          // TVM: overdamping (CBF when >= 1)
    double lambda;     // TVM: CBF gain
  };

  CylinderLimitsConstraint(const mc_rbdyn::Robots & robots,
                           unsigned int rIndex,
                           double dt,
                           const mc_rbdyn::CompoundJointConstraintDescriptionVector & descs,
                           const Damper & damper)
  : robots_(robots), rIndex_(rIndex), dt_(dt), damper_(damper)
  {
    const auto & robot = robots.robot(rIndex);
    for(const auto & d : descs)
    {
      for(const auto & j : {d.j1, d.j2})
      {
        if(!robot.hasJoint(j) || robot.mb().joint(static_cast<int>(robot.jointIndexByName(j))).dof() != 1)
        {
          mc_rtc::log::error_and_throw("[CylinderLimitsConstraint] {} is not a 1-dof joint of {}", j, robot.name());
        }
      }
      Line l;
      l.q1 = robot.jointIndexByName(d.j1);
      l.q2 = robot.jointIndexByName(d.j2);
      const Eigen::Vector2d P = d.p2 - d.p1;
      const double n = P.norm();
      l.a = P.y() / n;
      l.b = -P.x() / n;
      l.c = (d.p1.y() * P.x() - d.p1.x() * P.y()) / n;
      const double range = std::min(robot.qu()[l.q1][0] - robot.ql()[l.q1][0], robot.qu()[l.q2][0] - robot.ql()[l.q2][0]);
      l.di = damper.diPercent * range;
      l.ds = damper.dsPercent * range;
      lines_.push_back(l);
    }
  }

  size_t size() const noexcept { return lines_.size(); }

protected:
  void addToSolverImpl(mc_solver::QPSolver & solver) override
  {
    switch(backend_)
    {
      case mc_solver::QPSolver::Backend::Tasks:
        tasks_ = std::make_unique<TasksConstr>(robots_, rIndex_, dt_, lines_, damper_.xsiOff);
        tasks_->addToSolver(solver.robots().mbs(), mc_solver::tasks_solver(solver).solver());
        break;
      case mc_solver::QPSolver::Backend::TVM:
      {
        auto & problem = mc_solver::tvm_solver(solver).problem();
        auto & tvmRobot = robots_.robot(rIndex_).tvmRobot();
        for(const auto & l : lines_)
        {
          Eigen::MatrixXd A1(1, 1), A2(1, 1);
          Eigen::VectorXd c(1);
          A1 << l.a;
          A2 << l.b;
          c << l.c;
          std::vector<tvm::MatrixConstRef> A{A1, A2};
          auto f = std::make_shared<tvm::function::BasicLinearFunction>(
              A, std::vector<tvm::VariablePtr>{tvmRobot.qJoint(l.q1), tvmRobot.qJoint(l.q2)}, c);
          const tvm::task_dynamics::VelocityDamper::Config cfg(l.di, l.ds, 0.0, 0.0, damper_.m, damper_.lambda);
          tvm_.push_back(problem.add(f <= 0., tvm::task_dynamics::VelocityDamper(dt_, cfg),
                                     {tvm::requirements::PriorityLevel(0)}));
        }
        break;
      }
      default:
        mc_rtc::log::error_and_throw("[CylinderLimitsConstraint] unsupported solver backend");
    }
  }

  void removeFromSolverImpl(mc_solver::QPSolver & solver) override
  {
    switch(backend_)
    {
      case mc_solver::QPSolver::Backend::Tasks:
        if(tasks_) { tasks_->removeFromSolver(mc_solver::tasks_solver(solver).solver()); }
        tasks_.reset();
        break;
      case mc_solver::QPSolver::Backend::TVM:
        for(const auto & t : tvm_) { mc_solver::tvm_solver(solver).problem().remove(*t); }
        tvm_.clear();
        break;
      default:
        break;
    }
  }

private:
  struct Line
  {
    size_t q1, q2;  // mbc joint indices
    double a, b, c; // g(q) = a q1 + b q2 + c, (a, b) unit
    double di, ds;  // damper distances [rad]
  };

  /** Tasks backend: a q1dd + b q2dd <= ub, the tightest of
   *  - the line one tick ahead: g + dt gd + dt^2/2 (a q1dd + b q2dd) <= 0
   *  - inside di, the damper on the distance d = -g: d_dot(next) >= -xi (d - ds) / (di - ds), xi set on entry
   *    from the approach speed (tasks::qp::DamperJointLimitsConstr). */
  struct TasksConstr : public tasks::qp::ConstraintFunction<tasks::qp::Inequality>
  {
    TasksConstr(const mc_rbdyn::Robots & robots, unsigned int rIndex, double dt, std::vector<Line> lines, double xsiOff)
    : rIndex_(rIndex), dt_(dt), xsiOff_(xsiOff), lines_(std::move(lines)), in_(lines_.size(), false),
      xi_(lines_.size(), 0.0)
    {
      const auto & mb = robots.robot(rIndex).mb();
      for(const auto & l : lines_)
      {
        dof1_.push_back(mb.jointPosInDof(static_cast<int>(l.q1)));
        dof2_.push_back(mb.jointPosInDof(static_cast<int>(l.q2)));
      }
    }

    void updateNrVars(const std::vector<rbd::MultiBody> &, const tasks::qp::SolverData & data) override
    {
      const int n = static_cast<int>(lines_.size());
      A_.setZero(n, data.nrVars());
      b_.setZero(n);
      const int begin = data.alphaDBegin(static_cast<int>(rIndex_));
      for(int i = 0; i < n; ++i)
      {
        A_(i, begin + dof1_[i]) = lines_[i].a;
        A_(i, begin + dof2_[i]) = lines_[i].b;
      }
    }

    void update(const std::vector<rbd::MultiBody> &,
                const std::vector<rbd::MultiBodyConfig> & mbcs,
                const tasks::qp::SolverData &) override
    {
      const auto & mbc = mbcs[rIndex_];
      for(size_t i = 0; i < lines_.size(); ++i)
      {
        const auto & l = lines_[i];
        const double g = l.a * mbc.q[l.q1][0] + l.b * mbc.q[l.q2][0] + l.c;
        const double gd = l.a * mbc.alpha[l.q1][0] + l.b * mbc.alpha[l.q2][0];
        double ub = -2.0 * (g + dt_ * gd) / (dt_ * dt_);
        const double d = -g;
        if(d < l.di)
        {
          if(!in_[i])
          {
            xi_[i] = std::abs((l.di - l.ds) / std::max(d - l.ds, 1e-6) * gd) + xsiOff_;
            in_[i] = true;
          }
          const double damper = xi_[i] * (d - l.ds) / (l.di - l.ds);
          ub = std::min(ub, (damper - gd) / dt_);
        }
        else { in_[i] = false; }
        b_(static_cast<int>(i)) = ub;
      }
    }

    const Eigen::MatrixXd & AInEq() const override { return A_; }
    const Eigen::VectorXd & bInEq() const override { return b_; }
    int maxInEq() const override { return static_cast<int>(lines_.size()); }
    std::string nameInEq() const override { return "CylinderLimitsConstraint"; }
    std::string descInEq(const std::vector<rbd::MultiBody> &, int i) override
    {
      return "cylinder limit half-plane " + std::to_string(i);
    }

    unsigned int rIndex_;
    double dt_, xsiOff_;
    std::vector<Line> lines_;
    std::vector<int> dof1_, dof2_;
    std::vector<bool> in_;
    std::vector<double> xi_;
    Eigen::MatrixXd A_;
    Eigen::VectorXd b_;
  };

  const mc_rbdyn::Robots & robots_;
  unsigned int rIndex_;
  double dt_;
  Damper damper_;
  std::vector<Line> lines_;
  std::unique_ptr<TasksConstr> tasks_;
  std::vector<tvm::TaskWithRequirementsPtr> tvm_;
};
