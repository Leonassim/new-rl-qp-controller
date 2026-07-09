#include "NewRLQPController_Initial.h"

#include "../NewRLQPController.h"

void NewRLQPController_Initial::configure(const mc_rtc::Configuration & config) {}

void NewRLQPController_Initial::start(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<NewRLQPController &>(ctl_);
  ctl.utilsClass.start_rl_state(ctl, "RL_State");
}

bool NewRLQPController_Initial::run(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<NewRLQPController &>(ctl_);
  ctl.utilsClass.run_rl_state(ctl);

  auto pt = ctl.getPostureTask(ctl.robot().name());
  auto posture = pt->posture();
  for(int i = 0; i < ctl.nbActuatedJoints; ++i)
  {
    const int mcIdx = ctl.robot().jointIndexByName(ctl.jointNames[i]);
    posture[mcIdx][0] = ctl.q_rl[i];
  }
  pt->posture(posture);
  return false;
}

void NewRLQPController_Initial::teardown(mc_control::fsm::Controller & ctl_)
{
  auto & ctl = static_cast<NewRLQPController &>(ctl_);
}

EXPORT_SINGLE_STATE("NewRLQPController_Initial", NewRLQPController_Initial)
